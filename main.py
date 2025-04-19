from torchvision import transforms, utils
from torchvision.utils import save_image
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn.functional as F
from random import randint
import torch.nn as nn
from utils import Config, create_ckpt_dir, save_ckpt, load_ckpt, to_items
from PIL import Image
from dataset import InitDataset
from glob import glob
import oyaml as yaml
import numpy as np
import datetime
import random
import torch
from pconv import PConvLayer, PartialConvolution
from loss import InpaintingLoss, VGG16FeatureExtractor

def evaluate(model, dataset, device, filename):
    print('Start the evaluation')
    model.eval()

    # Get a mini-batch from the dataset
    image, mask, gt = zip(*[dataset[i] for i in range(8)])
    image = torch.stack(image).to(device)
    mask = torch.stack(mask).to(device)
    gt = torch.stack(gt).to(device)

    with torch.no_grad():
        output, _ = model(image, mask)

    # Move outputs to CPU
    image = image.cpu()
    mask = mask.cpu()
    gt = gt.cpu()
    output = output.cpu()
    output_comp = mask * image + (1 - mask) * output

    # Concatenate and save the visualization grid
    grid = make_grid(torch.cat([image, mask, output, output_comp, gt], dim=0), nrow=8, normalize=True)
    save_image(grid, filename)
    print(f"Saved evaluation result to {filename}")

class PConvUNet(nn.Module):
    def __init__(self, finetune=False, in_ch=3, layer_size=6):
        super().__init__()
        self.freeze_enc_bn = True if finetune else False
        self.layer_size = layer_size

        self.enc_1 = PConvLayer(in_ch, 64, 'down-7', bn=False)
        self.enc_2 = PConvLayer(64, 128, 'down-5')
        self.enc_3 = PConvLayer(128, 256, 'down-5')
        self.enc_4 = PConvLayer(256, 512, 'down-3')
        self.enc_5 = PConvLayer(512, 512, 'down-3')
        self.enc_6 = PConvLayer(512, 512, 'down-3')
        self.enc_7 = PConvLayer(512, 512, 'down-3')
        self.enc_8 = PConvLayer(512, 512, 'down-3')

        self.dec_8 = PConvLayer(512 + 512, 512, dec=True, active='leaky')
        self.dec_7 = PConvLayer(512 + 512, 512, dec=True, active='leaky')
        self.dec_6 = PConvLayer(512 + 512, 512, dec=True, active='leaky')
        self.dec_5 = PConvLayer(512 + 512, 512, dec=True, active='leaky')
        self.dec_4 = PConvLayer(512 + 256, 256, dec=True, active='leaky')
        self.dec_3 = PConvLayer(256 + 128, 128, dec=True, active='leaky')
        self.dec_2 = PConvLayer(128 + 64,   64, dec=True, active='leaky')
        self.dec_1 = PConvLayer(64 + 3,      3, dec=True, bn=False,
                                active=None, conv_bias=True)

    def forward(self, img, mask):
        enc_f, enc_m = [img], [mask]
        for layer_num in range(1, self.layer_size+1):
            if layer_num == 1:
                feature, update_mask = \
                    getattr(self, 'enc_{}'.format(layer_num))(img, mask)
            else:
                enc_f.append(feature)
                enc_m.append(update_mask)
                feature, update_mask = \
                    getattr(self, 'enc_{}'.format(layer_num))(feature,
                                                              update_mask)

        assert len(enc_f) == self.layer_size

        for layer_num in reversed(range(1, self.layer_size+1)):
            feature, update_mask = getattr(self, 'dec_{}'.format(layer_num))(
                    feature, update_mask, enc_f.pop(), enc_m.pop())

        return feature, mask

    def train(self, mode=True):
        super().train(mode)
        if not self.freeze_enc_bn:
            return 
        for name, module in self.named_modules():
            if isinstance(module, nn.BatchNorm2d) and 'enc' in name:
                module.eval()

class Trainer(object):
    def __init__(self, step, config, device, model, dataset_train,
                 dataset_val, criterion, optimizer):
        self.stepped = step
        self.config = config
        self.device = device
        self.model = model
        self.dataloader_train = DataLoader(dataset_train,
                                           batch_size=config.batch_size,
                                           shuffle=True)
        self.dataset_val = dataset_val
        self.criterion = criterion
        self.optimizer = optimizer
        self.evaluate = evaluate

    def iterate(self):
        print('Start the training')
        for step, (input, mask, gt) in enumerate(self.dataloader_train):
            loss_dict = self.train(step+self.stepped, input, mask, gt)
            # report the loss
            if step % self.config.log_interval == 0:
                self.report(step+self.stepped, loss_dict)

            # evaluation
            if (step+self.stepped + 1) % self.config.vis_interval == 0 \
                    or step == 0 or step + self.stepped == 0:
                
                self.model.eval()
                
                self.evaluate(self.model, self.dataset_val, self.device,
                              '{}/val_vis/{}.png'.format(self.config.ckpt,
                                                         step+self.stepped))

            # save the model
            if (step+self.stepped + 1) % self.config.save_model_interval == 0 \
                    or (step + 1) == self.config.max_iter:
                print('Saving the model...')
                save_ckpt('{}/models/{}.pth'.format(self.config.ckpt,
                                                    step+self.stepped + 1),
                          [('model', self.model)],
                          [('optimizer', self.optimizer)],
                          step+self.stepped + 1)

            if step >= self.config.max_iter:
                break

            

    def train(self, step, input, mask, gt):
        
        self.model.train()

        input = input.to(self.device)
        mask = mask.to(self.device)
        gt = gt.to(self.device)

        # forward
        output, _ = self.model(input, mask)
        loss_dict = self.criterion(input, mask, output, gt)
        loss = 0.0
        for key, val in loss_dict.items():
            coef = getattr(self.config, '{}_coef'.format(key))
            loss += coef * val

        # updates the model's params
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        loss_dict['total'] = loss
        return to_items(loss_dict)

    def report(self, step, loss_dict):
        print('[STEP: {:>6}] | Valid Loss: {:.6f} | Hole Loss: {:.6f}'\
              '| TV Loss: {:.6f} | Perc Loss: {:.6f}'\
              '| Style Loss: {:.6f} | Total Loss: {:.6f}'.format(
                        step, loss_dict['valid'], loss_dict['hole'],
                        loss_dict['tv'], loss_dict['perc'],
                        loss_dict['style'], loss_dict['total']))
        

config = Config("./config.yml")
config.ckpt = create_ckpt_dir()
print("Check Point is '{}'".format(config.ckpt))


device = torch.device("cuda:{}".format(config.cuda_id)
                      if torch.cuda.is_available() else "cpu")


print("Loading the Model...")
model = PConvUNet(finetune=config.finetune,
                  layer_size=config.layer_size)

if config.finetune:
    model.load_state_dict(torch.load(config.finetune)['model'])
model.to(device)

# Data Transformation
img_tf = transforms.Compose([
            transforms.ToTensor()
            ])

mask_tf = transforms.Compose([
            transforms.RandomResizedCrop(256),
            transforms.ToTensor()
            ])


print("Loading the Validation Dataset...")
dataset_val = InitDataset(config.data_root,
                      img_tf,
                      mask_tf,
                      data="test")

print("Loading the Training Dataset...")
dataset_train = InitDataset(config.data_root,
                        img_tf,
                        mask_tf,
                        data="train")

# Loss fucntion
criterion = InpaintingLoss(VGG16FeatureExtractor(),
                            tv_loss=config.tv_loss).to(device)
# Optimizer
lr = config.finetune_lr if config.finetune else config.initial_lr
if config.optim == "Adam":
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,
                                        model.parameters()),
                                    lr=lr,
                                    weight_decay=config.weight_decay)
elif config.optim == "SGD":
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad,
                                        model.parameters()),
                                lr=lr,
                                momentum=config.momentum,
                                weight_decay=config.weight_decay)

start_iter = 0
trainer = Trainer(start_iter, config, device, model, dataset_train,
                    dataset_val, criterion, optimizer)
trainer.iterate()