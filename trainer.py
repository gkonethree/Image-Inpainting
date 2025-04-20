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
        self.losses = []

    def iterate(self):
        print('Start the training')
        last_saved_model = 0
        iters = self.stepped + 1
        while(iters < self.config.max_iter):
            for step, (input, mask, gt) in enumerate(self.dataloader_train):
                iters += 1
                if iters > self.config.max_iter:
                    break
                loss_dict = self.train(step + self.stepped, input, mask, gt)
                self.losses.append(loss_dict)
                # report the loss
                if step % self.config.log_interval == 0:
                    self.report(step + self.stepped, loss_dict)

                # evaluation
                if (step + self.stepped + 1) % self.config.vis_interval == 0 \
                        or step == 0 or step + self.stepped == 0:
                    self.model.eval()

                    self.evaluate(self.model, self.dataset_val, self.device,
                                  '{}/val_vis/{}.png'.format(self.config.ckpt,
                                                             step + self.stepped))

                # save the model
                if (step + self.stepped + 1) % self.config.save_model_interval == 0 \
                        or (step + 1) == self.config.max_iter:
                    print('Saving the model...')
                    save_ckpt('{}/models/{}.pth'.format(self.config.ckpt,
                                                        step + self.stepped + 1),
                              [('model', self.model)],
                              [('optimizer', self.optimizer)],
                              step + self.stepped + 1)
                    last_saved_model = step + self.stepped + 1

                if step >= self.config.max_iter:
                    break
        return last_saved_model, self.losses

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
        print('[STEP: {:>6}] | Valid Loss: {:.6f} | Hole Loss: {:.6f}' \
              '| TV Loss: {:.6f} | Perc Loss: {:.6f}' \
              '| Style Loss: {:.6f} | Total Loss: {:.6f}'.format(
            step, loss_dict['valid'], loss_dict['hole'],
            loss_dict['tv'], loss_dict['perc'],
            loss_dict['style'], loss_dict['total']))
