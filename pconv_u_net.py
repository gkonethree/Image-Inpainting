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
