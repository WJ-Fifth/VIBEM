# -*- coding: utf-8 -*-
# By JInwu Wang u7354172
#
# Call more backbones to extract image features
#
# Contains Reset50, ResNeXt50 and swin transformer base


import math

import timm
import torch
import numpy as np
import os.path as osp
import torch.nn as nn
from lib.models.swin import swin_t


class Backbone(nn.Module):
    def __init__(self, base):
        super(Backbone, self).__init__()
        self.backbone = base
        if self.backbone == 'resnet50':
            m = timm.create_model(self.backbone, pretrained=True)
            backbone = nn.Sequential(*list(m.children())[:-1])
            self.model = backbone

        elif self.backbone == 'swin':
            m = timm.create_model('swin_base_patch4_window7_224', pretrained=True)
            backbone = nn.Sequential(*list(m.children())[:-1])
            self.model = backbone
        elif self.backbone == 'resnext50_32x4d':
            m = timm.create_model('resnext50_32x4d', pretrained=True)
            backbone = nn.Sequential(*list(m.children())[:-1])
            self.model = backbone
        self.avgpool = nn.AvgPool2d(7, stride=1)

    def forward(self, x):
        if self.backbone == 'resnet50':
            x = self.model(x)

        elif self.backbone == 'swin':
            x = self.model(x).view(-1, 1024, 7, 7)
            x = torch.cat([x, x], dim=1)
            x = self.avgpool(x)

        elif self.backbone == 'resnext50_32x4d':
            x = self.model(x)
        x = x.view(x.size(0), -1)
        return x


if __name__ == "__main__":
    import torch
    from torchsummary import summary

    input_size = 224
    model = Backbone('resnet50')
    # print(model)

    summary(model, input_size=[(3, input_size, input_size)], device="cpu")
