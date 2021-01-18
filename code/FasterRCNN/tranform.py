# -*- coding:utf-8 -*-

"""

Description:

"""
# **** modification history ***** #
# 2021/01/18,by junlu Ding,create #

from torch.nn import Module
from torchvision.transforms import Resize
from torchvision.transforms import Compose
from torchvision.transforms import ToTensor
from torchvision.transforms import Normalize


class Transform(Module):
    def __init__(self):
        super(Transform, self).__init__()
        self.transforms = Compose([
            ToTensor(),
            Resize((128, 128)),
            Normalize(mean=0, std=1)
        ])
        return

    def forward(self, x):
        return self.transforms(x)
