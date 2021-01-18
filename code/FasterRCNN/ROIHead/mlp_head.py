# -*- coding:utf-8 -*-

"""
two mlp head of roi head

Description:
   flatten output of roi pooling for faster-rcnn predict
"""
# **** modification history ****  #
# ******************************  #
# 2021/01/17,by Junlu Ding,create #

import torch
from torch.nn import Module
from torch.nn import Linear
import torch.nn.functional as F


class TwoMLPHead(Module):
    """
    process roi pooling output by mlp layers.
    args:
        num_in_channels (int): proproslas channel nums
        representation_size (int):

    """
    def __init__(self,num_in_channels,representation_size):
        super(TwoMLPHead, self).__init__()
        self.fc1 = Linear(num_in_channels,representation_size)
        self.fc2 = Linear(representation_size,representation_size)
        return

    def forward(self, x):
        """
        args:
            x (Tensor): output of roi pooling
        return:
            y (Tensor): mlp layer output.
        """
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        y = F.relu(self.fc2(x))
        return y


