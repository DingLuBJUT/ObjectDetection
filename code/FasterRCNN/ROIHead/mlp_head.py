# -*- coding:utf-8 -*-

"""
MLP layer for roi pooling output.

Description:
   process roi pooling output results and input to the fully
   connected layer.
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
    the roi pooing output to the result is flattened, and
    then input to the two fully connected layers.

    Args:
        num_in_channels (int): feature map channel nums
        representation_size (int): mlp layer output size.

    Return:
        mlp output tensor: the output size is (proposals num,
        representation_size).

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


