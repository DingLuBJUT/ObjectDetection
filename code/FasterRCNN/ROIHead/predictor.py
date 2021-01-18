# -*- coding:utf-8 -*-

"""
faster-rcnn predictor

Description:
    get cls scores and regression params.

"""
# **** modification history ****  #
# ******************************  #
# 2021/01/17,by Junlu Ding,create #

import torch
from torch.nn import Module
from torch.nn import Linear

class FasterRCNNPredictor(Module):
    """
    predict proposals class scores and regression params.

    args:
        representation_size (int): mlp output size.
        num_class (int): detection class num.

    """
    def __init__(self,representation_size, num_class):
        super(FasterRCNNPredictor, self).__init__()
        self.representation_size = representation_size
        self.num_class = num_class

        self.layer_cls_score = Linear(self.representation_size, self.num_class)
        self.layer_reg_param = Linear(self.representation_size, self.num_class * 4)
        return

    def forward(self,x):
        """
        args:
            x (Tensor): output of mlp.
        return:
            class score (Tenor):
            regression param (Tenor):
        """
        cls_score = self.layer_cls_score(x)
        reg_param = self.layer_reg_param(x)
        return cls_score, reg_param
