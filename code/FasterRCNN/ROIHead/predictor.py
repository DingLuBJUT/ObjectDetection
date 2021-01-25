# -*- coding:utf-8 -*-

"""
predict mlp layer outputs.

Description:
    perform classification and regression prediction on the
    output of mlp.

"""
# **** modification history ****  #
# ******************************  #
# 2021/01/17,by Junlu Ding,create #

from torch.nn import Module
from torch.nn import Linear


class FasterRCNNPredictor(Module):
    """
    input the results of mlp into two different fully connected
    layers to calculate the classification result score and position
    regression parameters.

    Args:
        representation_size (int): mlp output size.
        num_class (int): object category Number.

    Return:
        object classification score and box regression
        parameters. the outputs size is:

        classification: (proposals num, category Number)
        regression: (proposals num, category Number * 4)

    """
    def __init__(self, representation_size, num_class):
        super(FasterRCNNPredictor, self).__init__()
        self.representation_size = representation_size
        self.num_class = num_class

        self.layer_cls_score = Linear(self.representation_size, self.num_class)
        self.layer_reg_param = Linear(self.representation_size, self.num_class * 4)
        return

    def forward(self, x):
        cls_score = self.layer_cls_score(x)
        reg_param = self.layer_reg_param(x)
        return cls_score, reg_param
