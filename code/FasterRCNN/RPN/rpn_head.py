# -*- coding:utf-8 -*-

"""
Convolution processing feature maps.

Description:
    Two different convolution layers are used to process feature
    maps separately, and the convolution result is used to determine
    whether the box contains the target and the box position regression
    adjustment.

"""
# **** modification history ****  #
# ******************************  #
# 2021/01/17,by Junlu Ding,create #

import torch
from torch.nn import init
from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import functional


class RPNHead(Module):
    """
    Convolution processes the feature maps of different layers output by
    the backbone, and the feature maps with 4 and 1 output channels are
    used for box classification and position regression respectively.

    Args:
        in_channels (int): feature maps channel nums.
        num_pixel_anchors (int): base anchor nums.
    Return:
        class list (List[Tensor(b,9,w1,h1),Tensor(b,9,w2,h2)..]):
        different feature maps convolution output result list.

        regression (List[Tensor(b,36,w1,h1),Tensor(b,36,w2,h2)..]):
        different feature maps convolution output result list.
    """
    def __init__(self, in_channels, num_pixel_anchors):
        super(RPNHead, self).__init__()
        self.cov = Conv2d(in_channels,
                          in_channels,
                          kernel_size=3,
                          stride=1,
                          padding=1)
        self.cls = Conv2d(in_channels,
                          num_pixel_anchors,
                          kernel_size=1,
                          stride=1)
        self.reg = Conv2d(in_channels,
                          num_pixel_anchors * 4,
                          kernel_size=1,
                          stride=1)

        for layer in self.children():
            if isinstance(layer, Conv2d):
                init.normal_(layer.weight, std=0.01)
                init.constant_(layer.bias, 0)
        return

    def forward(self, list_features):
        list_cls = []
        list_regs = []
        for feature in list_features:
            convert_feature = functional.relu(self.cov(feature))
            list_cls.append(self.cls(convert_feature))
            list_regs.append(self.reg(convert_feature))
        return list_cls, list_regs


if __name__ == '__main__':
    input_feature_maps = [torch.randn(2, 3, 112, 112), torch.randn(2, 3, 64, 64)]
    input_in_channels = 3
    input_num_anchors = 9
    input_rpn_head = RPNHead(input_in_channels, input_num_anchors)
    list_cls, list_regs = input_rpn_head(input_feature_maps)
    print([cls.size() for cls in list_cls])
    print([reg.size() for reg in list_regs])