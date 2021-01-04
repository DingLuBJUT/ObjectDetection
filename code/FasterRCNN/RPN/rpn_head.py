# -*- coding:utf-8 -*-

import torch
from torch.nn import init
from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import functional

class RPNHead(Module):
    """
    class and regression anchor on feature maps.

    args:
        images,feature_maps

    return:

    """
    def __init__(self, in_channels, num_anchors):
        super(RPNHead, self).__init__()
        self.cov = Conv2d(in_channels,
                          in_channels,
                          kernel_size=3,
                          stride=1,
                          padding=1)
        self.cls = Conv2d(in_channels,
                          num_anchors,
                          kernel_size=1,
                          stride=1)
        self.reg = Conv2d(in_channels,
                          num_anchors * 4,
                          kernel_size=1,
                          stride=1)

        for layer in self.children():
            if isinstance(layer, Conv2d):
                init.normal_(layer.weight, std=0.01)
                init.constant_(layer.bias, 0)
        return

    def forward(self, feature_maps):
        # type List[Tensor] -> List[Tensor]
        cls_list = []
        reg_list = []
        for feature in feature_maps:
            convert_feature = functional.relu(self.cov(feature))
            cls_list.append(self.cls(convert_feature))
            reg_list.append(self.reg(convert_feature))
        return cls_list,reg_list


if __name__ == '__main__':
    feature_maps = [torch.randn(2, 3, 112, 112), torch.randn(2, 3, 64, 64)]
    in_channels = 3
    num_anchors = 9
    rpn_head = RPNHead(in_channels, num_anchors)
    cls_list, reg_list = rpn_head(feature_maps)
    print(cls_list[0].size())
    print(reg_list[0].size())