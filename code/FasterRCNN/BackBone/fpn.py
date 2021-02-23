# -*- coding:utf-8 -*-

"""
Feature Pyramid Networks for Object Detection

Description:
    build feature pyramid networks on the top of resnet.
"""
# ***** modification history *****
# ********************************
# 2021/02/11, by junlu Ding, create



from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import ModuleList
import torch.nn.functional as F
from torch.nn.init import constant_
from torch.nn.init import kaiming_normal_


class FeaturePyramidNetworks(Module):
    """
    construct efficient feature maps by fusing different levels
    of feature maps.

    Args:
        num_output_channels (int): Feature Pyramid Networks
        output dimensions num.

        list_num_channels (List[int]): feature maps channel
        num that output by different layers of feature extraction
        network.

    """
    def __init__(self, num_output_channels, list_num_channels, is_down_sampling_top):
        super(FeaturePyramidNetworks, self).__init__()
        self.is_down_sampling_top = is_down_sampling_top

        self.list_top_module = ModuleList()
        self.list_down_module = ModuleList()

        for num_channels in list_num_channels:
            self.list_top_module.append(Conv2d(num_channels,
                                               num_output_channels,
                                               kernel_size=1,
                                               stride=1,
                                               padding=0))

            self.list_down_module.append(Conv2d(num_output_channels,
                                                num_output_channels,
                                                kernel_size=3,
                                                stride=1,
                                                padding=1))

        # init model params
        for children in self.children():
            if isinstance(children,Conv2d):
                kaiming_normal_(children.weight,a=1)
                constant_(children.bias,0)
        return


    def forward(self, list_feature_map):
        # type : (List[Tensor]) -> List[Tensor]
        """
        Feature Pyramid Networks inference.

        Args:
            dict_feature_map : different level feature map, from high to
            low level.
        Return:
            semantically strong and high-resolution feature map list.

        """
        result = []
        top_feature_map = self.list_top_module[0](list_feature_map[0])
        result.append(self.list_down_module[0](top_feature_map))

        for i in range(1,len(list_feature_map)):
            up_feature_map = self.list_top_module[i](list_feature_map[i])
            # up sampling
            down_feature_map = F.interpolate(result[-1],
                                             size=up_feature_map.size()[-2:],
                                             mode="nearest")
            fpn_feature_map = self.list_down_module[i](down_feature_map + up_feature_map)
            result.append(fpn_feature_map)


        if self.is_down_sampling_top:
            top_down_sampling_fm = F.max_pool2d(result[0],
                                                kernel_size=1,
                                                stride=2,
                                                padding=0)
            result.insert(0, top_down_sampling_fm)

        return result


import torch

if __name__ == '__main__':
    # c1: torch.Size([1, 64, 56, 56])
    # c2: torch.Size([1, 256, 56, 56])
    # c3: torch.Size([1, 512, 28, 28])
    # c4: torch.Size([1, 1024, 14, 14])
    # c5: torch.Size([1, 2048, 7, 7])
    list_feature_map = [
        torch.randn(2, 2048, 7, 7),
        torch.randn(2, 1024, 14, 14),
        torch.randn(2, 512, 28, 28),
        torch.randn(2, 256, 56, 56)
    ]
    num_output_channels = 256
    list_num_channels = [2048,1024,512,256]
    is_down_sampling_top = True
    fpn = FeaturePyramidNetworks(num_output_channels,list_num_channels,is_down_sampling_top)

    for fm in fpn(list_feature_map):
        print(fm.size())




