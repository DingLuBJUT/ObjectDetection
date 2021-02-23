# -*- coding:utf-8 -*-

"""
faster r-cnn backbone.

Description:
    build faster r-cnn backbone that ResNet50 with FPN.
"""
# ***** modification history *****
# ********************************
# 2021/02/11, by junlu Ding, create

from torch.nn import Module
from torchvision.models import resnet50
from BackBone.fpn import FeaturePyramidNetworks


class ResNetWithFPN(Module):
    """
    Create pre-trained ResNet50 and FPN

    """
    def __init__(self, res_net, fpn):
        super(ResNetWithFPN, self).__init__()

        # freeze ResNet base layer params
        for name, params in res_net.named_parameters():
            if name.split(".")[0] not in ['layer2', 'layer3', 'layer4']:
                params.requires_grad_(False)

        self.dict_module = dict()
        for name, children in res_net.named_children():
            self.dict_module[name] = children
            if name == "layer4":
                break

        self.fpn = fpn
        self.list_feature_layer = ['layer1', 'layer2', 'layer3', 'layer4']

        return

    def forward(self, x):
        list_feature_maps = []
        for name,children in self.dict_module.items():
            x = children(x)
            if name in self.list_feature_layer:
                list_feature_maps.append(x)

        list_feature_maps = self.fpn(list(reversed(list_feature_maps)))
        return list_feature_maps


if __name__ == '__main__':
    res_net = resnet50(pretrained=True)
    fpn = FeaturePyramidNetworks(num_output_channels = 256,
                                 list_num_channels = [2048, 1024, 512, 256],
                                 is_down_sampling_top = True)

    backbone = ResNetWithFPN(res_net,fpn)
    import torch
    batch_images = torch.randn(2,3,512,782)
    list_feature_maps = backbone(batch_images)
    print([feature_map.size() for feature_map in list_feature_maps])




