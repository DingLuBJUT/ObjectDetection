# -*- coding:utf-8 -*-

"""
roi pooling

Description:
   perform roi pooling on proposals output by rpn, map
   each proposals to a different feature map and adjust
   the mapping result to a same size.
"""
# **** modification history ****  #
# ******************************  #
# 2021/01/17,by Junlu Ding,create #


import torch
from torch.nn import Module
from torchvision.ops import MultiScaleRoIAlign


class ROIPooling(Module):
    """
    input the feature maps dict and rpn output result list
    into MultiScaleRoIAlign to perform roi pooling, and
    return the same size to the mapping result of proposals
    on feature maps.

    Args:
        list_feature_names (List[str]): name list of feature
        map names.

        output_size (List[int,int]): output size of roi pooling.
        sampling_ratio (int) :
    Return:
        roi pooling output tensor the size is:
        (proposals num, output size, output size)

    """
    def __init__(self, list_feature_names, output_size, sampling_ratio):
        super(ROIPooling, self).__init__()
        self.list_feature_names = list_feature_names
        self.output_size = output_size
        self.sampling_ratio = sampling_ratio
        self.roi_pooling = MultiScaleRoIAlign(self.list_feature_names,
                                              self.output_size,
                                              self.sampling_ratio)
        return

    def forward(self,dict_feature_maps, list_proposals, image_size):

        output = self.roi_pooling(dict_feature_maps,
                                  list_proposals,
                                  image_size)
        return output


if __name__ == '__main__':
    list_feature_names = ['1', '2']
    output_size = [7, 7]
    sampling_ratio = 2
    roi_pooling = ROIPooling(list_feature_names, output_size, sampling_ratio)
    dict_feature_maps = {
        '1': torch.rand(size=(2, 3, 112, 112)),
        '2': torch.rand(size=(2, 3, 64, 64))
    }
    list_proposals = [
        torch.randint(low=0, high=256, size=(64, 4)).float(),
        torch.randint(low=0, high=256, size=(32, 4)).float()
    ]
    image_size = [(128, 128)]
    output = roi_pooling(dict_feature_maps, list_proposals, image_size)
    print(output.size())

