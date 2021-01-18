# -*- coding:utf-8 -*-

"""
faster-rcnn ROI Head

Description:


"""
# **** modification history ****  #
# ******************************  #
# 2021/01/17,by Junlu Ding,create #

import torch
from torch.nn import Module
from roi_pooling import ROIPooling
from mlp_head import TwoMLPHead
from predictor import FasterRCNNPredictor


from RPN.rpn import rpn_test_data


class ROIHead(Module):
    def __init__(self,params):
        super(ROIHead, self).__init__()
        self.roi_pooling = params['roi_pooling']
        self.mlp_head = params['mlp_head']
        self.predictor = params['predictor']
        # self.pos_threshold = params['pos_threshold']
        # self.neg_threshold = params['neg_threshold']
        # self.num_sample_per_image = params['num_sample_per_image']
        # self.pos_fraction = params['pos_fraction']
        return

    def _get_proposals_class_label(self,list_rpn_proposals, list_targets):

        for proposal, target in zip(list_rpn_proposals,list_targets):
            label =  torch.tensor(target['class'])
            ground_truth = torch.tensor(target['box'])
            print(label.size(),ground_truth.size())


        return

    def _sample_data_index(self):

        return

    def _get_proposals_regression_label(self):

        return

    def smooth_l1_loss(self):
        return

    def cross_entropy_loss(self):
        return

    def forward(self, dict_feature_map, list_rpn_proposals, list_targets, image_size):
        proposal_features = self.roi_pooling(dict_feature_map,list_rpn_proposals,image_size)
        mlp_features = self.mlp_head(proposal_features)
        cls_score, reg_param = self.predictor(mlp_features)
        print(cls_score.size(),reg_param.size())
        return


if __name__ == '__main__':

    list_rpn_proposals, list_targets, list_feature_maps, images = rpn_test_data()

    params = dict()
    list_feature_names = ['1', '2']
    roi_pooling_size = [7, 7]
    sampling_ratio = 2
    roi_pooling = ROIPooling(list_feature_names, roi_pooling_size, sampling_ratio)
    params['roi_pooling'] = roi_pooling
    num_in_channels = images.size(1) * roi_pooling_size[0] ** 2
    representation_size = 1024
    mlp_head = TwoMLPHead(num_in_channels, representation_size)
    params['mlp_head'] = mlp_head
    num_class = 3
    predictor = FasterRCNNPredictor(representation_size, num_class)
    params['predictor'] = predictor

    # params['pos_threshold'] = 0.5
    # params['neg_threshold'] = 0.5
    # params['num_sample_per_image'] = 512
    # params['pos_fraction'] = 0.25

    roi_head = ROIHead(params)
    dict_feature_map = dict(zip(list_feature_names,list_feature_maps))
    image_size = [(500,500)]
    roi_head(dict_feature_map, list_rpn_proposals, list_targets, image_size)


