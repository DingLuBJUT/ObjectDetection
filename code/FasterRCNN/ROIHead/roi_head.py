# -*- coding:utf-8 -*-

"""
faster rcnn roi head module.

Description:
    this module input is proposals that rpn outputs, firstly resize images size
    the same size by ROI Pooling, and than flatten it to classify and regression
    mlp layer. secondly compute loss of class and regression by mlp outputs and
    post_process detection result. finally map predicted bbox back to original
    image.



"""
# **** modification history ****  #
# ******************************  #
# 2021/01/17,by Junlu Ding,create #

import torch
from torch.nn import Module
import torch.nn.functional as F

from mlp_head import TwoMLPHead
from roi_pooling import ROIPooling
from predictor import FasterRCNNPredictor
from utils import box_iou
from utils import box_deviation
from RPN.rpn import rpn_test_data


class ROIHead(Module):
    def __init__(self, params):
        super(ROIHead, self).__init__()
        self.roi_pooling = params['roi_pooling']
        self.mlp_head = params['mlp_head']
        self.predictor = params['predictor']
        self.pos_threshold = params['pos_threshold']
        self.neg_threshold = params['neg_threshold']
        self.num_sampling_per_image = params['num_sampling_per_image']
        self.pos_fraction = params['pos_fraction']
        self.regression_weights = params['regression_weights']
        if params['regression_weights'] is None:
            self.regression_weights = (10., 10., 5., 5.)
        return

    def _get_proposals_label(self, list_proposals, list_targets):
        """
        labeled proposal by iou value of ground truth and proposal

        args:
           list_proposals_label (List[Tensor]): rpn output proposals of per images.
           list_targets (List[Dict]): ground truth、label of per image.
        return:
            a list of proposal label
        """
        list_proposals_label = []
        for proposal, target in zip(list_proposals, list_targets):
            # num_proposals = proposal.size(0)
            label = torch.as_tensor(target['class'])
            ground_truth = torch.as_tensor(target['box'])

            iou = box_iou(proposal, ground_truth)
            max_value, max_index = torch.max(iou, dim=1)

            proposal_label = torch.zeros_like(max_value, dtype=torch.float)
            pos_index = torch.where(torch.gt(max_value, self.pos_threshold))[0]
            neg_index = torch.where(torch.lt(max_value, self.neg_threshold))[0]
            discard_index = torch.where(torch.logical_and(torch.ge(max_value, self.neg_threshold),
                                                          torch.le(max_value, self.pos_threshold)))[0]
            proposal_label[discard_index] = -1
            proposal_label[neg_index] = 0
            proposal_label[pos_index] = label[max_index[pos_index]].float()
            list_proposals_label.append(proposal_label)
        return list_proposals_label

    def _sampling_data(self, list_proposals_label, list_rpn_proposals, list_targets):
        """
        sampling data

        args:
            list_proposals_label (List(Tensor)): proposal label.
            list_rpn_proposals (List[Tensor]): rpn output proposals of per images.
            list_targets (List[Dict]): ground truth、label of per image.
        return:
            proposals, label, regression deviation
        """

        list_proposals = []
        list_labels = []
        list_deviation = []
        for proposals_label, proposals, target in zip(list_proposals_label, list_rpn_proposals, list_targets):

            ground_truth = torch.as_tensor(target['box'], dtype=torch.float)

            num_pos = int(self.num_sampling_per_image * self.pos_fraction)
            num_neg = int(self.num_sampling_per_image - num_pos)

            neg_index = torch.where(torch.eq(proposals_label, 0))[0]
            pos_index = torch.where(torch.gt(proposals_label, 0))[0]

            # shuffle index
            sampling_pos_index = pos_index[torch.randperm(pos_index.size(0))][:num_pos]
            sampling_neg_index = neg_index[torch.randperm(neg_index.size(0))][:num_neg]

            sampling_index = torch.cat([sampling_pos_index, sampling_neg_index])
            proposals = proposals[sampling_index]
            proposal_labels = proposals_label[sampling_index]
            regression_deviation = box_deviation(proposals, ground_truth, self.regression_weights)

            list_proposals.append(proposals)
            list_labels.append(proposal_labels)
            list_deviation.append(regression_deviation)
        return list_proposals, list_labels, list_deviation

    def compute_loss(self, cls_score, reg_params, list_proposals_label, list_deviation):

        num_proposals = cls_score.size(0)
        labels = torch.cat(list_proposals_label,dim=0).long()
        cls_loss = F.cross_entropy(cls_score,labels)
        reg_params = reg_params.view(num_proposals, -1, 4)
        print(list_proposals_label[0],list_proposals_label[1])
        print(reg_params.size())
        print([deviation.size() for deviation in list_deviation])
        # for proposals_label, deviation in zip(list_proposals_label,list_deviation):
        #     print(proposals_label)
        #     pos_index = torch.where(torch.gt(proposals_label,0))[0]
        #     deviation_index = (proposals_label[pos_index] - torch.as_tensor([1])).long()
        #     print(deviation.size())
        #     print(deviation_index)
        #     deviation = deviation[pos_index, deviation_index]
        #     print("*******")

        return

    def forward(self, dict_feature_map, list_proposals, list_targets, image_size):
        if self.training:
            list_proposals_label = self._get_proposals_label(list_proposals, list_targets)
            list_proposals, list_proposals_label, list_deviation = self._sampling_data(list_proposals_label,
                                                                                       list_proposals,
                                                                                       list_targets)

        proposal_features = self.roi_pooling(dict_feature_map, list_proposals, image_size)
        mlp_features = self.mlp_head(proposal_features)
        cls_score, reg_params = self.predictor(mlp_features)
        if self.training:
            self.compute_loss(cls_score, reg_params, list_proposals_label, list_deviation)
        return


if __name__ == '__main__':

    rpn_proposals, targets, feature_maps, images = rpn_test_data()

    params = dict()
    params['roi_pooling'] = ROIPooling(['1', '2'], [7, 7], 2)
    params['mlp_head'] = TwoMLPHead(images.size(1) * [7, 7][0] ** 2, 1024)
    params['predictor'] = FasterRCNNPredictor(1024, 3)
    params['pos_threshold'] = 0.2
    params['neg_threshold'] = 0.05
    params['num_sampling_per_image'] = 512
    params['pos_fraction'] = 0.25
    params['regression_weights'] = None
    roi_head = ROIHead(params)
    roi_head(dict(zip(['1', '2'], feature_maps)), rpn_proposals, targets, [(500, 500)])




