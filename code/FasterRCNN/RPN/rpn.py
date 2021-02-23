# -*- coding:utf-8 -*-
"""
faster r-cnn rpn module.

Description:
    the faster r-cnn rpn get anchors by anchor generator and
    process anchors by rpn head, and than get proposals by process
    anchor by regression param, filter proposals by clip、remove small
    before nms. finally if training, randomly sample train data and
    compute anchor classification and regression loss.
"""
# ***** modification history *****
# ********************************
# 2021/01/16, by junlu Ding, create

import torch
from torch.nn import Module
from torchvision.ops import nms
from torch.nn import functional as F

from rpn_head import RPNHead
from anchor_generator import AnchorGenerator
from utils import box_regression
from utils import remove_small_box
from utils import box_clip
from utils import tagging_box
from utils import box_deviation
from utils import get_sampling_factor
from utils import smooth_l1_loss

class RPN(Module):
    def __init__(self, dict_params):
        super(RPN, self).__init__()
        self.anchor_generator = dict_params['anchor_generator']
        self.rpn_head = dict_params['rpn_head']
        self.num_pre_nms = dict_params['num_pre_nms']
        self.num_post_nms = dict_params['num_post_nms']
        self.nms_threshold = dict_params['nms_threshold']
        self.proposal_min_size = dict_params['proposal_min_size']
        self.pos_threshold = dict_params['pos_threshold']
        self.neg_threshold = dict_params['neg_threshold']
        self.num_sample = dict_params['num_sample']
        self.pos_fraction = dict_params['pos_fraction']
        return

    def forward(self, batch_image_info, list_features, list_targets):
        batch_size = batch_image_info[0]
        device, data_type = list_features[0].device,list_features[0].dtype

        # get anchor and rpn head output for per feature
        list_anchors = self.anchor_generator(batch_image_info, list_features)
        list_cls, list_regs = self.rpn_head(list_features)

        # merge anchor and rpn head output of per feature in a image
        anchors = torch.cat([anchor.flatten(1,2) for anchor in list_anchors], dim=1)
        cls = torch.cat([cls.permute(0,2,3,1).flatten(1,3) for cls in list_cls],dim=1)
        reg = torch.cat([reg.permute(0,2,3,1).flatten(1,2) for reg in list_regs],dim=1)

        # get cls、reg、anchor whit top score for per image
        list_top_cls = []
        list_top_reg = []
        list_top_anchor = []
        for image_score, image_reg, image_anchor in zip(cls, reg, anchors):
            num_anchor = image_score.size(0)
            num_pre_nums = min(self.num_pre_nms, num_anchor)
            _, top_index = torch.topk(image_score, k=num_pre_nums)
            list_top_cls.append(image_score[top_index])
            list_top_reg.append(image_reg.view(-1,4)[top_index])
            list_top_anchor.append(image_anchor[top_index])


        # get proposal for per image
        list_top_proposal = []
        for reg, anchor in zip(list_top_reg, list_top_anchor):
            list_top_proposal.append(box_regression(anchor, reg))


        # NMS proposals for per image, but do clip、remove small box before NMS
        list_nms_proposals = []
        for score,proposal in zip(list_top_cls,list_top_proposal):
            proposal = box_clip(proposal,(batch_image_info[1],batch_image_info[2]))
            proposal, index = remove_small_box(proposal,self.proposal_min_size)
            score = score[index]
            nms_index = nms(proposal, score, self.nms_threshold)[:self.num_post_nms]
            list_nms_proposals.append(proposal[nms_index])

        rpn_loss = None
        if self.training:

            list_cls = [cls.permute(0,2,3,1).flatten(1,3)
                        for cls in list_cls]
            list_regs = [reg.permute(0,2,3,1).flatten(1,2).reshape(batch_size,-1,4)
                         for reg in list_regs]

            # concat different feature anchors、scores、regs
            anchors = torch.cat(list_anchors, dim=1).flatten(1, 2)
            scores = torch.cat(list_cls, dim=1)
            regs = torch.cat(list_regs, dim=1)

            # tagging cls and reg label for anchor of per image and random sampling
            list_reg_label = []
            list_reg_prediction = []
            list_cls_label = []
            list_cls_prediction = []
            for anchor, score, reg, target in zip(anchors,scores,regs,list_targets):
                ground_truth = torch.tensor(target['box'],device=device,dtype=data_type)
                # tagging cls label
                cls_label = tagging_box(anchor, ground_truth,
                                        self.neg_threshold,
                                        self.pos_threshold)
                # tagging reg label
                reg_label = box_deviation(anchor, ground_truth)

                # random sampling
                neg_index = torch.where(torch.eq(cls_label, 0))[0]
                pos_index = torch.where(torch.eq(cls_label, 1))[0]
                pos_factor, neg_factor = get_sampling_factor(self.num_sample,
                                                             self.pos_fraction,
                                                             pos_index.size(0),
                                                             neg_index.size(0))

                random_index = torch.cat([neg_index[neg_factor],pos_index[pos_factor]])

                list_reg_label.append(reg_label[random_index])
                list_cls_label.append(cls_label[random_index])
                list_reg_prediction.append(reg[random_index])
                list_cls_prediction.append(score[random_index])

            # concat cls、reg label and rpn head output for per image
            reg_label = torch.cat(list_reg_label)
            reg_y = torch.cat(list_reg_prediction)
            cls_label = torch.cat(list_cls_label)
            cls_y = torch.cat(list_cls_prediction)

            # compute loss
            cls_loss = F.binary_cross_entropy_with_logits(cls_y, cls_label)
            reg_loss = smooth_l1_loss(reg_y,reg_label)
            rpn_loss = {
                "cls_loss": cls_loss,
                "reg_loss": reg_loss
            }
        return list_nms_proposals, rpn_loss


def rpn_test_data():
    batch_image_info = (2, 256, 256)

    list_features = [
        torch.rand(size=(2, 3, 112, 112)),
        torch.rand(size=(2, 3, 64, 64)),
        torch.rand(size=(2, 3, 32, 32)),
    ]

    list_targets = [
        {'size': [3.0, 281, 500], 'class': ['dog'], 'box': [[104, 78, 375, 183], [133, 88, 197, 123]]},
        {'size': [3.0, 500.0, 375.0], 'class': ['cat'], 'box': [[104, 78, 375, 183]]}
    ]

    params = dict()
    params['anchor_generator'] = AnchorGenerator()
    params['rpn_head'] = RPNHead(in_channels=3, num_pixel_anchors=9)
    params['num_pre_nms'] = 2000
    params['num_post_nms'] = 2000
    params['nms_threshold'] = 0.7
    params['proposal_min_size'] = 1e-3
    params['pos_threshold'] = 0.7
    params['neg_threshold'] = 0.3
    params['num_sample'] = 256
    params['pos_fraction'] = 0.5

    rpn = RPN(params)
    list_nms_proposals, rpn_loss = rpn(batch_image_info, list_features, list_targets)
    return list_nms_proposals, rpn_loss, list_features


if __name__ == '__main__':
    # input_images = torch.randn(size=(2, 3, 281, 500))
    batch_image_info = (2, 256, 256)

    list_features = [
        torch.rand(size=(2, 3, 112, 112)),
        torch.rand(size=(2, 3, 64, 64)),
        torch.rand(size=(2, 3, 32, 32)),
    ]

    list_targets = [
        {'size': [3.0, 281, 500], 'class': ['dog'], 'box': [[104, 78, 375, 183], [133, 88, 197, 123]]},
        {'size': [3.0, 500.0, 375.0], 'class': ['cat'], 'box': [[104, 78, 375, 183]]}
    ]

    params = dict()
    params['anchor_generator'] = AnchorGenerator()
    params['rpn_head'] = RPNHead(in_channels=3, num_pixel_anchors=9)
    params['num_pre_nms'] = 2000
    params['num_post_nms'] = 2000
    params['nms_threshold'] = 0.7
    params['proposal_min_size'] = 1e-3
    params['pos_threshold'] = 0.7
    params['neg_threshold'] = 0.3
    params['num_sample'] = 256
    params['pos_fraction'] = 0.5

    rpn = RPN(params)
    list_nms_proposals, rpn_loss = rpn(batch_image_info, list_features, list_targets)
    # print([proposals.size() for proposals in list_nms_proposals])

