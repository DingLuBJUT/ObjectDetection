# -*- coding:utf-8 -*-
"""
Faster-RCNN RPN Module.

Description:
    The faster-rcnn rpn will get anchors by AnchorGenerator and classify、regression
    anchors by RPNHead, and than get proposals by adjust anchor by RPNHead regression
    output, filter proposals by classify score before NMS, lastly return NMS result.
    Finally if training, will random sample train data and compute anchor classify loss
    and regression loss.
"""
# ***** modification history *****
# ********************************
# 2021/01/16, by junlu Ding, create


import math
import torch
from torch.nn import Module
from torchvision.ops import nms
from torch.nn import functional as F

from rpn_head import RPNHead
from anchor_generator import AnchorGenerator


class RPN(Module):
    def __init__(self, dict_params):
        super(RPN, self).__init__()
        self.anchor_generator = dict_params['anchor_generator']
        self.rpn_head = dict_params['rpn_head']
        self.num_pre_nms = dict_params['num_pre_nms']
        self.num_post_nms = dict_params['num_post_nms']
        self.nms_threshold = dict_params['nms_threshold']
        self.nms_min_size = dict_params['nms_min_size']
        self.pos_threshold = dict_params['pos_threshold']
        self.neg_threshold = dict_params['neg_threshold']
        self.num_sample = dict_params['num_sample']
        self.pos_fraction = dict_params['pos_fraction']
        return

    def _adjust_anchor_by_regression(self, regs, anchors):
        """
        adjust anchor by regression params.
        Args:
            list_regs: rpn head output regs for per feature.
            anchors: anchor generator output anchor.
        Return:
            proposals [n, 4]
        """
        dx, dy, dh, dw = regs[:, 0], regs[:, 1], regs[:, 2], regs[:, 3]
        dw = torch.clamp(dw, max=math.log(1000. / 16))
        dh = torch.clamp(dh, max=math.log(1000. / 16))

        anchor_h = anchors[:, 3] - anchors[:, 1]
        anchor_w = anchors[:, 2] - anchors[:, 0]
        center_x = anchor_w / 2
        center_y = anchor_h / 2

        proposal_x = center_x + dx * anchor_w
        proposal_y = center_y + dy * anchor_h
        proposal_h = torch.exp(dh) * anchor_h
        proposal_w = torch.exp(dw) * anchor_w

        min_x = (proposal_x - proposal_w / 2)[:, None]
        min_y = (proposal_y - proposal_h / 2)[:, None]
        max_x = (proposal_x + proposal_w / 2)[:, None]
        max_y = (proposal_y + proposal_h / 2)[:, None]
        proposals = torch.cat([min_x, min_y, max_x, max_y],dim=1)
        return proposals

    def _get_top_score_index(self, list_cls, list_num_anchors, batch_size):
        """
        get top k anchor score index in per feature and index num of per feature
        args:
            list_regs: rpn head output regs for per feature.
            list_num_anchors: anchor num for per batch feature.
            batch_size: batch image num.
        return:
            index list [index,]
            index num list [num1,num2,.]
        """
        list_top_index = []
        list_num_pre_nms = []
        for score, num_anchors in zip(list_cls, list_num_anchors):
            score = score.permute(0, 2, 3, 1).flatten(1, 3)
            num_pre_nms = min(self.num_pre_nms, num_anchors)
            _, index = torch.topk(score, k=num_pre_nms, dim=1)
            index = index.flatten()
            offset = torch.cat([torch.full((num_pre_nms,), i) * num_anchors for i in range(batch_size)])
            index += offset
            list_top_index.append(index)
            list_num_pre_nms.append(num_pre_nms)
        return list_top_index, list_num_pre_nms

    def _get_top_cls(self, list_cls, list_top_index):
        """
        get top k anchor score in per feature.
        args:
            list_cls: rpn head output cls for per feature.
            list_top_index: top k score index in per feature.
        return:
            top cls list
        """
        list_top_cls = []
        for score, index in zip(list_cls, list_top_index):
            score = score.permute(0, 2, 3, 1).flatten(0, 3)
            score = score[index]
            list_top_cls.append(score)
        return list_top_cls

    def _get_top_proposals(self, list_regs, list_anchors, list_top_index, list_num_anchors):
        """
        get per feature proposals by adjust feature anchor with regression params.
        args:
            list_regs: rpn head output regs for per feature.
            list_anchors: anchor generator output anchor for per image.
            list_top_index: top k score index in per feature.
            list_num_anchors: anchor num for per batch feature.
        return:
            proposal list
        """
        anchors = torch.cat([anchors[None, :] for anchors in list_anchors], dim=0)
        anchors = anchors.flatten(1, 2)

        list_top_proposals = []
        # get anchors of per feature
        feature_anchors = torch.split(anchors, list_num_anchors, dim=1)
        for anchors, regs, index in zip(feature_anchors, list_regs, list_top_index):
            regs = regs.permute(0, 2, 3, 1).flatten(0, 2).reshape(-1, 4)
            anchors = anchors.flatten(0, 1)
            regs, anchors = regs[index], anchors[index]
            proposals = self._adjust_anchor_by_regression(regs, anchors)
            list_top_proposals.append(proposals)
        return list_top_proposals

    def _nms(self, list_top_cls, list_top_proposals, batch_size, list_num_pre_nms, image_h, image_w):
        """
        clip and filter proposals and NMS
        args:
            list_top_cls: top score for per feature.
            list_top_proposals: top proposals for per feature.
            list_num_pre_nms: proposal num of per feature before nms.
            batch_size: batch image num.
            image_h: image height.
            image_w: image width.
        return:
            list_nms_proposals: per images nms proposal.
        """
        list_nms_proposals = []
        cls = torch.cat([cls.view(batch_size, -1) for cls in list_top_cls], dim=1)
        proposals = torch.cat([proposal.view(batch_size, -1, 4) for proposal in list_top_proposals], dim=1)
        for score, proposal in zip(cls, proposals):

            offset = torch.cat([torch.full((num,), i) for i, num in enumerate(list_num_pre_nms)])
            offset = offset[:, None] * proposal.max().to(offset)

            min_x = torch.clamp(proposal[:, 0], min=0, max=image_w)
            min_y = torch.clamp(proposal[:, 1], min=0, max=image_h)
            max_x = torch.clamp(proposal[:, 2], min=0, max=image_w)
            max_y = torch.clamp(proposal[:, 3], min=0, max=image_h)
            proposal_w, proposal_h = max_x - min_x, max_y - min_y
            keep = torch.logical_and(torch.ge(proposal_w, self.nms_min_size), torch.ge(proposal_h, self.nms_min_size))
            index = torch.where(keep)[0]
            score = score[index]
            proposal = proposal[index]
            offset = offset[index]
            proposal = proposal + offset
            # nms return proposal index
            nms_index = nms(proposal, score, self.nms_threshold)[:self.num_post_nms]
            list_nms_proposals.append(proposal[nms_index])
        return list_nms_proposals

    def _get_anchor_class_labels(self, list_anchors, list_targets):
        """
        labeled anchors by iou of anchor and ground truth for per image.
            if iou > pos_threshold as pos
            if iou < neg_threshold as neg
            if neg_threshold <= iou <= pos_threshold as mid
        args:
            list_anchors: anchor generator output anchor for per image.
            list_targets: per image target label.
        return:
            list_class_label: anchor label(1:pos、0:net、-1:mid) for per image.
            list_best_ground_truth: the best matched ground truth for per anchor.
        """
        list_class_label = []
        list_best_ground_truth = []
        for anchors, targets in zip(list_anchors, list_targets):

            anchors = anchors.flatten(0, 1)
            num_anchors = anchors.size(0)
            ground_truth = torch.tensor(targets['box'])

            anchors_w = anchors[:, 2] - anchors[:, 0]
            anchors_h = anchors[:, 3] - anchors[:, 1]
            anchors_area = (anchors_w * anchors_h)[:, None]

            ground_truth_w = ground_truth[:, 2] - ground_truth[:, 0]
            ground_truth_h = ground_truth[:, 3] - ground_truth[:, 1]
            ground_truth_area = (ground_truth_w * ground_truth_h)[None, :]
            total_area = anchors_area + ground_truth_area

            join_top_left = torch.max(anchors[:, None, :2], ground_truth[:, :2]).flatten(0, 1)
            join_bottom_right = torch.min(anchors[:, None, 2:], ground_truth[:, 2:]).flatten(0, 1)
            join_w = (join_bottom_right[:, 0] - join_top_left[:, 0]).clamp(min=0)
            join_h = (join_bottom_right[:, 1] - join_top_left[:, 1]).clamp(min=0)
            join_area = (join_w * join_h).view(num_anchors, -1)

            iou = join_area / (total_area - join_area)
            max_value, max_index = iou.max(dim=1)
            pos_position = torch.gt(max_value, self.pos_threshold)
            neg_position = torch.lt(max_value, self.neg_threshold)
            mid_position = torch.logical_and(torch.ge(max_value, self.neg_threshold),
                                             torch.le(max_value, self.pos_threshold))

            max_value[pos_position] = 1
            max_value[neg_position] = 0
            max_value[mid_position] = -1
            class_label = max_value
            best_match_ground_truth = ground_truth[max_index]
            list_class_label.append(class_label)
            list_best_ground_truth.append(best_match_ground_truth)
        return list_class_label, list_best_ground_truth

    def _get_anchor_regression_label(self, list_anchors, list_best_ground_truth):
        """
        get per anchor regression label with the best matched ground truth for per image.
        args:
            list_anchors: anchor generator output anchor for per image.
            list_best_ground_truth: the best matched ground truth for per anchor.
        return:
            list_regression_label: the deviation of the anchor and the best ground truth for per image.

        """
        list_regression_label = []
        for anchor, ground_truth in zip(list_anchors, list_best_ground_truth):
            anchor = anchor.flatten(0,1)
            anchor_w = anchor[:,2] - anchor[:,0]
            anchor_h = anchor[:,3] - anchor[:,1]
            anchor_center_x = anchor[:, 0] + anchor_w / 2
            anchor_center_y = anchor[:, 1] + anchor_h / 2

            ground_truth_w = ground_truth[:, 2] - ground_truth[:, 0]
            ground_truth_h = ground_truth[:, 3] - ground_truth[:, 1]
            ground_truth_center_x = ground_truth[:, 0] + ground_truth_w / 2
            ground_truth_center_y = ground_truth[:, 1] + ground_truth_h / 2

            dx = ((ground_truth_center_x - anchor_center_x) / anchor_w)[:, None]
            dy = ((ground_truth_center_y - anchor_center_y) / anchor_h)[:, None]
            dw = torch.log(ground_truth_w / anchor_w)[:, None]
            dh = torch.log(ground_truth_h / anchor_h)[:, None]

            regression_label = torch.cat([dx, dy, dh, dw], dim=1)
            list_regression_label.append(regression_label)

        return list_regression_label

    def _sample_data_index(self, list_class_label):
        """
        sample neg、pos data index by class label for per image.

        args:
            list_class_label: anchors class label for per image.
        return:
            pos_index: total image pos data index.
            neg_index: total image neg data index.
        """
        list_pos_index = []
        list_neg_index = []
        for i, class_label in enumerate(list_class_label):
            num_label = class_label.size(0)
            pos_index = torch.where(torch.eq(class_label, 1))[0]
            neg_index = torch.where(torch.eq(class_label, 0))[0]
            num_pos_sample = int(min(self.num_sample * self.pos_fraction, pos_index.size(0)))
            num_neg_sample = int(min(self.num_sample * (1 - self.pos_fraction), neg_index.size(0)))
            # shuffle pos、neg index
            pos_random_factor = torch.randperm(pos_index.size(0))[:num_pos_sample]
            neg_random_factor = torch.randperm(neg_index.size(0))[:num_neg_sample]

            pos_index = pos_index[pos_random_factor]
            neg_index = neg_index[neg_random_factor]
            # add offset for per image
            pos_offset = torch.full((num_pos_sample,), i) * num_label
            neg_offset = torch.full((num_neg_sample,), i) * num_label

            pos_index += pos_offset
            neg_index += neg_offset

            list_pos_index.append(pos_index)
            list_neg_index.append(neg_index)
        pos_index = torch.cat(list_pos_index, dim=0)
        neg_index = torch.cat(list_neg_index, dim=0)
        return pos_index, neg_index

    def smooth_l1_loss(self, pos_index, list_regs, list_regression_label, batch_size, beta=1./9, is_mean=True):
        """
        get the smooth l1 loss of anchor regression label and anchor regression.
        args:
            pos_index: total image pos data index.
            list_regs: rpn head output regs for per feature.
            list_regression_label:  anchors regression label for per image.
            batch_size: images batch size.
            beta: smooth l1 loss params, default is 1./9.
            is_mean: is return mean smooth l1 loss, default is True.
        return:
            mean smooth l1 loss(Tensor).
        """
        index = pos_index
        y = []
        for reg in list_regs:
            reg = reg.permute(0, 2, 3, 1).reshape(batch_size, -1, 4)
            y.append(reg)
        y = torch.cat(y, dim=1).flatten(0, 1)[index]
        label = torch.cat(list_regression_label,dim=0)[index]

        abs_difference = torch.abs(y - label)
        condition = torch.lt(abs_difference, beta)
        loss = torch.where(condition, 0.5 * abs_difference ** 2 / beta, abs_difference - 0.5 * beta)
        loss = loss.mean()
        return loss

    def binary_cross_entropy_loss(self, pos_index, neg_index,list_cls,list_class_label):
        """
        get the binary cross entropy loss of anchor cls label and anchor cls score.
        args:
            pos_index: total image pos data index.
            neg_index: total image neg data index.
            list_cls: rpn head output cls for per feature.
            list_class_label:
        return:
            binary cross entropy loss(Tensor).
        """
        index = torch.cat([pos_index, neg_index],dim=0)
        y = []
        for cls in list_cls:
            cls = cls.permute(0, 2, 3, 1).flatten(1, 2)
            y.append(cls)
        y = torch.cat(y, dim=1).flatten()[index]
        label = torch.cat(list_class_label, dim=0)[index].float()
        loss = F.binary_cross_entropy_with_logits(y, label)
        return loss

    def forward(self, images, list_features, list_targets):
        """
        args:
            images: batch images.
            list_features: batch features list.
            list_targets: batch image targets.
        return:
            proposals and loss.
        """
        batch_size, image_h, image_w = images.size()[0], images.size()[2], images.size()[3]
        list_anchors = self.anchor_generator(images,list_features)
        list_cls, list_regs = self.rpn_head(list_features)

        list_num_anchors = [cls.size()[1] * cls.size()[2] * cls.size()[3] for cls in list_cls]

        # get top score anchor index
        list_top_index, list_num_pre_nms = self._get_top_score_index(list_cls, list_num_anchors, batch_size)

        # get top cls by top index
        list_top_cls = self._get_top_cls(list_cls, list_top_index)

        # get top proposal by top index
        list_top_proposals = self._get_top_proposals(list_regs, list_anchors, list_top_index, list_num_anchors)

        # nms for top proposal
        list_nms_proposals = self._nms(list_top_cls, list_top_proposals, batch_size, list_num_pre_nms, image_h, image_w)

        rpn_loss = None
        if self.training:
            list_class_label, list_best_ground_truth = self._get_anchor_class_labels(list_anchors, list_targets)
            list_regression_label = self._get_anchor_regression_label(list_anchors, list_best_ground_truth)
            pos_index, neg_index = self._sample_data_index(list_class_label)
            cls_loss = self.binary_cross_entropy_loss(pos_index, neg_index,list_cls, list_class_label)
            reg_loss = self.smooth_l1_loss(pos_index, list_regs, list_regression_label, batch_size)
            rpn_loss = {
                "cls_loss": cls_loss,
                "reg_loss": reg_loss
            }
        return list_nms_proposals, rpn_loss


def rpn_test_data():
    input_images = torch.randn(size=(2, 3, 281, 500))

    input_feature_list = [
        torch.rand(size=(2, 3, 112, 112)),
        torch.rand(size=(2, 3, 64, 64))
    ]
    input_targets = [
        {'size': [3.0, 281, 500], 'class': ['1','1'], 'box': [[104, 78, 375, 183], [133, 88, 197, 123]]},
        {'size': [3.0, 500.0, 375.0], 'class': ['2'], 'box': [[104, 78, 375, 183]]}
    ]

    params = dict()
    params['anchor_generator'] = AnchorGenerator()
    params['rpn_head'] = RPNHead(in_channels=3, num_pixel_anchors=9)
    params['num_pre_nms'] = 2000
    params['num_post_nms'] = 2000
    params['nms_threshold'] = 0.7
    params['nms_min_size'] = 1e-3
    params['pos_threshold'] = 0.7
    params['neg_threshold'] = 0.3
    params['num_sample'] = 256
    params['pos_fraction'] = 0.5

    rpn = RPN(params)
    list_nms_proposals, rpn_loss = rpn(input_images, input_feature_list, input_targets)
    return list_nms_proposals, input_targets, input_feature_list, input_images


if __name__ == '__main__':
    input_images = torch.randn(size=(2, 3, 281, 500))
    input_feature_list = [
        torch.rand(size=(2, 3, 112, 112)),
        torch.rand(size=(2, 3, 64, 64))
    ]
    input_targets = [
        {'size': [3.0, 281, 500], 'class': ['dog'], 'box': [[104, 78, 375, 183], [133, 88, 197, 123]]},
        {'size': [3.0, 500.0, 375.0], 'class': ['cat'], 'box': [[104, 78, 375, 183]]}
    ]

    params = dict()
    params['anchor_generator'] = AnchorGenerator()
    params['rpn_head'] = RPNHead(in_channels=3, num_pixel_anchors=9)
    params['num_pre_nms'] = 2000
    params['num_post_nms'] = 2000
    params['nms_threshold'] = 0.7
    params['nms_min_size'] = 1e-3
    params['pos_threshold'] = 0.7
    params['neg_threshold'] = 0.3
    params['num_sample'] = 256
    params['pos_fraction'] = 0.5


    rpn = RPN(params)
    list_nms_proposals, rpn_loss = rpn(input_images, input_feature_list, input_targets)
    print([proposals for proposals in list_nms_proposals])