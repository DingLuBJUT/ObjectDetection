# -*- coding:utf-8 -*-

import math
import torch
from torch.nn import Module
from torchvision.ops import nms
from rpn_head import RPNHead
from anchor_generator import AnchorGenerator


class RPN(Module):
    def __init__(self, anchor_generator, rpn_head, pre_nmn_top_num, post_nms_top_num, fg_iou_thresh, bg_iou_thresh, nms_thresh,
                 sample_num, positive_fraction):
        super(RPN, self).__init__()
        self.anchor_generator = anchor_generator
        self.rpn_head = rpn_head
        self.pre_nmn_top_num = pre_nmn_top_num
        self.post_nms_top_num = post_nms_top_num
        self.nms_thresh = nms_thresh
        self.fg_iou_thresh = fg_iou_thresh
        self.bg_iou_thresh = bg_iou_thresh
        self.sample_num = sample_num
        self.positive_fraction = positive_fraction
        self.min_size = 1e-3
        return

    def adjust_anchor_by_reg(self):
        return

    def forward(self, images, feature_maps, targets=None):
        batch_size = images.size()[0]
        image_height, image_width = images.size()[2:]

        # ** get all anchors ** #
        anchor_list = self.anchor_generator(images, feature_maps)

        # *** get rpn head result and convert data format. can't direct reshape. *** #
        num_reg = 4
        num_cls = 1
        cls_list, reg_list = self.rpn_head(feature_maps)

        format_cls_list = []
        format_reg_list = []
        for cls, reg in zip(cls_list, reg_list):
            cls_size, cls_channels, cls_h, cls_w = cls.size()
            reg_size, reg_channels, reg_h, reg_w = reg.size()

            cls = cls.view(cls_size, -1, num_cls, cls_h, cls_w)
            cls = cls.permute(0, 3, 4, 1, 2)
            cls = cls.reshape(cls_size, -1, num_cls)

            reg = reg.view(reg_size, -1, num_reg, reg_h, reg_w)
            reg = reg.permute(0, 3, 4, 1, 2)
            reg = reg.reshape(reg_size, -1, num_reg)

            format_cls_list.append(cls)
            format_reg_list.append(reg)
        cls = torch.cat(format_cls_list, dim=1).reshape(-1, num_cls)
        reg = torch.cat(format_reg_list, dim=1).reshape(-1, num_reg)

        # *** apply regression on anchors. *** #
        anchor_position_num = sum([anchor.size()[0] for anchor in anchor_list])
        total_anchor = torch.cat(anchor_list, dim=0)
        reg = reg.reshape(anchor_position_num, -1)

        width = total_anchor[:, :, 2] - total_anchor[:, :, 0]
        height = total_anchor[:, :, 3] - total_anchor[:, :, 1]
        center_x = total_anchor[:, :, 0] + width / 2
        center_y = total_anchor[:, :, 1] + height / 2

        dx = reg[:, 0::4]
        dy = reg[:, 1::4]
        dw = reg[:, 2::4]
        dh = reg[:, 3::4]
        # limit max value for exp.
        dw = torch.clamp(dw, max=math.log(1000. / 16))
        dh = torch.clamp(dh, max=math.log(1000. / 16))
        pre_x = center_x + dx * width
        pre_y = center_y + dy * height
        pre_w = torch.exp(dw) * width
        pre_h = torch.exp(dh) * height

        min_x = pre_x - torch.tensor(0.5) * pre_w
        min_y = pre_y - torch.tensor(0.5) * pre_h
        max_x = pre_x + torch.tensor(0.5) * pre_w
        max_y = pre_y + torch.tensor(0.5) * pre_h

        proposals = torch.cat([min_x, min_y, max_x, max_y], dim=1)
        proposals = proposals.reshape(anchor_position_num, -1, 4)
        proposals = proposals.view(batch_size, -1, 4)

        # *** select top score anchor *** #
        # detach from graph
        cls = cls.detach()
        cls = cls.reshape(batch_size, -1)

        anchor_num_list = [cls.size()[1] * cls.size()[2] * cls.size()[3] for cls in cls_list]
        index_list = []
        offset = 0
        for anchor_score in torch.split(cls, anchor_num_list, dim=1):
            anchor_num = anchor_score.size()[1]
            pre_nmn_top_num = min(self.pre_nmn_top_num, anchor_num)
            _, top_index = anchor_score.topk(pre_nmn_top_num, dim=1)
            # offset for different feature maps.
            top_index += offset
            index_list.append(top_index)
            offset += anchor_num
        top_index = torch.cat(index_list, dim=1)
        cls = cls[batch_size - 1,  top_index]
        proposals = proposals[batch_size - 1, top_index]

        # *** do NMS *** #
        # make Mask - for NMS: make distance of different feature map anchor as far as possible.
        feature_mask = [torch.full((anchor_num,), i) for i, anchor_num in enumerate(anchor_num_list)]
        feature_mask = torch.cat(feature_mask, dim=0).reshape(1, -1)
        feature_mask = feature_mask.expand_as(torch.randn(size=(batch_size, feature_mask.size()[1])))

        filter_proposals = []
        filter_scores = []
        for box, score, mask in zip(proposals, cls, feature_mask):
            # clip box
            min_x = torch.clamp(box[:, 0], min=0, max=image_width)[:, None]
            min_y = torch.clamp(box[:, 1], min=0, max=image_height)[:, None]
            max_x = torch.clamp(box[:, 2], min=0, max=image_width)[:, None]
            max_y = torch.clamp(box[:, 3], min=0, max=image_height)[:, None]
            box = torch.cat([min_x, min_y, max_x, max_y], dim=1)
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            # remove small box
            keep = torch.logical_and(torch.ge(box_w, self.min_size), torch.ge(box_h, self.min_size))
            keep = torch.where(keep)[0]
            box = box[keep]
            score = score[keep]
            mask = mask[keep]
            # make distance of different feature map anchor as far as possible
            offset = mask.to(box) * (box.max() + 1)
            box_for_mns = box + offset[:, None]
            # mns
            keep = nms(box_for_mns, score, self.nms_thresh)[:self.post_nms_top_num]

            box, score = box[keep], score[keep]
            filter_proposals.append(box)
            filter_scores.append(score)

        losses = {}
        if self.training:
            # *** 1、get the best matched ground truth for every anchor *** #
            # *** 2、classify anchor in (background,foreground,middle) *** #
            anchor_label_list = []
            ground_truth_list = []
            for anchor, target in zip(anchor_list, targets):
                # compute anchors iou value with per ground truth.
                ground_truth = torch.tensor(target['box'])
                anchor = anchor.view(-1, 4)
                anchor_w = anchor[:, 2] - anchor[:, 0]
                anchor_h = anchor[:, 3] - anchor[:, 1]
                anchor_area = anchor_w * anchor_h

                ground_truth_w = ground_truth[:, 2] - ground_truth[:, 0]
                ground_truth_h = ground_truth[:, 3] - ground_truth[:, 1]
                ground_truth_area = ground_truth_w * ground_truth_h
                # get left top point and right bottom point.
                left_top = torch.max(anchor[:, None, :2], ground_truth[:, :2])
                right_bottom = torch.min(anchor[:, None, 2:], ground_truth[:, 2:])
                join_part = (right_bottom - left_top).clamp(min=0)
                join_area = join_part[:, :, 0] * join_part[:, :, 1]
                iou = join_area / (anchor_area[:, None] + ground_truth_area - join_area)

                max_iou, max_index = iou.max(dim=1)
                neg_position = torch.lt(max_iou, self.bg_iou_thresh)
                pos_position = torch.gt(max_iou, self.fg_iou_thresh)
                mid_position = torch.logical_and(torch.gt(max_iou, self.bg_iou_thresh),
                                                  torch.lt(max_iou, self.fg_iou_thresh))
                max_iou[neg_position] = 0
                max_iou[pos_position] = 1
                max_iou[mid_position] = -1

                label = max_iou
                anchor_label_list.append(label)
                ground_truth_list.append(ground_truth[max_index])

            # *** get regression labels by anchor and matched ground truth *** #
            anchors = torch.cat(anchor_list, dim=0).view(-1, 4)
            labels = torch.cat(anchor_label_list, dim=0)
            ground_truths = torch.cat(ground_truth_list, dim=0)

            anchor_min_x = anchors[:,0][:,None]
            anchor_min_y = anchors[:,1][:,None]
            anchor_max_x = anchors[:,2][:,None]
            anchor_max_y = anchors[:,3][:,None]
            anchor_w = anchor_max_x - anchor_min_x
            anchor_h = anchor_max_y - anchor_min_y
            anchor_center_x = anchor_min_x + 0.5 * anchor_w
            anchor_center_y = anchor_min_y + 0.5 * anchor_h

            gt_min_x = ground_truths[:,0][:,None]
            gt_min_y = ground_truths[:,1][:, None]
            gt_max_x = ground_truths[:,2][:, None]
            gt_max_y = ground_truths[:,3][:, None]
            gt_w = gt_max_x - gt_min_x
            gt_h = gt_max_y - gt_min_y
            gt_center_x = gt_min_x + 0.5 * gt_w
            gt_center_y = gt_min_y + 0.5 * gt_h

            dx = (gt_center_x - anchor_center_x) / anchor_w
            dy = (gt_center_y - anchor_center_y) / anchor_h
            dw = torch.log(gt_w / anchor_w)
            dh = torch.log(gt_h / anchor_h)
            regressions = torch.cat([dx, dy, dw, dh],dim=1)

            # *** compute regression loss and classify loss *** #
            pos_idx = []
            neg_idx = []
            for anchor_label in anchor_label_list:
                positive = torch.where(torch.eq(anchor_label, 1))[0]
                negative = torch.where(torch.eq(anchor_label, 0))[0]
                num_pos = int(min(positive.numel(), self.sample_num * self.positive_fraction))
                num_neg = int(min(negative.numel(), self.sample_num - num_pos))
                # random pos index and neg index
                pos_idx_per_image = positive[torch.randperm(positive.numel())[:num_pos]]
                neg_idx_per_image = negative[torch.randperm(negative.numel())[:num_neg]]

                pos_idx_per_image_mask = torch.zeros_like(anchor_label, dtype=torch.uint8)
                neg_idx_per_image_mask = torch.zeros_like(anchor_label, dtype=torch.uint8)
                pos_idx_per_image_mask[pos_idx_per_image] = 1
                neg_idx_per_image_mask[neg_idx_per_image] = 1
                pos_idx.append(pos_idx_per_image_mask)
                neg_idx.append(neg_idx_per_image_mask)
            sampled_pos_inds = torch.where(torch.cat(pos_idx, dim=0))[0]
            sampled_neg_inds = torch.where(torch.cat(neg_idx, dim=0))[0]
            sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)
            labels = torch.cat(anchor_label_list, dim=0)
            print(anchor_label_list[0].size(),anchor_label_list[1].size())
            print(labels.size(),sampled_inds.size())
        return filter_proposals, losses


if __name__ == '__main__':
    input_images = torch.randn(size=(2, 3, 281, 500))
    input_feature_maps = [torch.randn(2, 3, 112, 112), torch.randn(2, 3, 64, 64)]
    input_targets = [{'size': [3.0, 281, 500], 'class': ['dog'], 'box': [[104, 78, 375, 183],[133, 88, 197, 123]]}, {'size': [3.0, 500.0, 375.0], 'class': ['cat'], 'box': [[104, 78, 375, 183]]}]

    input_in_channels = 3
    input_num_anchors = 9
    input_rpn_head = RPNHead(input_in_channels, input_num_anchors)
    input_anchor_generator = AnchorGenerator()

    input_pre_nmn_top_num = 2000
    input_post_nms_top_num = 2000

    input_nms_thresh = 0.7

    input_fg_iou_thresh = 0.7
    input_bg_iou_thresh = 0.3

    input_sample_num = 256
    input_positive_fraction = 0.5

    rpn = RPN(input_anchor_generator, input_rpn_head,
              input_pre_nmn_top_num, input_post_nms_top_num,
              input_nms_thresh, input_fg_iou_thresh,
              input_bg_iou_thresh, input_sample_num,
              input_positive_fraction)
    rpn(input_images, input_feature_maps, input_targets)