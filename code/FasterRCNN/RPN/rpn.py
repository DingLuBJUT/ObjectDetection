# -*- coding:utf-8 -*-

import math
import torch
from torch.nn import Module
from torchvision.ops import nms
from  code.FasterRCNN.RPN.rpn_head import RPNHead
from  code.FasterRCNN.RPN.anchor_generator import AnchorGenerator

class RPN(Module):
    def __init__(self,anchor_generator,rpn_head,
                 pre_nmn_top_num,post_nms_top_num,
                 fg_iou_thresh,bg_iou_thresh,
                 nms_thresh):
        super(RPN, self).__init__()
        self.anchor_generator = anchor_generator
        self.rpn_head = rpn_head
        self.pre_nmn_top_num = pre_nmn_top_num
        self.post_nms_top_num = post_nms_top_num
        self.nms_thresh = nms_thresh
        self.fg_iou_thresh = fg_iou_thresh
        self.bg_iou_thresh = bg_iou_thresh
        self.min_size = 1e-3
        return

    def adjust_anchor_by_reg(self):
        return

    def forward(self,images, feature_maps, targets=None):
        batch_size = images.size()[0]
        image_height, image_width = images.size()[2: ]

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
        total_anchor_num = sum([anchor.size()[0] for anchor in anchor_list])
        total_anchor = torch.cat(anchor_list, dim=0)
        reg = reg.reshape(total_anchor_num, -1)

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

        proposals = torch.cat([min_x,min_y,max_x,max_y],dim=1)
        proposals = proposals.reshape(total_anchor_num,-1,4)
        proposals = proposals.view(batch_size ,-1, 4)

        # *** filter proposals. *** #
        anchor_num_list = [cls.size()[1] * cls.size()[2] * cls.size()[3] for cls in cls_list]

        # detach from graph
        cls = cls.detach()
        cls = cls.reshape(batch_size, -1)

        # make Mask
        feature_mask = [torch.full((anchor_num,), i)
                        for i, anchor_num in enumerate(anchor_num_list)]
        feature_mask = torch.cat(feature_mask, dim=0)
        feature_mask = feature_mask.reshape(1, -1).expand_as(cls)

        # get pre mns proposals in different feature maps.
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

            # get proposals that do clip、remove small and NMS
        filter_proposals = []
        filter_scores = []
        for box, score, mask in zip(proposals, cls, feature_mask):
            min_x = torch.clamp(box[:, 0], min=0, max=image_width)[:,None]
            min_y = torch.clamp(box[:, 1], min=0, max=image_height)[:,None]
            max_x = torch.clamp(box[:, 2], min=0, max=image_width)[:,None]
            max_y = torch.clamp(box[:, 3], min=0, max=image_height)[:,None]
            box = torch.cat([min_x,min_y,max_x,max_y], dim=1)
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            keep = torch.logical_and(torch.ge(box_w, self.min_size),
                                     torch.ge(box_h, self.min_size))
            keep = torch.where(keep)[0]
            box = box[keep]
            score = score[keep]
            mask = mask[keep]
            offset = mask.to(box) * (box.max() + 1)

            box_for_mns = box + offset[:, None]
            keep = nms(box_for_mns,score, self.nms_thresh)[:self.post_nms_top_num]

            box, score = box[keep], score[keep]
            filter_proposals.append(box)
            filter_scores.append(score)


        losses= {}
        if self.training:
            # *** get pos and neg data sample. *** #
            for anchor, target in zip(anchor_list, targets):
                # get iou value that anchors for per ground truth.
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
                neg_condition = torch.lt(max_iou,self.bg_iou_thresh)
                pos_condition = torch.gt(max_iou,self.fg_iou_thresh)
                mid_condition = torch.logical_and(torch.gt(max_iou,self.bg_iou_thresh),
                                                  torch.lt(max_iou,self.fg_iou_thresh))
                max_iou[neg_condition] = 0
                max_iou[pos_condition] = 1
                max_iou[mid_condition] = -1

                anchor_label = max_iou
                anchor_match_gt = ground_truth[max_index]
                break
        return filter_proposals, losses



if __name__ == '__main__':
    images = torch.randn(size=(2, 3, 281, 500))
    feature_maps = [torch.randn(2, 3, 112, 112), torch.randn(2, 3, 64, 64)]
    targets = [{'size': [3.0, 281, 500], 'class': ['dog'], 'box': [[104, 78, 375, 183],[133, 88, 197, 123]]}, {'size': [3.0, 500.0, 375.0], 'class': ['cat'], 'box': [[104, 78, 375, 183]]}]
    in_channels = 3
    num_anchors = 9
    rpn_head = RPNHead(in_channels, num_anchors)
    anchor_generator = AnchorGenerator()

    pre_nmn_top_num = 2000
    post_nms_top_num = 2000

    nms_thresh = 0.7

    fg_iou_thresh = 0.7
    bg_iou_thresh = 0.3

    rpn = RPN(anchor_generator,rpn_head,
              pre_nmn_top_num,post_nms_top_num,
              nms_thresh,
              fg_iou_thresh, bg_iou_thresh)
    rpn(images,feature_maps,targets)