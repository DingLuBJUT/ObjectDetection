import math
import torch
from torch.nn import Module
from torchvision.ops import nms

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
            proposal with nms (*, 4)
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
        nms_proposals = torch.cat(list_nms_proposals, dim=0)
        return nms_proposals

    def smooth_l1_loss(self):
        return

    def _get_anchor_class_labels(self, list_anchors, list_targets):
        """

        args:
        return:
        """
        cls_label = None
        reg_label = None
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
            cls_label = max_value
            best_match_ground_truth = ground_truth[max_index]


            break

        return cls_label, reg_label

    def _get_anchor_regression_label(self):
        return

    def forward(self, images, list_features, list_targets):
        batch_size, image_h, image_w = images.size()[0], images.size()[2], images.size()[3]
        list_anchors = self.anchor_generator(images,list_features)
        list_cls, list_regs = self.rpn_head(list_features)

        list_num_anchors = [cls.size()[1] * cls.size()[2] * cls.size()[3] for cls in list_cls]

        # get top score anchor index
        list_top_index, list_num_pre_nms = self._get_top_score_index(list_cls, list_num_anchors, batch_size)

        # get top cls cls top index
        list_top_cls = self._get_top_cls(list_cls, list_top_index)

        # get top proposal by top index
        list_top_proposals = self._get_top_proposals(list_regs, list_anchors, list_top_index, list_num_anchors)

        # nms for top proposal
        nms_proposals = self._nms(list_top_cls, list_top_proposals, batch_size, list_num_pre_nms,image_h, image_w)

        if self.training:
            self._get_anchor_labels(list_anchors, list_targets)
        return


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
    params['num_pre_nms'] = 5
    params['num_post_nms'] = 5
    params['nms_threshold'] = 0.7
    params['nms_min_size'] = 1e-3
    params['pos_threshold'] = 0.7
    params['neg_threshold'] = 0.3
    params['pos_fraction'] = 0.5

    rpn = RPN(params)
    rpn(input_images, input_feature_list, input_targets)

