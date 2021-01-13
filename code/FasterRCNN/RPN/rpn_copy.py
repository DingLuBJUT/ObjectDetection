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
        self.nms_threshold = dict_params['nms_threshold']
        self.nms_min_size = dict_params['nms_min_size']

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

        min_x = (proposal_x - proposal_w / 2)[:,None]
        min_y = (proposal_y - proposal_h / 2)[:,None]
        max_x = (proposal_x + proposal_w / 2)[:,None]
        max_y = (proposal_y + proposal_h / 2)[:,None]
        proposals = torch.cat([min_x,min_y,max_x,max_y],dim=1)
        return proposals

    def _get_top_score_index(self, list_cls, list_num_anchors, batch_size):
        """
        get top k anchor score index in per feature.
        args:
            list_regs: rpn head output regs for per feature.
            list_num_anchors: anchor num for per batch feature.
            batch_size: batch image num.
        return:
            index list [index,]
        """
        list_top_index = []
        list_num_pre_nms = []
        for score, num_anchors in zip(list_cls,list_num_anchors):
            score = score.permute(0, 2, 3, 1).flatten(1,3)
            num_pre_nms = min(self.num_pre_nms, num_anchors)
            _, index = torch.topk(score, k=num_pre_nms, dim=1)
            index = index.flatten()
            offset = torch.cat([torch.full((num_pre_nms,),i) * num_anchors for i in range(batch_size)])
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
        anchors = torch.cat([anchors[None,:] for anchors in list_anchors],dim=0)
        anchors = anchors.flatten(1, 2)

        list_top_proposals = []
        # get anchors of per feature
        feature_anchors = torch.split(anchors, list_num_anchors, dim=1)
        for anchors, regs, index in zip(feature_anchors, list_regs, list_top_index):
            regs = regs.permute(0, 2, 3, 1).flatten(0, 2).reshape(-1,4)
            anchors = anchors.flatten(0, 1)
            regs, anchors = regs[index], anchors[index]
            proposals = self._adjust_anchor_by_regression(regs, anchors)
            list_top_proposals.append(proposals)
        return list_top_proposals

    def _nms(self, list_top_cls, list_top_proposals, batch_size, list_num_pre_nms, image_h, image_w):

        list_nms_proposals = []
        cls = torch.cat([cls.view(batch_size, -1) for cls in list_top_cls], dim=1)
        proposals = torch.cat([proposal.view(batch_size, -1, 4) for proposal in list_top_proposals], dim=1)
        for score, proposal in zip(cls,proposals):

            offset = torch.cat([torch.full((num,), i) for i, num in enumerate(list_num_pre_nms)])
            offset = offset[:, None] * proposal.max().to(offset)

            min_x = torch.clamp(proposal[:, 0], min=0, max=image_w)
            min_y = torch.clamp(proposal[:, 1], min=0, max=image_h)
            max_x = torch.clamp(proposal[:, 2], min=0, max=image_w)
            max_y = torch.clamp(proposal[:, 3], min=0, max=image_h)
            proposal_w, proposal_h = max_x - min_x, max_y - min_y
            keep = torch.logical_and(torch.ge(proposal_w,self.nms_min_size),torch.ge(proposal_h,self.nms_min_size))
            index = torch.where(keep)[0]
            score = score[index]
            proposal = proposal[index]
            offset = offset[index]
            proposal = proposal + offset
            nms_proposal = nms(proposal,score,self.nms_threshold)
            list_nms_proposals.append(nms_proposal)
        nms_proposals = torch.cat(list_nms_proposals,dim=0).view(batch_size, -1, 4)
        return nms_proposals

    def forward(self, images, feature_list):
        batch_size, image_h, image_w = images.size()[0], images.size()[2], images.size()[3]
        list_anchors = self.anchor_generator(images,feature_list)
        list_cls, list_regs = self.rpn_head(feature_list)

        list_num_anchors = [cls.size()[1] * cls.size()[2] * cls.size()[3] for cls in list_cls]

        list_top_index,list_num_pre_nms = self._get_top_score_index(list_cls, list_num_anchors, batch_size)
        list_top_proposals = self._get_top_proposals(list_regs, list_anchors, list_top_index, list_num_anchors)
        list_top_cls = self._get_top_cls(list_cls, list_top_index)
        self._nms(list_top_cls, list_top_proposals, batch_size, list_num_pre_nms,image_h, image_w)

        return



if __name__ == '__main__':
    input_images = torch.randn(size=(2, 3, 281, 500))
    input_feature_list = [
        torch.rand(size=(2,3, 112, 112)),
        torch.rand(size=(2,3, 64, 64))
    ]
    params = dict()
    params['anchor_generator'] = AnchorGenerator()
    params['rpn_head'] = RPNHead(in_channels=3, num_pixel_anchors=9)
    params['num_pre_nms'] = 5
    params['nms_threshold'] = 0.7
    params['nms_min_size'] = 1e-3
    rpn = RPN(params)
    rpn(input_images,input_feature_list)

