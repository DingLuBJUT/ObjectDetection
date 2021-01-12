import math
import torch
from torch.nn import Module
from anchor_generator import AnchorGenerator
from rpn_head import RPNHead

class RPN(Module):
    def __init__(self, dict_params):
        super(RPN, self).__init__()
        self.anchor_generator = dict_params['anchor_generator']
        self.rpn_head = dict_params['rpn_head']
        self.num_pre_nms_top = dict_params['num_pre_nms_top']
        self.nms_threshold = dict_params['nms_threshold']

        return

    def _adjust_anchor_by_regression(self, regs, anchors):
        """
        adjust anchor by regression params.
        Args:
            list_regs: rpn head output regs for per feature.
            list_anchors: anchor generator output anchor for per img.
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

    # def _make_proposals(self, list_regs, list_anchors, batch_size):
    #     """
    #     adjust anchor by regression params.
    #     Args:
    #         list_regs: rpn head output regs for per feature.
    #         list_anchors: anchor generator output anchor for per img.
    #         batch_size: input batch image num
    #     Return:
    #         proposals [batch_size, n, 4]
    #     """
    #     anchors = torch.cat(list_anchors, dim=0).view(-1, 4)
    #     regs = torch.cat([reg.permute(0, 2, 3, 1).reshape(-1, 4)
    #         for reg in list_regs
    #     ], dim=0)
    #     dx, dy, dh, dw = regs[:, 0], regs[:, 1], regs[:, 2], regs[:, 3]
    #     dw = torch.clamp(dw, max=math.log(1000. / 16))
    #     dh = torch.clamp(dh, max=math.log(1000. / 16))
    #
    #     anchor_h = anchors[:, 3] - anchors[:, 1]
    #     anchor_w = anchors[:, 2] - anchors[:, 0]
    #     center_x = anchor_w / 2
    #     center_y = anchor_h / 2
    #     proposal_x = center_x + dx * anchor_w
    #     proposal_y = center_y + dy * anchor_h
    #     proposal_h = torch.exp(dh) * anchor_h
    #     proposal_w = torch.exp(dw) * anchor_w
    #     min_x = (proposal_x - proposal_w / 2)[:,None]
    #     min_y = (proposal_y - proposal_h / 2)[:,None]
    #     max_x = (proposal_x + proposal_w / 2)[:,None]
    #     max_y = (proposal_y + proposal_h / 2)[:,None]
    #     proposals = torch.cat([min_x,min_y,max_x,max_y],dim=1)
    #     proposals = proposals.view(batch_size,-1,4)
    #     return proposals


    def _get_top_score_index(self,list_cls, batch_size):
        """
        get top k anchor score index in per feature.
        args:
            list_regs: rpn head output regs for per feature.
            batch_size: batch image num
        return:
            index list [(batch_size, top_k)]
        """
        top_index_list = []
        offset = 0
        for score in list_cls:
            score = score.permute(0, 2, 3, 1).reshape(batch_size, -1, 1)
            num_anchor = score.size()[1]
            num_pre_nms_top = min(self.num_pre_nms_top, num_anchor)
            _, index = torch.topk(score, k=num_pre_nms_top, dim=1)
            index += offset
            offset += num_anchor
            top_index_list.append(index.squeeze()-1)
        print([index.size() for index in top_index_list])
        return top_index_list

    def _get_proposals(self,list_regs, top_index_list):
        for reg, index in zip(list_regs,top_index_list):
            batch_size, channels, _ ,_ = reg.size()
            reg = reg.permute(0, 2, 3, 1).reshape(batch_size, -1, channels)
            print(index.size(),reg.size())

        return



    # def _get_top_proposal(self, list_cls, proposals, batch_size):
    #     """
    #     get top k proposal by score in per img、per feature
    #     Args:
    #         list_cls: rpn head output cls for per feature.
    #         proposals: _make_proposals output.
    #         batch_size: input batch image num.
    #     Return:
    #         proposals [batch_size, n, 4]
    #     """
    #     offset = 0
    #     list_index = []
    #     for score in list_cls:
    #         score = score.permute(0, 2, 3, 1).reshape(batch_size, -1, 1)
    #         num_anchor = score.size()[1]
    #         num_pre_nms_top = min(self.num_pre_nms_top, num_anchor)
    #         # get top score in per batch
    #         _, index = torch.topk(score, k=num_pre_nms_top, dim=1)
    #         index += offset
    #         offset += num_anchor
    #         list_index.append(index)
    #     # cat features index
    #     top_index = torch.cat(list_index, dim=1).view(batch_size,-1)
    #     list_top_proposal = []
    #     for index, proposal in zip(top_index, proposals):
    #         list_top_proposal.append(proposal[index][None,:])
    #     proposals = torch.cat(list_top_proposal, dim=0)
    #     return proposals

    def _nms(self, list_cls, proposals):

        return

    def forward(self, images, feature_list):
        batch_size = images.size()[0]
        list_anchors = self.anchor_generator(images,feature_list)
        list_cls, list_regs = self.rpn_head(feature_list)
        top_index_list = self._get_top_score_index(list_cls, batch_size)
        # self._get_proposals(list_regs,top_index_list)



        # self._get_top_score_index(list_cls,batch_size)
        # proposals = self._make_proposals(list_regs, list_anchors, batch_size)
        # proposals = self._get_top_proposal(list_cls,proposals,batch_size)
        # print(proposals.size())
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
    params['num_pre_nms_top'] = 2000
    params['nms_threshold'] = 0.7
    rpn = RPN(params)
    rpn(input_images,input_feature_list)

