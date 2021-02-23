# -*- coding:utf-8 -*-

"""
faster r-cnn roi head module.

Description:

    if in training mode, tagging the class and regression label, and then randomly
    sample each image proposals that output by rpn according to a proportion.and
    then perform roi pooling and mlp on the sampled proposal to obtain class and
    regression output, and finally the regression and classification loss are
    computed by tagging label and class and regression output.

    if in testing mode, firstly perform roi pooling and mlp similar to the training,
    and than perform different filtering operations including softMax、assign label、
    regression proposal、clip proposal、remove low score、small、label is '0' proposal、
    nms proposal、select top nms result.

"""
# **** modification history ****  #
# ******************************  #
# 2021/01/17,by Junlu Ding,create #

import torch
from torch.nn import Module
from torchvision.ops import nms
import torch.nn.functional as F

from mlp_head import TwoMLPHead
from roi_pooling import ROIPooling
from predictor import FasterRCNNPredictor
from utils import box_deviation
from utils import smooth_l1_loss
from RPN.rpn import rpn_test_data
from utils import tagging_box
from utils import get_sampling_factor
from utils import box_clip
from utils import box_regression
from utils import remove_small_box


class ROIHead(Module):
    def __init__(self, params):
        super(ROIHead, self).__init__()
        self.roi_pooling = params['roi_pooling']
        self.mlp_head = params['mlp_head']
        self.predictor = params['predictor']
        self.pos_threshold = params['pos_threshold']
        self.neg_threshold = params['neg_threshold']
        self.num_sample = params['num_sample']
        self.pos_fraction = params['pos_fraction']
        self.regression_weights = params['regression_weights']
        self.score_thresh = params['score_thresh']
        self.proposal_min_size = params['proposal_min_size']
        self.nms_threshold = params['nms_threshold']
        self.num_post_nms = params['num_post_nms']
        return


    def forward(self, list_rpn_proposals, list_features, batch_image_info, list_targets, list_image_size):
        """
        roi head inference.

        Args:
            list_rpn_proposals（List[Tensor]): rpn output proposal for per image.

            list_features (List[Tensor]): backbone different output feature for
            batch image.

            batch_image_info (Tuple): batch transformed image size info.

            list_targets (List[Dict]): batch original image info including original
            size、object class、ground truth.

            list_image_size (List[Tuple]):

        Example:
            list_rpn_proposals: [(64, 4]), (35, 4)]
            list_features: [(2, 3, 112, 112),(2, 3, 64, 64),(2, 3, 32, 32)]
            batch_image_info: (2, 256, 256)
            list_targets: [{'size':[3.0, 500.0, 375.0],'class':[2.0],'box':[[104, 78, 375, 183]]}]
            list_image_size: [(267, 226),(289, 271)]

        Return:
            roi head predict proposal、score、label.
            roi head cls and reg train loss.

        """

        self.train(mode=False)
        image_height = batch_image_info[1]
        image_width = batch_image_info[2]
        device, data_type = list_rpn_proposals[0].device, list_rpn_proposals[0].dtype

        list_cls_label = []
        list_reg_label = []
        list_proposals = []
        if self.training:
            for proposal, target in zip(list_rpn_proposals, list_targets):
                labels = torch.as_tensor(target['class'],device=device)
                ground_truth = torch.as_tensor(target['box'],device=device,
                                               dtype=data_type)
                # tagging cls label
                cls_label = tagging_box(proposal,ground_truth,
                                        self.neg_threshold,
                                        self.pos_threshold,
                                        labels)
                # tagging reg label
                reg_label = box_deviation(proposal, ground_truth,
                                          self.regression_weights)

                # random sampling
                pos_index = torch.where(torch.gt(cls_label, 0))[0]
                neg_index = torch.where(torch.eq(cls_label, 0))[0]

                pos_factor, neg_factor = get_sampling_factor(self.num_sample,
                                                             self.pos_fraction,
                                                             pos_index.size(0),
                                                             neg_index.size(0))

                random_index = torch.cat([neg_index[neg_factor], pos_index[pos_factor]])
                list_cls_label.append(cls_label[random_index])
                list_reg_label.append(reg_label[random_index])
                list_proposals.append(proposal[random_index])
                list_rpn_proposals = list_proposals

        dict_feature = dict(zip([str(i + 1) for i in range(len(list_features))], list_features))
        # roi pooling - mapping proposal to different features, and uniform output size
        roi_features = self.roi_pooling(dict_feature, list_rpn_proposals,
                                        [(image_height, image_width)])
        # mlp head - flatten roi pooling and input fully connected layer
        mlp_features = self.mlp_head(roi_features)
        # predictor - predict class score and reg params.
        predict_cls, predict_reg = self.predictor(mlp_features)
        predict_reg = predict_reg.view(predict_cls.size(0), -1, 4)

        roi_loss = {}
        result = []

        if self.training:
            cls_label = torch.cat(list_cls_label)
            reg_label = torch.cat(list_reg_label)
            # compute cls loss
            cls_loss = F.cross_entropy(predict_cls, cls_label.long())

            # compute reg loss
            pos_index = torch.where(torch.gt(cls_label,0))[0]
            pos_index_label = cls_label[pos_index].long()

            predict_reg = predict_reg[pos_index,pos_index_label]
            reg_label = reg_label[pos_index]
            reg_loss = smooth_l1_loss(predict_reg,reg_label)
            roi_loss = {
                "cls_loss": cls_loss,
                "reg_loss": reg_loss
            }
        else:
            # 1、softMax cls.
            # 2、assign label for proposal.
            # 3、regression proposal by reg param.
            # 4、clip proposal.
            # 5、remove low score proposal.
            # 5、remove small proposal.
            # 6、remove label is '0' proposal.
            # 7、nms proposal.
            # 8、select top nms result.
            num_class = predict_cls.size(-1)
            # softMax cls
            predict_score = F.softmax(predict_cls, -1)
            num_proposals_per_image = [proposal.size(0) for proposal in list_rpn_proposals]

            list_predict_score = torch.split(predict_score,num_proposals_per_image)
            list_predict_reg = torch.split(predict_reg,num_proposals_per_image)

            list_proposal = []
            list_score = []
            list_label = []
            for i in range(len(list_rpn_proposals)):
                proposal = list_rpn_proposals[i]
                score = list_predict_score[i]
                reg = list_predict_reg[i]
                image_size = list_image_size[i]

                # assign label for proposal
                label = torch.arange(num_class,device=device)
                label = label[None:,].expand_as(score)

                # regression proposal by reg param
                proposal = box_regression(proposal,reg.flatten(1,2)).view(-1,4)
                # clip proposal
                proposal = box_clip(proposal, image_size)

                score = score.flatten()
                label = label.flatten()

                # remove low score proposal
                index = torch.where(torch.gt(score,self.score_thresh))[0]
                proposal = proposal[index]
                score = score[index]
                label = label[index]

                # remove small proposal
                _, index = remove_small_box(proposal,self.proposal_min_size)
                proposal = proposal[index]
                score = score[index]
                label = label[index]

                # remove label is '0' proposal
                index = torch.where(torch.gt(label,0))[0]
                proposal = proposal[index]
                score = score[index]
                label = label[index]

                # nms proposal and select top nms result
                index = nms(proposal, score, self.nms_threshold)[:self.num_post_nms]
                proposal = proposal[index]
                score = score[index]
                label = label[index]

                list_proposal.append(proposal)
                list_score.append(score)
                list_label.append(label)

            for i in range(len(list_proposal)):
                result.append({
                    'proposal': list_proposal[i],
                    'score': list_score[i],
                    'label': list_label[i]
                })

        return result, roi_loss


if __name__ == '__main__':

    batch_image_info = (2, 256, 256)
    list_rpn_proposals, rpn_loss, list_features = rpn_test_data()
    list_targets = [
        {'size': [3.0, 281, 500], 'class': [1.0], 'box': [[104, 78, 375, 183], [133, 88, 197, 123]]},
        {'size': [3.0, 500.0, 375.0], 'class': [2.0], 'box': [[104, 78, 375, 183]]}
    ]

    params = dict()
    # default 0.5
    params['pos_threshold'] = 0.01
    # default 0.5
    params['neg_threshold'] = 0.01
    params['num_sample'] = 512
    params['pos_fraction'] = 0.25
    params['regression_weights'] = [10., 10., 5., 5.]
    params['score_thresh'] = 0.05
    params['proposal_min_size'] = 1e-3
    params['nms_threshold'] = 0.5
    params['num_post_nms'] = 1000


    roi_feature_names = [str(i + 1) for i in range(len(list_features))]
    roi_output_size = [7, 7]
    roi_sampling_ratio = 2
    params['roi_pooling'] = ROIPooling(roi_feature_names, roi_output_size, roi_sampling_ratio)

    representation_size = 1024
    num_channels = 3
    num_in_pixels = num_channels * roi_output_size[0] * roi_output_size[1]
    params['mlp_head'] = TwoMLPHead(num_in_pixels, representation_size)

    num_class = 3
    params['predictor'] = FasterRCNNPredictor(representation_size, num_class)

    roi_head = ROIHead(params)
    print(roi_head.state_dict().keys())
    list_image_size = [(267, 226),(289, 271)]
    result, roi_loss= roi_head(list_rpn_proposals, list_features, batch_image_info, list_targets, list_image_size)
    print(result)
    print(roi_loss)