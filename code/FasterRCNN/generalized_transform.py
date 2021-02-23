# -*- coding:utf-8 -*-

"""
generalized process batch image.

Description:
  perform a generalized process on the batch image to make
  the batch image size same.

"""
# ***** modification history *****
# ********************************
# 2021/02/10, by junlu Ding, create

import torch
from torch.nn import Module
from torch.nn.functional import interpolate

from utils import batched_images

class GeneralizedTransform(Module):
    """
    Firstly, the bilinear interpolation algorithm is used to process
    per image in the batch, the scale factor of image is determined
    by the ratio of the maximum(minimum) size with the real size of image.
    In addition, the image box is also scaled according to the factor.
    Finally, make the batch image size same.

    Args:
        min_size (int): the minimum size in batch images.
        max_size (int): the maximum size in batch images.

    Return:
        images (Tensor): generally processed batch images.
        targets (List[Dict]): generally processed targets.

    """
    def __init__(self,min_size,max_size):
        super(GeneralizedTransform, self).__init__()
        self.min_size = min_size
        self.max_size = max_size
        return

    def forward(self,images,targets):

        for i in range(len(images)):
            image = images[i]
            target = targets[i]

            height, width = image.size(1), image.size(2)
            min_size = float(min(height, width))
            max_size = float(max(height, width))

            scale_factor = self.min_size / min_size
            if min_size * scale_factor >= self.min_size:
                scale_factor = self.max_size / max_size

            image = interpolate(image[None, :, :, :],
                                scale_factor=scale_factor,
                                mode='bilinear',
                                recompute_scale_factor=True,
                                align_corners=False)[0]

            images[i] = image

            if target is not None:
                box = target['box']

                ratio_height = torch.tensor([height/image.size(-2)],
                                             dtype=box.dtype,
                                             device=box.device)
                ratio_width = torch.tensor([width/image.size(-1)],
                                            dtype=box.dtype,
                                            device=box.device)

                min_x = box[:, 0] * ratio_width
                min_y = box[:, 1] * ratio_height
                max_x = box[:, 2] * ratio_width
                max_y = box[:, 3] * ratio_height

                target['box'] = torch.cat([min_x,min_y,max_x,max_y])
        images = batched_images(images,divide_size=32)
        return images, targets


if __name__ == '__main__':
    generalized_transform = GeneralizedTransform(max_size=1000,min_size=800)
