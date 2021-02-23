# -*- coding:utf-8 -*-
"""
image data pre-process transform

Description:
    define different image pre-process transform.

"""
# **** modification history ***** #
# 2021/01/18,by junlu Ding,create #

import torch
import numpy as np
from torchvision.transforms import functional as F


class Compose:
    """
    compose different image transform to list, and process
    image,target by per transform in list.

    Args:
       transforms (List): different transform list.
    Return:
       processed image and target by transforms.
    """
    def __init__(self,transforms):
        self.transforms = transforms
        return

    def __call__(self, image, target):
        for transform in self.transforms:
            image, target = transform(image, target)
        return image, target

# convert PIL image to Tensor
class ToTensor:
    def __call__(self, image, target):
        # to_tensor —> image channel dimension position advance
        # tensor -> image channel dimension position remains unchanged
        image =  F.to_tensor(image)
        target['label'] = torch.tensor(target['label'], dtype=torch.int)
        target['box'] = torch.tensor(target['box'], dtype=torch.float)
        return image, target

class Normalize:
    def __init__(self,mean,std,device=None,d_type=None):
        self.mean = torch.tensor(mean,device=device,dtype=d_type)
        self.std = torch.tensor(std,device=device,dtype=d_type)
        return

    def __call__(self, image, target):
        image = (image - self.mean[:,None,None]) / self.std[:,None,None]
        return image, target

# horizontal flip image and box position
class RandomHorizontalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if np.random.random() > self.prob:
            # flip image
            image = image.flip(-1)
            # flip box
            image_width = image.size(2)
            box = target['box']
            min_x = image_width - box[:, 0]
            min_y = box[:, 1]
            max_x = image_width - box[:, 2]
            max_y = box[:, 3]
            target['box'] = torch.cat([min_x[:,None],
                                       min_y[:,None],
                                       max_x[:,None],
                                       max_y[:,None]],dim=1)
        return image, target
