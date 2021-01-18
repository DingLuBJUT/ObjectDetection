# -*- coding:utf-8 -*-

"""
create Dataset for voc 2012 data set

Description:
    create voc 2012 Dataset by extends torch.utils.data.Dataset, and pre_process
    image by Composed different transforms for per image.
"""
# **** modification history ***** #
# 2021/01/18,by junlu Ding,create #

import os
import json
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

from utils import parse_xml
from tranform import Transform

ROOT_PATH = "/Users/dingjunlu/PycharmProjects/ObjectDetection/data/VOC2012/"


class VOCDataSet(Dataset):
    """
    make voc2012 data to Dataset object for data batch loading, and return
    transformed batch images and image box、class label.
    args:
      image_dir (str): jpeg images directory path in voc2012 data set.
      annotation_dir (str): annotation xml directory path in voc2012 data set.
      file_name_path (str):
      class_json_path (str): object label mapping json data path in in voc2012 data set.
      transforms : compose transform object.

    """
    def __init__(self, image_dir, annotation_dir, file_name_path, class_json_path, transforms=None):
        super(VOCDataSet, self).__init__()
        self.image_dir = os.path.join(ROOT_PATH, image_dir)
        self.annotation_dir = os.path.join(ROOT_PATH, annotation_dir)

        self.list_file_names = []
        with open(os.path.join(ROOT_PATH, file_name_path)) as f:
            lines = f.readlines()
            for name in lines:
                self.list_file_names.append(name.strip())
        self.dict_class = json.loads(open(os.path.join(ROOT_PATH, class_json_path), 'r').read())
        self.transforms = transforms
        return

    def __len__(self):
        return len(self.list_file_names)

    def __getitem__(self, index):
        file_name = self.list_file_names[index]
        numpy_image = np.array(Image.open(os.path.join(self.image_dir, file_name + ".jpg")))
        image = self.transforms(numpy_image)
        boxes, labels, _ = parse_xml(self.annotation_dir, file_name, self.dict_class)
        return image, {"box": boxes, "label": labels}


def collate_fn(batch_data):
    images, targets = list(zip(*batch_data))
    images = torch.cat([image[None, :] for image in images])
    return images, targets


if __name__ == '__main__':
    image_dir = "JPEGImages/"
    annotation_dir = "Annotations/"
    file_name_path = "ImageSets/Main/train.txt"
    class_json_path = "ImageSets/Main/class.json"
    transforms = Transform()
    data_set = VOCDataSet(image_dir, annotation_dir, file_name_path, class_json_path, transforms)
    data_loader = DataLoader(data_set, batch_size=2, shuffle=True, collate_fn=collate_fn)
    for batch_images, batch_targets in data_loader:
        print(batch_images.size())
        print(batch_targets)
        break

