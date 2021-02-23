# -*- coding:utf-8 -*-

"""
voc 2012 Dataset

Description:
    create voc 2012 Dataset by extends torch.utils.data.Dataset.
"""
# ***** modification history *****
# ********************************
# 2021/02/10, by junlu Ding, create

import os
import json
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from utils import parse_xml
from tranforms import Compose
from tranforms import ToTensor
from tranforms import Normalize
from tranforms import RandomHorizontalFlip
from generalized_transform import GeneralizedTransform


ROOT_PATH = "/Users/dinglu/Documents/code/ObjectDetection/data/voc_2012/"


class VOCDataSet(Dataset):
    """
    implement voc2012 Dataset and pre_process images by different transforms.

    args:
        image_dir (str): jpeg images directory path.
        annotation_dir (str): annotation xml directory path.
        file_name_path (str): the file path of total data name.
        class_json_path (str): object label json path.
        transforms : Compose transforms.

    return:
        images (Tensor): batch images.
        targets (dict): batch box、label.
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
        boxes, labels, _ = parse_xml(self.annotation_dir, file_name, self.dict_class)
        target = {"box": boxes, "label": labels}
        image = Image.open(os.path.join(self.image_dir, file_name + ".jpg"))
        image, target = self.transforms(image, target)
        return image, target


def collate_fn(batch_data):
    """
    convert the data structure returned by the DataSet.

    Args:
        batch_data (List[Tuple]) :

    Return:
        images (list)
        targets (list)

    Example:
        batch_data: [(image_1,target_1),(image_1,target_1)]
        images: (image_1,image_2)
        targets: (target_1,target_2)
    """
    images, targets = list(zip(*batch_data))
    return list(images), list(targets)


if __name__ == '__main__':
    image_dir = "JPEGImages/"
    annotation_dir = "Annotations/"
    class_json_path = "ImageSets/Main/class.json"
    train_path = "ImageSets/Main/train.txt"
    image_mean = [0.485, 0.456, 0.406]
    image_std = [0.229, 0.224, 0.225]
    transforms = Compose([
        ToTensor(),
        RandomHorizontalFlip(prob=0.5),
        Normalize(image_mean,image_std)
    ])
    data_set = VOCDataSet(image_dir,
                          annotation_dir,
                          train_path,
                          class_json_path,
                          transforms=transforms)
    data_loader = DataLoader(data_set,
                             batch_size=5,
                             shuffle=True,
                             collate_fn=collate_fn)

    generalized_transform = GeneralizedTransform(min_size=800,max_size=1000)

    for batch_images, batch_targets in data_loader:
        images, targets = generalized_transform(batch_images, batch_targets)
        print(images.size())
        break
