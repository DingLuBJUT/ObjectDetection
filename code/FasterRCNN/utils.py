# -*- coding:utf-8 -*-

import os
import cv2
import json
import numpy as np
from numpy.random import randint
import xml.etree.ElementTree as ET


def draw_anchor_box(image, positions, object_name=None):
    """

    draw anchor box on image and show image until press Enter key.
    Args:
        image: cv2 image
        positions: anchor box position(x_min, y_min, x_max, x_max)
        object_name: object name list
    Return:

    """
    if not isinstance(image, np.ndarray) or not isinstance(positions[0], (list, tuple)):
        return -1

    if object_name is None:
        for pos in positions:
            x_min, y_min, x_max, y_max = pos
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    else:
        color_dict = dict([(name, (randint(0, 255), randint(0, 255), randint(0, 255)))
                           for name in object_name])

        for pos, name in zip(positions, object_name):
            min_x = pos[0]
            min_y = pos[1]
            max_x = pos[2]
            max_y = pos[3]
            font_x = min_x
            font_y = min_y - 1
            box_color = color_dict[name]
            cv2.rectangle(image, (min_x, min_y), (max_x, max_y), box_color, 2)
            # image, text, text_pos, text_font, text_size, text_color
            cv2.putText(image, name, (font_x, font_y), cv2.FONT_HERSHEY_PLAIN, 1, box_color)
            while True:
                cv2.imshow("image", image)
                if cv2.waitKey(1) == 13:
                    break
    return


# def box_iou(box_1, box_2):
#
#     return


def parse_xml(xml_dir, xml_name, dict_label=None, use_difficult=True):
    """
    parse image annotations xml file, return box、label and so on.

    args:
      xml_dir (str):
      xml_name (str):
      dict_label (dict):
      use_difficult (bool):
    return:
        bbox (List[List[int]]): image ground truth position.
        label (List[int]): image ground truth class.
    """
    annotations = ET.parse(os.path.join(xml_dir, xml_name + '.xml'))
    bbox = list()
    label = list()
    difficult = list()
    for obj in annotations.findall('object'):
        if not use_difficult and int(obj.find('difficult').text) == 1:
            continue
        difficult.append(int(obj.find('difficult').text))
        box = obj.find('bndbox')
        bbox.append([int(box.find(tag).text) - 1 for tag in ('xmin', 'ymin', 'xmax', 'ymax')])
        name = obj.find('name').text.lower().strip()
        if dict_label is not None:
            label.append(dict_label.get(name))
    bbox = np.stack(bbox).astype(np.float32)
    label = np.stack(label).astype(np.int32)
    difficult = np.array(difficult, dtype=np.bool).astype(np.uint8)
    return bbox, label, difficult


if __name__ == '__main__':
    json_class_path = "/Users/dingjunlu/PycharmProjects/ObjectDetection/data/VOC2012/ImageSets/Main/class.json"
    dict_label = json.loads(open(json_class_path, 'r').read())
    xml_dir = "/Users/dingjunlu/PycharmProjects/ObjectDetection/data/VOC2012/Annotations/"
    xml_name = "2007_000033"
    bbox, label, difficult = parse_xml(xml_dir, xml_name, dict_label)
    print(bbox)
    print(label)
    # list_file_names = os.listdir("/Users/dingjunlu/PycharmProjects/ObjectDetection/data/VOC2012/ImageSets/Main/")
    # dict_label = dict()
    # set_labels = set()
    # for name in list_file_names:
    #     if name not in ['trainval.txt', 'val.txt', 'train.txt']:
    #         label = name.split("_")[0]
    #         set_labels.add(label)
