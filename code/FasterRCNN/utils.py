# -*- coding:utf-8 -*-


import os
import cv2
import math
import numpy as np
from numpy.random import randint
import xml.etree.ElementTree as et

import torch


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


def parse_xml(xml_dir, xml_name, dict_label=None, use_difficult=True):
    """
    parse image annotations xml file.

    args:
      xml_dir (str):
      xml_name (str):
      dict_label (dict):
      use_difficult (bool):
    return:
        bbox (List[List[int]]): image ground truth position.
        label (List[int]): image ground truth class.
    """
    annotations = et.parse(os.path.join(xml_dir, xml_name + '.xml'))
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


def box_iou(box_1, box_2):
    """
    return iou values of per box_1 and per box_2.

    args:
        box_1 (Tenor (N, 4))
        box_1 (Tenor (M, 4))
    return:
        N * M iou value matrix
    """

    num_box_1 = box_1.size(0)

    box_1_w = box_1[:, 2] - box_1[:, 0]
    box_1_h = box_1[:, 3] - box_1[:, 1]
    box_1_area = (box_1_w * box_1_h)[:, None]

    box_2_w = box_2[:, 2] - box_2[:, 0]
    box_2_h = box_2[:, 3] - box_2[:, 1]
    box_2_area = (box_2_w * box_2_h)[:, None]

    left_top = torch.max(box_1[:, :2][:, None, :], box_2[:, :2])
    right_bottom = torch.min(box_1[:, 2:][:, None, :], box_2[:, 2:])
    join_w = (right_bottom[:, :, 0] - left_top[:, :, 0]).clamp(min=0)
    join_h = (right_bottom[:, :, 1] - left_top[:, :, 1]).clamp(min=0)
    join_area = join_w * join_h
    iou = join_area / ((box_1_area[:, None, :] + box_2_area).view(num_box_1, -1) - join_area)
    return iou


def box_deviation(box_1, box_2, weights=None):
    """
    return regression distance deviation of box_1 and box_2.
    args:
        box_1 (Tenor (N, 4)):
        box_1 (Tenor (M, 4)):
    return:
        (N * M * 4) regression deviation matrix of box_1 and box_2.
    """

    iou = box_iou(box_1, box_2)
    _, max_index = torch.max(iou, dim=1)

    num_box_1 = box_1.size(0)
    weight_x = weights[0]
    weight_y = weights[1]
    weight_w = weights[2]
    weight_h = weights[3]

    width_1 = (box_1[:, 2] - box_1[:, 0])[:, None]
    height_1 = (box_1[:, 3] - box_1[:, 1])[:, None]
    center_x_1 = box_1[:, 0][:, None] + width_1 / 2
    center_y_1 = box_1[:, 1][:, None] + height_1 / 2

    width_2 = (box_2[:, 2] - box_2[:, 0])[:, None]
    height_2 = (box_2[:, 3] - box_2[:, 1])[:, None]
    center_x_2 = box_2[:, 0][:, None] + width_2 / 2
    center_y_2 = box_2[:, 1][:, None] + height_2 / 2

    deviation_x = weight_x * ((center_x_2 - center_x_1[:, None, :]).view(num_box_1, -1) / width_1)
    deviation_y = weight_y * ((center_y_2 - center_y_1[:, None, :]).view(num_box_1, -1) / height_1)
    deviation_w = weight_w * torch.log(width_2 / width_1[:, None, :]).view(num_box_1, -1)
    deviation_h = weight_h * torch.log(height_2 / height_1[:, None, :]).view(num_box_1, -1)

    deviations = torch.cat([deviation_x[:, :, None],
                            deviation_y[:, :, None],
                            deviation_w[:, :, None],
                            deviation_h[:, :, None]], dim=2)

    index_1 = torch.range(0, max_index.size(0) - 1).long()
    index_2 = max_index.long()
    deviations = deviations[index_1, index_2]
    return deviations


def box_regression(box, regression_params):
    """
    shifting box by regression params.

    Args:
        box (Tensor(n, 4)): box to be shifted.
        regression_params (Tensor(n, 4)): shifting params.
    Return:
        shifted box (Tensor(n, 4)).
    """

    regs = regression_params
    dx, dy, dh, dw = regs[:, 0], regs[:, 1], regs[:, 2], regs[:, 3]
    dw = torch.clamp(dw, max=math.log(1000. / 16))
    dh = torch.clamp(dh, max=math.log(1000. / 16))

    box_h = box[:, 3] - box[:, 1]
    box_w = box[:, 2] - box[:, 0]
    center_x = box_w / 2
    center_y = box_h / 2

    box_x = center_x + dx * box_w
    box_y = center_y + dy * box_h
    box_h = torch.exp(dh) * box_h
    box_w = torch.exp(dw) * box_w

    min_x = (box_x - box_w / 2)[:, None]
    min_y = (box_y - box_h / 2)[:, None]
    max_x = (box_x + box_w / 2)[:, None]
    max_y = (box_y + box_h / 2)[:, None]
    box = torch.cat([min_x, min_y, max_x, max_y], dim=1)
    return box


def remove_useless_box(box, score_thresh, size_thresh):

    return


def clip_box():

    return


def smooth_l1_loss(y_hat, y, beta=1./9, size_average=True):
    """
    compute smooth l1 loss:

            0.5 * x^2 if |x| < 1
        l1 =
            |x| - 0.5 if x<-1 or x>1
    Args:
        y_hat (Tensor(n,4)): prediction proposal deviation.
        y (Tensor(n,4)): deviations of proposal and ground truth.
        beta (double): loss hyper parameter.
        size_average (bool): if return mean loss.
    Returns:
        smooth l1 loss (tensor)
    """

    x = torch.abs(y_hat - y)
    condition = torch.lt(x, beta)
    loss = torch.where(condition, 0.5 * x ** 2 / beta, x - 0.5 * beta)
    if size_average:
        return loss.mean()
    return loss.sum()

