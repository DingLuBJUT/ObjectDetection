# -*- coding:utf-8 -*-


import os
import cv2
import math
import numpy as np
import numpy
from numpy.random import randint
import xml.etree.ElementTree as et

import torch


def draw_anchor_box(image, positions, object_name=None):
    """

    draw anchor box on image and show image until press Enter key.
    Args:
        image (numpy): cv2 image
        positions (tuple): anchor box position(x_min, y_min, x_max, x_max)
        object_name (list): object name list
    Return:

    """

    if not isinstance(image, numpy.ndarray) or not isinstance(positions[0], (list, tuple)):
        return -1

    if object_name is None:
        for pos in positions:
            x_min, y_min, x_max, y_max = pos
            cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0),2)
            while True:
                cv2.imshow("image", image)
                if cv2.waitKey(1) == 13:
                    break

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


def parse_xml(xml_dir, xml_name, class_mapping=None, use_difficult=True):
    """
    parse image annotations xml file.

    args:
      xml_dir (str): xml files root directory.
      xml_name (str): xml file name.
      class_mapping (dict): object class mapping dict.
      use_difficult (bool): is use difficult label.
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
        if class_mapping is not None:
            label.append(class_mapping.get(name))
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

    Args:
        box_1 (Tenor (N, 4)):
        box_1 (Tenor (M, 4)):
        weights (List[float]):
    Return:
        (N * M * 4) regression deviation matrix of box_1 and box_2.
    """

    iou = box_iou(box_1, box_2)
    _, max_index = torch.max(iou, dim=1)

    num_box_1 = box_1.size(0)
    if weights is None:
        weights = torch.tensor([1.0] * 4, dtype=box_1.dtype, device=box_1.device)

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
        regression_params (Tensor(n, 12) or Tensor(n, 4)): shifting params.
    Return:
        shifted box (Tensor(n, 12) or Tensor(n, 4)).
    """

    regs = regression_params

    box_h = (box[:, 3] - box[:, 1])[:,None]
    box_w = (box[:, 2] - box[:, 0])[:,None]
    center_x = box_w / 2
    center_y = box_h / 2

    dx, dy, dw, dh = regs[:, 0::4], regs[:, 1::4], regs[:, 2::4], regs[:, 3::4]

    dw = torch.clamp(dw, max=math.log(1000. / 16))
    dh = torch.clamp(dh, max=math.log(1000. / 16))

    box_x = center_x + dx * box_w
    box_y = center_y + dy * box_h
    box_h = torch.exp(dh) * box_h
    box_w = torch.exp(dw) * box_w

    min_x = box_x - box_w / 2
    min_y = box_y - box_h / 2
    max_x = box_x + box_w / 2
    max_y = box_y + box_h / 2

    box = torch.stack([min_x, min_y, max_x, max_y], dim=1)
    box = box.permute(0,2,1).flatten(1)
    return box


def remove_small_box(box, size_thresh, box_score=None, score_thresh=None):
    """
    remove boxes with size and score less than the specified
    threshold.

    Args:
        box (Tensor(n, 4)): box to be filtered.
        box_score (Tensor(n)): box score.
        score_thresh (int): score minimum threshold.
        size_thresh (int): size minimum threshold.

    Return:
        filtered box by score and size (Tensor(m,4)) and index (Tensor(m,)).

    """
    # filter by score
    if box_score is not None and score_thresh is not None:
        index = torch.where(torch.gt(box_score, score_thresh))[0]
        box = box[index]

    # filter by size
    box_w = (box[:, 2] - box[:, 0])[:, None]
    box_h = (box[:, 3] - box[:, 1])[:, None]

    condition_w = torch.gt(box_w, size_thresh)
    condition_h = torch.gt(box_h, size_thresh)
    index = torch.where(torch.logical_and(condition_w, condition_h))[0]
    box = box[index]
    return box, index


def box_clip(box, image_size):
    """
    clip the box that exceeds the original size of the image.

    Args:
        box (Tensor(n, 4)): box to be clip.
        image_size (Tuple(int,int)):
    Return:
        clip box (Tensor(n,4))
    """

    image_h, image_w = image_size
    min_x = torch.clamp(box[:, 0][:,None], min=0, max=image_w)
    min_y = torch.clamp(box[:, 1][:,None], min=0, max=image_h)
    max_x = torch.clamp(box[:, 2][:,None], min=0, max=image_w)
    max_y = torch.clamp(box[:, 3][:,None], min=0, max=image_h)
    box = torch.cat([min_x,min_y,max_x,max_y],dim=1)
    return box

def tagging_box(box_1, box_2, neg_threshold, pos_threshold, labels=None):
    """
    tagging classification label by iou value of box_1 with box_2.
        if iou < neg_threshold: tagging 0 label.
        if iou > pos_threshold: tagging 1 label.
        else: tagging -1 label.
    Args:
        box_1 (Tensor(n, 4)) : anchor tensor.
        box_2 (Tensor(m, 4)) : ground truth tensor.
        neg_threshold float: neg label threshold.
        pos_threshold float: pos label threshold.
    Return:
        classification label (Tensor(n,))
    """
    classification_label = torch.zeros(box_1.size(0))
    iou = box_iou(box_1, box_2)
    max_value, index = iou.max(dim=1)

    pos_position = torch.gt(max_value, pos_threshold)
    neg_position = torch.lt(max_value, neg_threshold)
    mid_position = torch.logical_and(torch.ge(max_value, neg_threshold),
                                     torch.le(max_value, pos_threshold))

    if labels is not None:
        classification_label[pos_position] = labels[index[pos_position]]
    else:
        classification_label[pos_position] = 1

    classification_label[neg_position] = 0
    classification_label[mid_position] = -1
    return classification_label

def get_sampling_factor(sampling_num, pos_fraction, pos_num, neg_num):
    """
    get random sampling factor that random sampling index.

    Args:
        sampling_num (int): random sample num.
        pos_fraction (int): pos sample fraction.
        pos_num (int): total pos data num.
        neg_num (int): total neg data num.
    Return:
        pos_factor (int) pos random sampling factor.
        neg_factor (int) neg random sampling factor.
    """
    neg_fraction = 1 - pos_fraction
    pos_sampling_num = int(min(sampling_num * pos_fraction,pos_num))
    neg_sampling_num = int(min(sampling_num * neg_fraction,neg_num))

    pos_factor = torch.randperm(pos_num)[:pos_sampling_num]
    neg_factor = torch.randperm(neg_num)[:neg_sampling_num]
    return pos_factor, neg_factor


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


def batched_images(images, divide_size):
    """
    resize different size image in batch to same size.

    Args:
        images (List[Tensor]): different size image list.
        divide_size (int): divide base size.
    Return:
        images (Tensor): batched images.

    """

    max_height = 0
    max_width = 0
    for image in images:
        if image.size(1) > max_height:
            max_height = image.size(1)
        elif image.size(2) > max_width:
            max_width = image.size(2)

    batch_height = math.ceil(max_height / divide_size) * divide_size
    batch_width = math.ceil(max_width / divide_size) * divide_size

    list_batch_image = []
    for image in images:
        image_channels = image.size(0)
        image_height = image.size(1)
        image_width = image.size(2)
        batch_image = torch.zeros(image_channels,batch_height,batch_width)
        batch_image[:,:image_height,:image_width] = image
        list_batch_image.append(batch_image[None,:,:,:])

    images = torch.cat(list_batch_image)
    return images




