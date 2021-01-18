# -*- coding:utf-8 -*-

import cv2
import numpy
from numpy.random import randint


def draw_anchor_box(image, positions, object_name=None):
    """

    draw anchor box on image and show image until press Enter key.
    Args:
        image: cv2 image
        positions: anchor box position(x_min, y_min, x_max, x_max)
        object_name: object name list
    Return:

    """
    if not isinstance(image, numpy.ndarray) or not isinstance(positions[0], (list, tuple)):
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


def box_iou(box_1, box_2):

    return
