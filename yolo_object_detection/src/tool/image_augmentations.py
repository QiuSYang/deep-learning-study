"""
# 图像增强库函数，结合Torch torchvision transforms 使用
"""
import os
import logging
import cv2
import numpy as np
from scipy import ndimage
import torch
from torchvision import transforms

_logger = logging.getLogger(__name__)


class CustomCompose(transforms.Compose):
    """ transform.Compose 一个作用
    Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        super(CustomCompose, self).__init__(transforms=transforms)

    def __call__(self, img, boxes=None, labels=None):
        for t in self.transforms:
            img, boxes, labels = t(img, boxes, labels)
        return img, boxes, labels


class CustomImageNormalize(object):
    """图像数据读取归一化
    .. note::
        This transform acts out of place, i.e., it does not mutate the input tensor.

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation in-place."""
    def __init__(self):
        pass

    def __call__(self, image, gt_bbox=None, label=None):
        if image.dtype is not np.float:
            image = image.astype(np.float32)
        image = (image - np.min(image)) / (np.max(image) - np.min(image))

        return image, gt_bbox, label


class DistortionLessResize(object):
    """图像长宽等比例缩放，周围没有地方0补齐"""
    def __init__(self, max_width=512):
        self.max_side_length = max_width

    def __call__(self, image, gt_bbox=None, label=None):
        """
        :param image: 原始图像数据
        :param gt_bbox: 原始图片的bbox位置
        :param label: 原始bbox的标签（变化之后一般还是保持不变的）
        :return:
        """
        # 图像等比例resize
        image, window, scale, padding = self.resize_image(image,
                                                          square_max=self.max_side_length)
        # gt_bbox 的变化
        gt_bbox = self.resize_coordinate(coordinate_arr=gt_bbox,
                                         padding=padding,
                                         scale=scale)

        return image, gt_bbox, label

    def resize_image(self, image, square_max=512.0):
        image_dtype = image.dtype
        h, w = image.shape[:2]
        scale = 1
        # 三个维度，每个维度填充多少行
        padding = [(0, 0), (0, 0), (0, 0)]
        window = (0, 0, h, w)

        image_max = max(w, h)

        scale = square_max / float(image_max)

        if scale is not 1.0:
            image = cv2.resize(image, dsize=(round(w * scale), round(h * scale)))

        h, w = image.shape[:2]
        top_pad = int((square_max - h) // 2)
        bottom_pad = int(square_max - h - top_pad)
        left_pad = int((square_max - w) // 2)
        right_pad = int(square_max - w - left_pad)

        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        image = np.pad(image, padding, mode='constant', constant_values=0)
        # 包含图像的范围
        window = (top_pad, left_pad, h + top_pad, w + left_pad)

        return image.astype(image_dtype), window, scale, padding

    def resize_coordinate(self, coordinate_arr, padding, scale=1.0):
        # 坐标缩放
        if not isinstance(coordinate_arr, np.ndarray):
            coordinate_arr = np.array(coordinate_arr) * scale
        else:
            coordinate_arr = coordinate_arr * scale

        # 坐标平移
        coordinate_arr[:, 0] = coordinate_arr[:, 0] + padding[1][0]
        coordinate_arr[:, 2] = coordinate_arr[:, 2] + padding[1][0]
        coordinate_arr[:, 1] = coordinate_arr[:, 1] + padding[0][0]
        coordinate_arr[:, 3] = coordinate_arr[:, 3] + padding[0][0]

        return coordinate_arr
