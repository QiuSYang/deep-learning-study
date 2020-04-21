"""
# anchor generator
"""
import os
import logging
import tensorflow as tf

from detectron.utils.misc import *


class AnchorGenerator(object):
    def __init__(self,
                 scales=(32, 64, 128, 256, 512),
                 ratios=(0.5, 1, 2),
                 feature_strides=(4, 8, 16, 32, 64)):
        """Anchor Generator

        Attributes
        ---
            scales: 1D array of anchor sizes in pixels.
            ratios: 1D array of anchor ratios of width/height.
            feature_strides: Stride of the feature map relative to the image in pixels.
        """
        self.scales = scales
        self.ratios = ratios
        self.feature_strides = feature_strides

    def generate_pyramid_anchors(self, img_metas):
        """Generate the multi-level anchors for Region Proposal Network

        Args
        ---
            img_metas: [batch_size, 11]

        Returns
        ---
            anchors: [num_anchors, (y1, x1, y2, x2)] in image coordinates.
            valid_flags: [batch_size, num_anchors]
        """
        # generate anchors
        pad_shape = calc_batch_padded_shape(img_metas)

        feature_shapes = [(pad_shape[0]//stride, pad_shape[1]//stride)
                          for stride in self.feature_strides]
        anchors = None

    def _generate_valid_flags(self, anchors, img_shape):
        """
        Args
        ---
            anchors: [num_anchors, (y1, x1, y2, x2)] in image coordinates.
            img_shape: Tuple. (height, width, channels)

        Returns
        ---
            valid_flags: [num_anchors]
        """
        y_center = (anchors[:, 2] + anchors[:, 0]) / 2
        x_center = (anchors[:, 3] + anchors[:, 1]) / 2

        valid_flags = tf.ones(anchors.shape[0], dtype=tf.int32)
        zeros = tf.zeros(anchors.shape[0], dtype=tf.int32)

        """
        # tf.where(condition): condition是bool型值，True/False, 返回值，是condition中元素为True对应的索引
        # tf.where(condition, x=None, y=None, name=None): condition， x, y 相同维度，condition是bool型值，True/False
        #    返回值是对应元素，condition中元素为True的元素替换为x中的元素，为False的元素替换为y中对应元素
        #    x只负责对应替换True的元素，y只负责对应替换False的元素，x，y各有分工
        #    由于是替换，返回值的维度，和condition，x ， y都是相等的。
        """
        valid_flags = tf.where(y_center <= img_shape[0], valid_flags, zeros)
        valid_flags = tf.where(x_center <= img_shape[1], valid_flags, zeros)

        return valid_flags

    def _generate_level_anchors(self, level, feature_shape):
        """Generate the anchors given the spatial shape of feature map.

        Args
        ---
            feature_shape: (height, width)

        Returns
        ---
            numpy.ndarray [anchors_num, (y1, x1, y2, x2)]
        """
        scale = self.scales[level]
        ratios = self.ratios
        feature_stride = self.feature_strides[level]

        # Get all combinations of scales and ratios
        # tf.meshgrid(x, y): 生产网格，x所有列坐标，y所有横坐标
        scales, ratios = tf.meshgrid([float(scale)], ratios)
        scales = tf.reshape(scales, [-1])
        ratios = tf.reshape(ratios, [-1])
