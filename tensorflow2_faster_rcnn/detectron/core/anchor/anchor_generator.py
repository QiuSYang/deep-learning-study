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
