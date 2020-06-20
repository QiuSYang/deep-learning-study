"""
# 模型预测模块，主要包括predict(), nms(), decoder()等function
"""
import os
import logging
import PIL
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms

from src.datasets.pascalvoc_dataset import VOC_CLASSES
from src.yolo_v1.model import YoloV1Net

_logger = logging.getLogger(__name__)


class YoloV1Predict(object):
    def __init__(self):
        pass

    def decoder(self):
        pass

    def _nms(self):
        pass

    def predict_one_image(self):
        pass

    def predict(self):
        pass


if __name__ == "__main__":
    pass
