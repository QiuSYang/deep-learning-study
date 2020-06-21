"""
# YOLO-V1 一些超参数设置
"""
import os
import logging
import torch
from typing import List

_logger = logging.getLogger(__name__)


class YoloV1Config(object):
    # 网格大小
    GRID_NUM = 7
    # 每个网格bbox num
    ANCHOR_NUM = 2
    # 数据集类别数
    CLASSES_NUM = 20
    # 损失超参数
    L_COORD = 5
    L_NOOBJ = 0.5

    # 设置训练资源
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 设置训练超参数
    LEARNING_RATE: float = 0.001
    MOMENTUM: float = 0.9
    WEIGHT_DECAY: float = 0.0005
    STEP_LR_SIZES: List[int] = [200000, 400000]
    STEP_LR_GAMMA: float = 0.1
    WARM_UP_FACTOR: float = 0.1
    WARM_UP_NUM_ITERS: int = 1000

    NUM_STEPS_TO_SAVE: int = 100
    NUM_STEPS_TO_SNAPSHOT: int = 10000
    NUM_STEPS_TO_FINISH: int = 600000

    def __init__(self, dataset_classes_num=20):
        # 根据数据集更新类别数
        self.CLASSES_NUM = dataset_classes_num

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")


if __name__ == "__main__":
    config = YoloV1Config()
    config.display()
