"""
# 用于YOLO-V1 model 评测
"""
import os
import logging
import argparse
import torch
from torch.utils import data

from src.tool.image_augmentations import CustomCompose, DistortionLessResize
from src.datasets.pascalvoc_dataset import PascalVocDataset, detection_collate

_logger = logging.getLogger(__name__)


class YoloV1Test(object):
    def __init__(self, args):
        self.args = args

    def get_test_data_loader(self, year=None):
        if year == '2007':
            image_sets = (('2007test', 'test'),)
        elif year == '2012':
            image_sets = (('2012test', 'test'),)
        else:
            image_sets = (('2007test', 'test'), ('2012test', 'test'))

        dataset = PascalVocDataset(data_path_root=self.args.voc_data_set_root,
                                   image_sets=image_sets,
                                   transform=CustomCompose(
                                       [DistortionLessResize(max_width=self.args.max_image_size)]))

        return data.DataLoader(dataset,
                               batch_size=self.args.batch_size,
                               num_workers=self.args.num_workers,
                               shuffle=True,
                               collate_fn=detection_collate,
                               pin_memory=False)

    def test_data_evaluate(self):
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector '
                                                 'Testing With Pytorch - YOLOV1')
    # parser.add_argument()

    args = parser.parse_args()

    _logger.info("all the super parameters are loaded.")

