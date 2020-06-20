"""
# yolo-v1 加载数据训练
"""
import os
import argparse
import logging
import time
import torch
from torch.utils import data
from torch import optim
from src.yolo_v1.model import YoloV1Net
from src.yolo_v1.loss import YoloV1Loss
from src.yolo_v1.custom_lr_scheduler import WarmUpMultiStepLR
from src.datasets.pascalvoc_dataset import VOC_CLASSES

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(filename)s - %(levelname)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S')
_logger = logging.getLogger(__name__)


class YoloV1Train(object):
    def __init__(self, args):
        self.args = args
        pass

    def train(self):
        """ 训练模型
        :return:
        """
        # 设置模型
        model = YoloV1Net(num_classes=len(VOC_CLASSES))
        # 设置优化器
        optimizer = optim.SGD(model.parameters(),
                              lr=self.args.learning_rate,
                              momentum=self.args.momentum,
                              weight_decay=self.args.weight_decay)
        scheduler = WarmUpMultiStepLR(optimizer,
                                      milestones=self.args.step_lr_sizes,
                                      gamma=self.args.step_lr_gamma,
                                      warm_up_factor=self.args.warm_up_factor,
                                      warm_up_iters=self.args.warm_up_num_iters)
        step = model.load_model()

        return model

    def get_data_loader(self,
                        test=False,
                        year=None):
        """ 加载数据生产器(Loader)
        :param test: 是否是test dataset
        :param year: 数据集年份
        :return:
        """
        if not test:
            # 非测试数据集(训练数据集)
            image_sets = (('2007', 'trainval'), ('2012', 'trainval'))
        else:
            # 测试数据集
            if year is None:
                image_sets = (('2007test', 'test'), ('2012test', 'test'))
            elif year == '2007':
                image_sets = (('2007test', 'test'),)
            elif year == '2012':
                image_sets = (('2012test', 'test'),)

        from src.tool.image_augmentations import CustomCompose, DistortionLessResize
        from src.datasets.pascalvoc_dataset import PascalVocDataset, detection_collate

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector '
                                                 'Training With Pytorch - YOLOV1')
    # parser.add_argument()

    args = parser.parse_args()

    _logger.info("all the super parameters are loaded.")
