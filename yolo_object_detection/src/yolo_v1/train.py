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
from src.yolo_v1.config import YoloV1Config
from src.datasets.pascalvoc_dataset import VOC_CLASSES
from src.tool.image_augmentations import CustomCompose, DistortionLessResize
from src.datasets.pascalvoc_dataset import PascalVocDataset, detection_collate

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(filename)s - %(levelname)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S')
_logger = logging.getLogger(__name__)


class YoloV1Train(object):
    def __init__(self, args):
        self.args = args
        self.config = YoloV1Config(dataset_classes_num=len(len(VOC_CLASSES)))
        # 显示超参数
        self.config.display()

    def train(self):
        """ 训练模型
        :return:
        """
        # 设置模型
        model = YoloV1Net(num_classes=self.config.CLASSES_NUM)
        # 设置优化器
        optimizer = optim.SGD(model.parameters(),
                              lr=self.config.LEARNING_RATE,
                              momentum=self.config.MOMENTUM,
                              weight_decay=self.config.WEIGHT_DECAY)
        scheduler = WarmUpMultiStepLR(optimizer,
                                      milestones=self.config.STEP_LR_SIZES,
                                      gamma=self.config.STEP_LR_GAMMA,
                                      warm_up_factor=self.config.WARM_UP_FACTOR,
                                      warm_up_iters=self.config.WARM_UP_NUM_ITERS)
        step = model.load_model(self.args.model_dir_path,
                                self.args.model_file_name,
                                optimizer=optimizer,
                                lr_scheduler=scheduler)
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(self.config.DEVICE)
        model.train()

        criterion = YoloV1Loss()
        while step < self.config.NUM_STEPS_TO_FINISH:
            data_loader = self.get_data_loader()
            start_time = time.perf_counter()
            for i, (images, gt_boxes, get_labels, gt_outs) in enumerate(data_loader):
                # images - batch image data, gt_boxes - batch gt boxes,
                # get_labels - batch gt labels, gt_outs - encoder batch gt model out, 即yolo最后输出的标签
                step += 1
                scheduler.step()
                images = images.to(self.config.DEVICE)
                gt_outs = gt_outs.to(self.config.DEVICE)

                predict_outs = model(images)
                # 计算损失，反向传播训练模型
                loss = criterion(predict_outs, gt_outs)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                end_time = time.perf_counter()
                _logger.info("step: {}, loss: {:.8f}, time: {:.4f}".format(step, loss.item(),
                                                                           end_time-start_time))
                start_time = time.perf_counter()
                if step is not 0 and step % self.args.save_step is 0:
                    # 固定步长保存模型
                    model.save_model(self.args.model_dir_path,
                                     self.args.model_file_name,
                                     step=step,
                                     optimizer=optimizer,
                                     lr_scheduler=scheduler)

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
