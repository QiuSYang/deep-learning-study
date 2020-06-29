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
from src.tool.image_augmentations import CustomCompose, CustomImageNormalize, DistortionLessResize
from src.datasets.pascalvoc_dataset import PascalVocDataset, detection_collate

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(filename)s - %(levelname)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S')
_logger = logging.getLogger(__name__)


class YoloV1Train(object):
    def __init__(self, args):
        self.args = args
        self.config = YoloV1Config(dataset_classes_num=len(VOC_CLASSES))
        # 显示超参数
        self.config.display()

    def train(self):
        """ 训练模型
        :return:
        """
        # 设置模型
        model = YoloV1Net(s=self.config.GRID_NUM,
                          b=self.config.ANCHOR_NUM,
                          num_classes=self.config.CLASSES_NUM)
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
        # step = 0
        # if os.path.exists(os.path.join(self.args.model_dir_path,
        #                                '.pkl'.format(self.args.model_file_name))):
        #     # 已有之前的模型路径，加载模型
        #     step = model.load_model(self.args.model_dir_path,
        #                             self.args.model_file_name,
        #                             optimizer=optimizer,
        #                             lr_scheduler=scheduler)

        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(self.config.DEVICE)
        model.train()

        # 设置损失函数
        criterion = YoloV1Loss(s=self.config.GRID_NUM,
                               b=self.config.ANCHOR_NUM,
                               l_coord=self.config.L_COORD,
                               l_noobj=self.config.L_NOOBJ,
                               device=self.config.DEVICE)
        # 训练开始
        if self.args.steps is 0:
            self.args.steps = self.config.NUM_STEPS_TO_FINISH

        _logger.info("start step is {}.".format(step))
        while step < self.args.steps:
            data_loader = self.get_data_loader()
            start_time = time.perf_counter()
            for i, (images, gt_boxes, get_labels, gt_outs) in enumerate(data_loader):
                # images - batch image data, gt_boxes - batch gt boxes,
                # get_labels - batch gt labels, gt_outs - encoder batch gt model out, 即yolo最后输出的标签
                step += 1
                # scheduler.step()
                images = images.to(self.config.DEVICE)
                gt_outs = gt_outs.to(self.config.DEVICE)

                predict_outs = model(images)
                # 计算损失，反向传播训练模型
                loss = criterion(predict_outs, gt_outs)
                optimizer.zero_grad()
                loss.backward()

                optimizer.step()
                scheduler.step()

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
                                       [CustomImageNormalize(),
                                        DistortionLessResize(max_width=self.args.image_max_size)]))

        return data.DataLoader(dataset,
                               batch_size=self.args.batch_size,
                               num_workers=self.args.num_workers,
                               shuffle=True,
                               collate_fn=detection_collate,
                               pin_memory=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector '
                                                 'Training With Pytorch - YOLOV1')
    parser.add_argument('-data_set', '--voc_data_set_root',
                        default='../../data/pascalvoc/VOCdevkit',
                        type=str,
                        help='data set root directory path.')
    parser.add_argument('-m_d_p', '--model_dir_path',
                        default='../../model/yolo_v1',
                        type=str,
                        help='yolo-v1 model saving directory path.')
    parser.add_argument('-m_f_n', '--model_file_name',
                        default='{}_model'.format(str(time.strftime('%y%m%d', time.localtime(time.time())))),
                        type=str,
                        help='yolo-v1 model saving file name.')
    parser.add_argument('-b_s', '--batch_size',
                        default=2, type=int,
                        help='batch size for training')
    parser.add_argument('-n_w', '--num_workers',
                        default=0, type=int,
                        help='number of workers used in data loading.')
    parser.add_argument('-s_s', '--save_step',
                        default=100, type=int,
                        help='directory for saving checkpoint models.')
    parser.add_argument('-i_m_s', '--image_max_size',
                        default=448, type=int,
                        help='image max size, image resize max size.')
    parser.add_argument('-s', '--steps',
                        default=0, type=int,
                        help='the training steps, steps = epochs * (image_numbers / batch_size)')

    args = parser.parse_args()

    _logger.info("all the super parameters are loaded.")

    yolo_train = YoloV1Train(args=args)

    model = yolo_train.train()
