"""
# yolo-v1 损失函数建立
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class YoloV1Loss(nn.Module):
    """yolo-v1 损失函数定义实现"""
    def __init__(self, s=7, b=2, l_coord=5, l_noobj=0.5):
        """ 为了更重视8维的坐标预测，给这些算是前面赋予更大的loss weight
        对于有物体的记为λcoord，在pascal VOC训练中取5，对于没有object的bbox的confidence loss，
        前面赋予更小的loss weight 记为 λnoobj,
        在pascal VOC训练中取0.5, 有object的bbox的confidence loss"""
        super(YoloV1Loss, self).__init__()
        self.s = s  # 正方形网格数
        self.b = b  # 每个格的预测框数
        self.l_coored = l_coord  # 损失函数坐标回归权重
        self.l_boobj = l_noobj  # 损失函数类别分类权重
        pass

    def forward(self, predict_tensor, target_tensor):
        pass

    def compute_iou(self, box1, box2):
        """iou的作用是，当一个物体有多个框时，选一个相比ground truth最大的执行度的为物体的预测，然后将剩下的框降序排列，
        如果后面的框中有与这个框的iou大于一定的阈值时则将这个框舍去（这样就可以抑制一个物体有多个框的出现了），
        目标检测算法中都会用到这种思想。
        Compute the intersection over union of two set of boxes, each box is [x1,y1,x2,y2].
        Args:
          box1: (tensor) bounding boxes, sized [N,4].
          box2: (tensor) bounding boxes, sized [M,4].
        Return:
          (tensor) iou, sized [N,M]."""
        pass
