"""
# yolo-v1 损失函数建立
# 参考链接：https://zhuanlan.zhihu.com/p/70387154
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
        """
        :param predict_tensor:
            (tensor) size(batch_size, S, S, Bx5+20=30) [x, y, w, h, c]---预测对应的格式
        :param target_tensor:
            (tensor) size(batch_size, S, S, 30) --- 标签的准确格式
        :return:
        """
        N = predict_tensor.size()[0]

        # 具有目标标签的索引
        coo_mask = target_tensor[:, :, :, 4] > 0
        # 不具有目标的标签索引
        noo_mask = target_tensor[:, :, :, 4] == 0

        # 得到含物体的坐标等信息
        coo_mask = coo_mask.unsqueeze(-1).expand_as(target_tensor)
        # 得到不含物体的坐标等信息
        noo_mask = noo_mask.unsqueeze(-1).expand_as(target_tensor)

        return None

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

        N = box1.size(0)
        M = box2.size(0)

        # torch.max(input, other, out=None) → Tensor
        # Each element of the tensor input is compared with the corresponding element
        # of the tensor other and an element-wise maximum is taken.
        # left top
        lt = torch.max(
            box1[:, :2].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
            box2[:, :2].unsqueeze(0).expand(N, M, 2),  # [M,2] -> [1,M,2] -> [N,M,2]
        )
        # right bottom
        rb = torch.min(
            box1[:, 2:].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
            box2[:, 2:].unsqueeze(0).expand(N, M, 2),  # [M,2] -> [1,M,2] -> [N,M,2]
        )

        wh = rb - lt  # [N,M,2]
        wh[wh < 0] = 0  # clip at 0
        inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

        area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])  # [N,]
        area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])  # [M,]
        area1 = area1.unsqueeze(1).expand_as(inter)  # [N,] -> [N,1] -> [N,M]
        area2 = area2.unsqueeze(0).expand_as(inter)  # [M,] -> [1,M] -> [N,M]

        iou = inter / (area1 + area2 - inter)

        return iou
