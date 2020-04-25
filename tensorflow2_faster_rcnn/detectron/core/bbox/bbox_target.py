"""
# 计算 proposal 与 gt_box 的差值
"""
import numpy as np
import tensorflow as tf

from detectron.core.bbox import geometry, transforms
from detectron.utils.misc import *


class ProposalTarget(object):
    def __init__(self,
                 target_means=(0., 0., 0., 0.),
                 target_stds=(0.1, 0.1, 0.2, 0.2),
                 num_rcnn_deltas=256,
                 positive_fraction=0.25,
                 pos_iou_thr=0.5,
                 neg_iou_thr=0.5,
                 num_classes=81):
        """Compute regression and classification targets for proposals.

        Attributes
        ---
            target_means: [4]. Bounding box refinement mean for RCNN.
            target_stds: [4]. Bounding box refinement standard deviation for RCNN.
            num_rcnn_deltas: int. Maximal number of RoIs per image to feed to bbox heads.
            positive_fraction: float.
            pos_iou_thr: float.
            neg_iou_thr: float.
            num_classes: int.
        """
        self.target_means = target_means
        self.target_stds = target_stds
        self.num_rcnn_deltas = num_rcnn_deltas
        self.positive_fraction = positive_fraction
        self.pos_iou_thr = pos_iou_thr
        self.neg_iou_thr = neg_iou_thr
        self.num_classes = num_classes
