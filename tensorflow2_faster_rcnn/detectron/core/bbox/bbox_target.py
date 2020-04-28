"""
# rcnn layer 计算 proposal 与 gt_box 的差值, proposal rois 相当于 rcnn 网络的anchor，
# 相当于对anchor进行二次修正
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

    def build_targets(self, proposals, gt_boxes, gt_class_ids, img_metas):
        """Generates detection targets for images. Subsamples proposals and
        generates target class IDs, bounding box deltas for each.

        Args
        ---
            proposals: [batch_size * num_proposals, (batch_ind, y1, x1, y2, x2)] in normalized coordinates.
            gt_boxes: [batch_size, num_gt_boxes, (y1, x1, y2, x2)] in image coordinates.
            gt_class_ids: [batch_size, num_gt_boxes] Integer class IDs.
            img_metas: [batch_size, 11]

        Returns
        ---
            rcnn_rois: [batch_size * num_rois, (batch_ind, y1, x1, y2, x2)] in normalized coordinates
            rcnn_labels: [batch_size * num_rois].
                Integer class IDs.
            rcnn_label_weights: [batch_size * num_rois].
            rcnn_delta_targets: [batch_size * num_rois, num_classes, (dy, dx, log(dh), log(dw))].
                ROI bbox deltas.
            rcnn_delta_weights: [batch_size * num_rois, num_classes, 4].
        """

        pad_shapes = calc_pad_shapes(img_metas)
        batch_size = img_metas.shape[0]

        proposals = tf.reshape(proposals[:, :5], (batch_size, -1, 5))

        rcnn_rois = []
        rcnn_labels = []
        rcnn_label_weights = []
        rcnn_delta_targets = []
        rcnn_delta_weights = []

        for i in range(batch_size):
            rois, labels, label_weights, delta_targets, delta_weights = self._build_single_target(
                proposals[i], gt_boxes[i], gt_class_ids[i], pad_shapes[i], i)
            rcnn_rois.append(rois)
            rcnn_labels.append(labels)
            rcnn_label_weights.append(label_weights)
            rcnn_delta_targets.append(delta_targets)
            rcnn_delta_weights.append(delta_weights)

        rcnn_rois = tf.concat(rcnn_rois, axis=0)
        rcnn_labels = tf.concat(rcnn_labels, axis=0)
        rcnn_label_weights = tf.concat(rcnn_label_weights, axis=0)
        rcnn_delta_targets = tf.concat(rcnn_delta_targets, axis=0)
        rcnn_delta_weights = tf.concat(rcnn_delta_weights, axis=0)

        return rcnn_rois, rcnn_labels, rcnn_label_weights, rcnn_delta_targets, rcnn_delta_weights

    def _build_single_target(self, proposals, gt_boxes, gt_class_ids, img_shape, batch_ind):
        """
        Args
        ---
            proposals: [num_proposals, (batch_ind, y1, x1, y2, x2)] in normalized coordinates.
            gt_boxes: [num_gt_boxes, (y1, x1, y2, x2)]
            gt_class_ids: [num_gt_boxes]
            img_shape: np.ndarray. [2]. (img_height, img_width)
            batch_ind: int.

        Returns
        ---
            rois: [num_rois, (batch_ind, y1, x1, y2, x2)]
            labels: [num_rois]
            label_weights: [num_rois]
            target_deltas: [num_rois, num_classes, (dy, dx, log(dh), log(dw))]
            delta_weights: [num_rois, num_classes, 4]
        """
        H, W = img_shape
