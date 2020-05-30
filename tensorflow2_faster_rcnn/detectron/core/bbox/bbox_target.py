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
        """训练过程二次bbox时，在再一次对proposals进行筛选，一般选取256个proposal, 一般前景框64个，背景框192个
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

        # 去除扩充的预选框，扩充的预选框的四个坐标都设置为0
        trimmed_proposals, _ = trim_zeros(proposals[:, 1:])
        # 去除无效gt box
        gt_boxes, non_zeros = trim_zeros(gt_boxes)
        # 去除无效 box class label
        gt_class_ids = tf.boolean_mask(gt_class_ids, non_zeros)

        # gt box 数据归一化, 因为proposal是归一化的结果
        gt_boxes = gt_boxes/tf.constant([H, W, H, W], dtype=tf.float32)

        # overlaps是一个什么样的结果？是每个proposal与每个gt box 的IOU吗？(是的)
        overlaps = geometry.compute_overlaps(trimmed_proposals, gt_boxes)
        roi_iou_argmax = tf.argmax(overlaps, axis=1)  # 2d tensor
        roi_iou_max = tf.reduce_max(overlaps, axis=1)  # 2d tensor

        positive_roi_bool = (roi_iou_max >= self.pos_iou_thr)  # 2d tensor
        positive_indices = tf.where(positive_roi_bool)[:, 0]  # 2d tensor(n, 2)第二列全为0 to 1d tensor

        negative_indices = tf.where(roi_iou_max < self.neg_iou_thr)[:, 0]

        # Subsample ROIs. Aim for 33% positive
        # Positive ROIs
        positive_count = int(self.num_rcnn_deltas*self.positive_fraction)
        # 随机选取前positive_count个box
        positive_indices = tf.random.shuffle(positive_indices)[:positive_count]
        positive_count = tf.shape(positive_indices)[0]

        # Negative ROIs. Add enough to maintain positive:negative ratio.
        r = 1.0 / self.positive_fraction
        negative_count = tf.cast(r * tf.cast(positive_count, tf.float32), tf.int32) - positive_count
        negative_indices = tf.random.shuffle(negative_indices)[:negative_count]

        # Gather selected ROIs(positive_rois + negative_rois = self.num_rcnn_deltas(bbox提取的rois数目))
        positive_rois = tf.gather(proposals, positive_indices)  # [num_positive_rois, 5]
        negative_rois = tf.gather(proposals, negative_indices)  # [num_negative_rois, 5]

        # Assign positive ROIs to GT boxes.
        positive_overlaps = tf.gather(overlaps, positive_indices)
        roi_gt_box_assignment = tf.argmax(positive_overlaps, axis=1)
        roi_gt_boxes = tf.gather(gt_boxes, roi_gt_box_assignment)
        labels = tf.gather(gt_class_ids, roi_gt_box_assignment)

        delta_targets = transforms.bbox2delta(positive_rois[:, 1:],
                                              roi_gt_boxes, self.target_means, self.target_stds)

        # 前景和背景框拼接，组成所有rois box
        rois = tf.concat([positive_rois, negative_rois], axis=0)

        N = tf.shape(negative_rois)[0]
        P = tf.maximum(self.num_rcnn_deltas - tf.shape(rois)[0], 0)
        num_bfg = rois.shape[0]

        rois = tf.pad(rois, [(0, P), (0, 0)])  # [num_rois, 5]
        labels = tf.pad(labels, [(0, N)], constant_values=0)
        labels = tf.pad(labels, [(0, P)], constant_values=-1)  # [num_rois]
        delta_targets = tf.pad(delta_targets, [(0, N + P), (0, 0)])  # [num_rois, 4]

        # Compute weights
        label_weights = tf.zeros((self.num_rcnn_deltas,), dtype=tf.float32)
        delta_weights = tf.zeros((self.num_rcnn_deltas,), dtype=tf.float32)
        if num_bfg > 0:
            label_weights = tf.where(labels >= 0,
                                     tf.ones((self.num_rcnn_deltas,), dtype=tf.float32) / num_bfg,
                                     label_weights)

            # delta_weights = tf.where(labels > 0,
            #                          tf.ones((self.num_rcnn_deltas,), dtype=tf.float32) / num_bfg,
            #                          label_weights)

            delta_weights = tf.where(labels > 0,
                                     tf.ones((self.num_rcnn_deltas,), dtype=tf.float32) / num_bfg,
                                     delta_weights)

        # tf.tile(tensor, [1, 4]) tensor沿第0维复制扩展1倍，沿第1维复制扩展4倍
        delta_weights = tf.tile(tf.reshape(delta_weights, (-1, 1)), [1, 4])  # [num_rois, 4]

        # Organize Box targets and weights
        tile_delta_targets = tf.zeros((self.num_rcnn_deltas, self.num_classes, 4))
        tile_delta_weights = tf.zeros((self.num_rcnn_deltas, self.num_classes, 4))
        ids = tf.stack([tf.range(self.num_rcnn_deltas, dtype=labels.dtype), labels], axis=1)
        # tf.tensor_scatter_nd_update(tensor1, ids_tensor, tensor2)
        # 按照ids_tensor的索引位置在tensor1中找到位置用tensor2更新，tensor2的维度比tensor1维度小1
        delta_targets = tf.tensor_scatter_nd_update(tile_delta_targets, ids,
                                                    delta_targets)  # [num_rois, self.num_classes, 4]
        delta_weights = tf.tensor_scatter_nd_update(tile_delta_weights, ids,
                                                    delta_weights)  # [num_rois, self.num_classes, 4]

        return rois, labels, label_weights, delta_targets, delta_weights

