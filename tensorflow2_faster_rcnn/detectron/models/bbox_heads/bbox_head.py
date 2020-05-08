"""
# rcnn layer, roi map分类和 roi box二次回归
"""

import tensorflow as tf

from detectron.core.bbox import transforms
from detectron.core.loss import losses
from detectron.utils.misc import *

layers = tf.keras.layers


class BBoxHead(tf.keras.Model):
    def __init__(self, num_classes,
                 pool_size=(7, 7),
                 target_means=(0., 0., 0., 0.),
                 target_stds=(0.1, 0.1, 0.2, 0.2),
                 min_confidence=0.05,
                 nms_threshold=0.5,
                 max_instances=100,
                 **kwags):
        super(BBoxHead, self).__init__(**kwags)

        self.num_classes = num_classes
        self.pool_size = tuple(pool_size)
        self.target_means = target_means
        self.target_stds = target_stds
        self.min_confidence = min_confidence
        self.nms_threshold = nms_threshold
        self.max_instances = max_instances

    def __call__(self, inputs, training=True):
        """
        Args
        ---
            pooled_rois: [batch_size * num_rois, pool_size, pool_size, channels]

        Returns
        ---
            rcnn_class_logits: [batch_size * num_rois, num_classes]
            rcnn_probs: [batch_size * num_rois, num_classes]
            rcnn_deltas: [batch_size * num_rois, num_classes, (dy, dx, log(dh), log(dw))]
        """
        pass
