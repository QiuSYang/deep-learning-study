"""
# rpn network
"""
import os
import logging
import tensorflow as tf
layers = tf.keras.layers

from detectron.utils.misc import *


class RPNHead(tf.keras.Model):
    def __init__(self,
                 anchor_scales=(32, 64, 128, 256, 512),
                 anchor_ratios=(0.5, 1, 2),
                 anchor_feature_strides=(4, 8, 16, 32, 64),
                 proposal_count=2000,
                 nms_threshold=0.7,
                 target_means=(0., 0., 0., 0.),
                 target_stds=(0.1, 0.1, 0.2, 0.2),
                 num_rpn_deltas=256,
                 positive_fraction=0.5,
                 pos_iou_thr=0.7,
                 neg_iou_thr=0.3,
                 **kwags):
        """Network head of Region Proposal Network.

                                      / - rpn_cls (1x1 conv)
        input - rpn_conv (3x3 conv) -
                                      \ - rpn_reg (1x1 conv)

        Attributes
        ---
            anchor_scales: 1D array of anchor sizes in pixels.
            anchor_ratios: 1D array of anchor ratios of width/height.
            anchor_feature_strides: Stride of the feature map relative
                to the image in pixels.
            proposal_count: int. RPN proposals kept after non-maximum
                supression.
            nms_threshold: float. Non-maximum suppression threshold to
                filter RPN proposals.
            target_means: [4] Bounding box refinement mean.
            target_stds: [4] Bounding box refinement standard deviation.
            num_rpn_deltas: int.
            positive_fraction: float.
            pos_iou_thr: float.
            neg_iou_thr: float.
        """
        super(RPNHead, self).__init__(**kwags)

        self.proposal_count = proposal_count
        self.nms_threshold = nms_threshold
        self.target_means = target_means
        self.target_stds = target_stds

        # Shared convolutional base of the RPN
        self.rpn_conv_shared = layers.Conv2D(512, (3, 3), padding='same',
                                             kernel_initializer='he_normal',
                                             name='rpn_conv_shared')

        self.rpn_class_raw = layers.Conv2D(len(anchor_ratios) * 2, (1, 1),
                                           kernel_initializer='he_normal',
                                           name='rpn_class_raw')

        self.rpn_delta_pred = layers.Conv2D(len(anchor_ratios) * 4, (1, 1),
                                            kernel_initializer='he_normal',
                                            name='rpn_bbox_pred')

    def __call__(self, inputs, training=True):
        """
        Args
        ---
            inputs: [batch_size, feat_map_height, feat_map_width, channels]
                one level of pyramid feat-maps.

        Returns
        ---
            rpn_class_logits: [batch_size, num_anchors, 2]
            rpn_probs: [batch_size, num_anchors, 2]
            rpn_deltas: [batch_size, num_anchors, 4]
        """
        layer_outputs = []
        for feat in inputs:
            # share convolution layer 3X3
            shared = self.rpn_conv_shared(feat)
            shared = tf.nn.relu(shared)

            # classify branch
            x = self.rpn_class_raw(shared)
            # reshape convolution tensor shape size [batch_size, w*h, 2]
            # every anchor two score
            rpn_class_logits = tf.reshape(x, [tf.shape(x)[0], -1, 2])
            rpn_probs = tf.nn.softmax(rpn_class_logits)

            x = self.rpn_delta_pred(shared)
            # reshape convolution tensor shape size [batch_size, w*h, 4]
            # every anchor four revise item(paper corresponding t)
            rpn_deltas = tf.reshape(x, [tf.shape(x)[0], -1, 4])

            layer_outputs.append([rpn_class_logits, rpn_probs, rpn_deltas])

        # zip(*layer_outputs)解压layer_outputs, 将所有feature layer rpn_class_logits,
        # 保存到同一个tuple中, rpn_probs, rpn_deltas同理
        outputs = list(zip(*layer_outputs))
        # 各个层数据组合到一起
        outputs = [tf.concat(list(o), axis=1) for o in outputs]
        rpn_class_logits, rpn_probs, rpn_deltas = outputs

        return rpn_class_logits, rpn_probs, rpn_deltas


