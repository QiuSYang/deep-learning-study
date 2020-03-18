"""
# 搭建faster-rcnn网络流程图(pipeline)
"""
import os 
import sys 
import keras 
import keras.layers as KL 
import keras.models as KM 
import keras.backend as KB 
import numpy as np 
import tensorflow as tf 

from layer_utils.snippets import generate_anchors_pre, generate_anchors_pre_tf 
from layer_utils.proposal_layer import proposal_layer, proposal_layer_tf 
from layer_utils.anchor_target_layer import anchor_target_layer
from layer_utils.proposal_target_layer import proposal_target_layer
from layer_utils.proposal_top_layer import proposal_top_layer, proposal_top_layer_tf

from model.config import cfg

class NetWork(object):
    def __init__(self):
        self._predictions = {} 
        self._losses = {} 
        self._anchor_targets = {} 
        self._proposal_targets = {} 
        self._layers = {} 
        self._gt_image = None 
        self._act_summaries = [] 
        self._score_summaries = {} 
        self._train_summaries = [] 
        self._event_summaries = {} 
        self._variables_to_fix = {} 

        # 特征提取网络经过一系列卷积之后图像缩小的倍数
        self._feat_stride = [16, ] 

    def _build_network(self, is_training=True):
        # select initializers 
        if True: # cfg.TRAIN.TRUNCATED:
            initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
            initializer_box = tf.truncated_normal_initializer(mean=0.0, stddev=0.001)
        else:
            initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
            initializer_box = tf.random_normal_initializer(mean=0.0, stddev=0.001)
        
        net_conv = None
        with tf.variable_scope():
            # build the anchors for the image 
            # 在特征图的每个像素点上都生产9个anchor, anchor尺寸大小与原image尺寸相同
            self._anchor_component()
            # region proposal network (RPN 网络)
            rois = self._region_proposal(net_conv, is_training, initializer)
            # region of interest pooling
        if True: # cfg.POOLING_MODE == 'crop':
            pool5 = self._crop_pool_layer(net_conv, rois, "pool5")
        else:
            raise NotImplementedError
    
    # 为特征图生产anchors
    def _anchor_component(self):
        with tf.variable_scope('ANCHOR_' + self._tag) as scope:
            # just to get the shape right 
            height = tf.to_int32(tf.ceil(self._im_info[0] / np.float32(self._feat_stride[0])))
            width = tf.to_int32(tf.ceil(self._im_info[1] / np.float32(self._feat_stride[0])))
            if cfg.USE_E2E_TF:
                anchors, anchor_length = generate_anchors_pre_tf(
                height,
                width,
                self._feat_stride,
                self._anchor_scales,
                self._anchor_ratios
                )
            else:
                anchors, anchor_length = tf.py_func(generate_anchors_pre,
                                                    [height, width,
                                                    self._feat_stride, self._anchor_scales, self._anchor_ratios],
                                                    [tf.float32, tf.int32], name="generate_anchors")
            anchors.set_shape([None, 4])
            anchor_length.set_shape([])
            self._anchors = anchors
            self._anchor_length = anchor_length
    
    # RPN网络
    def _region_proposal(self, net_conv, is_training, initializer):
        # 添加weight的正则化项
        rpn = KL.Conv2D(512, kernel_size=7, strides=(1, 1), trainable=is_training, 
                        kernel_initializer=initializer, name='rpn_conv/3x3')(net_conv)
        # relu激活层
        rpn = KL.Activation('relu')(rpn)

        # 前后景分类
        rpn_cls_score = KL.Conv2D(self._num_anchors*2, kernel_size=1, strides=(1, 1), 
                                trainable=is_training, kernel_initializer = initializer, 
                                padding='valid', name='rpn_cls_score')(rpn)
        # reshape
        # change it so that the score has 2 as its channel size 
        rpn_cls_score_reshape = self._reshape_layer(rpn_cls_score, 2, 'rpn_cls_score_reshape')
        # softmax
        rpn_cls_prob_reshape = self._softmax_layer(rpn_cls_score_reshape, 'rpn_cls_prob_reshape')
        rpn_cls_pred = tf.argmax(tf.reshape(rpn_cls_score_reshape, [-1, 2]), axis=1, name='rpn_cls_pred')
        # reshape 
        rpn_cls_prob = self._reshape_layer(rpn_cls_prob_reshape, self._num_anchors*2, 'rpn_cls_prob')
        # 第一次边框回归
        rpn_bbox_pred = KL.Conv2D(self._num_anchors*4, kernel_size=1, strides=(1, 1), 
                                trainable=is_training, kernel_initializer = initializer, 
                                padding='valid', name='rpn_bbox_pred')(rpn)
        # 线性(linear)激活层
        rpn_bbox_pred = KL.Activation('linear')(rpn_bbox_pred)

        if is_training:
            rois, roi_scores = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred, 'rois')
            rpn_labels = self._anchor_target_layer(rpn_cls_score, 'anchor')
            # Try to have a deterministic order for the computing graph, for reproducibility 
            '''
            # tf.control_dependencies(control_inputs)
            # 此函数指定某些操作执行的依赖关系

            # 返回一个控制依赖的上下文管理器，使用 with 关键字可以让在这个上下文环境中的操作都在 control_inputs 执行

            # 1 with tf.control_dependencies([a, b]):
            # 2     c = ....
            # 3     d = ...
            # 在执行完 a，b 操作之后，才能执行 c，d 操作。意思就是 c，d 操作依赖 a，b 操作

            # 1 with tf.control_dependencies([train_step, variable_averages_op]):
            # 2     train_op = tf.no_op(name='train')
            # tf.no_op()表示执行完 train_step, variable_averages_op 操作之后什么都不做
            '''
            with tf.control_dependencies([rpn_labels]):
                rois, _ = self._proposal_target_layer(rois, roi_scores, 'rpn_rois')
        else:
            if cfg.TEST.MODE == 'nms':
                rois, _ = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
            elif cfg.TEST.MODE == 'top':
                rois, _ = self._proposal_top_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
            else:
                raise NotImplementedError
                
        self._predictions['rpn_cls_score'] = rpn_cls_score
        self._predictions['rpn_cls_score_reshape'] = rpn_cls_score_reshape 
        self._predictions['rpn_cls_prob'] = rpn_cls_prob 
        self._predictions['rpn_cls_pred'] = rpn_cls_pred 
        self._predictions['rpn_bbox_pred'] = rpn_bbox_pred 
        self._predictions['rois'] = rois

        return rois 

    # 为某层数据层进行形状转换
    def _reshape_layer(self, bottom, numDim, name):
        inputShape = tf.shape(bottom) #[None(0), height(1), weight(2), channel(3)]
        with tf.variable_scope(name) as scope:
            # change the channel to caffe format 
            toCaffe = tf.transpose(bottom, [0, 3, 1, 2])
            # then force it to have channel 2 
            reshaped = KL.Reshape(tf.concat(axis=0, 
                                            values=[[1, numDim, -1], [inputShape[2]]]))(toCaffe)
            # then swap the channel back 
            toTf = tf.transpose(reshaped, [0, 2, 3, 1])

        return toTf

    # softmax层
    def _softmax_layer(self, bottom, name):
        if name.startswith('rpn_cls_prob_reshape'):
            inputShape = tf.shape(bottom)
            bottomReshaped = KL.Reshape([-1, inputShape[-1]])(bottom)
            reshapedScore = tf.nn.softmax(bottomReshaped, name=name)
            backReshaped = KL.Reshape(inputShape)(reshapedScore)
            return backReshaped

        return tf.nn.softmax(bottom, name=name)

    # get anchor score and coor
    def _proposal_layer(self, rpn_cls_prob, rpn_bbox_pred, name):
        with tf.variable_scope(name) as scope:
            if cfg.USE_E2E_TF:
                rois, rpn_scores = proposal_layer_tf(
                    rpn_cls_prob, 
                    rpn_bbox_pred, 
                    self._im_info, 
                    self._mode, 
                    self._feat_stride, 
                    self._anchors, 
                    self._num_anchors
                )
            else:
                rois, rpn_scores = tf.py_func(proposal_layer,
                            [rpn_cls_prob, rpn_bbox_pred, self._im_info, self._mode,
                            self._feat_stride, self._anchors, self._num_anchors],
                            [tf.float32, tf.float32], name="proposal")
            
            rois.set_shape([None, 5])
            rpn_scores.set_shape([None, 1])
        
        return rois, rpn_scores 
    
    def _proposal_top_layer(self, rpn_cls_prob, rpn_bbox_pred, name):
        with tf.variable_scope(name) as scope:
            if cfg.USE_E2E_TF:
                rois, rpn_scores = proposal_top_layer_tf(
                rpn_cls_prob,
                rpn_bbox_pred,
                self._im_info,
                self._feat_stride,
                self._anchors,
                self._num_anchors
                )
            else:
                rois, rpn_scores = tf.py_func(proposal_top_layer,
                                    [rpn_cls_prob, rpn_bbox_pred, self._im_info,
                                    self._feat_stride, self._anchors, self._num_anchors],
                                    [tf.float32, tf.float32], name="proposal_top")
                
            rois.set_shape([cfg.TEST.RPN_TOP_N, 5])
            rpn_scores.set_shape([cfg.TEST.RPN_TOP_N, 1])

        return rois, rpn_scores

    def _anchor_target_layer(self, rpn_cls_score, name):
        with tf.variable_scope(name) as scope:
            rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = tf.py_func(
                anchor_target_layer, 
                [rpn_cls_score, self._gt_boxes, self._im_info, self._feat_stride, self._anchors, self._num_anchors], 
                [tf.float32, tf.float32, tf.float32, tf.float32], 
                name='anchor_target'
            )

            rpn_labels.set_shape([1, 1, None, None])
            rpn_bbox_targets.set_shape([1, None, None, self._num_anchors * 4])
            rpn_bbox_inside_weights.set_shape([1, None, None, self._num_anchors * 4])
            rpn_bbox_outside_weights.set_shape([1, None, None, self._num_anchors * 4])

            rpn_labels = tf.to_int32(rpn_labels, name="to_int32")
            self._anchor_targets['rpn_labels'] = rpn_labels
            self._anchor_targets['rpn_bbox_targets'] = rpn_bbox_targets
            self._anchor_targets['rpn_bbox_inside_weights'] = rpn_bbox_inside_weights
            self._anchor_targets['rpn_bbox_outside_weights'] = rpn_bbox_outside_weights

            # self._score_summaries.update(self._anchor_targets)

        return rpn_labels

    def _proposal_target_layer(self, rois, roi_scores, name):
        with tf.variable_scope(name) as scope:
            rois, roi_scores, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights = tf.py_func(
                proposal_target_layer,
                [rois, roi_scores, self._gt_boxes, self._num_classes],
                [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32],
                name="proposal_target")

            rois.set_shape([cfg.TRAIN.BATCH_SIZE, 5])
            roi_scores.set_shape([cfg.TRAIN.BATCH_SIZE])
            labels.set_shape([cfg.TRAIN.BATCH_SIZE, 1])
            bbox_targets.set_shape([cfg.TRAIN.BATCH_SIZE, self._num_classes * 4])
            bbox_inside_weights.set_shape([cfg.TRAIN.BATCH_SIZE, self._num_classes * 4])
            bbox_outside_weights.set_shape([cfg.TRAIN.BATCH_SIZE, self._num_classes * 4])

            self._proposal_targets['rois'] = rois
            self._proposal_targets['labels'] = tf.to_int32(labels, name="to_int32")
            self._proposal_targets['bbox_targets'] = bbox_targets
            self._proposal_targets['bbox_inside_weights'] = bbox_inside_weights
            self._proposal_targets['bbox_outside_weights'] = bbox_outside_weights

            self._score_summaries.update(self._proposal_targets)

        return rois, roi_scores
    
    def _crop_pool_layer(self, bottom, rois, name):
        with tf.variable_scope(name) as scope:
            batch_ids = tf.squeeze(tf.slice(rois, [0, 0], [-1, 1], name="batch_id"), [1])
            # Get the normalized coordinates of bounding boxes
            bottom_shape = tf.shape(bottom)
            height = (tf.to_float(bottom_shape[1]) - 1.) * np.float32(self._feat_stride[0])
            width = (tf.to_float(bottom_shape[2]) - 1.) * np.float32(self._feat_stride[0])
            x1 = tf.slice(rois, [0, 1], [-1, 1], name="x1") / width
            y1 = tf.slice(rois, [0, 2], [-1, 1], name="y1") / height
            x2 = tf.slice(rois, [0, 3], [-1, 1], name="x2") / width
            y2 = tf.slice(rois, [0, 4], [-1, 1], name="y2") / height
            # Won't be back-propagated to rois anyway, but to save time
            bboxes = tf.stop_gradient(tf.concat([y1, x1, y2, x2], axis=1))
            pre_pool_size = cfg.POOLING_SIZE * 2
            crops = tf.image.crop_and_resize(bottom, bboxes, tf.to_int32(batch_ids), [pre_pool_size, pre_pool_size], name="crops")

        # return slim.max_pool2d(crops, [2, 2], padding='SAME')
        return KL.MaxPool2D(pool_size=(2, 2), padding='same')(crops)

    def create_architecture(self, mode, num_classes, tag=None, 
                            anchor_scales=(8, 16, 32), anchor_ratios=(0.5, 1, 2)):
        # 数据输入占位符，Keras中将被取代
        self._image = tf.placeholder(tf.float32, shape=[1, None, None, 3])
        self._im_info = tf.placeholder(tf.float32, shape=[3])
        self._gt_boxes = tf.placeholder(tf.float32, shape=[None, 5])
        self._tag = tag 

        self._num_classes = num_classes 
        self._mode = mode 
        self._anchor_scales = anchor_scales 
        self._num_scales = len(anchor_scales)
        self._anchor_ratios = anchor_ratios 
        self._num_ratios = len(anchor_ratios) 
        self._num_anchors = self._num_scales * self._num_ratios 

        training = mode == 'TRAIN'
        testing = mode =='TEST'

        assert tag != None  


