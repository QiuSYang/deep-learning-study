# -----------------------------------------
# 获取锚点
# -----------------------------------------

import tensorflow as tf 
import numpy as np 
from layer_utils.generate_anchors import generate_anchors 

def generate_anchors_pre(height, width, feat_stride, 
                        anchor_scales=(8, 16, 32), anchor_ratios=(0.5, 1, 2)):
    ''' A wrapper function to generate anchors given different scales, 
        Also return the number of anchors in variable 'length'
    '''
    # 锚框记录的两个坐标点是左下角右上角
    anchors = generate_anchors(ratios=np.array(anchor_ratios), scales=np.array(anchor_scales))
    A = anchors.shape[0] 
    shift_x = np.arange(0, width) * feat_stride 
    shift_y = np.arange(0, height) * feat_stride 
    # ----------------- #
    # x = np.array([0, 1, 2])
    # y = np.array([0, 1])
    # np.meshgrid()函数说明
    # X, Y = np.meshgrid(x, y)
    # X = [[0 1 2]
    #      [0 1 2]]
    # Y = [[0 0 0]
    #      [1 1 1]]
    # ------------------ # 生成图像每个点坐标位置
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    #******************************#
    # np.hstack() np.vstack()实例说明
    # a = np.array([[1, 2],[2, 3],[3, 4]])
    # b = np.array([[2, 3],[3, 4],[4, 5]])
    # np.hstack((a,b)) = 
    #                 array([[1, 2, 2, 3],
    #                     [2, 3, 3, 4],
    #                     [3, 4, 4, 5]])
    # np.vstack((a,b)) = 
    #                 array([[1, 2],
    #                     [2, 3],
    #                     [3, 4],
    #                     [2, 3],
    #                     [3, 4],
    #                     [4, 5]])
    # np.ravel() 解释
    # a = arange(12).reshape(3,4) = [[ 0  1  2  3]
    #                                [ 4  5  6  7]
    #                                [ 8  9 10 11]]
    # a.ravel() = [ 0  1  2  3  4  5  6  7  8  9 10 11]
    #******************************#
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), 
                        shift_x.ravel(), shift_y.ravel())).transpose()
    K = shifts.shape[0]

    # width changes faster, so here it is H, W, C 
    anchors = anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2))
    anchors = anchors.reshape((K * A, 4)).astype(np.float32, copy=False)
    length = np.int32(anchors.shape[0])

    return anchors, length 


def generate_anchors_pre_tf(height, width, feat_stride=16, 
                            anchor_scales=(8,16,32), anchor_ratios=(0.5,1,2)):
    shift_x = tf.range(width) * feat_stride # width 
    shift_y = tf.range(height) * feat_stride # height 
    shift_x, shift_y = tf.meshgrid(shift_x, shift_y) 
    sx = tf.reshape(shift_x, shape=(-1, ))
    sy = tf.reshape(shift_y, shape=(-1, ))
    shifts = tf.transpose(tf.stack([sx, sy, sx, sy]))

    K = tf.multiply(width, height)
    shifts = tf.transpose(tf.reshape(shifts, shape=[1, K, 4]), perm=(1, 0, 2))

    anchors = generate_anchors(ratios=np.array(anchor_ratios), scales=np.array(anchor_scales))
    A = anchors.shape[0]
    anchor_constant = tf.constant(anchors.reshape((1, A, 4)), dtype=tf.int32)

    length = K * A
    anchors_tf = tf.reshape(tf.add(anchor_constant, shifts), shape=(length, 4))
    
    return tf.cast(anchors_tf, dtype=tf.float32), length
