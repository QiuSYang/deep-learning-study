"""
# 搭建faster-rcnn网络流程图(pipeline)
"""
import os 
import sys 
import keras 
import keras.layers as KL 
import keras.models as KL 
import keras.backend as KB 
import numpy as np 
import tensorflow as tf 

class NetWork(object):
    def __init__(self):
        pass 

    def _build_network(self, is_training=True):
        # select initializers 
        if True:
            initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
            initializer_box = tf.truncated_normal_initializer(mean=0.0, stddev=0.001)
        else:
            initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
            initializer_box = tf.random_normal_initializer(mean=0.0, stddev=0.001)
        
        net_conv = None
        with tf.variable_scope():
            # build the anchors for the image 
            pass 
        