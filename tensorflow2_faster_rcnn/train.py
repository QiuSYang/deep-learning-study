"""
# train coco data
"""
import os
import tensorflow as tf
import numpy as np
from detectron.utils import visualize

# tensorflow config - using one gpu and extending the GPU
# memory region needed by the TensorFlow process
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# session = tf.Session(config=config)

from detectron.datasets import coco, data_generator

