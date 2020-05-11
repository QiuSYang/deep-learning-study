"""
# train coco data
"""
import os
import tensorflow as tf
import numpy as np
from detectron.utils import visualize

# tensorflow config - using one gpu and extending the GPU
# memory region needed by the TensorFlow process
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# session = tf.Session(config=config)

from detectron.datasets import coco, data_generator

img_mean = (123.675, 116.28, 103.53)
# img_std = (58.395, 57.12, 57.375)
img_std = (1., 1., 1.)

train_dataset = coco.CocoDataSet('./datasets/', 'val',
                                 flip_ratio=0.5,
                                 pad_mode='fixed',
                                 mean=img_mean,
                                 std=img_std,
                                 scale=(800, 1024))

train_generator = data_generator.DataGenerator(train_dataset)

# batch_size = 2
#
# train_tf_dataset = tf.data.Dataset.from_generator(
#     train_generator, (tf.float32, tf.float32, tf.float32, tf.int32))
# train_tf_dataset = train_tf_dataset.padded_batch(
#     batch_size, padded_shapes=([None, None, None], [None], [None, None], [None]))
# train_tf_dataset = train_tf_dataset.prefetch(100).shuffle(100)
#
# for (batch, inputs) in enumerate(train_tf_dataset):
#     batch_imgs, batch_metas, batch_bboxes, batch_labels = inputs
#     print(batch_imgs.shape)

from detectron.datasets.utils import get_original_image
img, img_meta, bboxes, labels = train_dataset[3]

rgb_img = np.round(img + img_mean)
ori_img = get_original_image(img, img_meta, img_mean)

visualize.display_instances(rgb_img, bboxes, labels, train_dataset.get_categories())

from detectron.models.detectors import faster_rcnn

model = faster_rcnn.FasterRCNN(
    num_classes=len(train_dataset.get_categories()))

batch_imgs = tf.Variable(np.expand_dims(img, 0), dtype=tf.float32)
batch_metas = tf.Variable(np.expand_dims(img_meta, 0), dtype=tf.float32)
batch_bboxes = tf.Variable(np.expand_dims(bboxes, 0), dtype=tf.float32)
batch_labels = tf.Variable(np.expand_dims(labels, 0), dtype=tf.int32)

_ = model((batch_imgs, batch_metas), training=False)

model.load_weights('weights/faster_rcnn.h5', by_name=True)

proposals = model.simple_test_rpn(img, img_meta)
res = model.simple_test_bboxes(img, img_meta, proposals)

visualize.display_instances(ori_img, res['rois'], res['class_ids'],
                            train_dataset.get_categories(), scores=res['scores'])


