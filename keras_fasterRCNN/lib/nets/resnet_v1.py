"""
# 使用Keras搭建faster-rcnn特征提取基础网络Conv_5: resnet
"""
import os 
import keras
import keras.models as KM 
import keras.layers as KL 
import keras.backend as KB
import tensorflow as tf
import numpy as np

from model.config import cfg
from nets.network import Network

class Resnet(Network):
    def __init__(self):
        super(Resnet, self).__init__()

    def _bottleneckBlock(self, InputTensor, KernelSize, Filters, Strides,
                        Stage, Block, Trainable=True, ShutCut=False):
        # 创建说明信息
        conv_name_base = 'res' + str(Stage) + Block + '_branch'
        bn_name_base = 'bn' + str(Stage) + Block + '_branch'

        # 获取卷积核的大小
        Ft1, Ft2, Ft3 = Filters

        x = KL.Conv2D(Ft1, kernel_size=1, strides=Strides, padding='same',
                        name=conv_name_base+'2a', trainable=Trainable)(InputTensor)
        x = KL.BatchNormalization(name=bn_name_base+'2a')(x)
        x = KL.Activation('relu')(x)

        x = KL.Conv2D(Ft2, kernel_size=KernelSize, padding='same', name=conv_name_base+'2b',
                        trainable=Trainable)(x)
        x = KL.BatchNormalization(name=bn_name_base+'2b')(x)
        x = KL.Activation('relu')(x)

        x = KL.Conv2D(Ft3, kernel_size=1, padding='same', name=conv_name_base+'2c',
                        trainable=Trainable)(x)
        x = KL.BatchNormalization(name=bn_name_base+'2c')(x)

        if ShutCut:
            shutcut_x = KL.Conv2D(Ft3, kernel_size=1, strides=Strides, padding='same',
                            name=conv_name_base+'1', trainable=Trainable)(InputTensor)
            shutcut_x = KL.BatchNormalization(name=bn_name_base+'1')(shutcut_x)
        else:
            shutcut_x = InputTensor

        x = KL.Add()([x, shutcut_x])
        x = KL.Activation('relu')(x)

        return x

    def nnBase(self, image_input=KL.Input(shape=(256, 256, 3)),
               BlockSizes=[3, 4, 6, 3], Trainable=False):
        """
        :param image_input: KL.Input() 对象， a tensor variable
        :param BlockSizes:
        :param Trainable:
        :return:
        """
        # image_input = KL.Input(shape=InputShape)

        # 基础模块 stage=1
        x = KL.ZeroPadding2D(padding=(3, 3))(image_input)
        x = KL.Conv2D(64, kernel_size=7, strides=(2, 2),
                    padding='valid', name='conv1', trainable=Trainable)(x)
        x = KL.BatchNormalization(name='bn_conv1')(x)
        x = KL.Activation('relu')(x)
        x = KL.MaxPooling2D(3, strides=(2, 2))(x)

        # stage=2
        x = self._bottleneckBlock(x, 3, [64, 64, 256], Strides=(1, 1),
                            Stage=2, Block=str(1), Trainable=Trainable, ShutCut=True)
        for i in range(1, BlockSizes[0]):
            x = self._bottleneckBlock(x, 3, [64, 64, 256], Strides=(1, 1),
                                Stage=2, Block=str(i+1), Trainable=Trainable, ShutCut=False)

        # stage=3
        x = self._bottleneckBlock(x, 3, [128, 128, 512], Strides=(2, 2),
                            Stage=3, Block=str(1), Trainable=Trainable, ShutCut=True)
        for i in range(1, BlockSizes[1]):
            x = self._bottleneckBlock(x, 3, [128, 128, 512], Strides=(1, 1),
                                Stage=3, Block=str(i+1), Trainable=Trainable, ShutCut=False)

        # stage=4
        x = self._bottleneckBlock(x, 3, [256, 256, 1024], Strides=(2, 2),
                            Stage=4, Block=str(1), Trainable=Trainable, ShutCut=True)
        for i in range(1, BlockSizes[2]):
            x = self._bottleneckBlock(x, 3, [256, 256, 1024], Strides=(1, 1),
                                Stage=4, Block=str(i+1), Trainable=Trainable, ShutCut=False)

        # # stage=5
        # x = self._bottleneckBlock(x, 3, [512, 512, 2048], Strides=(2, 2),
        #                     Stage=5, Block=str(1), Trainable=Trainable, ShutCut=True)
        # for i in range(1, BlockSizes[3]):
        #     x = self._bottleneckBlock(x, 3, [512, 512, 2048], Strides=(1, 1),
        #                         Stage=5, Block=str(i+1), Trainable=Trainable, ShutCut=False)

        return x

    def _crop_pool_layer(self, bottom, rois, name):
        with tf.variable_scope(name) as scope:
            batch_ids = tf.squeeze(tf.slice(rois, [0, 0], [-1, 1], name="batch_id"), [1])
            # Get the normalized coordinates of bboxes
            bottom_shape = tf.shape(bottom)
            height = (tf.to_float(bottom_shape[1]) - 1.) * np.float32(self._feat_stride[0])
            width = (tf.to_float(bottom_shape[2]) - 1.) * np.float32(self._feat_stride[0])
            x1 = tf.slice(rois, [0, 1], [-1, 1], name="x1") / width
            y1 = tf.slice(rois, [0, 2], [-1, 1], name="y1") / height
            x2 = tf.slice(rois, [0, 3], [-1, 1], name="x2") / width
            y2 = tf.slice(rois, [0, 4], [-1, 1], name="y2") / height
            # Won't be back-propagated to rois anyway, but to save time
            bboxes = tf.stop_gradient(tf.concat([y1, x1, y2, x2], 1))
            if cfg.RESNET.MAX_POOL:
                pre_pool_size = cfg.POOLING_SIZE * 2
                crops = tf.image.crop_and_resize(bottom, bboxes, tf.to_int32(batch_ids), [pre_pool_size, pre_pool_size],
                                                 name="crops")
                crops = KL.MaxPooling2D(crops, [2, 2], padding='SAME')
            else:
                crops = tf.image.crop_and_resize(bottom, bboxes, tf.to_int32(batch_ids),
                                                 [cfg.POOLING_SIZE, cfg.POOLING_SIZE],
                                                 name="crops")
        return crops

    def _image_to_head(self, input_tensor, is_training):
        """
        # 基础网络
        :param input_tensor:
        :param is_training:
        :return:
        """
        conv_net = self.nnBase(input_tensor, Trainable=is_training)

        return conv_net

    def _head_to_tail(self, pool5, is_training=False, BlockSizes=[3, 4, 6, 3]):
        """
        # 全连接层，分类前最后参数的学习
        :param pool5:
        :param is_training:
        :param BlockSizes:
        :return:
        """
        # stage=5
        x = self._bottleneckBlock(pool5, 3, [512, 512, 2048], Strides=(2, 2),
                            Stage=5, Block=str(1), Trainable=is_training, ShutCut=True)
        for i in range(1, BlockSizes[3]):
            x = self._bottleneckBlock(x, 3, [512, 512, 2048], Strides=(1, 1),
                                Stage=5, Block=str(i+1), Trainable=is_training, ShutCut=False)

        # fc7 = KL.AveragePooling2D((7, 7), padding='same')(x)
        # average pooling done by reduce_mean
        fc7 = tf.reduce_mean(x, axis=[1, 2])

        return fc7


if __name__ == "__main__":
    resnet = Resnet()
    InputShape = (256, 256, 3)
    image_input = KL.Input(shape=InputShape)
    feature_output = resnet.nnBase(image_input=image_input, Trainable=True)

    model = KM.Model(inputs=image_input, outputs=feature_output)

    model.summary() 

    # model graph 
    keras.utils.plot_model(model, to_file='resnet50_model.png')

    for layer in model.layers:
        print(layer.name, layer.trainable)

