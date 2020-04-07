"""
# 使用Keras搭建faster-rcnn特征提取基础网络Conv_5: vggnet
"""
import os 
import keras 
import keras.models as KM 
import keras.layers as KL 
import keras.backend as KB 

from nets.network import Network
from model.config import cfg

class Vgg(Network):
    def __init__(self):
        super(Vgg, self).__init__()

    def nnBase(self,
               image_input=KL.Input(shape=(256, 256, 3)),
               Trainable=False):
        """
        :param image_input: KL.Input() 对象， a tensor variable
        :param Trainable:
        :return:
        """
        # image_input = KL.Input(shape=InputShape)

        # block 1
        x = KL.Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding='same',
                    name='block1_conv1', trainable=Trainable)(image_input)
        x = KL.Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding='same',
                    name='block1_conv2', trainable=Trainable)(x)
        x = KL.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

        # block 2
        x = KL.Conv2D(128, (3, 3), strides=(1, 1), activation='relu', padding='same',
                    name='block2_conv1', trainable=Trainable)(x)
        x = KL.Conv2D(128, (3, 3), strides=(1, 1), activation='relu', padding='same',
                    name='block2_conv2', trainable=Trainable)(x)
        x = KL.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

        # block 3
        x = KL.Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='same',
                    name='block3_conv1', trainable=Trainable)(x)
        x = KL.Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='same',
                    name='block3_conv2', trainable=Trainable)(x)
        x = KL.Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='same',
                    name='block3_conv3', trainable=Trainable)(x)
        x = KL.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

        # block 4
        x = KL.Conv2D(512, (3, 3), strides=(1, 1), activation='relu', padding='same',
                    name='block4_conv1', trainable=Trainable)(x)
        x = KL.Conv2D(512, (3, 3), strides=(1, 1), activation='relu', padding='same',
                    name='block4_conv2', trainable=Trainable)(x)
        x = KL.Conv2D(512, (3, 3), strides=(1, 1), activation='relu', padding='same',
                    name='block4_conv3', trainable=Trainable)(x)
        x = KL.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

        # block 5
        x = KL.Conv2D(512, (3, 3), strides=(1, 1), activation='relu', padding='same',
                    name='block5_conv1', trainable=Trainable)(x)
        x = KL.Conv2D(512, (3, 3), strides=(1, 1), activation='relu', padding='same',
                    name='block5_conv2', trainable=Trainable)(x)
        x = KL.Conv2D(512, (3, 3), strides=(1, 1), activation='relu', padding='same',
                    name='block5_conv3', trainable=Trainable)(x)
        #x = KL.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

        return x

    def _image_to_head(self, input_tensor, is_training):
        """
        # 基础网络
        :param input_tensor:
        :param is_training:
        :return:
        """

        conv_net = self.nnBase(input_tensor, Trainable=is_training)

        return conv_net

    def _head_to_tail(self, pool5, is_training):
        """
        # 全连接层
        :param pool5:
        :param is_training:
        :return:
        """
        pool5_flat = KL.Flatten(name='flatten')(pool5)
        fc6 = KL.Dense(4096, name='fc6')(pool5_flat)
        if is_training:
            fc6 = KL.Dropout(keep_prob=0.5, is_training=True,
                             name='dropout6')(fc6)
        fc7 = KL.Dense(4096, name='fc7')(fc6)
        if is_training:
            fc7 = KL.Dropout(keep_prob=0.5, is_training=True,
                             name='dropout7')(fc7)

        return fc7


if __name__ == "__main__":
    vgg = Vgg()
    InputShape = (256, 256, 3)
    image_input = KL.Input(shape=InputShape)
    feature_output = vgg.nnBase(image_input=image_input, Trainable=True)

    model = KM.Model(inputs=image_input, outputs=feature_output)

    model.summary() 

    # model graph 
    keras.utils.plot_model(model, to_file='vgg16_model.png')
