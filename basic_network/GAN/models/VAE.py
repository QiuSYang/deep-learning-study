# Only support version: python3.x
"""
# VAE: 变分自编码器实现特征向量转换(即从A向量转为B向量)
    关键点：1. 模型encode器需要从标准向量B中提取出mean and var
            2. 模型decode器将A向量向B向量生成, 并且保证A向量mean and var 也向B的 mean and var靠近
# 1. 训练过程:
    A. 我们从迭代数据集开始
    B. 在每次迭代期间，我们将标准向量(例如向量B)传递给编码器，以获得近似后验 q(z|x) 的一组均值和对数方差参数
    C. 然后，我们应用 重参数化技巧 从 q(z|x) 中采样(或者省去这一步, 采样样本直接由外界传入, 例如 A 向量(或者A向量进行编码之后的输出),
        即将A向量的均值和方差向B向量均值和方差靠近)
    D. 最后，我们将重新参数化的样本传递给解码器，以获取生成分布 p(x|z) 的 logit
# 2. 生成过程:
    A. 进行训练后，可以生成一些图片了
    B. 我们首先从单位高斯先验分布 p(z) 中采样一组潜在向量(个人感觉可以使用训练B, C过程通过编码过程直接生成样本 z,
        或者直接外部出入一个向量, 例如向量A-即需要进行转化的向量(A向量进行编码之后的输出).
        即通过原始样本先生产 z 样本, 之后再解码生成观测值Logit.)
    C. 随后生成器将潜在样本 z 转换为观测值的 logit, 得到分布 p(x|z)
# 参考链接：
    A. https://tensorflow.google.cn/tutorials/generative/cvae#%E8%AE%AD%E7%BB%83
"""
import os
import logging
import numpy as np
import tensorflow as tf

layers = tf.keras.layers
logger = logging.getLogger(__name__)


class LinearVAE(tf.keras.Model):
    """线性层变分自编码器"""
    def __init__(self, feature_size=1024):
        super(LinearVAE, self).__init__()
        self.feature_size = feature_size

        # encoder 3 dense layer
        en_fc_1_output = self.feature_size//2
        self.encoder_fc_1 = layers.Dense(units=en_fc_1_output,
                                         activation='relu',
                                         name='encoder_dense_layer_1')
        en_fc_2_output = en_fc_1_output//2
        self.encoder_fc_2 = layers.Dense(units=en_fc_2_output,
                                         activation='relu',
                                         name='encoder_dense_layer_2')
        en_fc_3_output = en_fc_2_output//2
        if en_fc_3_output % 2 == 1:
            # en_fc_3_output不是双数, -1
            en_fc_3_output = en_fc_3_output - 1
        # NO activation(mean_output size: en_fc_3_output//2, logvar size: en_fc_3_output//2)
        self.encoder_fc_3 = layers.Dense(units=en_fc_3_output,
                                         # activation='relu',
                                         # activation='sigmoid',
                                         name='encoder_dense_layer_3')

        # generate encoder
        generate_en_fc_1_output = self.feature_size // 2
        self.generate_en_fc_1 = layers.Dense(units=generate_en_fc_1_output,
                                             activation='relu',
                                             name='generate_en_dense_layer_1')
        generate_en_fc_2_output = generate_en_fc_1_output // 2
        self.generate_en_fc_2 = layers.Dense(units=generate_en_fc_2_output,
                                             activation='relu',
                                             name='generate_en_dense_layer_2')
        generate_en_fc_3_output = generate_en_fc_2_output // 2
        if generate_en_fc_3_output % 2 == 1:
            # generate_en_fc_3_output不是双数, -1(保证与标准向量B编码统一)
            generate_en_fc_3_output = generate_en_fc_3_output - 1
        self.generate_en_fc_3 = layers.Dense(units=generate_en_fc_3_output,
                                             activation='relu',
                                             name='generate_en_dense_layer_3')
        # 待转换向量A多添加一层编码层, 将编码最后输出维度与B向量编码输出mean shape一致(是否需要添加激活层有待考虑)
        generate_en_fc_4_output = generate_en_fc_3_output // 2
        self.generate_en_fc_4 = layers.Dense(units=generate_en_fc_4_output,
                                             # activation='relu',
                                             # activation='sigmoid',
                                             name='generate_en_dense_layer_4')

        # generate decoder
        generate_de_fc_1_output = generate_en_fc_4_output * 2  # generate_en_fc_3_output
        self.generate_de_fc_1 = layers.Dense(units=generate_de_fc_1_output,
                                             activation='relu',
                                             name='generate_de_dense_layer_1')
        generate_de_fc_2_output = generate_de_fc_1_output * 2  # generate_en_fc_2_output
        self.generate_de_fc_2 = layers.Dense(units=generate_de_fc_2_output,
                                             activation='relu',
                                             name='generate_de_dense_layer_2')
        generate_de_fc_3_output = generate_de_fc_2_output * 2  # generate_en_fc_1_output
        self.generate_de_fc_3 = layers.Dense(units=generate_de_fc_3_output,
                                             activation='relu',
                                             name='generate_de_dense_layer_3')
        generate_de_fc_4_output = self.feature_size  # 原始向量大小
        self.generate_de_fc_4 = layers.Dense(units=generate_de_fc_4_output,
                                             name='generate_de_dense_layer_4')
        self.bn = True

    def _layer_regularization(self, x):
        if self.bn:
            x = layers.BatchNormalization()(x)
        else:
            x = layers.Dropout(0.2)(x)

        return x

    def reparameterize(self, eps, mean, logvar):
        """合并编码得到mean and var --- 重参数化"""
        # eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * 0.5) + mean

    def encode(self, x):
        """编码, 通过训练得到标准集均值和方差"""
        x = self.encoder_fc_1(x)
        x = self._layer_regularization(x)
        x = self.encoder_fc_2(x)
        x = self._layer_regularization(x)
        x = self.encoder_fc_3(x)
        # 训练产生分布均值与方差
        mean, logvar = tf.split(x, num_or_size_splits=2, axis=1)

        return mean, logvar

    def generate_encode(self, z):
        """生成网络编码器, 对待转换向量Z进行编码(相当于教程中输入给解码网络的正态分布随机初始值)
        使用网络自动从待转换向量中生成随机初始值用于解码 --- 只是使用简单的网络生成"""
        z_en = self.generate_en_fc_1(z)
        z_en = self._layer_regularization(z_en)
        z_en = self.generate_en_fc_2(z_en)
        z_en = self._layer_regularization(z_en)
        z_en = self.generate_en_fc_3(z_en)
        z_en = self._layer_regularization(z_en)
        z_en = self.generate_en_fc_4(z_en)

        return z_en

    def generate_decode(self, z_en, apply_sigmoid=False):
        """生成网络解码器, 通过编码网络生成随机向量解码之后生成与标准向量B一致向量"""
        z_de = self.generate_de_fc_1(z_en)
        z_de = self._layer_regularization(z_de)
        z_de = self.generate_de_fc_2(z_de)
        z_de = self._layer_regularization(z_de)
        z_de = self.generate_de_fc_3(z_de)
        z_de = self._layer_regularization(z_de)
        logits = self.generate_de_fc_4(z_de)
        if apply_sigmoid:
            # inference, 先进行归一化
            probs = tf.sigmoid(logits)

            return probs

        return logits

    def sample(self, eps=None):
        """直接从单位高斯先验分布 p(z) 中采样一组潜在向量"""
        if eps is None:
            eps = tf.random.normal(shape=(100, self.feature_size))

        return self.generate_decode(eps, apply_sigmoid=True)

    def call(self, inputs, training=True):
        """ 流程函数
            inputs = (z, x) --- inference 不包含 x
            z: 待转换向量(例如向量A), 用于解码的输入
            x: 标准向量(例如向量B), 用于编码生产mean and var
        """
        if training:
            # training
            z, x = inputs  # 解析输入数据
            apply_sigmoid = False

            if x is None:
                logger.info("Have to input x tensor.")
                raise ValueError("Have to input x tensor.")
            means, logvars = self.encode(x)
            z_en = self.generate_encode(z)
            eps = self.reparameterize(z_en, mean=means, logvar=logvars)  # 重参数化
            logits = self.generate_decode(eps, apply_sigmoid=apply_sigmoid)

            return means, logvars, eps, logits
        else:
            # inference
            z = inputs  # 解析输入数据
            apply_sigmoid = True

            z_en = self.generate_encode(z)
            # 预测时候不在进行reparameterize(直接使用待转换A向量编码向量作为解码的输入 --- 无重参数化)
            logits = self.generate_decode(z_en, apply_sigmoid=apply_sigmoid)

            return logits
