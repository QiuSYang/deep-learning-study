"""
# support version: python3.x
# VAE: 变分自编码器实现特征向量转换(即从A向量转为B向量)
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
                                         name='encoder dense layer 1')
        en_fc_2_output = en_fc_1_output//2
        self.encoder_fc_2 = layers.Dense(units=en_fc_2_output,
                                         activation='relu',
                                         name='encoder dense layer 2')
        en_fc_3_output = en_fc_2_output//2
        if en_fc_3_output % 2 == 1:
            # en_fc_3_output不是双数, -1
            en_fc_3_output = en_fc_3_output - 1
        # NO activation(mean_output=en_fc_3_output//2, logvar=en_fc_3_output//2)
        self.encoder_fc_3 = layers.Dense(units=en_fc_3_output,
                                         name='encoder dense layer 3')

        # decoder
        de_fc_1_output = en_fc_3_output
        self.decoder_fc_1 = layers.Dense(units=de_fc_1_output,
                                         activation='relu',
                                         name='decoder dense layer 1')
        de_fc_2_output = de_fc_1_output*2
        self.decoder_fc_2 = layers.Dense(units=de_fc_2_output,
                                         activation='relu',
                                         name='decoder dense layer 2')
        de_fc_3_output = de_fc_2_output * 2
        self.decoder_fc_3 = layers.Dense(units=de_fc_3_output,
                                         activation='relu',
                                         name='decoder dense layer 3')
        de_fc_4_output = self.feature_size
        self.decoder_fc_4 = layers.Dense(units=de_fc_4_output,
                                         name='decoder dense layer 3')

    def reparameterize(self, mean, logvar):
        """合并编码得到mean and var"""
        eps = tf.random.normal(shape=mean.shape)

        return eps * tf.exp(logvar * 5) + mean

    def encode(self, x):
        """编码"""
        x = self.encoder_fc_1(x)
        x = self.encoder_fc_2(x)
        x = self.encoder_fc_3(x)
        mean, logvar = tf.split(x, num_or_size_splits=2, axis=1)

        return mean, logvar

    def decode(self, z, apply_sigmoid=False):
        """解码"""
        z = self.decoder_fc_1(z)
        z = self.decoder_fc_2(z)
        z = self.decoder_fc_3(z)
        logits = self.decoder_fc_4(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)

            return probs

        return logits
