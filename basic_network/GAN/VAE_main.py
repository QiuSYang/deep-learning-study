# Only support version: python3.x
"""
# VAE(变分自编码器)-构建VAE网络主程序
"""
import os
import logging
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from models.VAE import LinearVAE

# set logger
fmt = '%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=fmt)
logger = logging.getLogger(__name__)


class MainVAE(object):
    """VAE 主程序类"""
    def __init__(self, args):
        self.args = args
        # 1. 构建模型
        if self.args.model_path and os.path.exists(self.args.model_path):
            # 导入已经trained模型
            try:
                self.model = tf.keras.models.load_model(self.args.model_path)
            except OSError:
                # 模型load error
                self.model = LinearVAE(feature_size=self.args.feature_size)
        else:
            # 从头构建模型
            self.model = LinearVAE(feature_size=self.args.feature_size)

        # 2. 设置训练必须参数
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.args.lr)
        self.train_loss = tf.keras.metrics.Mean(name='train_average_loss')
        self.valid_loss = tf.keras.metrics.Mean(name='valid_average_loss')

    def train(self, train_data_generator, vail_data_generator):
        """模型训练"""
        best_valid_loss = np.inf
        for epoch in tf.range(1, self.args.epochs+1):
            logger.info("The {} epoch model training.".format(epoch))
            for batch_id, (transform_features, target_features) in tqdm(enumerate(train_data_generator)):
                with tf.GradientTape() as tape:
                    mean, logvar, eps, logit = self.model(transform_features, target_features)
                    loss = self.loss_function(eps, logits=logit,
                                              means=mean, logvars=logger)
                # backward
                gradients = tape.gradient(loss, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
                # 更新epoch train loss
                self.train_loss.update_state(loss)

            logger.info("The {} epoch model evaluating.".format(epoch))
            valid_loss = self.evaluate(vail_data_generator)
            if valid_loss > best_valid_loss:
                logger.info("Save the best model: epoch is {}.".format(epoch))
                self.save_model(self.args.model_path)
                best_valid_loss = valid_loss

        return self.model

    def evaluate(self, vail_data_generator):
        """模型评估"""
        for batch_id, (transform_features, target_features) in tqdm(enumerate(vail_data_generator)):
            mean, logvar, eps, logit = self.model(transform_features, target_features)
            loss = self.loss_function(eps, logits=logit,
                                      means=mean, logvars=logger)
            self.valid_loss.update_state(loss)

        return self.valid_loss.result()

    def predict(self, test_data):
        """模型 inference"""
        pass

    def save_model(self, model_path='./checkpoints'):
        """保存模型
            model_path: 模型保存路径
        """
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        self.model.save(model_path)

    @tf.function
    def loss_function(self, eps, logits, labels, means, logvars):
        """损失函数
            eps: 待转换向量A经过编码以及转换生产正态分布随机向量(重参数化)
            logits: 待转换向量A编解码计算的结果
            labels: 标准向量B
            means: 标准向量B编码生产的均值
            logvars: 标准向量B编码生产的方差
        """
        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
        logpx_z = -tf.reduce_mean(cross_ent, axis=[1, 2, 3])
        logpz = self.log_normal_pdf(eps, 0.0, 0.0)
        logqz_x = self.log_normal_pdf(eps, mean=means, logvar=logvars)

        return -tf.reduce_mean(logpx_z + logpz - logqz_x)

    def log_normal_pdf(self, sample, mean, logvar, raxis=1):
        """"""
        log2pi = tf.math.log(2. * np.pi)

        return tf.reduce_sum(-0.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
                             axis=raxis)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("设置训练基本参数.")
    parser.add_argument("--epochs", type=int, default=10,
                        help="训练轮数.")
    parser.add_argument("--feature_size", type=int, default=384,
                        help="训练轮数.")
    parser.add_argument("--model_path", type=str,
                        default="./checkpoints",
                        help="模型保存路径.")
    parser.add_argument("--inference", action="store_true",
                        default=False,
                        help="是否为预测模式.")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="学习率大小.")
    args = parser.parse_args()

    main = MainVAE(args=args)
    # main.model.build(input_shape=(4, args.feature_size))
    # main.model.summary()
