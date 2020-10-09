# Only support version: python3.x
"""
# VAE(变分自编码器)-构建VAE网络主程序
"""
import os
import time
import logging
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from models.VAE import LinearVAE
from util import *

# set logger
fmt = '%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=fmt)
logger = logging.getLogger(__name__)


class MainVAE(object):
    """VAE 主程序类"""
    def __init__(self, args):
        self.args = args
        # 1. 构建模型
        # if self.args.model_path and os.path.exists(self.args.model_path):
        #     # 导入已经trained模型
        #     try:
        #         self.model = tf.keras.models.load_model(self.args.model_path)
        #     except OSError:
        #         # 模型load error
        #         self.model = LinearVAE(feature_size=self.args.feature_size)
        # else:
        #     # 从头构建模型
        #     self.model = LinearVAE(feature_size=self.args.feature_size)

        self.model = LinearVAE(feature_size=self.args.feature_size)
        if (self.args.model_dir and
                os.path.exists(self.args.model_dir) and
                os.listdir(self.args.model_dir)):
            # 导入已经trained模型
            try:
                self.load_model(self.args.model_dir, self.args.model_name)
            except OSError:
                # 模型load weight error
                pass

        # 2. 设置训练必须参数
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.args.lr)
        self.train_loss = tf.keras.metrics.Mean(name='train_average_loss')
        self.valid_loss = tf.keras.metrics.Mean(name='valid_average_loss')

    def train(self, train_data_generator, vail_data_generator):
        """模型训练"""
        best_valid_loss = np.inf
        for epoch in tf.range(1, self.args.epochs+1):
            logger.info("The {} epoch model training.".format(epoch))
            for batch_id, (transform_features, target_features) in enumerate(tqdm(train_data_generator, ncols=80)):
                with tf.GradientTape() as tape:
                    mean, logvar, eps, logit = self.model(transform_features, target_features)
                    loss = self.loss_function(eps, logits=logit, labels=target_features,
                                              means=mean, logvars=logvar)
                # backward
                gradients = tape.gradient(loss, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
                # 更新epoch train loss
                self.train_loss.update_state(loss)

            logger.info("The {} epoch model evaluating.".format(epoch))
            valid_loss = self.evaluate(vail_data_generator)
            if valid_loss < best_valid_loss:
                logger.info("Save the best model: epoch is {}.".format(epoch))
                self.save_model(self.args.model_dir, self.args.model_name)
                best_valid_loss = valid_loss

            if epoch % 1 == 0:
                logger.info("Epoch={}, Loss:{}, Valid Loss:{}, Best Valid Loss:{}".format(
                        epoch, self.train_loss.result(), self.valid_loss.result(), best_valid_loss))

        return self.model

    def evaluate(self, vail_data_generator):
        """模型评估"""
        for batch_id, (transform_features, target_features) in enumerate(tqdm(vail_data_generator, ncols=80)):
            mean, logvar, eps, logit = self.model(transform_features, target_features)
            loss = self.loss_function(eps, logits=logit, labels=target_features,
                                      means=mean, logvars=logvar)
            self.valid_loss.update_state(loss)

        return self.valid_loss.result()

    def predict(self, test_transform_features):
        """模型 inference
            test_transform_features: 一个batch的预测数据"""
        batch_logit = self.model(test_transform_features, apply_sigmoid=True)

        return batch_logit.numpy()

    def save_model(self, model_dir='./checkpoints/default_dir', model_name='default_name'):
        """保存模型
            model_dir: 模型保存路径
        """
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model_path = os.path.join(model_dir, model_name)
        self.model.save_weights(model_path)

    def load_model(self, model_dir='./checkpoints/default_dir', model_name='default_name'):
        """加载模型"""
        model_path = os.path.join(model_dir, model_name)
        self.model.load_weights(model_path)

    def loss_function(self, eps, logits, labels, means, logvars):
        """损失函数
            eps: 待转换向量A经过编码以及转换生产正态分布随机向量(重参数化)
            logits: 待转换向量A编解码计算的结果
            labels: 标准向量B
            means: 标准向量B编码生产的均值
            logvars: 标准向量B编码生产的方差
        """
        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
        logpx_z = -tf.reduce_mean(cross_ent, axis=[1])
        logpz = self.log_normal_pdf(eps, 0.0, 0.0)
        logqz_x = self.log_normal_pdf(eps, mean=means, logvar=logvars)

        return -tf.reduce_mean(logpx_z + logpz - logqz_x)

    def log_normal_pdf(self, sample, mean, logvar, raxis=-1):
        """"""
        log2pi = tf.math.log(2.0 * np.pi)

        return tf.reduce_sum(-0.5 * ((sample - mean) ** 2.0 * tf.exp(-logvar) + logvar + log2pi),
                             axis=raxis)


def get_data_from_numpy(args):
    """从numpy array 中构建数据"""
    from sklearn.model_selection import train_test_split
    DATA_SIZE = 200000
    transform_feature = np.load(args.feature_1210)[:DATA_SIZE]
    target_feature = np.load(args.feature_0220)[:DATA_SIZE]
    # 归一化
    transform_feature = data_pre_process(transform_feature)
    target_feature = data_pre_process(target_feature)

    (transform_feature_train, transform_feature_test,
     target_feature_train, target_feature_test) = train_test_split(transform_feature,
                                                                   target_feature,
                                                                   test_size=args.test_data_size)
    train_dataset = tf.data.Dataset.from_tensor_slices((transform_feature_train, target_feature_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((transform_feature_test, target_feature_test))

    SHUFFLE_BUFFER_SIZE = 100
    train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(args.batch_size)
    test_dataset = test_dataset.batch(args.batch_size)

    return train_dataset, test_dataset


def main():
    """主处理逻辑"""
    import argparse
    parser = argparse.ArgumentParser("设置训练基本参数.")
    # model
    parser.add_argument("--epochs", type=int, default=10,
                        help="训练轮数.")
    parser.add_argument("--feature_size", type=int, default=384,
                        help="特征大小.")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="学习率大小.")
    parser.add_argument("--model_dir", type=str,
                        default="./checkpoints/default_dir/",
                        help="模型保存路径.")
    parser.add_argument("--model_name", type=str,
                        default="default_name",
                        help="模型保存名称.")
    parser.add_argument("--inference", action="store_true",
                        default=True,
                        help="是否为预测模式.")

    # data
    parser.add_argument("--feature_1210", type=str,
                        default="./datasets/features_1.npy",
                        help="待转换特征数据1210.")
    parser.add_argument("--feature_0220", type=str,
                        default="./datasets/features_2.npy",
                        help="待转换特征数据0220.")
    parser.add_argument("--test_data_size", type=float, default=0.2,
                        help="数据集拆分-验证数据集的大小.")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="训练batch大小.")
    args = parser.parse_args()

    train_dataset, test_dataset = get_data_from_numpy(args)

    vae = MainVAE(args=args)
    if not args.inference:
        # train
        # vae.model.build(input_shape=(4, args.feature_size))
        # vae.model.summary()
        vae_model = vae.train(train_dataset, test_dataset)
    else:
        # inference
        test_result = []
        for batch_id, (transform_features, target_features) in enumerate(tqdm(test_dataset, ncols=80)):
            batch_result = vae.predict(transform_features)
            batch_result_convert_normalization = data_post_process(batch_result)
            batch_target_convert_normalization = data_post_process(target_features.numpy())
            logger.info("predict result: {}".format(batch_result_convert_normalization[0]))
            logger.info("target result: {}".format(batch_target_convert_normalization[0]))
            test_result.append(batch_result_convert_normalization)


if __name__ == "__main__":
    main()
