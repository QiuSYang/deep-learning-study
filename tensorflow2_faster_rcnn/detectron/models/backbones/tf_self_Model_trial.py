"""
# tf build model trial
# 1. API build model
# 2. Custom model
"""
import os
import logging
import tensorflow as tf
layers = tf.keras.layers

_logger = logging.getLogger(__name__)

# 超参
num_words = 2000
num_tags = 12
num_departments = 4


def api_build_model():
    """
    # 建议使用这种方式任意搭建网络
    :return:
    """
    # 构建一个根据文档内容、标签和标题，预测文档优先级和执行部门的网络

    # 输入, Input就相当于占位符
    body_input = tf.keras.Input(shape=(None,), name='body')
    title_input = tf.keras.Input(shape=(None,), name='title')
    tag_input = tf.keras.Input(shape=(num_tags,), name='tag')

    # 嵌入层
    body_feat = layers.Embedding(num_words, 64)(body_input)
    title_feat = layers.Embedding(num_words, 64)(title_input)

    # 特征提取层
    body_feat = layers.LSTM(32)(body_feat)
    title_feat = layers.LSTM(128)(title_feat)
    features = layers.concatenate([title_feat, body_feat, tag_input])

    # 分类层
    priority_pred = layers.Dense(1, activation='sigmoid', name='priority')(features)
    department_pred = layers.Dense(num_departments, activation='softmax', name='department')(features)

    # 构建模型
    model = tf.keras.Model(inputs=[body_input, title_input, tag_input],
                           outputs=[priority_pred, department_pred])
    model.summary()
    tf.keras.utils.plot_model(model, 'multi_model.png', show_shapes=True)

    # 训练，各个数据指定name
    # history = model.fit(
    #     {'title': title_data, 'body': body_data, 'tag': tag_data},
    #     {'priority': priority_label, 'department': department_label},
    #     batch_size=32,
    #     epochs=5
    # )


class SelfModel(tf.keras.Model):
    def __init__(self, **kwargs):
        super(SelfModel, self).__init__(**kwargs)

        # 嵌入层
        self.body_feat = layers.Embedding(num_words, 64)
        self.title_feat = layers.Embedding(num_words, 64)

        # 特征提取层
        self.body_feat_ = layers.LSTM(32)
        self.title_feat_ = layers.LSTM(128)

        # 分类层
        self.priority_pred = layers.Dense(1, activation='sigmoid', name='priority')
        self.department_pred = layers.Dense(num_departments, activation='softmax', name='department')

    def call(self, body_input, title_input, tag_input):
        """ 此处就是在搭建graph，是否使用__call__()魔法函数更好, 之后不能使用summary()函数
        :param body_input: 可以是tensor variable，应该也可是tf.keras.Input(shape=(元组))-占位符
        :param title_input:
        :param tag_input:
        :return:
        """
        body_feature = self.body_feat(body_input)
        title_feature = self.title_feat(title_input)

        body_feature  = self.body_feat_(body_feature)
        title_feature = self.title_feat_(title_feature)
        features = layers.concatenate([title_feature, body_feature, tag_input])

        priority_predict = self.priority_pred(features)
        department_predict = self.department_pred(features)

        return priority_predict, department_predict


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    _logger.info("{}".format('build model start.'))
    # api_build_model()
    outputs = SelfModel()
    # 输入, Input就相当于占位符
    body_input = tf.keras.Input(shape=(None,), name='body')
    title_input = tf.keras.Input(shape=(None,), name='title')
    tag_input = tf.keras.Input(shape=(num_tags,), name='tag')
    priority_predict, department_predict = outputs(body_input, title_input, tag_input)
    # model = tf.keras.Model(inputs=[body_input, title_input, tag_input],
    #                        outputs=[priority_predict, department_predict])
    # model.summary()
    outputs.summary()
