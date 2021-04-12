"""
使用飞桨(paddle)实现基于LSTM的情感分析模型
接下来让我们看看如何使用飞桨实现一个基于长短时记忆网络的情感分析模型。在飞桨中，不同深度学习模型的训练过程基本一致，流程如下：
    1. 数据处理：选择需要使用的数据，并做好必要的预处理工作。
    2. 网络定义：使用飞桨定义好网络结构，包括输入层，中间层，输出层，损失函数和优化算法。
    3. 网络训练：将准备好的数据送入神经网络进行学习，并观察学习的过程是否正常，如损失函数值是否在降低，也可以打印一些中间步骤的结果出来等。
    4. 网络评估：使用测试集合测试训练好的神经网络，看看训练效果如何
"""
import os
import sys
import logging
import re
import random
import tarfile
import requests
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

logger = logging.getLogger(__name__)
work_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(work_root)  # 设置当前路径为工作路径

from src.utils import (
    init_logs,
    save_dict_obj,
    load_json_obj,
)


def download(data_file):
    """下载IMDB的电影评论数据"""
    # 通过python的requests类，下载存储在
    # https://dataset.bj.bcebos.com/imdb%2FaclImdb_v1.tar.gz的文件
    corpus_url = "https://dataset.bj.bcebos.com/imdb%2FaclImdb_v1.tar.gz"
    web_request = requests.get(corpus_url)
    corpus = web_request.content

    # 将下载的文件写在当前目录的aclImdb_v1.tar.gz文件内
    with open(data_file, "wb") as f:
        f.write(corpus)
    f.close()


def load_imdb(data_file, is_training=False):
    """加载数据集"""
    data_set = []

    # aclImdb_v1.tar.gz解压后是一个目录
    # 我们可以使用python的rarfile库进行解压
    # 训练数据和测试数据已经经过切分，其中训练数据的地址为：
    # ./aclImdb/train/pos/ 和 ./aclImdb/train/neg/，分别存储着正向情感的数据和负向情感的数据
    # 我们把数据依次读取出来，并放到data_set里
    # data_set中每个元素都是一个二元组，（句子，label），其中label=0表示负向情感，label=1表示正向情感

    for label in ["pos", "neg"]:
        with tarfile.open(data_file) as tarf:
            path_pattern = "aclImdb/train/" + label + "/.*\.txt$" if is_training \
                else "aclImdb/test/" + label + "/.*\.txt$"
            path_pattern = re.compile(path_pattern)
            tf = tarf.next()
            while tf != None:
                if bool(path_pattern.match(tf.name)):
                    sentence = tarf.extractfile(tf).read().decode()
                    sentence_label = 0 if label == 'neg' else 1
                    data_set.append((sentence, sentence_label))
                tf = tarf.next()

    return data_set


def data_preprocess(corpus):
    """数据预处理"""
    data_set = []
    for sentence, sentence_label in corpus:
        # 这里有一个小trick是把所有的句子转换为小写，从而减小词表的大小
        # 一般来说这样的做法有助于效果提升
        sentence = sentence.strip().lower()
        sentence = sentence.split(" ")  # 英文按空格切词

        data_set.append((sentence, sentence_label))

    return data_set


def build_dict(corpus):
    """构建词表"""
    word_freq_dict = dict()
    for sentence, _ in corpus:
        for word in sentence:
            if word not in word_freq_dict:
                word_freq_dict[word] = 0
            word_freq_dict[word] += 1

    word_freq_dict = sorted(word_freq_dict.items(), key=lambda x: x[1], reverse=True)  # 使用词频排序

    word2id_dict = dict()
    word2id_freq = dict()
    id2word_dict = dict()

    # 一般来说，我们把oov和pad放在词典前面，给他们一个比较小的id，这样比较方便记忆，并且易于后续扩展词表
    word2id_dict["[pad]"] = 0
    id2word_dict[0] = "[pad]"
    word2id_freq[0] = 1e10

    word2id_dict["[oov]"] = 1
    id2word_dict[1] = "[oov]"
    word2id_freq[1] = 1e10

    for word, freq in word_freq_dict:
        current_id = len(word2id_dict)
        word2id_dict[word] = current_id
        id2word_dict[current_id] = word
        assert current_id == word2id_dict[word]
        word2id_freq[word2id_dict[word]] = freq

    return word2id_freq, word2id_dict, id2word_dict


def convert_corpus_to_id(corpus, word2id_dict):
    """把语料转换为id序列"""
    data_set = []
    for sentence, sentence_label in corpus:
        # 将句子中的词逐个替换成id，如果句子中的词不在词表内，则替换成oov
        # 这里需要注意，一般来说我们可能需要查看一下test-set中，句子oov的比例，
        # 如果存在过多oov的情况，那就说明我们的训练数据不足或者切分存在巨大偏差，需要调整
        sentence = [word2id_dict[word] if word in word2id_dict else word2id_dict['[oov]'] for word in sentence]
        data_set.append((sentence, sentence_label))
    return data_set


def build_batch(word2id_dict, corpus, batch_size, epoch_num, max_seq_len, shuffle=True, drop_last=False):
    # 模型将会接受的两个输入：
    # 1. 一个形状为[batch_size, max_seq_len]的张量，sentence_batch，代表了一个mini-batch的句子。
    # 2. 一个形状为[batch_size, 1]的张量，sentence_label_batch，每个元素都是非0即1，代表了每个句子的情感类别（正向或者负向）
    sentence_batch = []
    sentence_label_batch = []

    for _ in range(epoch_num):

        # 每个epoch前都shuffle一下数据，有助于提高模型训练的效果
        # 但是对于预测任务，不要做数据shuffle
        if shuffle:
            random.shuffle(corpus)

        for sentence, sentence_label in corpus:
            sentence_sample = sentence[:min(max_seq_len, len(sentence))]
            if len(sentence_sample) < max_seq_len:
                for _ in range(max_seq_len - len(sentence_sample)):
                    sentence_sample.append(word2id_dict['[pad]'])  # 数据补齐

            sentence_sample = [[word_id] for word_id in sentence_sample]

            sentence_batch.append(sentence_sample)
            sentence_label_batch.append([sentence_label])

            if len(sentence_batch) == batch_size:
                yield np.array(sentence_batch).astype("int64"), np.array(sentence_label_batch).astype("int64")
                sentence_batch = []
                sentence_label_batch = []
        if not drop_last and len(sentence_batch) > 0:
            yield np.array(sentence_batch).astype("int64"), np.array(sentence_label_batch).astype("int64")


class SentimentClassifier(nn.Layer):
    """定义一个用于情感分类的网络实例"""

    def __init__(self, hidden_size, vocab_size, class_num=2, num_steps=128, num_layers=1, init_scale=0.1, dropout=None):
        # 参数含义如下：
        # 1.hidden_size，表示embedding-size，hidden和cell向量的维度
        # 2.vocab_size，模型可以考虑的词表大小
        # 3.class_num，情感类型个数，可以是2分类，也可以是多分类
        # 4.num_steps，表示这个情感分析模型最大可以考虑的句子长度
        # 5.num_layers，表示网络的层数
        # 6.init_scale，表示网络内部的参数的初始化范围
        # 长短时记忆网络内部用了很多Tanh，Sigmoid等激活函数，这些函数对数值精度非常敏感，
        # 因此我们一般只使用比较小的初始化范围，以保证效果

        super(SentimentClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.class_num = class_num
        self.init_scale = init_scale
        self.num_layers = num_layers
        self.num_steps = num_steps
        self.dropout = dropout

        # 声明一个embedding层，用来把句子中的每个词转换为向量
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=hidden_size, sparse=False,
                                      weight_attr=paddle.ParamAttr(
                                       initializer=paddle.nn.initializer.Uniform(low=-init_scale, high=init_scale)))

        # 声明一个LSTM模型，用来把每个句子抽象成向量
        self.simple_lstm_rnn = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers)

        # 在得到一个句子的向量表示后，需要根据这个向量表示对这个句子进行分类
        # 一般来说，可以把这个句子的向量表示乘以一个大小为[self.hidden_size, self.class_num]的W参数，
        # 并加上一个大小为[self.class_num]的b参数，从而达到把句子向量映射到分类结果的目的

        # 我们需要声明最终在使用句子向量映射到具体情感类别过程中所需要使用的参数
        # 这个参数的大小一般是[self.hidden_size, self.class_num]
        self.cls_fc = nn.Linear(in_features=self.hidden_size, out_features=self.class_num,
                                weight_attr=None, bias_attr=None)
        self.dropout_layer = nn.Dropout(p=self.dropout, mode='upscale_in_train')

    def forward(self, inputs, label=None):
        # 首先我们需要定义LSTM的初始hidden和cell，这里我们使用0来初始化这个序列的记忆
        batch_size = inputs.shape[0]
        init_hidden_data = np.zeros(
            (self.num_layers, batch_size, self.hidden_size), dtype='float32')
        init_cell_data = np.zeros(
            (self.num_layers, batch_size, self.hidden_size), dtype='float32')

        # 将这些初始记忆转换为飞桨可计算的向量
        # 设置stop_gradient=True，避免这些向量被更新，从而影响训练效果
        init_hidden = paddle.to_tensor(init_hidden_data)
        init_hidden.stop_gradient = True
        init_cell = paddle.to_tensor(init_cell_data)
        init_cell.stop_gradient = True

        init_h = paddle.reshape(
            init_hidden, shape=[self.num_layers, -1, self.hidden_size])
        init_c = paddle.reshape(
            init_cell, shape=[self.num_layers, -1, self.hidden_size])

        # 将输入的句子的mini-batch转换为词向量表示
        x_emb = self.embedding(inputs)
        x_emb = paddle.reshape(
            x_emb, shape=[-1, self.num_steps, self.hidden_size])
        if self.dropout is not None and self.dropout > 0.0:
            x_emb = self.dropout_layer(x_emb)

        # 使用LSTM网络，把每个句子转换为向量表示
        rnn_out, (last_hidden, last_cell) = self.simple_lstm_rnn(x_emb, (init_h, init_c))
        last_hidden = paddle.reshape(
            last_hidden[-1], shape=[-1, self.hidden_size])

        # 将每个句子的向量表示映射到具体的情感类别上
        projection = self.cls_fc(last_hidden)
        pred = F.softmax(projection, axis=-1)

        if label is not None:
            # 根据给定的标签信息，计算整个网络的损失函数，这里我们可以直接使用分类任务中常使用的交叉熵来训练网络
            loss = F.softmax_with_cross_entropy(
                logits=projection, label=label, soft_label=False)
            loss = paddle.mean(loss)

            # 最终返回预测结果pred，和网络的loss
            return pred, loss
        else:
            return pred


def train(model, opt, word2id_dict, corpus, epochs=10, batch_size=32, max_seq_len=128):
    """模型训练"""
    model.train()  # 设置训练模式
    step = 0
    for sentences, labels in build_batch(
            word2id_dict, corpus, batch_size, epochs, max_seq_len):

        sentences_var = paddle.to_tensor(sentences)
        labels_var = paddle.to_tensor(labels)
        pred, loss = model(sentences_var, labels_var)

        # 后向传播
        loss.backward()
        # 最小化loss
        opt.step()
        # 清除梯度
        opt.clear_grad()

        step += 1
        if step % 100 == 0:
            logger.info("step %d, loss %.3f" % (step, loss.numpy()[0]))


def evaluate(model, word2id_dict, corpus, epochs=10, batch_size=32, max_seq_len=128):
    """模型评估"""
    model.eval()
    tp = 0.
    tn = 0.
    fp = 0.
    fn = 0.
    for sentences, labels in build_batch(
            word2id_dict, corpus, batch_size, epochs, max_seq_len):

        sentences_var = paddle.to_tensor(sentences)
        labels_var = paddle.to_tensor(labels)

        # 获取模型对当前batch的输出结果
        pred, loss = model(sentences_var, labels_var)

        # 把输出结果转换为numpy array的数据结构
        # 遍历这个数据结构，比较预测结果和对应label之间的关系，并更新tp，tn，fp和fn
        pred = pred.numpy()
        for i in range(len(pred)):
            if labels[i][0] == 1:
                if pred[i][1] > pred[i][0]:
                    tp += 1
                else:
                    fn += 1
            else:
                if pred[i][1] > pred[i][0]:
                    fp += 1
                else:
                    tn += 1

    # 输出最终评估的模型效果
    logger.info("the acc in the test set is %.3f" % ((tp + tn) / (tp + tn + fp + fn)))


def main():
    """主函数"""
    logger.info("下载数据")
    data_dir = os.path.join(work_root, "data")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    data_file = os.path.join(data_dir, "aclImdb_v1.tar.gz")
    download(data_file)

    logger.info("加载数据集")
    train_corpus = load_imdb(data_file, True)
    test_corpus = load_imdb(data_file, False)
    for i in range(5):
        logger.info("sentence %d, %s" % (i, train_corpus[i][0]))
        logger.info("sentence %d, label %d" % (i, train_corpus[i][1]))

    logger.info("数据预处理")
    train_corpus = data_preprocess(train_corpus)
    test_corpus = data_preprocess(test_corpus)

    logger.info("构建词表")
    word2id_freq, word2id_dict, id2word_dict = build_dict(train_corpus)  # 输入仅仅包含句子内容就好
    assert len(word2id_freq) == len(word2id_dict) == len(id2word_dict)
    vocab_size = len(word2id_freq)
    logger.info("there are totoally %d different words in the corpus" % vocab_size)
    for _, (word, word_id) in zip(range(10), word2id_dict.items()):
        logger.info("word %s, its id %d, its word freq %d" % (word, word_id, word2id_freq[word_id]))
    save_dict_obj(word2id_dict, "sentiment_word2id.json")
    save_dict_obj(id2word_dict, "sentiment_id2word.json")

    logger.info("Words 2 ids")
    train_corpus = convert_corpus_to_id(train_corpus, word2id_dict)
    test_corpus = convert_corpus_to_id(test_corpus, word2id_dict)
    logger.info("%d tokens in the corpus" % len(train_corpus))
    logger.info(train_corpus[:5])
    logger.info(test_corpus[:5])

    logger.info("开始训练")
    paddle.set_device('gpu:1')
    batch_size = 128
    epoch_num = 5
    embedding_size = 256
    learning_rate = 0.01
    max_seq_len = 128
    sentiment_classifier = SentimentClassifier(
        embedding_size, vocab_size, num_steps=max_seq_len, num_layers=1)
    # 创建优化器Optimizer，用于更新这个网络的参数
    optimizer = paddle.optimizer.Adam(learning_rate=learning_rate, beta1=0.9, beta2=0.999,
                                      parameters=sentiment_classifier.parameters())
    train(sentiment_classifier, optimizer, word2id_dict, train_corpus,
          epochs=epoch_num, batch_size=batch_size, max_seq_len=max_seq_len)
    evaluate(sentiment_classifier, word2id_dict, test_corpus,
             epochs=1, batch_size=batch_size, max_seq_len=max_seq_len)

    logger.info("保存模型")
    model_path = os.path.join(work_root, "models")
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    paddle.save(sentiment_classifier.state_dict(),
                os.path.join(model_path, "{}.pdparams".format("sentiment_classifier")))


def inference(model_file, word2id_dict, text_list: list):
    """前向推理"""
    vocab_size = len(word2id_dict)
    embedding_size = 256
    max_seq_len = 128
    sentiment_classifier = SentimentClassifier(
        embedding_size, vocab_size, num_steps=max_seq_len, num_layers=1)
    params_dict = paddle.load(model_file)
    # sentiment_classifier.set_state_dict(params_dict)
    sentiment_classifier.load_dict(params_dict)
    sentiment_classifier.eval()

    inputs = []
    for char in text_list:
        if char:
            inputs.append(word2id_dict[char] if char in word2id_dict else word2id_dict['[oov]'])
    if len(inputs) > max_seq_len:
        inputs = inputs[:max_seq_len]
    if len(inputs) < max_seq_len:
        for _ in range(max_seq_len - len(inputs)):
            inputs.append(word2id_dict['[pad]'])  # 数据补齐
    inputs = paddle.to_tensor([inputs])

    pred = sentiment_classifier(inputs)
    labels = paddle.argmax(pred, axis=-1)

    return labels.numpy()


if __name__ == '__main__':
    init_logs()
    is_train = False
    if is_train:
        main()
    else:
        text = " Zentropa is the most original movie I've seen in years. " \
               "If you like unique thrillers that are influenced by film noir, " \
               "then this is just the right cure for all of those Hollywood summer blockbusters clogging the theaters these days. " \
               "Von Trier's follow-ups like Breaking the Waves have gotten more acclaim, but this is really his best work." \
               " It is flashy without being distracting and offers the perfect combination of suspense and dark humor. " \
               "It's too bad he decided handheld cameras were the wave of the future. " \
               "It's hard to say who talked him away from the style he exhibits here, " \
               "but it's everyone's loss that he went into his heavily theoretical dogma direction instead."
        model_file = model_path = os.path.join(work_root, "models/{}.pdparams".format("sentiment_classifier"))
        word2id_dict = load_json_obj("sentiment_word2id.json")
        logging.info("result: {}".format(inference(model_file, word2id_dict, text_list=text.split(" "))))
