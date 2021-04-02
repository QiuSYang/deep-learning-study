"""
使用飞桨实现Skip-gram --- A kind of the word embedding model
接下来我们将学习使用飞桨实现Skip-gram模型的方法。在飞桨中，不同深度学习模型的训练过程基本一致，流程如下：
    1. 数据处理：选择需要使用的数据，并做好必要的预处理工作。
    2. 网络定义：使用飞桨定义好网络结构，包括输入层，中间层，输出层，损失函数和优化算法。
    3. 网络训练：将准备好的数据送入神经网络进行学习，并观察学习的过程是否正常，如损失函数值是否在降低，也可以打印一些中间步骤的结果出来等。
    4. 网络评估：使用测试集合测试训练好的神经网络，看看训练效果如何。
"""
import io
import os
import json
import logging
import sys
import requests
from collections import OrderedDict
from tqdm import tqdm
import math
import random
import numpy as np
import paddle
from paddle.nn import Embedding
import paddle.nn.functional as F

logger = logging.getLogger(__name__)
work_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def download():
    """数据下载"""
    data_root = os.path.join(work_root, "data")
    if not os.path.exists(data_root):
        os.makedirs(data_root)
    data_file = os.path.join(data_root, "text8.txt")

    if not os.path.exists(data_file):
        # 可以从百度云服务器下载一些开源数据集（dataset.bj.bcebos.com）
        corpus_url = "https://dataset.bj.bcebos.com/word2vec/text8.txt"
        # 使用python的requests包下载数据集到本地
        web_request = requests.get(corpus_url)
        corpus = web_request.content
        # 把下载后的文件存储在当前目录的text8.txt文件内
        with open(data_file, "wb") as f:
            f.write(corpus)
        f.close()

    return data_file


def load_data(data_file):
    """获取数据"""
    with open(data_file, "r") as f:
        corpus = f.read().strip("\n")
    f.close()

    return corpus


def data_preprocess(corpus):
    """对语料进行预处理（分词）"""
    # 由于英文单词出现在句首的时候经常要大写，所以我们把所有英文字符都转换为小写，
    # 以便对语料进行归一化处理（Apple vs apple等）
    corpus = corpus.strip().lower()
    corpus = corpus.split(" ")
    return corpus


def build_vocab(corpus):
    """构造词典，统计每个词的频率，并根据频率将每个词转换为一个整数id"""
    # 首先统计每个不同词的频率（出现的次数），使用一个词典记录
    word_freq_dict = dict()
    for word in corpus:
        if word not in word_freq_dict:
            word_freq_dict[word] = 0
        word_freq_dict[word] += 1

    # 将这个词典中的词，按照出现次数排序，出现次数越高，排序越靠前
    # 一般来说，出现频率高的高频词往往是：I，the，you这种代词，而出现频率低的词，往往是一些名词，如：nlp

    word_freq_dict = sorted(word_freq_dict.items(), key=lambda x:x[1], reverse=True)

    # 构造3个不同的词典，分别存储，
    # 每个词到id的映射关系：word2id_dict
    # 每个id出现的频率：word2id_freq
    # 每个id到词的映射关系：id2word_dict
    word2id_dict = dict()
    word2id_freq = dict()
    id2word_dict = dict()

    # 按照频率，从高到低，开始遍历每个单词，并为这个单词构造一个独一无二的id
    for word, freq in word_freq_dict:
        current_id = len(word2id_dict)
        word2id_dict[word] = current_id
        assert word2id_dict[word] == current_id
        word2id_freq[current_id] = freq
        id2word_dict[current_id] = word

    return word2id_freq, word2id_dict, id2word_dict


def convert_corpus_to_id(corpus, word2id_dict):
    """把语料转换为id序列"""
    # 使用一个循环，将语料中的每个词替换成对应的id，以便于神经网络进行处理
    corpus = [word2id_dict[word] for word in corpus]
    return corpus


def subsampling(corpus, word2id_freq):
    """二次采样法的主要思想是降低高频词在语料中出现的频次。
    方法是随机将高频的词抛弃，频率越高，被抛弃的概率就越大；频率越低，被抛弃的概率就越小。
    标点符号或冠词这样的高频词就会被抛弃，从而优化整个词表的词向量训练效果"""
    # 使用二次采样算法（subsampling）处理语料，强化训练效果
    # 这个discard函数决定了一个词会不会被替换，这个函数是具有随机性的，每次调用结果不同
    # 如果一个词的频率很大，那么它被遗弃的概率就很大
    def discard(word_id):
        return random.uniform(0, 1) < 1 - math.sqrt(
            1e-4 / word2id_freq[word_id] * len(corpus))

    corpus = [word for word in corpus if not discard(word)]
    return corpus


def build_data(corpus, vocab_size=5000, max_window_size=3, negative_sample_num=4):
    """在完成语料数据预处理之后，需要构造训练数据。根据上面的描述，
       我们需要使用一个滑动窗口对语料从左到右扫描，在每个窗口内，中心词需要预测它的上下文，并形成训练数据。
       在实际操作中，由于词表往往很大（50000，100000等），对大词表的一些矩阵运算（如softmax）需要消耗巨大的资源，
       因此可以通过负采样的方式模拟softmax的结果。
            1. 给定一个中心词和一个需要预测的上下文词，把这个上下文词作为正样本。
            2. 通过词表随机采样的方式，选择若干个负样本。
            3. 把一个大规模分类问题转化为一个2分类问题，通过这种方式优化计算速度。
    """
    # 构造数据，准备模型训练
    # max_window_size代表了最大的window_size的大小，程序会根据max_window_size从左到右扫描整个语料
    # negative_sample_num代表了对于每个正样本，我们需要随机采样多少负样本用于训练，
    # 一般来说，negative_sample_num的值越大，训练效果越稳定，但是训练速度越慢。
    dataset = []
    # 从左到右，开始枚举每个中心点的位置
    for center_word_idx in range(len(corpus)):
        # 以max_window_size为上限，随机采样一个window_size，这样会使得训练更加稳定
        window_size = random.randint(1, max_window_size)
        # 当前的中心词就是center_word_idx所指向的词
        center_word = corpus[center_word_idx]

        # 以当前中心词为中心，左右两侧在window_size内的词都可以看成是正样本
        positive_word_range = (max(0, center_word_idx - window_size),
                               min(len(corpus) - 1, center_word_idx + window_size))  # 边界处理
        positive_word_candidates = [corpus[idx] for idx in range(positive_word_range[0], positive_word_range[1] + 1)
                                    if idx != center_word_idx]
        # 对于每个正样本来说，随机采样negative_sample_num个负样本，用于训练
        for positive_word in positive_word_candidates:
            # 首先把（中心词，正样本，label=1）的三元组数据放入dataset中，
            # 这里label=1表示这个样本是个正样本
            dataset.append((center_word, positive_word, 1))

            # 开始负采样
            i = 0
            while i < negative_sample_num:
                negative_word_candidate = random.randint(0, vocab_size - 1)

                if negative_word_candidate not in positive_word_candidates:
                    # 把（中心词，正样本，label=0）的三元组数据放入dataset中，
                    # 这里label=0表示这个样本是个负样本
                    dataset.append((center_word, negative_word_candidate, 0))
                    i += 1
    return dataset


def build_batch(dataset, batch_size, epoch_num):
    """构造mini-batch，准备对模型进行训练
       我们将不同类型的数据放到不同的tensor里，便于神经网络进行处理
       并通过numpy的array函数，构造出不同的tensor来，并把这些tensor送入神经网络中进行训练"""
    # center_word_batch缓存batch_size个中心词
    center_word_batch = []
    # target_word_batch缓存batch_size个目标词（可以是正样本或者负样本）
    target_word_batch = []
    # label_batch缓存了batch_size个0或1的标签，用于模型训练
    label_batch = []

    for epoch in range(epoch_num):
        # 每次开启一个新epoch之前，都对数据进行一次随机打乱，提高训练效果
        random.shuffle(dataset)

        for center_word, target_word, label in dataset:
            # 遍历dataset中的每个样本，并将这些数据送到不同的tensor里
            center_word_batch.append([center_word])
            target_word_batch.append([target_word])
            label_batch.append(label)

            # 当样本积攒到一个batch_size后，我们把数据都返回回来
            # 在这里我们使用numpy的array函数把list封装成tensor
            # 并使用python的迭代器机制，将数据yield出来
            # 使用迭代器的好处是可以节省内存
            if len(center_word_batch) == batch_size:
                yield np.array(center_word_batch).astype("int64"), \
                      np.array(target_word_batch).astype("int64"), \
                      np.array(label_batch).astype("float32")
                center_word_batch = []
                target_word_batch = []
                label_batch = []

    if len(center_word_batch) > 0:
        yield np.array(center_word_batch).astype("int64"), \
              np.array(target_word_batch).astype("int64"), \
              np.array(label_batch).astype("float32")


class SkipGram(paddle.nn.Layer):
    """使用paddle实现skip-gram模型"""
    def __init__(self, vocab_size, embedding_size, init_scale=0.1):
        # vocab_size定义了这个skipgram这个模型的词表大小
        # embedding_size定义了词向量的维度是多少
        # init_scale定义了词向量初始化的范围，一般来说，比较小的初始化范围有助于模型训练
        super(SkipGram, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size

        # 使用Embedding函数构造一个词向量参数
        # 这个参数的大小为：[self.vocab_size, self.embedding_size]
        # 数据类型为：float32
        # 这个参数的名称为：embedding_para
        # 这个参数的初始化方式为在[-init_scale, init_scale]区间进行均匀采样
        self.embedding = Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.embedding_size,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Uniform(low=-0.5/embedding_size,
                                                          high=0.5/embedding_size)))
        # 使用Embedding函数构造另外一个词向量参数
        # 这个参数的大小为：[self.vocab_size, self.embedding_size]
        # 这个参数的初始化方式为在[-init_scale, init_scale]区间进行均匀采样
        self.embedding_out = Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.embedding_size,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Uniform(low=-0.5 / embedding_size,
                                                          high=0.5 / embedding_size)))

    def forward(self, center_words, target_words, label=None):
        """# 定义网络的前向计算逻辑
        # center_words是一个tensor（mini-batch），表示中心词
        # target_words是一个tensor（mini-batch），表示目标词
        # label是一个tensor（mini-batch），表示这个词是正样本还是负样本（用0或1表示）
        # 用于在训练中计算这个tensor中对应词的同义词，用于观察模型的训练效果"""
        # 首先，通过embedding_para（self.embedding）参数，将mini-batch中的词转换为词向量
        # 这里center_words和eval_words_emb查询的是一个相同的参数
        # 而target_words_emb查询的是另一个参数
        center_words_emb = self.embedding(center_words)
        target_words_emb = self.embedding_out(target_words)

        # 我们通过点乘的方式计算中心词到目标词的输出概率，并通过sigmoid函数估计这个词是正样本还是负样本的概率。
        word_sim = paddle.multiply(center_words_emb, target_words_emb)  # 向量点乘, 对应元素相乘
        word_sim = paddle.sum(word_sim, axis=-1)
        word_sim = paddle.reshape(word_sim, shape=[-1])

        if label is not None:
            # 通过估计的输出概率定义损失函数，注意我们使用的是binary_cross_entropy_with_logits函数
            # 将sigmoid计算和cross entropy合并成一步计算可以更好的优化，所以输入的是word_sim，而不是pred
            loss = F.binary_cross_entropy_with_logits(word_sim, label)
            loss = paddle.mean(loss)
            return loss
        else:
            return F.sigmoid(word_sim)


def train(model, opt, dataset, word2id_dict, id2word_dict, batch_size=512, epoch_num=3):
    """训练模型"""
    model.train()

    def get_similar_tokens(query_token, k, embed):
        """ #定义一个使用word-embedding查询同义词的函数
            #这个函数query_token是要查询的词，k表示要返回多少个最相似的词，embed是我们学习到的word-embedding参数
            #我们通过计算不同词之间的cosine距离，来衡量词和词的相似度
            #具体实现如下，x代表要查询词的Embedding，Embedding参数矩阵W代表所有词的Embedding
            #两者计算Cos得出所有词对查询词的相似度得分向量，排序取top_k放入indices列表"""
        W = embed.numpy()
        x = W[word2id_dict[query_token]]
        cos = np.dot(W, x) / np.sqrt(np.sum(W * W, axis=-1) * np.sum(x * x) + 1e-9)
        flat = cos.flatten()
        indices = np.argpartition(flat, -k)[-k:]
        indices = indices[np.argsort(-flat[indices])]
        for i in indices:
            logger.info("for word {}, the similar word is {}".format(query_token, str(id2word_dict[i])))

    # 使用build_batch函数，以mini-batch为单位，遍历训练数据，并训练网络
    for step, (center_words, target_words, label) in enumerate(tqdm(build_batch(dataset,
                                                                                batch_size=batch_size,
                                                                                epoch_num=epoch_num))):
        # 使用paddle.to_tensor，将一个numpy的tensor，转换为飞桨可计算的tensor
        center_words_var = paddle.to_tensor(center_words)
        target_words_var = paddle.to_tensor(target_words)
        label_var = paddle.to_tensor(label)

        loss = model(center_words_var, target_words_var, label_var)

        # 反向传播
        loss.backword()
        # 程序根据loss，完成一步对参数的优化更新
        opt.step()
        # 清空模型中的梯度，以便于下一个mini-batch进行更新
        opt.clear_grad()

        # 每经过100个mini-batch，打印一次当前的loss，看看loss是否在稳定下降
        if step % 1000 == 0:
            logger.info("step {}, loss {}".format(step, loss.numpy()[0]))

        # 每隔10000步，打印一次模型对以下查询词的相似词，
        # 这里我们使用词和词之间的向量点积作为衡量相似度的方法，只打印了5个最相似的词
        if step % 10000 == 0:
            get_similar_tokens('movie', 5, model.embedding.weight)
            get_similar_tokens('one', 5, model.embedding.weight)
            get_similar_tokens('chip', 5, model.embedding.weight)


def evaluate(model, valid_dataset):
    """评估模型"""
    model.eval()
    correct_num, total_num = 0, len(valid_dataset)
    for step, (center_words, target_words, label) in enumerate(tqdm(valid_dataset)):
        center_words_var = paddle.to_tensor([center_words])
        target_words_var = paddle.to_tensor([target_words])

        pred = model(center_words_var, target_words_var)
        pred_label = 1 if pred.numpy()[0] > 0.5 else 0
        if pred_label == label:
            correct_num += 1

    return float(correct_num) / float(total_num)


def main():
    """主函数"""
    logger.info("1. Load data")
    data_file = download()
    logger.info("2. Data preprocess")
    corpus = load_data(data_file)
    corpus = data_preprocess(corpus)
    logger.info("3. Build vocabulary")
    word2id_freq, word2id_dict, id2word_dict = build_vocab(corpus)
    vocab_size = len(word2id_dict)
    logger.info("there are totoally {} different words in the corpus".format(vocab_size))

    logger.info("4. words to ids")
    corpus = convert_corpus_to_id(corpus, word2id_dict)
    logger.info("{} tokens in the corpus".format(len(corpus)))
    corpus = subsampling(corpus, word2id_freq)
    logger.info("{} tokens in the corpus".format(len(corpus)))

    logger.info("5. build dataset")
    dataset = build_data(corpus, vocab_size)
    logger.info("6. build batch dataset")
    batch_size = 512
    epoch_num = 3
    for i in range(3):
        random.shuffle(dataset)
    data_num = len(dataset)
    train_dataset = dataset[:int(data_num*0.9)]
    valid_dataset = dataset[int(data_num*0.9):]

    logger.info("7. build model")
    model = SkipGram(vocab_size=vocab_size, embedding_size=200)
    # 构造训练这个网络的优化器
    opt = paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters())

    logger.info("8. train model")
    train(model, opt, dataset=train_dataset,
          word2id_dict=word2id_dict, id2word_dict=id2word_dict,
          batch_size=batch_size, epoch_num=epoch_num)

    logger.info("9. evaluate model")
    accuracy = evaluate(model, valid_dataset=valid_dataset)
    logger.info("accuracy value: {}".format(accuracy))

    logger.info("10. save model")
    model_path = os.path.join(work_root, "models")
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    paddle.save(model.state_dict(), os.path.join(model_path, "{}.pdparams".format("lasted")))

    return word2id_dict, id2word_dict


def inference(model_file, word2id_dict, id2word_dict):
    """前向推理"""
    vocad_size = len(word2id_dict)
    model = SkipGram(vocad_size=vocad_size, embedding_size=200)
    params_dict = paddle.load(model_file)
    model.state_dict(params_dict)
    model.eval()

    word = input("input word: ")
    word_id = word2id_dict[word]
    word_ = input("input word_: ")
    word_id_ = word2id_dict[word_]

    center_words_var = paddle.to_tensor([word_id])
    target_words_var = paddle.to_tensor([word_id_])

    pred = model(center_words_var, target_words_var)
    pred_label = 1 if pred.numpy()[0] > 0.5 else 0

    return pred_label


def save_json_data(data_file, data_obj):
    with open(data_file, mode="w", encoding="utf-8") as fw:
        json.dump(data_obj, fw, ensure_ascii=False, indent=2)


def load_json_data(data_file):
    with open(data_file, mode="r", encoding="utf-8") as fp:
        return json.load(fp)


if __name__ == '__main__':
    logging.basicConfig(format="[%(asctime)s %(filename)s: %(lineno)s] %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",
                        level=logging.INFO,
                        filename=None,
                        filemode="a")
    logger.info("set gpu id")
    paddle.set_device("gpu:2")
    is_train = True
    if is_train:
        word2id_dict, id2word_dict = main()
        word2id_dict_file = os.path.join(work_root, "data/word2id.json")
        id2word_dict_file = os.path.join(work_root, "data/id2word.json")
        save_json_data(word2id_dict_file, word2id_dict)
        save_json_data(id2word_dict_file, id2word_dict)
    else:
        pass
