"""
# 使用GRU(LSTM的变种)网络进行个性化推荐召回模型实现
"""
import os
import logging
import numpy as np
import paddle
import paddle.nn as nn

from tqdm import tqdm

logger = logging.getLogger(__name__)
EOS = "<eos>"
PAD = "<pad>"


class PtbModel(nn.Layer):
    """召回网络"""
    def __init__(self, hidden_size, vocab_size, num_layers=1, init_scale=0.1, dropout=None):
        super(PtbModel, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=hidden_size, sparse=False,
                                      weight_attr=paddle.ParamAttr(
                                          name="embedding_para",
                                          initializer=paddle.nn.initializer.Uniform(low=-init_scale, high=init_scale)))
        self.rnn = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers)
        self.classifier = nn.Linear(in_features=hidden_size, out_features=vocab_size)

        self.dropout_layer = nn.Dropout(p=dropout, mode='upscale_in_train') \
                                if dropout is not None and dropout > 0.0 else None

    def forward(self, inputs, labels=None):
        """
            inputs: [btc, time_steps]
            labels: [btc, time_steps]
        """
        batch_size, seq_length = inputs.shape[0], inputs.shape[1]
        init_hidden_data = np.zeros(
            (self.num_layers, batch_size, self.hidden_size), dtype='float32')
        init_hidden = paddle.to_tensor(
            data=init_hidden_data, dtype=None, place=None, stop_gradient=True)

        x_emb = self.embedding(inputs)
        # 添加dropout 随机丢失
        if self.dropout_layer is not None:
            x_emb = self.dropout_layer(x_emb)

        rnn_out, last_hidden = self.rnn(x_emb, init_hidden)

        output = self.classifier(rnn_out)

        if labels is not None:
            logits = paddle.reshape(output, shape=[-1, self.vocab_size])
            labels = paddle.reshape(labels, shape=[-1, 1])
            loss = nn.functional.softmax_with_cross_entropy(logits=logits, label=labels,
                                                            soft_label=False, ignore_index=1)
            # 计算recall@20 指标
            acc = paddle.metric.accuracy(input=logits, label=labels, k=20)
            loss = paddle.reshape(loss, shape=[-1, seq_length])
            # 综合所有batch和序列长度的loss
            loss = paddle.mean(loss, axis=[0])
            loss = paddle.sum(loss)
            # loss = paddle.mean(loss)

            return loss, acc

        return output


def build_vocab(data_file):
    """构建词汇表"""
    vocab_dict = {}
    ids = 0
    vocab_dict[EOS] = ids
    ids += 1
    vocab_dict[PAD] = ids
    ids += 1
    with open(data_file, mode="r") as fp:
        for line in tqdm(fp.readlines()):
            for word in line.strip().split():
                if word not in vocab_dict:
                    vocab_dict[word] = ids
                    ids += 1

    assert len(vocab_dict) == ids
    logger.info("vocab word nums: {}".format(len(vocab_dict)))

    return vocab_dict


def file_to_ids(data_file, vocabs, max_length):
    """根据词表将文本转成ID序列，每一个短句结束添加0作为标识符"""
    src_data = []
    with open(data_file, mode="r") as fp:
        for line in fp.readlines():
            arr = line.strip().split()
            ids = [vocabs[word] for word in arr if word in vocabs]
            if len(ids) > max_length - 1:
                ids = ids[:max_length - 1]  # 截断
            ids += [vocabs[EOS]]
            extra = max_length - len(ids)
            if extra > 0:
                # ids += [vocabs[EOS]] * extra
                ids += [vocabs[PAD]] * extra
            src_data.append(ids)

    return src_data


def get_ptb_data(data_path, max_length):
    """获取数据"""
    # 训练输入文件
    train_file = os.path.join(data_path, "ptb.train.txt")
    # 可以不加valid文件
    valid_file = os.path.join(data_path, "ptb.valid.txt")
    # 测试输入文件
    test_file = os.path.join(data_path, "ptb.test.txt")
    # 构建词典
    vocab_dict = build_vocab(train_file)
    train_ids = file_to_ids(train_file, vocab_dict, max_length)
    valid_ids = file_to_ids(valid_file, vocab_dict, max_length)
    test_ids = file_to_ids(test_file, vocab_dict, max_length)

    return train_ids, valid_ids, test_ids, vocab_dict


def get_data_iter(raw_data, batch_size):
    """生产实际数据"""
    raw_data = np.asarray(raw_data, dtype="int64")
    x, y = [], []

    for single_data in raw_data:
        x.append(single_data[:-1])
        y.append(single_data[1:])  # 根据历史预测未来
        if len(x) == batch_size:
            assert len(x) == len(y)
            yield (x, y)
            x, y = [], []
    yield (x, y)  # 返回最后一组数据


def train(model, opt, train_data, batch_size=32, epochs=5):
    """模型训练"""
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        iter = 0
        train_data_iter = get_data_iter(train_data, batch_size)  # 每个epoch初始化一次啊
        for idx, batch in enumerate(tqdm(train_data_iter)):
            x_data, y_data = batch
            x = paddle.to_tensor(
                data=x_data, dtype=None, place=None, stop_gradient=True)
            y = paddle.to_tensor(
                data=y_data, dtype=None, place=None, stop_gradient=True)
            max_length = x.shape[1]

            # 执行前向训练逻辑，
            # loss值-- dy_loss
            # 当前时刻的隐变量输出-- last_hidden
            # recall@20 计算指标-- acc
            #
            dy_loss, acc = model(x, y)

            out_loss = dy_loss.numpy()
            acc_ = acc.numpy()[0]

            # 执行反向梯度
            dy_loss.backward()
            # 最小化loss
            opt.step()
            # 清除梯度
            opt.clear_grad()

            total_loss += out_loss
            iter += max_length

            # 每隔一段时间可以打印信息，包括ppl ，acc 和学习率Lr
            if idx > 0 and idx % 1000 == 1:
                ppl = total_loss / float(iter)
                logger.info(
                    "-- Epoch:[%d]; Batch:[%d]; ppl: %.5f, acc: %.5f"
                    % (epoch, idx, ppl, acc_))


def evaluate(model, eval_data, batch_size=32):
    """模型评估"""
    model.eval()
    eval_data_iter = get_data_iter(eval_data, batch_size)
    accum_num_recall = 0.0
    for idx, batch in enumerate(eval_data_iter):
        x_data, y_data = batch
        x = paddle.to_tensor(
            data=x_data, dtype=None, place=None, stop_gradient=True)
        y = paddle.to_tensor(
            data=y_data, dtype=None, place=None, stop_gradient=True)

        # 执行前向训练逻辑，
        # loss值-- dy_loss
        # 当前时刻的隐变量输出-- last_hidden
        # recall@20 计算指标-- acc
        #
        dy_loss, acc = model(x, y)

        acc_ = acc.numpy()[0]
        accum_num_recall += acc_

    logger.info("recall@20:{}".format(accum_num_recall / (idx + 1)))
    return accum_num_recall/(idx + 1)


if __name__ == '__main__':
    logging.basicConfig(format="[%(asctime)s - %(filename)s - %(lineno)s]: %(levelname)s - %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",
                        level=logging.INFO,
                        filemode="a",
                        filename=None)
    logger.info("初始化日志配置.")

    logger.info("加载数据")
    data_path = "/home/yckj2453/workspace/baidu-dl/natural_language_processing/data/data"
    max_length = 10  # 序列最大长度
    train_ids, valid_ids, test_ids, vocabs = get_ptb_data(data_path, max_length)

    vocab_size = len(vocabs)  # 词典大小
    num_layers = 1  # gru层数
    batch_size = 500  #
    hidden_size = 10  # 隐变量
    init_scale = 0.1  # 初始化
    max_grad_norm = 5.0  # grad超参数
    epoch_start_decay = 3  # decay开始的epoch
    max_epoch = 5  # 训练的epoch数
    dropout = 0.0  # dropout
    lr_decay = 0.5  # 学习率decay
    base_learning_rate = 0.05  # 基础学习率

    paddle.set_device("gpu:1")
    model = PtbModel(hidden_size, vocab_size,
                     num_layers=num_layers, init_scale=init_scale, dropout=dropout)
    # 打印定义好的参数
    logger.info("parameters:--------------------------------")
    for para in model.parameters():
        logger.info(para.name)
    logger.info("parameters:--------------------------------")
    # grad_clip = nn.ClipGradByGlobalNorm(max_grad_norm)
    # 定义优化器，需要将参数列表传入
    # sgd = paddle.optimizer.Adagrad(
    #     parameters=model.parameters(),
    #     learning_rate=base_learning_rate,
    #     grad_clip=grad_clip)
    optimizer = paddle.optimizer.Adam(learning_rate=base_learning_rate, beta1=0.9, beta2=0.999,
                                      parameters=model.parameters())

    logger.info("训练模型")
    train(model, optimizer, train_ids, batch_size=batch_size, epochs=max_epoch)
    evaluate(model, test_ids)
