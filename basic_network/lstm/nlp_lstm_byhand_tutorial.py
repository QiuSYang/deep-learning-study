"""
# 使用pytorch 实现LSTM上手教程
# 预测一句话的意图为 positive or negative
"""
import os
import logging
import torch
import torch.nn as nn

from custom_lstm import CustomLSTM

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(filename)s - %(levelname)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S')
_logger = logging.getLogger(__name__)


class Net(nn.Module):
    def __init__(self, vocab_size):
        super(Net, self).__init__()
        self.embedding = nn.Embedding(vocab_size, 32)
        self.lstm = CustomLSTM(32, 32)  # nn.LSTM(32, 32, batch_first=True)
        self.fc1 = nn.Linear(32, 2)

    def forward(self, x):
        x_ = self.embedding(x)
        x_, (h_n, c_n) = self.lstm(x_)
        # 获取最后一个隐层
        x_ = x_[:, -1, :]
        x_ = self.fc1(x_)

        return x_

    def save_model(self, model_path):
        """保存模型"""
        if os.path.exists(model_path):
            # 模型文件存在先删除已存在模型之后再保存模型
            os.remove(model_path)
        torch.save(self.state_dict(), model_path)

    def load_model(self, model_path):
        """加载模型"""
        try:
            model_para = torch.load(model_path)
            self.load_state_dict(model_para)
        except Exception:
            _logger.info("Loading model weight fail.")


if __name__ == "__main__":
    _logger.info("Load data.")
    import pandas as pd
    import numpy as np

    df = pd.read_csv('Reviews.csv')

    # drop useless data
    df = df.drop(['Id', 'ProductId', 'UserId', 'ProfileName', 'HelpfulnessNumerator',
                  'HelpfulnessDenominator', 'Time', 'Summary', ], axis=1)

    # remove ambiguous 3 and 4 stars for balancing
    df = df[df['Score'] != 3]  # 删除等于3的行

    # create labels and preprocess
    df['Score'] = df['Score'].apply(lambda i: 'positive' if i > 4 else 'negative')
    df['Text'] = df['Text'].apply(lambda x: x.lower())

    # set names for beautiful df
    df.columns = ['labels', 'text']

    idx_positive = df[df['labels'] == 'positive'].index
    nbr_to_drop = len(df) - len(idx_positive)

    # 保持数据平衡, 删除多余数据条目
    drop_indices = np.random.choice(idx_positive, nbr_to_drop, replace=False)
    df = df.drop(drop_indices)

    _logger.info((df['labels'] == 'negative').mean())

    text_as_list = df['text'].tolist()
    labels_as_list = df['labels'].tolist()

    from torchnlp.encoders.text import SpacyEncoder, pad_tensor
    from sklearn.model_selection import train_test_split
    from tqdm.notebook import tqdm

    # 使用pytorch NLP 工具对txt进行编码
    encoder = SpacyEncoder(text_as_list)
    _logger.info("{} encoder is {}".format(text_as_list[0], encoder.encode(text_as_list[0])))

    encoded_texts = []
    for i in tqdm(range(len(text_as_list))):
        encoded_texts.append(encoder.encode(text_as_list[i]))

    lengths = [len(i) for i in tqdm(encoded_texts)]

    import seaborn as sns
    import matplotlib.pyplot as plt
    length_as_series = pd.Series(lengths)
    plt.title("Probability Density Function for text lengths")
    sns.distplot(length_as_series)

    max_pad_length = length_as_series.quantile(0.9)

    reviews = []
    labels = []

    # 丢弃过长的数据
    for i in tqdm(range(len(encoded_texts))):
        if len(encoded_texts[i]) < max_pad_length:
            reviews.append(encoded_texts[i])
            labels.append(1 if labels_as_list[i] == "positive" else 0)

    assert len(reviews) == len(labels), "The labels and feature lists should have the same time"

    # 扩充单个sequence到最大序列长度
    padded_dataset = []
    for i in tqdm(range(len(reviews))):
        padded_dataset.append(pad_tensor(reviews[i], int(max_pad_length)))

    # preparing the final dataset:
    X = torch.stack(padded_dataset)
    y = torch.tensor(labels)

    (y == 1).float().mean()

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=.25,
                                                        random_state=42)

    X_train, y_train = torch.tensor(X_train), torch.tensor(y_train)
    X_test, y_test = torch.tensor(X_test), torch.tensor(y_test)

    _logger.info("Create torch data loader.")
    ds_train = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(ds_train, batch_size=128, shuffle=True)

    ds_test = torch.utils.data.TensorDataset(X_test, y_test)
    test_loader = torch.utils.data.DataLoader(ds_test, batch_size=256, shuffle=True)

    _logger.info("Create loss and optimizer.")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    classifier = Net(vocab_size=encoder.vocab_size).to(device)
    import torch.optim as optim
    optimizer = optim.Adam(classifier.parameters(), lr=0.005)  # 0.002 dives 85% acc
    criterion = nn.CrossEntropyLoss()

    epoch_bar = tqdm(range(10),
                     desc="Training",
                     position=0,
                     total=2)

    acc = 0

    _logger.info("Start training.")
    for epoch in epoch_bar:
        batch_bar = tqdm(enumerate(train_loader),
                         desc="Epoch: {}".format(str(epoch)),
                         position=1,
                         total=len(train_loader))

        for i, (datapoints, labels) in batch_bar:

            optimizer.zero_grad()

            preds = classifier(datapoints.long().to(device))
            loss = criterion(preds, labels.to(device))
            loss.backward()
            optimizer.step()
            # acc = (preds.argmax(dim=1) == labels).float().mean().cpu().item()

            if (i + 1) % 50 == 0:
                acc = 0

                with torch.no_grad():
                    for i, (datapoints_, labels_) in enumerate(test_loader):
                        preds = classifier(datapoints_.to(device))
                        acc += (preds.argmax(dim=1) == labels_.to(device)).float().sum().cpu().item()
                acc /= len(X_test)

            batch_bar.set_postfix(loss=loss.cpu().item(),
                                  accuracy="{:.2f}".format(acc),
                                  epoch=epoch)
            batch_bar.update()

        epoch_bar.set_postfix(loss=loss.cpu().item(),
                              accuracy="{:.2f}".format(acc),
                              epoch=epoch)
        epoch_bar.update()

    classifier.save_model(model_path="./lstm_txt_classifier.pkl")

    _logger.info("Training end.")
