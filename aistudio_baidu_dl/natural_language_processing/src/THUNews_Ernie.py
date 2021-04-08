"""
# 使用百度ERNIE模型进行news title classify
"""
import os
import sys
import logging
import random
import numpy as np
import paddle
import paddle.nn.functional as F
import paddlenlp
from paddlenlp.data import Stack, Pad, Tuple, Dict
from paddlenlp.datasets import load_dataset, MapDataset
from paddlenlp.transformers import LinearDecayWithWarmup
from functools import partial
from tqdm import tqdm
from flask import Flask, json, jsonify, request, abort  # server

logger = logging.getLogger(__name__)
work_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(work_root)  # 设置工作路径
from src.utils import init_logs, save_dict_obj, load_json_obj

app = Flask(__name__)


def set_seed(seed):
    """set random seed"""
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)


def create_classification_model(num_classes, training_steps):
    """创建模型, 损失函数, 以及优化器"""
    model = paddlenlp.transformers.ErnieForSequenceClassification.from_pretrained("ernie-tiny",
                                                                                  num_classes=num_classes)
    lr_scheduler = LinearDecayWithWarmup(learning_rate=5e-5,
                                         total_steps=training_steps,
                                         warmup=0.0)

    # Generate parameter names needed to perform weight decay.
    # All bias and LayerNorm parameters are excluded.
    decay_params = [
        p.name for n, p in model.named_parameters()
        if not any(nd in n for nd in ["bias", "norm"])
    ]
    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        parameters=model.parameters(),
        weight_decay=0.0,
        apply_decay_param_fun=lambda x: x in decay_params)

    criterion = paddle.nn.loss.CrossEntropyLoss()

    return model, optimizer, criterion, lr_scheduler


def load_data(data_file):
    """读取原始数据"""
    with open(data_file, mode="r", encoding="utf-8") as fp:
        datasets = []
        labels = {}
        for id, line in enumerate(tqdm(fp.readlines())):
            contents = line.strip().split("\t")
            if len(contents) == 3:
                # 0-label_id, 1-label_name, 2-context
                if contents[0] not in labels:
                    labels[contents[0]] = contents[1]
                datasets.append({
                    "text": contents[2],
                    "label": contents[0]
                })
            else:
                datasets.append({
                    "text": contents[0]
                })

        return datasets, labels


def convert_example(example, tokenizer, max_seq_length=256):
    """数据转换"""
    encoded_inputs = tokenizer(text=example["text"], max_seq_len=max_seq_length)
    input_ids = encoded_inputs["input_ids"]
    token_type_ids = encoded_inputs["token_type_ids"]
    label = np.array([example.get("label", 0)], dtype="int64")

    return input_ids, token_type_ids, label


def create_dataloader(dataset,
                      mode='train',
                      batch_size=1,
                      batchify_fn=None,
                      trans_fn=None):
    """创建data loader"""
    if trans_fn:
        dataset = dataset.map(trans_fn)

    shuffle = True if mode == 'train' else False
    batch_sampler = paddle.io.BatchSampler(
        dataset, batch_size=batch_size, shuffle=shuffle)

    return paddle.io.DataLoader(
        dataset=dataset,
        batch_sampler=batch_sampler,
        collate_fn=batchify_fn,
        return_list=True)


@paddle.no_grad()
def evaluate(model, metric, valid_loader):
    """模型评估"""
    model.eval()
    metric.reset()
    for id, batch in enumerate(tqdm(valid_loader)):
        input_ids, token_type_ids, labels = batch
        logits = model(input_ids, token_type_ids)
        correct = metric.compute(logits, labels)
        metric.update(correct)
    accu = metric.accumulate()
    model.train()
    metric.reset()

    return accu


def train(model, optimizer, criterion, lr_scheduler, metric, tokenizer,
          train_loader, valid_loader, epochs=3):
    """模型训练"""
    model.train()
    best_accuracy = 0.0
    for epoch in range(epochs):
        losses = []
        for id, batch in enumerate(tqdm(train_loader)):
            input_ids, token_type_ids, labels = batch
            logits = model(input_ids, token_type_ids)
            loss = criterion(logits, labels)
            losses.append(loss.numpy())
            probs = F.softmax(logits, axis=-1)

            # 计算分类准确率
            correct = metric.compute(probs, labels)
            metric.update(correct)

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.clear_grad()  # 梯度清理

        # 模型评估
        acc = metric.accumulate()
        logger.info("train loss: {}, accuracy: {}".format(np.mean(losses), acc))
        evaluate_acc = evaluate(model, metric, valid_loader)
        logger.info("eval accuarcy: {}".format(evaluate_acc))
        if evaluate_acc > best_accuracy:
            best_accuracy = evaluate_acc
            model_dir = os.path.join(work_root, "models/epoch_{}".format(epoch))
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            logger.info("保存第{} epoch 模型".format(epoch))
            model.save_pretrained(model_dir)
            tokenizer.save_pretrained(model_dir)


def main():
    """主函数"""
    paddle.set_device("gpu:1")
    set_seed(2021)  # 设置随机数种子
    logger.info("构建数据集")
    data_file = os.path.join(work_root, "data/NewsTrain.txt")
    datasets, labels = load_data(data_file)
    save_dict_obj(labels, os.path.join(work_root, "data/news_labels_info.json"))
    for i in range(3):
        random.shuffle(datasets)
    train_data_num = int(len(datasets) * 0.8)
    train_dataset, valid_dataset = datasets[:train_data_num], datasets[train_data_num:]
    train_dataset, valid_dataset = MapDataset(train_dataset), MapDataset(valid_dataset)
    logger.info("数据转换word2id")
    tokenizer = paddlenlp.transformers.ErnieTinyTokenizer.from_pretrained('ernie-tiny')
    trans_func = partial(
        convert_example,
        tokenizer=tokenizer,
        max_seq_length=64)
    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # segment
        Stack(dtype="int64")  # label
    ): fn(samples)  # 有点难以理解, 没有Torch便于理解, pad_val非常好, 可以动态设置batch最大序列长度

    train_loader = create_dataloader(train_dataset, mode="train", batch_size=512,
                                     batchify_fn=batchify_fn, trans_fn=trans_func)
    valid_loader = create_dataloader(valid_dataset, mode="valid", batch_size=256,
                                     batchify_fn=batchify_fn, trans_fn=trans_func)
    epochs = 5  # 训练epoch number
    num_training_steps = len(train_loader) * epochs
    num_classes = len(labels)
    model, optimizer, criterion, lr_scheduler = create_classification_model(num_classes, num_training_steps)

    logger.info("训练模型")
    metric = paddle.metric.Accuracy()
    train(model, optimizer, criterion, lr_scheduler, metric, tokenizer,
          train_loader, valid_loader, epochs=epochs)


@paddle.no_grad()
def inference(model, data_loader):
    """模型预测"""
    model.eval()
    total_labels = []
    for id, batch in enumerate(data_loader):
        input_ids, token_type_ids, labels = batch
        logits = model(input_ids, token_type_ids)

        probs = F.softmax(logits, axis=-1)
        batch_label = paddle.argmax(probs, axis=-1)
        total_labels.extend(batch_label.numpy().tolist())

    return total_labels


@app.route("/news_title_cls", methods=["POST"])
def news_title_classification():
    """文本分类---新闻标题分类"""
    if request.method == "POST":
        data = json.loads(request.data)
        if isinstance(data["texts"], str):
            data["texts"] = [data["texts"]]

        if isinstance(data["texts"], list):
            datasets = []
            for text in data["texts"]:
                datasets.append({
                    "text": text
                })
            datasets = MapDataset(datasets)
            trans_func = partial(
                convert_example,
                tokenizer=tokenizer,
                max_seq_length=64)
            batchify_fn = lambda samples, fn=Tuple(
                Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input
                Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # segment
                Stack(dtype="int64")  # label
            ): fn(samples)
            data_loader = create_dataloader(datasets, mode="test", batch_size=32,
                                            batchify_fn=batchify_fn, trans_fn=trans_func)

            labels = inference(model, data_loader)
            labels_text = []
            for id, label in enumerate(labels):
                labels_text.append(labels_info[str(label)])

            return jsonify(status="Success",
                           results=labels_text)
        else:
            return jsonify(status="Failure",
                           message="参数格式不对.")
    else:
        return jsonify(status="Failure",
                       message="请求方式不对, 仅支持POST请求.")


if __name__ == '__main__':
    init_logs()
    is_train = False
    if is_train:
        main()
    else:
        paddle.set_device("gpu:2")
        is_server = True
        model_dir = os.path.join(work_root, "models/epoch_{}".format(3))
        labels_info = load_json_obj(os.path.join(work_root, "data/news_labels_info.json"))
        model = paddlenlp.transformers.ErnieForSequenceClassification.from_pretrained(model_dir)
        tokenizer = paddlenlp.transformers.ErnieTinyTokenizer.from_pretrained(model_dir)
        if is_server:
            app.run(host="0.0.0.0", port=8280)  # 启动服务
        else:
            test_data_file = os.path.join(work_root, "data/NewsTest.txt")
            test_datasets, _ = load_data(test_data_file)
            test_datasets = MapDataset(test_datasets[:1000])
            trans_func = partial(
                convert_example,
                tokenizer=tokenizer,
                max_seq_length=64)
            batchify_fn = lambda samples, fn=Tuple(
                Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input
                Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # segment
                Stack(dtype="int64")  # label
            ): fn(samples)
            test_loader = create_dataloader(test_datasets, mode="test", batch_size=32,
                                            batchify_fn=batchify_fn, trans_fn=trans_func)

            labels = inference(model, test_loader)
            for id, label in enumerate(labels):
                logger.info("分类标签: {}".format(labels_info[str(label)]))
