# -*- coding: utf-8 -*-
"""
# 工具包
"""

from datetime import datetime
import json
import logging
import os
import tarfile
import tempfile
import socket
import PIL
import copy
import numpy as np
from itertools import chain
from collections import defaultdict

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from transformers import cached_path

logger = logging.getLogger(__name__)

NO_PARTICIPLE_KEY = ['img_list', 'sid']
SPECIAL_TOKENS = ["<bos>", "<eos>", "<speaker1>", "<speaker2>", "<pad>"]
ATTR_TO_SPECIAL_TOKEN = {'bos_token': '<bos>', 'eos_token': '<eos>', 'pad_token': '<pad>',
                         'additional_special_tokens': ['<speaker1>', '<speaker2>']}
PADDED_INPUTS = ["input_ids", "input_token_type_ids"]
PADDED_LABELS = ["lm_labels"]
MODEL_INPUTS = ["input_ids", "input_token_type_ids", "input_images_name", "input_images_id",
                "lm_labels", "mc_token_ids", "mc_labels"]
IMG_TOKEN = '<img>'

# temporarily use resent18 image statistics
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


def get_dataset(tokenizer, dataset_path, dataset_cache,
                single_sentence_token_max_num=80, seg_voc=False, inference=False):
    """ Get tokenized PERSONACHAT dataset from S3 or cache.
        tokenizer: 词汇object
        seg_voc: 是否进行分词
    """
    dataset_path = dataset_path
    dataset_cache = dataset_cache
    if dataset_cache and os.path.isfile(dataset_cache):
        logger.info("Load tokenized dataset from cache at %s", dataset_cache)
        dataset = torch.load(dataset_cache)
    else:
        logger.info("Download dataset from %s", dataset_path)
        data_file = cached_path(dataset_path)
        with open(data_file, "r", encoding="utf-8") as f:
            dataset = json.loads(f.read())

        logger.info("Tokenize and encode the dataset")

        def tokenize(obj):
            if isinstance(obj, str):
                if seg_voc:
                    # 已经被分好词的string, 按空格拆分, 并转为token
                    sentence_tokens = tokenizer.sent2id(obj.split())
                else:
                    sentence_tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
                if len(sentence_tokens) > single_sentence_token_max_num:
                    # 单个句子的最大token数量超出阈值, 从前置后截断
                    sentence_tokens = sentence_tokens[:single_sentence_token_max_num]
                return sentence_tokens
            if isinstance(obj, dict):
                return dict((n, tokenize(o)) if n not in NO_PARTICIPLE_KEY else (n, o)
                            for n, o in obj.items())
            if isinstance(obj, list):
                return list(tokenize(o) for o in obj)
            # 其他对象不做任何处理直接返回
            return obj
        dataset = tokenize(dataset)
        if not inference and dataset_cache:
            # 仅仅训练的时候存储数据, (并且保存路径需要存在)
            torch.save(dataset, dataset_cache)
    return dataset


def pad_dataset(dataset, padding=0):
    """ Pad the dataset. This could be optimized by defining a
    Dataset class and padding at the batch level, but this is simpler. """
    # Bart输入输出是分离的, 各自有各自的长度
    max_input_length = max(len(x) for x in dataset["input_ids"])
    max_label_length = max(len(x) for x in dataset["lm_labels"])
    for name in PADDED_INPUTS:
        # 输入数据padding
        dataset[name] = [x + [padding] * (max_input_length - len(x))
                         for x in dataset[name]]
    for name in PADDED_LABELS:
        # label数据padding(不能使用-100制作padding)
        # dataset[name] = [x + [padding if name != "lm_labels" else -100] * (max_label_length - len(x))
        #                  for x in dataset[name]]
        dataset[name] = [x + [padding] * (max_label_length - len(x))
                         for x in dataset[name]]

    return dataset


def add_special_tokens_(model, tokenizer):
    """ Add special tokens to the tokenizer and the model if they have not already been added. """
    orig_num_tokens = len(tokenizer.encoder)
    num_added_tokens = tokenizer.add_special_tokens(ATTR_TO_SPECIAL_TOKEN)  # doesn't add if they are already there
    if num_added_tokens > 0:
        model.resize_token_embeddings(new_num_tokens=orig_num_tokens + num_added_tokens)


def build_input_from_segments(config, tokenizer, utterance, reply, image_dir, persona=None, lm_labels=False):
    """ Build a sequence of input from 3 segments: persona, history and last reply. """
    if config.seg_voc:
        bos, eos, speaker1, speaker2 = tokenizer.sent2id(SPECIAL_TOKENS[:-1])
    else:
        bos, eos, speaker1, speaker2 = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[:-1])
    # 1. 获取输入数据
    history = utterance.get('history')
    if persona:
        # 历史turn的最后一句添加[eos]结束符
        sequence = [[bos] + persona] + history[:-1] + [history[-1]+[eos]]
    else:
        # 没有persona属性信息
        sequence = [[bos]] + history[:-1] + [history[-1]+[eos]]
    sequence = [sequence[0]] + \
               [[speaker2 if i % 2 else speaker1] + s for i, s in enumerate(sequence[1:])]
    # # 倒退历史轮标志(预防历史轮不是偶数, 第一轮不是Q)
    # sequence_temp = [sequence[0]] + \
    #             [[speaker2 if (len(sequence[1:])-1 - i) % 2 else speaker1] + s for i, s in enumerate(sequence[1:])]

    instance = {}
    instance["input_ids"] = list(chain(*sequence))
    # persona信息的type使用[bos] ID表示
    instance["input_token_type_ids"] = [bos] * len(sequence[0]) + \
                                       [speaker2 if i % 2 else speaker1 for i, s in enumerate(sequence[1:]) for _ in s]
    # input_token_type_ids_temp = [bos] * len(sequence[0]) + \
    #         [speaker2 if (len(sequence[1:])-1 - i) % 2 else speaker1 for i, s in enumerate(sequence[1:]) for _ in s]
    if len(instance["input_token_type_ids"]) > config.ctx_length:
        # 上下文数据过长, 从后向前截断, 去除persona(ctx_length=176)
        token_type_one = instance["input_token_type_ids"][:len(sequence[0])]
        token_type_two = instance["input_token_type_ids"][len(token_type_one):][-(config.ctx_length-len(token_type_one)):]
        diff_idx = 0  # 第一种token type 的长度
        for idx, token_ids in enumerate(token_type_two[1:]):
            if token_ids != token_type_two[idx]:
                diff_idx = idx + 1
                break
        if diff_idx <= 3:
            # 删除过短token type
            token_type_two = token_type_two[diff_idx:]
        input_ids_one = instance.get("input_ids")[:len(token_type_one)]
        # 截取和token type two一样的长度
        input_ids_two = instance.get("input_ids")[-len(token_type_two):]
        input_ids_two[0] = token_type_two[0]  # 将首字符替换为与token type 第一字符一样的id区分这句话归属问题

        # 数据整合
        instance["input_ids"] = input_ids_one + input_ids_two
        instance["input_token_type_ids"] = token_type_one + token_type_two
    assert len(instance["input_ids"]) == len(instance["input_token_type_ids"])

    # 2. target首尾添加特殊字符(bart decoder不添加token type)
    target_sequence = [bos] + reply + [eos]
    if len(target_sequence) > config.ans_length:
        # 从前向后截取
        target_sequence = target_sequence[:config.ans_length]
        # 替换末尾字符
        target_sequence[-1] = eos
    instance["mc_token_ids"] = len(target_sequence) - 1
    instance["lm_labels"] = [tokenizer.word2id[SPECIAL_TOKENS[-1]]] * len(target_sequence)
    if lm_labels:
        instance["lm_labels"] = target_sequence

    # 3. 整合图像数据
    def get_image_chars_indexes(array, token):
        """获取图片在input ids 中位置"""
        if not isinstance(array, np.ndarray):
            array = np.array(array)

        # 查找图片字符在上下文中的位置
        indexes = np.argwhere(array == token)

        return indexes.reshape(1, -1).tolist()[0]

    images_name = [os.path.join(image_dir, image_name) for image_name in utterance['img_list'] if image_name != 'NULL']
    img_id = tokenizer.word2id[IMG_TOKEN]
    images_id = get_image_chars_indexes(instance["input_ids"], img_id)
    if not images_id and len(images_id) == 0:
        images_name = []
    else:
        images_name = images_name[-len(images_id):]  # 截取input ids还存在images
    instance["input_images_name"] = images_name
    instance["input_images_id"] = images_id
    assert len(instance["input_images_name"]) == len(instance["input_images_id"])

    return instance


class ConversationDataset(Dataset):
    def __init__(self, pre_dataset_json, data_type='val'):
        """
        @param pre_dataset_json: 预处理之后的json数据
        """
        self.size = pre_dataset_json.get('n_candidates')
        if self.size > 1:
            # 多任务需要将数据拆解为[n, n_candidates, sequence_length]
            self.pre_dataset_json = self.split_list(pre_dataset_json)
        else:
            # 单任务数据[n*n_candidates, sequence_length], 候选集只包含一条label数据
            self.pre_dataset_json = pre_dataset_json
        self.data_type = data_type

    def __getitem__(self, index):
        """ 返回的所有数据: ["input_ids", "input_token_type_ids", "input_images_name", "input_images_id",
                            "lm_labels", "mc_token_ids", "mc_labels"]
        @param index:
        @return:
        """
        single_input_ids = self.pre_dataset_json.get("input_ids")[index]
        single_input_token_type_ids = self.pre_dataset_json.get("input_token_type_ids")[index]
        # single_input_images_name = self.pre_dataset_json.get("input_images_name")[index]
        if self.size > 1:
            # 包含N个候选集(选取第一个作为sample, n个输入都是一样的, 只是label不一样)
            single_input_images = self.image_transform(self.pre_dataset_json.get("input_images_name")[index][0],
                                                       data_type=self.data_type)
            single_input_images_id = self.pre_dataset_json.get("input_images_id")[index][0]
        else:
            single_input_images = self.image_transform(self.pre_dataset_json.get("input_images_name")[index],
                                                       data_type=self.data_type)
            single_input_images_id = self.pre_dataset_json.get("input_images_id")[index]
        single_lm_labels = self.pre_dataset_json.get("lm_labels")[index]  # label在候选集中的索引号
        single_mc_token_ids = self.pre_dataset_json.get("mc_token_ids")[index]
        single_mc_labels = self.pre_dataset_json.get("mc_labels")[index]

        return single_input_ids, single_input_token_type_ids, single_input_images, single_input_images_id, \
            single_lm_labels, single_mc_token_ids, single_mc_labels

    def __len__(self):
        # 获取输入数据size
        return len(self.pre_dataset_json.get(MODEL_INPUTS[0]))

    def split_list(self, pre_dataset_json):
        """ 将一维列表拆分为二维列表
        @param pre_dataset_json:
        @return:
        """
        for input_name in MODEL_INPUTS:
            data = pre_dataset_json.get(input_name)
            if input_name != "mc_labels":
                assert len(data) == len(pre_dataset_json.get('mc_labels')) * self.size
                _data = []
                sample = len(data)//self.size  # 可以采样出多少组这样的数据
                for i in range(sample):
                    # 将一维列表拆分二维列表
                    start = i * self.size
                    end = (i+1) * self.size
                    _data.append(data[start:end])
                assert len(_data) == len(pre_dataset_json.get('mc_labels'))
                pre_dataset_json[input_name] = _data

        return pre_dataset_json

    def image_transform(self, images, data_type):
        """单条数据所有图片数据的读取"""
        resp_list = list()

        # image_sample = torch.zeros(3, 224, 224)  # 图片样例, 不包含图片的token全部使用样例代替
        for image in images:
            # img = image_sample.clone()
            img = torch.zeros(3, 224, 224)
            try:
                img_tmp = PIL.Image.open(image)
                img = data_transforms[data_type](img_tmp)
            except:
                # print("can't open image file: ", image)
                pass
            finally:
                resp_list.append(img)

        return resp_list   # 没有图片直接传空list

        # # 样例图像存在于单条数据的最后
        # resp_list.append(image_sample)
        #
        # return torch.stack(resp_list)

    def get_images_feature_pad(self, single_images_name, single_images_id, sentence_length):
        """获取image feature pad
        @param single_images_name:  images name(key)
        @param single_images_id:
        @param sentence_length:
        @return:
        """
        sample_image_embed = np.zeros(self.embed_size)  # image feature sample
        sentence_embeds = [sample_image_embed] * sentence_length
        assert len(single_images_name) == len(single_images_id)
        if single_images_name and len(single_images_name) > 0:
            for idx, image_name in enumerate(single_images_name):
                i = single_images_id[idx]
                image_feature = self.images_feature_json.get(image_name)
                sentence_embeds[i] = image_feature

        return sentence_embeds

    def get_images_feature(self, images_name):
        """获取image feature"""
        images_feature = []
        for image_name in images_name:
            images_feature.append(torch.from_numpy(self.images_feature_json.get(image_name,
                                                                                np.zeros((512,), dtype=np.float32))))

        return images_feature


def get_data_loader(config, tokenizer, input_dataset_json, image_dir, batch_size=1, data_type='val', multi_task=False):
    """ 数据生产器封装
    @param config: 数据超参数
    @param tokenizer: 词汇object
    @param input_dataset_json: 数据集
    @param image_dir:
    @param data_type:
    @param batch_size:
    @param multi_task:
    @return:
    """
    def collate_fn(batch):
        """Collate list of data in to batch"""
        input_ids, input_token_type_ids, input_images_name, input_images_id, \
            lm_labels, mc_token_ids, mc_labels = zip(*batch)

        return input_ids, input_token_type_ids, input_images_name, input_images_id, \
            lm_labels, mc_token_ids, mc_labels

    logger.info("Build inputs and labels")
    num_candidates = len(input_dataset_json[0]['utterances'][0]['candidates'])
    if multi_task and not config.inference:
        # 预测时根本就不存在target(即candidates)
        if config.num_candidates > 0 and data_type == 'train':
            num_candidates = min(config.num_candidates, num_candidates)
    else:
        # 单任务
        num_candidates = 1
    pre_dataset_json = defaultdict(list)
    # 组织单条数据样本
    for dialog in input_dataset_json:
        persona = copy.copy(dialog['personality'])
        for utterance in dialog['utterances']:
            for j, candidate in enumerate(utterance['candidates'][-num_candidates:]):
                # 候选集的最后一条为label(即reply-answer)
                lm_labels = bool(j == num_candidates-1)
                instance = build_input_from_segments(config, tokenizer, utterance, candidate, image_dir,
                                                     persona=persona, lm_labels=lm_labels)
                for input_name, input_array in instance.items():
                    pre_dataset_json[input_name].append(input_array)
            pre_dataset_json['mc_labels'].append(num_candidates - 1)  # mc_labels就是子候选最后一条的index
            pre_dataset_json['n_candidates'] = num_candidates  # 记录候选集的数量

    logger.info("Pad inputs")
    pre_dataset_json = pad_dataset(pre_dataset_json, padding=tokenizer.word2id[SPECIAL_TOKENS[-1]])
    logger.info("Build dataloader")
    data_batch_size = batch_size * max(1, config.gpu_nums)
    dataset = ConversationDataset(pre_dataset_json, data_type=data_type)
    if data_type == 'val':
        config.val_num_candidates = num_candidates  # 更新
    else:
        config.train_num_candidates = num_candidates
    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if config.distributed else None
    shuffle = False if data_type == 'val' else True
    data_loader = DataLoader(dataset,
                             sampler=sampler,
                             batch_size=data_batch_size,
                             collate_fn=collate_fn,
                             # num_workers=1,
                             pin_memory=True,
                             shuffle=shuffle)

    return data_loader


if __name__ == "__main__":
    # 测试脚本
    pass
