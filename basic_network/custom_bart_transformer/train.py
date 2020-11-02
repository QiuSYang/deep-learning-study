# -*- coding: utf-8 -*-
"""
# 自定义Bart生成模型训练模块
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "2"
import json
import argparse
import random
import numpy as np
import torch
from pathlib import Path

from configs import get_config
from vocab import Vocab
from utils import get_dataset, get_data_loader
from bart_solver import BartSolver

# set default path for data and test data
project_dir = Path(__file__).resolve().parent
datasets_dir = project_dir.joinpath('./data/')
images_train_dir = project_dir.joinpath('./data/images_train/')
images_valid_dir = project_dir.joinpath('./data/images_dev/')
images_test_dir = project_dir.joinpath('./online_test_data/images_test/')


def set_seed(config):
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if config.gpu_nums > 0:
        torch.cuda.manual_seed_all(config.seed)


def set_distributed(config):
    """进行分布式基本配置"""
    if config.local_rank == -1:
        # DataParallel形式的分布式, 负载不平衡
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("{}: {}".format(config.local_rank, device))
        config.gpu_nums = torch.cuda.device_count() if torch.cuda.is_available() else 0
    else:
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(config.local_rank)
        device = torch.device('cuda', config.local_rank)
        print("{}: {}.".format(config.local_rank, device))
        torch.distributed.init_process_group(backend="nccl")
        config.gpu_nums = 1
        config.distributed = True
    config.device = device


if __name__ == "__main__":
    config = get_config(mode='train')

    # 设置分布式模式
    config.distributed = False  # 初始不支持分布训练
    set_distributed(config)
    # 设置随机数种子
    set_seed(config)
    print("device: {}".format(config.device))
    print("available gpu nums: {}".format(config.gpu_nums))

    print('Loading Vocabulary...')
    vocab = Vocab()
    vocab.load(config.word2id_path, config.id2word_path)
    print(f'Vocabulary size: {vocab.vocab_size}')
    config.vocab_size = vocab.vocab_size

    # 获取训练数据以及测试数据
    train_dev_ids_json = get_dataset(tokenizer=vocab,
                                     dataset_path=config.train_ids_path,
                                     dataset_cache=config.train_cache_path,
                                     single_sentence_token_max_num=config.max_sentence_token_num,
                                     seg_voc=config.seg_voc,
                                     inference=config.inference)

    train_data_loader = get_data_loader(config, vocab, train_dev_ids_json.get('train'),
                                        image_dir=images_train_dir,
                                        batch_size=config.per_gpu_train_batch_size,
                                        data_type='train',
                                        multi_task=config.multi_task)

    eval_data_loader = get_data_loader(config, vocab, train_dev_ids_json.get('dev'),
                                       image_dir=images_valid_dir,
                                       batch_size=config.per_gpu_eval_batch_size,
                                       data_type='val',
                                       multi_task=config.multi_task)
    # 计算训练步长
    config.num_training_steps = config.n_epoch * len(train_data_loader) // config.gradient_accumulation_step
    print(config)
    with open(os.path.join(config.save_path, 'config.txt'), 'w') as f:
        print(config, file=f)

    # for index, batch in enumerate(eval_data_loader):
    #     temp = batch

    main = BartSolver(config=config,
                      vocab=vocab,
                      train_data_loader=train_data_loader,
                      eval_data_loader=eval_data_loader,
                      is_train=True)
    main.build_graph()
    main.train()
