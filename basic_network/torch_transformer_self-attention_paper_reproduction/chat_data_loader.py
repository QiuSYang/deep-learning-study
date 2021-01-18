"""
# 闲聊数据集
"""
import os
import logging
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


class ChatDataset(Dataset):
    def __init__(self, data_path, tokenizer):
        pass

    def __len__(self):
        pass

    def __getitem__(self, item):
        pass

    def get_samples(self, data_path):
        """获取训练样例"""
        with open(data_path, mode='r', encoding='utf-8') as f:
            data_lines = f.readlines()

            single_dialogue = []  # 保存单个对话数据
            samples = []  # 存储所有训练样例
            for idx, line in enumerate(tqdm(data_lines)):
                if line == '\n':
                    pass
                    single_dialogue = []
                else:
                    single_dialogue.append(line)
