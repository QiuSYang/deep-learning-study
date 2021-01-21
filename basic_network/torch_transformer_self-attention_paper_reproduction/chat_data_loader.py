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
    def __init__(self, data_path, tokenizer,
                 max_encode_len=256, max_decode_len=128, history_turns=3):
        self.tokenizer = tokenizer
        self.history_turns = history_turns
        self.max_encode_len = max_encode_len
        self.max_decode_len = max_decode_len

        self.PAD_ID = self.tokenizer._convert_token_to_id("[PAD]")

        self.samples = self.get_samples(data_path)  # all data

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]

    def _get_dialogue_samples(self, dialogue):
        if len(dialogue) < 2:
            return []

        dialogue_samples = []
        for idx, target in enumerate(dialogue):
            if idx == 0:
                continue
            context_tokens = ["[SEP]"]
            if idx > self.history_turns:
                history_dialogue = dialogue[idx-self.history_turns:idx]
            else:
                history_dialogue = dialogue[:idx]
            for history in history_dialogue[::-1]:  # 逆序访问
                current_tokens = self.tokenizer.tokenize(history)
                if len(context_tokens) + len(context_tokens) <= self.max_encode_len:
                    context_tokens = current_tokens + context_tokens  # 切词
                    context_tokens = ["[SEP]"] + context_tokens
                else:
                    break
            context_tokens[0] = "[CLS]"  # 将头部[SEP] token替换为[CLS] token

            target_tokens = self.tokenizer.tokenize(dialogue[idx])
            if len(target_tokens) > self.max_decode_len - 1:
                target_tokens = target_tokens[:self.max_decode_len - 1]
            target_input_tokens = ["[CLS]"] + target_tokens
            target_label_tokens = target_tokens + ["[SEP]"]

            input_ids = self.tokenizer.convert_tokens_to_ids(context_tokens)
            decode_input_ids = self.tokenizer.convert_tokens_to_ids(target_input_tokens)
            decode_label_ids = self.tokenizer.convert_tokens_to_ids(target_label_tokens)

            assert len(context_tokens) == len(input_ids)
            assert len(decode_label_ids) == len(decode_input_ids) == len(target_input_tokens) == len(target_label_tokens)

            extra = self.max_encode_len - len(input_ids)
            if extra > 0:
                input_ids += [self.PAD_ID] * extra

            extra = self.max_decode_len - len(decode_input_ids)
            if extra > 0:
                decode_input_ids += [self.PAD_ID] * extra
                decode_label_ids += [self.PAD_ID] * extra

            dialogue_samples.append({
                "input_ids": torch.tensor(input_ids).long(),
                "decode_input_ids": torch.tensor(decode_input_ids).long(),
                "decode_label_ids": torch.tensor(decode_label_ids).long()
            })

        return dialogue_samples

    def get_samples(self, data_path):
        """获取训练样例"""
        with open(data_path, mode='r', encoding='utf-8') as f:
            data_lines = f.readlines()

            single_dialogue = []  # 保存单个对话数据
            samples = []  # 存储所有训练样例
            for idx, line in enumerate(tqdm(data_lines)):
                if line == '\n':
                    samples.extend(self._get_dialogue_samples(single_dialogue))  # 单个对话数据集制作
                    single_dialogue = []
                else:
                    single_dialogue.append(line)

            return samples


if __name__ == "__main__":
    from transformers import BertTokenizer
    root = os.path.dirname(os.path.abspath(__file__))
    tokenizer = BertTokenizer(vocab_file=os.path.join(root, "vocabulary/vocab.txt"))

    data_path = os.path.join(root, "data/train.txt")

    dataset = ChatDataset(data_path, tokenizer)
    print(dataset[0])
