# -*- coding: utf-8 -*-
"""
# custom bert vocab, BertTokenizer上封装一层
"""
import torch
from transformers import BertTokenizer

PAD_TOKEN = '[PAD]'
UNK_TOKEN = '[UNK]'
SOS_TOKEN = '[CLS]'
EOS_TOKEN = '[SEP]'
IMG_TOKEN = '<img>'
MSP_TOKEN = '</s>'

PAD_ID, UNK_ID, SOS_ID, EOS_ID = [0, 100, 101, 102]


class CustomBertVocab(object):
    def __init__(self, lang='en'):
        """Basic Vocabulary object"""
        self.lang = lang
        self.vocab_size = 0
        self.tokenizer = None

    def load(self, bert_vocab_path):
        """load 词汇表"""
        self.tokenizer = BertTokenizer(vocab_file=bert_vocab_path,
                                       never_split=['<num>', '<url>', '<img>', '</s>'])
        self.vocab_size = self.tokenizer.vocab_size

    def encode(self, words: list):
        """words 编码"""
        ids = []
        for word in words:
            ids.append(self.tokenizer.convert_tokens_to_ids(word))

        return ids

    def decode(self, ids, decode_type: str):
        """ids 解码"""
        sentence = []
        for id in ids:
            if isinstance(id, torch.Tensor):
                word = self.tokenizer.convert_ids_to_tokens(id.item())
            else:
                word = self.tokenizer.convert_ids_to_tokens(id)
            if decode_type == 'predict':
                if word not in [EOS_TOKEN, SOS_TOKEN, PAD_TOKEN, IMG_TOKEN, MSP_TOKEN]:
                    sentence.append(word)
                if word == PAD_TOKEN or word == EOS_TOKEN:
                    break
            else:  # context question
                sentence.append(word)
                if word == PAD_TOKEN:
                    break
        if self.lang == 'zh':
            return ''.join(sentence)

        return ' '.join(sentence)
