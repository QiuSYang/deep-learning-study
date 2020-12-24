# -*- coding: utf-8 -*-
"""
# transformers库, 构建数据集专有词汇表
# 支持基于词汇表基础上, 根据数据集在原有词汇上添加新的词汇
# 使用BertTokenizer
"""
import os
import re
import json
import argparse
import itertools
from tqdm import tqdm
from collections import OrderedDict
import torch
from transformers import BertTokenizer


class BuildCustomTransformersVocabulary(object):
    def __init__(self, base_vocab_path='./vocab_small.txt',
                 additional_special_tokens={'additional_special_tokens': ['<num>', '<img>', '<url>', '#E-s', '|||']}):
        self.tokenizer = BertTokenizer(vocab_file=base_vocab_path, do_lower_case=False, do_basic_tokenize=True)
        self.tokenizer.add_special_tokens(additional_special_tokens)
        self.no_vocab_tokens = set()

    def get_no_vocab_token(self, text, unk_token='[UNK]', other_split=False):
        """ tokens compare
        @param text:
        @param unk_token:
        @param other_split:  原始拆分出来single token txt, bert tokenizer拆分之后依然拆解为多个token, 是否增加词汇
        @return:
        """
        # text_tokens = self.tokenizer.tokenize(text)  # bert tokenizer根据词汇表处理之后切分出来的token(包含unk)
        origin_tokens = self.tokenize(text)  # 切词之后结果, 不在词汇表中的词没有转为unk

        # # 第一种方法不能保证一一对应, 有些切分出来字符再次转换时候会被再次切分
        # assert len(text_tokens) == len(origin_tokens)
        for idx, token in enumerate(origin_tokens):
            # 使用transformer tokenizer根据基础词汇表转换
            bert_token = self.tokenizer.tokenize(token)
            # if token != origin_tokens[idx]:
            #     # 未知token添加进词汇表
            #     self.no_vocab_tokens.append(origin_tokens[idx])
            if len(bert_token) == 1 and bert_token[0] == unk_token:
                self.no_vocab_tokens.add(token)  # 借助set去重
            if other_split and len(bert_token) > 1:
                # 单个字符被bert tokenizer拆分为多个字符, 实际不需要拆分
                self.no_vocab_tokens.add(token)

    def _tokenize(self, text):
        """将text拆分为 token list"""
        tokens_list = self.tokenizer.basic_tokenizer.tokenize(text,
                                                              never_split=self.tokenizer.all_special_tokens)

        return tokens_list

    def tokenize(self, text: str, **kwargs):
        """ Converts a string in a sequence of tokens (string), using the tokenizer.
            Split in words for word-based vocabulary or sub-words for sub-word-based
            vocabularies (BPE/SentencePieces/WordPieces).

            Take care of added tokens.

            Args:
                text (:obj:`string`): The sequence to be encoded.
                **kwargs (:obj: `dict`): Arguments passed to the model-specific `prepare_for_tokenization` preprocessing method.
        """
        all_special_tokens = self.tokenizer.all_special_tokens
        text = self.tokenizer.prepare_for_tokenization(text, **kwargs)

        # TODO: should this be in the base class?
        def lowercase_text(t):
            # convert non-special tokens to lowercase
            escaped_special_toks = [re.escape(s_tok) for s_tok in all_special_tokens]
            pattern = r"(" + r"|".join(escaped_special_toks) + r")|" + r"(.+?)"
            return re.sub(pattern, lambda m: m.groups()[0] or m.groups()[1].lower(), t)

        if self.tokenizer.init_kwargs.get("do_lower_case", False):
            text = lowercase_text(text)

        def split_on_token(tok, text):
            result = []
            split_text = text.split(tok)
            for i, sub_text in enumerate(split_text):
                sub_text = sub_text.rstrip()
                if i == 0 and not sub_text:
                    result += [tok]
                elif i == len(split_text) - 1:
                    if sub_text:
                        result += [sub_text]
                    else:
                        pass
                else:
                    if sub_text:
                        result += [sub_text]
                    result += [tok]
            return result

        def split_on_tokens(tok_list, text):
            if not text.strip():
                return []
            if not tok_list:
                return self._tokenize(text)

            tokenized_text = []
            text_list = [text]
            for tok in tok_list:
                tokenized_text = []
                for sub_text in text_list:
                    if sub_text not in self.tokenizer.unique_added_tokens_encoder:
                        tokenized_text += split_on_token(tok, sub_text)
                    else:
                        tokenized_text += [sub_text]
                text_list = tokenized_text

            return list(
                itertools.chain.from_iterable(
                    (
                        self._tokenize(token) if token not in self.tokenizer.unique_added_tokens_encoder else [token]
                        for token in tokenized_text
                    )
                )
            )

        added_tokens = self.tokenizer.unique_added_tokens_encoder
        tokenized_text = split_on_tokens(added_tokens, text)
        return tokenized_text

    def update_vocab(self, new_vocab_tokens: list):
        """ 更新原有基础词汇表
        @param new_vocab_tokens:
        @return:
        """
        add_token_num = self.tokenizer.add_tokens(new_vocab_tokens)

        return add_token_num

    def custom_save_vocabulary(self, new_vocab_path):
        """保存新的词汇表"""
        if os.path.exists(new_vocab_path):
            os.remove(new_vocab_path)

        index = 0
        with open(new_vocab_path, mode='w', encoding='utf-8') as writer:
            for token, token_index in sorted(self.tokenizer.vocab.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    print(
                        "Saving vocabulary to {}: vocabulary indices are not consecutive."
                        " Please check that the vocabulary is not corrupted!".format(new_vocab_path)
                    )
                    index = token_index
                writer.write(token + "\n")
                index += 1

            # 将新增加的token添加到词汇表
            add_tokens_vocab = OrderedDict(self.tokenizer.added_tokens_encoder)
            for token, token_index in sorted(add_tokens_vocab.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    print(
                        "Saving vocabulary to {}: vocabulary indices are not consecutive."
                        " Please check that the vocabulary is not corrupted!".format(new_vocab_path)
                    )
                    index = token_index
                writer.write(token + "\n")
                index += 1

        return new_vocab_path

    def save_vocab_pretrained(self, vocab_pretrained_path):
        """保存词表预训练全部内容"""
        if not os.path.exists(vocab_pretrained_path):
            # 路径不存在, 创建路径
            os.makedirs(vocab_pretrained_path)
        all_file = self.tokenizer.save_pretrained(vocab_pretrained_path)  # 存储所有词汇内容
        # model.resize_token_embeddings(len(tokenizer)) -> 重新设置embedding大小(词汇表大小已经改变)
        # Notice: resize_token_embeddings expect to receive the full size of the new vocabulary,
        # i.e. the length of the tokenizer.
        return all_file


if __name__ == "__main__":
    main = BuildCustomTransformersVocabulary(base_vocab_path='./vocabulary/vocab_small.txt')

    text_path = './data/data_train_1w.txt'
    with open(text_path, mode='r', encoding='utf-8') as fp:
        lines = fp.readlines()
        for lines in tqdm(lines):
            text = lines.strip().split('\t')[3]
            if text.endswith('.jpg'):
                continue
            main.get_no_vocab_token(text)
    add_new_vocab_tokens = list(main.no_vocab_tokens)
    main.update_vocab(add_new_vocab_tokens)
    main.custom_save_vocabulary('./vocabulary/vocab_small_new.txt')
    main.save_vocab_pretrained('./vocabulary/new_vocab')

    new_tokenizer = BertTokenizer.from_pretrained('./vocabulary/new_vocab')
    print(new_tokenizer.tokenize('Jewel'))
    pass
