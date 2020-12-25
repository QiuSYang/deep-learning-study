"""
# drcm data + cmrc data processor
"""
import logging
import datasets
import torch
from transformers import PreTrainedTokenizer

from data_processor import DataProcessor

logger = logging.getLogger(__name__)


class DRMCDataProcessor(DataProcessor):
    def __init__(self,
                 tokenizer: PreTrainedTokenizer,
                 max_encode_length=512,
                 max_decode_length=32,
                 max_answer_length=100,
                 task="hl_ag",  # example--['hl_ag', 'a2q', 'q2a']
                 add_tokens: list = None):
        super(DRMCDataProcessor, self).__init__(tokenizer=tokenizer,
                                                max_encode_length=max_encode_length,
                                                max_decode_length=max_decode_length)
        self.task = task
        self.decode_start_token = "[unused1]"
        self.decode_end_token = "[unused2]"
        self.hl_token = "[unused3]"
        self.answer_split_token = "[unused4]"  # 也可以设置为
        additional_special_tokens = [self.decode_start_token, self.decode_end_token,
                                     self.hl_token, self.answer_split_token]
        if add_tokens:
            # 增加特殊token
            additional_special_tokens.extend(add_tokens)
        special_tokens_dict = {"additional_special_tokens": additional_special_tokens}
        self.tokenizer.add_special_tokens(special_tokens_dict)
        self.max_answer_length = max_answer_length

    def process(self, dataset: datasets.Dataset):
        dataset = dataset.filter(self._filter_cmrc_data)
        if self.task == 'hl_ag':
            dataset = dataset.filter(self._filter_task_hl)
        else:
            dataset = dataset.filter(self._filter_task_qa)
        dataset = dataset.map(self._convert_to_features)

        return dataset

    def _filter_cmrc_data(self, example):
        return example['data_name'] == 'cmrc'

    def _filter_task_qa(self, example):
        return example['task'] == 'a2q_or_q2a'

    def _filter_task_hl(self, example):
        return example['task'] == 'hl_ag'

    def _convert_to_features(self, example):
        """将文本数据转为ids"""
        context = example['context']
        answers = example['answer']
        questions = example['question']

        if self.task == "hl_ag":
            # highlight + context = answer
            return self._highlight_context_answer(context, answers)
        elif self.task == "a2q":
            # answer + context = question
            context_sentences = context['sentences']
            if isinstance(context_sentences, list):
                context_str = "".join(context_sentences)  # 数据合并

            if isinstance(answers, list) and isinstance(questions, list):
                assert len(answers) == len(questions)
                # samples = []
                # for idx, answer in enumerate(answers):
                #     question = questions[idx]
                #     sample = self._condition_context_target(context_str, condition=answer, target=question)
                #     samples.append(sample)
                # return samples  # 不支持list数据类型返回,仅仅支持dict数据类型返回
                return self._condition_context_target(context_str, answers[0], questions[0])  # 支持第一条数据编码转换
            elif isinstance(answers, str) and isinstance(questions, str):
                return self._condition_context_target(context_str, answers, questions)
            else:
                raise TypeError("输入数据类型错误")
        elif self.task == "q2a":
            # question + context = answer
            pass

    def _condition_context_target(self, context, condition, target):
        context_tokens = self.tokenizer.tokenize(context.replace("\n", " ").replace("\t", " ").replace("\\", ""))
        target_tokens = self.tokenizer.tokenize(target)
        condition_tokens = self.tokenizer.tokenize(condition)[:self.max_answer_length]

        input_tokens = ['[CLS]'] + condition_tokens + ['[SEP]'] + context_tokens
        if len(input_tokens) > self.max_encode_length - 1:
            input_tokens = input_tokens[:self.max_encode_length - 1]
        input_tokens += ['[SEP]']

        input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)
        input_mask_ids = [1.0] * len(input_ids)
        input_type_ids = [0] * (len(condition_tokens) + 2) + [1] * (len(input_ids) - 2 - len(condition_tokens))
        extra = self.max_encode_length - len(input_ids)
        if extra > 0:
            input_ids += [self.tokenizer.pad_token_id] * extra
            input_mask_ids += [0.0] * extra
            input_type_ids += [self.tokenizer.pad_token_id] * extra

        if len(target_tokens) > self.max_decode_length - 1:
            target_tokens = target_tokens[:self.max_decode_length - 1]
        target_ids = self.tokenizer.convert_tokens_to_ids(target_tokens)
        decode_input_ids = self.tokenizer.convert_tokens_to_ids([self.decode_start_token]) + target_ids
        decode_output_ids = target_ids + self.tokenizer.convert_tokens_to_ids([self.decode_end_token])
        assert len(decode_input_ids) == len(decode_output_ids)
        extra = self.max_decode_length - len(decode_input_ids)
        if extra > 0:
            decode_input_ids += [self.tokenizer.pad_token_id] * extra
            decode_output_ids += [self.tokenizer.pad_token_id] * extra

        return {
            "input_ids": torch.tensor(input_ids).long(),
            "input_mask": torch.tensor(input_mask_ids).float(),
            "input_seg": torch.tensor(input_type_ids).long(),
            "decode_input": torch.tensor(decode_input_ids).long(),
            "decode_target": torch.tensor(decode_output_ids).long(),
            "label": target
        }

    def _highlight_context_answer(self, context, answers):
        context_sentences = context['sentences']
        highlight_idx = context['highlight_idx']

        context_tokens = ['[CLS]']
        input_type_ids = [0]
        for idx, sentence in enumerate(context_sentences):
            sentence_tokens = self.tokenizer.tokenize(sentence.replace("\n", " ").replace("\t", " ").replace("\\", ""))
            type_id = 0
            if idx == highlight_idx:
                sentence_tokens = [self.hl_token] + sentence_tokens + [self.hl_token]
                type_id = 1
            context_tokens += sentence_tokens
            input_type_ids += [type_id] * len(sentence_tokens)

        assert len(context_tokens) == len(input_type_ids)
        if len(context_tokens) > self.max_encode_length - 1:
            context_tokens = context_tokens[:self.max_encode_length - 1]
            input_type_ids = input_type_ids[:self.max_encode_length - 1]
        context_tokens += ['[SEP]']
        input_type_ids += [0]

        input_ids = self.tokenizer.convert_tokens_to_ids(context_tokens)
        input_mask_ids = [1.0] * len(input_ids)

        extra = self.max_encode_length - len(input_ids)
        if extra > 0:
            input_ids += [self.tokenizer.pad_token_id] * extra
            input_mask_ids += [0.0] * extra
            input_type_ids += [self.tokenizer.pad_token_id] * extra

        answers_tokens = []
        for idx, answer in enumerate(answers):
            answer_tokens = self.tokenizer.tokenize(answer)
            answers_tokens += answer_tokens
            answers_tokens += [self.answer_split_token]
        # -2因为要处理self.answer_split_token位置
        if len(answer_tokens) > self.max_decode_length - 2:
            answers_tokens = answers_tokens[:self.max_decode_length - 2]
        if answers_tokens[-1] != self.answer_split_token:
            answers_tokens += [self.answer_split_token]  # 答案以self.answer_split_token结尾

        decode_input_tokens = [self.decode_start_token] + answers_tokens
        decode_output_tokens = answers_tokens + [self.decode_end_token]
        decode_input_ids = self.tokenizer.convert_tokens_to_ids(decode_input_tokens)
        decode_output_ids = self.tokenizer.convert_tokens_to_ids(decode_output_tokens)
        assert len(decode_input_ids) == len(decode_output_ids)
        extra = self.max_decode_length - len(decode_input_ids)
        if extra > 0:
            decode_input_ids += [self.tokenizer.pad_token_id] * extra
            decode_output_ids += [self.tokenizer.pad_token_id] * extra

        return {
            "input_ids": torch.tensor(input_ids).long(),
            "input_mask": torch.tensor(input_mask_ids).float(),
            "input_seg": torch.tensor(input_type_ids).long(),
            "decode_input": torch.tensor(decode_input_ids).long(),
            "decode_target": torch.tensor(decode_output_ids).long(),
            "label": "".join(answers_tokens)
        }
