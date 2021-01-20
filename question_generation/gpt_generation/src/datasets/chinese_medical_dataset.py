"""
# 天池算法大赛中医问题生成
# 参考链接: https://tianchi.aliyun.com/competition/entrance/531826/introduction
"""
import os
import json
import logging
from random import shuffle
from tqdm import tqdm
import torch
from torch.utils.data import Dataset

from src.utils import *

logger = logging.getLogger(__name__)
root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class DataPreprocessing(object):
    """原始数据结构化处理"""
    def __init__(self, a2q=True):
        self.chinese_medicine_output = {
            "train": [],
            "valid": []
        }

        self.a2q = a2q  # A + C = Q task

    def chinese_medical_json(self, data, key="train"):
        for item in tqdm(data):
            for jtem in item["annotations"]:
                if self.a2q:
                    self.chinese_medicine_output[key].append({
                            "context": item["text"],
                            "condition": jtem["A"],
                            "target": jtem["Q"]
                        })
                else:
                    self.chinese_medicine_output[key].append({
                            "context": item["text"],
                            "condition": jtem["Q"],
                            "target": jtem["A"]
                        })

    def chinese_medical_deal(self):
        """处理中医数据"""
        # train data
        train_data_path = os.path.join(root, "data/round1_train_0907.json")
        with open(train_data_path, mode='r', encoding='utf-8') as fp:
            train_data = json.load(fp)
            self.chinese_medical_json(train_data)

        for i in range(3):
            shuffle(self.chinese_medicine_output.get("train"))

        data_nums = len(self.chinese_medicine_output.get("train"))
        valid_data_nums = int(data_nums * 0.2) // 1000 * 1000

        self.chinese_medicine_output['valid'] = self.chinese_medicine_output['train'][:valid_data_nums]
        self.chinese_medicine_output['train'] = self.chinese_medicine_output['train'][valid_data_nums:]

        print("chinese medical train data numbers: {}".format(len(self.chinese_medicine_output.get("train"))))
        print("chinese medical  valid data numbers: {}".format(len(self.chinese_medicine_output.get("valid"))))

        test_data_path = os.path.join(root, "data/round1_test_0907.json")
        with open(test_data_path, mode='r', encoding='utf-8') as fp:
            test_data = json.load(fp)
            self.chinese_medicine_output['test'] = test_data

        # save cmrc data
        save_path = os.path.join(root, "data", "chinese_medical_dataset.pt")
        torch.save(self.chinese_medicine_output, save_path)


class ChineseMedicalDataset(Dataset):
    def __init__(self,
                 data,
                 tokenizer,
                 max_sequence_len=512,
                 max_condition_len=100,
                 max_target_len=50,
                 is_right_pad=True,
                 is_condition_first=False,
                 is_unilm_mask=False
                 ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_sequence_len = max_sequence_len
        self.max_condition_len = max_condition_len
        self.max_target_len = max_target_len

        self.SEG_CONDITION = 0 if is_condition_first else 1
        self.SEG_CONTEXT = 1 if is_condition_first else 0
        self.SEG_TARGET = 2
        self.ID_PAD = tokenizer._convert_token_to_id("[PAD]")

        self.is_right_pad = is_right_pad  # padding 放在句子末尾, 否则放在句首
        self.is_condition_first = is_condition_first  # 条件放在前面还是后面
        self.is_unilm_mask = is_unilm_mask  # 是否再构造数据时构造token mask

        self.samples = self.get_samples()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]

    def get_samples(self):
        samples = []
        for idx, item in enumerate(tqdm(self.data)):
            context, condition, target = item.get("context"), item.get("condition"), item.get("target")
            context_tokens = self.tokenizer.tokenize(context.replace("\n", " ").replace("\t", " ").replace("\\", ""))
            target_tokens = self.tokenizer.tokenize(target)
            condition_tokens = self.tokenizer.tokenize(condition)[:self.max_condition_len]

            max_context_len = self.max_sequence_len - self.max_condition_len - self.max_target_len
            if len(context_tokens) > max_context_len - 3:
                # 截取上下文
                context_tokens = context_tokens[:max_context_len-3]
            if self.is_condition_first:
                c = ["[CLS]"] + condition_tokens + ["[SEP]"] + context_tokens + ["[SEP]"]
            else:
                c = ["[CLS]"] + context_tokens + ["[SEP]"] + condition_tokens + ["[SEP]"]
            # if len(c) > max_context_len - 1:
            #     c = c[:max_context_len - 1]
            # c += ["[SEP]"]
            if len(target_tokens) > self.max_target_len - 1:
                target_tokens = target_tokens[: self.max_target_len - 1]
            target_tokens += ["[SEP]"]

            # all_tokens = c + query_tokens
            # token_ids = self.tokenizer.convert_tokens_to_ids(all_tokens)

            context_ids = self.tokenizer.convert_tokens_to_ids(c)
            target_ids = self.tokenizer.convert_tokens_to_ids(target_tokens)
            assert len(c) == len(context_ids)
            assert len(target_ids) == len(target_tokens)

            input_ids = context_ids + target_ids
            decode_target_ids = [-100.0] * len(context_ids) + target_ids  # context不计算loss
            input_mask = [1.0] * len(input_ids)
            if self.is_condition_first:
                input_seg = [self.SEG_CONDITION] * (len(condition_tokens) + 2) + \
                            [self.SEG_CONTEXT] * (len(c) - 2 - len(condition_tokens)) + \
                            [self.SEG_TARGET] * len(target_tokens)
            else:
                input_seg = [self.SEG_CONTEXT] * (len(context_tokens) + 2) + \
                            [self.SEG_CONDITION] * (len(c) - 2 - len(context_tokens)) + \
                            [self.SEG_TARGET] * len(target_tokens)
            assert len(input_ids) == len(decode_target_ids) == len(input_mask) == len(input_seg)
            if self.is_unilm_mask:
                unilm_token_type_ids = [0] * len(c) + [1] * (len(target_tokens) - 1)  # 全部上下文进行双向self-attention
                assert len(input_ids) == len(unilm_token_type_ids)

            extra = self.max_sequence_len - len(input_ids)
            if extra > 0:
                if self.is_right_pad:
                    input_ids += [self.ID_PAD] * extra
                    decode_target_ids += [-100.0] * extra
                    input_mask += [0.0] * extra
                    input_seg += [self.SEG_TARGET] * extra
                    if self.is_unilm_mask:
                        unilm_token_type_ids += [1] * extra
                else:
                    input_ids = [self.ID_PAD] * extra + input_ids
                    decode_target_ids = [-100.0] * extra + decode_target_ids
                    input_mask = [0.0] * extra + input_mask
                    pad_token_type_id = self.SEG_CONDITION if self.is_condition_first else self.SEG_CONTEXT
                    input_seg = [pad_token_type_id] * extra + input_seg
                    if self.is_unilm_mask:
                        unilm_token_type_ids = [0] * extra + unilm_token_type_ids

            input_ids = torch.tensor(input_ids).long()
            decode_target_ids = torch.tensor(decode_target_ids).long()
            input_mask = torch.tensor(input_mask).float()
            input_token_type_ids = torch.tensor(input_seg).long()

            if self.is_unilm_mask:
                unilm_token_type_ids = torch.tensor(unilm_token_type_ids).long()
                input_mask = comput_unilm_attention_mask(unilm_token_type_ids, input_mask)
                # for temp in input_mask[len(c)-2:len(c)+5, :]:
                #     print(temp)

            samples.append({
                "input_ids": input_ids,
                "attention_mask": input_mask,
                "token_type_ids": input_token_type_ids,
                "labels": decode_target_ids,
                # "label_text": target
                })

        return samples


if __name__ == "__main__":
    logging.basicConfig(format='[%(asctime)s %(filename)s:%(lineno)s] %(message)s', level=logging.INFO,
                        filename=None, filemode='a')
    logger.info("root dir: {}".format(root))
    data_obj = DataPreprocessing(a2q=True)
    logger.info("Chinese Medical dataset preprocessing.")
    data_obj.chinese_medical_deal()
