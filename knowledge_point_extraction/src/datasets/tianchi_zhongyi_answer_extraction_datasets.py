"""
# 使用天池中医问题生成数据制作答案抽(知识点)取数据集, 参考序列标注
"""
import json
import logging
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

logger = logging.getLogger(__file__)


class ChineseMedicalAnswerExtractionDataset(Dataset):
    def __init__(self, data_path: str,
                 tokenizer: BertTokenizer,
                 label2id: dict = {"<pad>": 0, "S": 1, "B": 2, "M": 3, "E": 4},  # BMES label
                 max_enc_len=512):
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_encode_len = max_enc_len

        self.samples = self.get_samples(data_path)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]

    def get_samples(self, data_path):
        data = self.get_data(data_path)
        samples = []
        for id, single_data in enumerate(tqdm(data)):
            context = single_data.get('text')
            process_context = context.replace("\n", " ").replace("\t", " ").replace("\\", "")
            context_tokens = self.tokenizer.tokenize(process_context)
            if len(context_tokens) == 0:
                # 忽略这条数据
                continue
            if len(context_tokens) > self.max_encode_len:
                context_tokens = context_tokens[:self.max_encode_len]

            labels = ['S'] * len(context_tokens)

            answers = single_data.get('annotations')
            nums = 0
            for answer in answers:
                answer_tokens = self.tokenizer.tokenize(answer.get("A"))
                if len(answer_tokens) == 0:
                    nums += 1
                    continue
                try:
                    # 确定答案在上下文的位置
                    start_id, end_id = self.get_answer_span(context_tokens, answer_tokens)
                except Exception:
                    nums += 1
                    continue
                labels[start_id] = "B"
                labels[start_id+1:end_id-1] = ["M"] * (len(answer_tokens) - 2)
                labels[end_id-1] = "E"
            if nums == len(answers):
                # 答案匹配全为空
                continue

            input_ids = self.tokenizer.convert_tokens_to_ids(context_tokens)
            # label_ids = list(map(lambda x: self.label2id[x], labels))
            # assert len(input_ids) == len(label_ids), "the input and label data length is inconsistent."
            mask_ids = [1.0] * len(input_ids)

            extra = self.max_encode_len - len(input_ids)
            if extra > 0:
                input_ids += [self.tokenizer.pad_token_id] * extra
                mask_ids += [0.0] * extra
                labels += ["<pad>"] * extra

            label_ids = list(map(lambda x: self.label2id[x], labels))
            assert len(input_ids) == len(label_ids) == len(mask_ids), \
                "the input and label data length is inconsistent."

            samples.append({
                "input_ids": torch.tensor(input_ids).long(),
                "labels": torch.tensor(label_ids).long(),
                "attention_mask": torch.tensor(mask_ids).float()
            })

        return samples

    def get_data(self, data_path):
        with open(data_path, mode='r', encoding='utf-8') as fp:
            return json.load(fp)

    def get_answer_span(self, context: list, answer: list):
        """获取答案在上下文中的位置"""
        length = len(answer)
        for i in range(len(context)):
            # 对上下文子串进行对比
            if context[i:i + length] == answer:
                # 返回起始结束位置
                return i, i + length


if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained("hfl/chinese-bert-wwm")
    data_path = "../../data/round1_train_0907.json"
    dataset = ChineseMedicalAnswerExtractionDataset(data_path, tokenizer)
    print(dataset[0])
