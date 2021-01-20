"""
# gpt model evaluate and inference
"""
import os
import logging
import json
from tqdm import tqdm
import numpy as np
import torch
from transformers import BertTokenizer

from src.models.custom_gpt import CustomGPTGeneration
from src.utils import *
from src.hyper_parameters import HyperParametersConfig

logger = logging.getLogger(__name__)
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def rouge_l(hypo, refer):
    """rouge_l值计算"""
    if len(hypo) == 0 or len(refer) == 0:
        return 0
    x = [[0 for _ in range(len(refer) + 1)] for _ in range(len(hypo) + 1)]
    lcs = 0
    for i in range(len(hypo)):
        for j in range(len(refer)):
            if hypo[i] == refer[j]:
                x[i + 1][j + 1] = x[i][j] + 1
                if x[i + 1][j + 1] > lcs:
                    lcs = x[i + 1][j + 1]
            else:
                x[i + 1][j + 1] = max(x[i + 1][j], x[i][j + 1])
    p, r = lcs / len(hypo), lcs / len(refer)
    if (p + r) == 0:
        return 0
    else:
        return (2 * p * r) / (p + r)


def text_to_ids(tokenizer, context, condition_text,
                is_condition_first=False, is_unilm_mask=False,
                max_sequence_len=512, max_condition_len=100, max_target_len=50):
    """文本转为id number"""
    context_tokens = tokenizer.tokenize(context.replace("\n", " ").replace("\t", " ").replace("\\", ""))
    condition_tokens = tokenizer.tokenize(condition_text)[:max_condition_len]

    max_context_len = max_sequence_len - max_condition_len - max_target_len
    if len(context_tokens) > max_context_len - 3:
        # 截取上下文
        context_tokens = context_tokens[:max_context_len - 3]
    if is_condition_first:
        c = ["[CLS]"] + condition_tokens + ["[SEP]"] + context_tokens + ["[SEP]"]
    else:
        c = ["[CLS]"] + context_tokens + ["[SEP]"] + condition_tokens + ["[SEP]"]

    input_ids = tokenizer.convert_tokens_to_ids(c)
    assert len(input_ids) == len(c)
    input_mask = [1.0] * len(input_ids)
    if is_condition_first:
        token_type_ids = [0] * (len(condition_tokens) + 2) + \
                         [1] * (len(c) - len(condition_tokens) - 2)
    else:
        token_type_ids = [0] * (len(context_tokens) + 2) + \
                         [1] * (len(c) - len(context_tokens) - 2)

    if is_unilm_mask:
        unilm_token_type_ids = [0] * len(c)
        assert len(input_mask) == len(token_type_ids)

    input_ids = torch.tensor(input_ids).long()
    input_mask = torch.tensor(input_mask).float()
    input_token_type_ids = torch.tensor(token_type_ids).long()

    if is_unilm_mask:
        unilm_token_type_ids = torch.tensor(unilm_token_type_ids).long()
        input_mask = comput_unilm_attention_mask(unilm_token_type_ids, input_mask)

    return {
        "input_ids": input_ids,
        "attention_mask": input_mask,
        "token_type_ids": input_token_type_ids
    }


def ids_to_text(ids, tokenizer):
    valid_ids = []
    for id_ in ids:
        if int(id_) == tokenizer._convert_token_to_id("[SEP]"):
            break
        else:
            valid_ids.append(int(id_))
    text = "".join(tokenizer.convert_ids_to_tokens(valid_ids))
    text = text.replace(", ", "").replace("[UNK]", "").replace("#", "")

    # Deduplication
    qlist = []
    pre_char = ""
    for char in text:
        if char != pre_char:
            qlist.append(char)
        pre_char = char
    text = "".join(qlist)

    return text


def predict(model, tokenizer, config, context, condition):
    """预测函数"""
    model.eval()
    with torch.no_grad():
        inputs = text_to_ids(tokenizer, context, condition,
                             is_condition_first=config.is_condition_first,
                             is_unilm_mask=config.is_unilm_mask,
                             max_sequence_len=config.max_sequence_len,
                             max_condition_len=config.max_condition_len,
                             max_target_len=config.max_target_len)
        input_ids, attention_mask, token_type_ids = (
            inputs["input_ids"].unsqueeze(0).to(config.device),
            inputs["attention_mask"].unsqueeze(0).to(config.device),
            inputs["token_type_ids"].unsqueeze(0).to(config.device)
        )
        current_len = input_ids.size(1)
        bos_token_id = tokenizer._convert_token_to_id("[CLS]")
        pad_token_id = tokenizer._convert_token_to_id("[PAD]")
        eos_token_id = tokenizer._convert_token_to_id("[SEP]")
        result = model.generate(input_ids,
                                max_length=current_len + config.max_target_len,
                                num_beams=1,
                                early_stopping=True,
                                bos_token_id=bos_token_id,
                                pad_token_id=pad_token_id,
                                eos_token_id=eos_token_id,
                                token_type_ids=token_type_ids,
                                attention_mask=attention_mask)
        predicts_ids = result.data.cpu().numpy().tolist()
        results = []
        for i, predict_ids in enumerate(predicts_ids):
            result_text = ids_to_text(predict_ids[current_len:], tokenizer)
            results.append(result_text)

    return results


def evaluate(model, tokenizer, config, dataset):
    """评估函数"""
    rouge_l_list = []
    generate_results = []
    for id, sample in enumerate(tqdm(dataset)):
        context, condition, target = sample['context'], sample['condition'], sample['target']
        results = predict(model, tokenizer, config,
                          context=context, condition=condition)
        # logger.info("results: {}, targets: {}".format(results, [target]))
        generate_results.append({
            "result": results[0],
            "target": target
        })
        rouge_l_list.append(rouge_l(results[0], target))  # 计算F1

    with open(os.path.join(root, 'logs/evaluate_results.json'), mode='w', encoding="utf-8") as fw:
        json.dump(generate_results, fw, ensure_ascii=False, indent=2)

    return np.average(rouge_l_list)


def main(do_eval=False):
    """主函数"""
    config = HyperParametersConfig()
    config.model_name_or_path = "/home/yckj2453/nlp_space/kbqa/gpt_generation/models/2021-01-19"
    logger.info("Load trained model.")
    tokenizer = BertTokenizer.from_pretrained(config.model_name_or_path)
    model = CustomGPTGeneration.from_pretrained(config.model_name_or_path)
    model.to(config.device)

    if do_eval:
        evaluate_dataset = torch.load(config.dataset_path)['valid']
        f1 = evaluate(model, tokenizer, config, dataset=evaluate_dataset)
        logger.info("F1 results: {}".format(f1))
    else:
        predict()


if __name__ == "__main__":
    logging.basicConfig(format='[%(asctime)s %(filename)s:%(lineno)s] %(message)s', level=logging.INFO,
                        filename=None, filemode='a')
    logger.info("root dir: {}".format(root))
    main(do_eval=True)
