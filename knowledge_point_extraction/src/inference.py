"""
# model inference
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import logging
import json
from rouge import Rouge  # F1---评测指标
import torch
from transformers import BertTokenizer
from src.models.knowledge_point_extraction import KnowledgePointExtractionModel

logger = logging.getLogger(__name__)
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ID2LABEL = {0: "<pad>", 1: "S", 2: "B", 3: "M", 4: "E"}
LABEL2ID = {"<pad>": 0, "S": 1, "B": 2, "M": 3, "E": 4}


def get_data_from_json_file(json_file):
    with open(json_file, mode='r', encoding='utf-8') as fp:
        return json.load(fp)


def text2id(tokenizer, context, max_encode_len=512):
    """context text 2 ids"""
    process_context = context.replace("\n", " ").replace("\t", " ").replace("\\", "")
    context_tokens = tokenizer.tokenize(process_context)
    if len(context_tokens) > max_encode_len:
        context_tokens = context_tokens[:max_encode_len]

    input_ids = tokenizer.convert_tokens_to_ids(context_tokens)
    mask_ids = [1.0] * len(input_ids)

    extra = max_encode_len - len(input_ids)
    if extra > 0:
        input_ids += [tokenizer.pad_token_id] * extra
        mask_ids += [0.0] * extra

    return {
        "input_ids": torch.tensor(input_ids).long(),
        "attention_mask": torch.tensor(mask_ids).float()
        }


def get_answers(predict_label, context):
    """获取知识点列表"""
    answers = []
    label_str = str()
    for label, char in zip(predict_label, context):
        if label == LABEL2ID["B"]:
            if len(label_str) > 0:
                answers.append(label_str)
                label_str = str()
            label_str = char
        elif label == LABEL2ID["M"] or label == LABEL2ID["E"]:
            if len(label_str) > 0:
                label_str += char
        else:
            if len(label_str) > 0:
                answers.append(label_str)
                label_str = str()
    if len(label_str) > 0:
        answers.append(label_str)

    return answers


def predict(model, tokenizer, context, device="cpu", max_encode_len=512):
    inputs = text2id(tokenizer, context, max_encode_len=max_encode_len)
    for key, value in inputs.items():
        inputs[key] = value.unsqueeze(0).to(device)

    results = model(**inputs)["pred"]

    return results[0]


def evaluate():
    pass


def main(do_eval=False):
    logger.info("Load self trained model")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = os.path.join(root, "checkpoints")
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    model = KnowledgePointExtractionModel.from_pretrained(model_dir)
    model.to(device)

    logger.info("Get test dataset")
    test_data_path = os.path.join(root, "data/juesai_1011.json")
    test_dataset = get_data_from_json_file(test_data_path)
    if do_eval:
        pass
    else:
        rouge_obj = Rouge()
        for id, single in enumerate(test_dataset):
            context = single["text"]
            labels = []
            qas = single["annotations"]
            for qa in qas:
                labels.append(qa["A"])

            predict_label = predict(model, tokenizer, context, device=device, max_encode_len=512)
            logger.info(predict_label)

            answers = get_answers(predict_label, context)
            logger.info(answers)

            for answer in answers:
                if len(answer) < 5:
                    continue
                for label in labels:
                    f1 = rouge_obj.get_scores(hyps=" ".join(label), refs=" ".join(answer))
                    logger.info("F1 value: {}".format(f1))


if __name__ == '__main__':
    logging.basicConfig(format='[%(asctime)s %(filename)s:%(lineno)s] %(message)s',
                        level=logging.INFO,
                        filename=None,
                        filemode='a')
    logger.info("root dir: {}".format(root))
    main()
