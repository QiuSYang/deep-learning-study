"""
# 使用bert模型进行文本向量提取
"""
import os
import argparse
import logging
import numpy as np
import torch
from transformers import (WEIGHTS_NAME, BertConfig,
                          BertForSequenceClassification, BertTokenizer,
                          RobertaConfig, XLNetConfig,
                          XLNetForSequenceClassification,
                          XLNetTokenizer,
                          AlbertForSequenceClassification)

logging.basicConfig(
    format="%(asctime)s - %(filename)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
_logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    # bert ernie bert_wwm bert_wwwm_ext
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    'roberta': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'albert': (BertConfig, AlbertForSequenceClassification, BertTokenizer)
}


class BertTextToVector(object):
    def __init__(self, args):
        self.args = args
        # 设置句子的最大长度
        self.args.max_length = 128
        self.args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        _logger.info("Load trained model.")
        self.args.model_type = self.args.model_type.lower()
        config_class, model_class, tokenizer_class = MODEL_CLASSES[self.args.model_type]
        self.config = config_class.from_pretrained(
            self.args.config_name if self.args.config_name else self.args.model_name_or_path)
        # 设置输出隐层
        self.config.output_hidden_states = self.args.output_hidden_states

        self.tokenizer = tokenizer_class.from_pretrained(
            args.tokenizer_name if args.tokenizer_name else args.model_name_or_path)
        self.model = model_class.from_pretrained(args.model_name_or_path,
                                                 from_tf=bool('.ckpt' in args.model_name_or_path),
                                                 config=self.config)

        self.model.to(self.args.device)

    def get_sentence_embedding(self, sentence):
        """获取句子的embedding向量"""
        self.model.eval()
        with torch.no_grad():
            inputs = self.tokenizer.encode_plus(sentence, max_length=self.args.max_length)
            for key in inputs.keys():
                inputs[key] = torch.tensor(inputs.get(key)).unsqueeze(dim=0).to(self.args.device)
            # 设置输出隐层
            # inputs['output_hidden_states'] = self.args.output_hidden_states  # 3.0版就可以使用
            # 解字典传参
            outputs = self.model(**inputs)

            if self.args.multi_layer_mean:
                # 获取每层 cls token做均值
                # token_encoders: [num_layer, batch, seq_length, hidden_size]
                token_encoders = torch.stack(outputs[1], dim=0)
                # cls_token_encoder: [num_layer, batch, 1, hidden_size]
                cls_token_encoder = token_encoders[:, :, 0, :]
                # 先删除batch维度, 因为每次batch都为1
                cls_token_encoder_mean = torch.mean(cls_token_encoder.squeeze(dim=1), dim=0)

                # 将tensor转为ndarray
                return cls_token_encoder_mean.data.cpu().numpy()
            else:
                # 获取最后一层的cls token embedding作为句子的表征
                last_cls_token_encoder = outputs[1][-1][:, 0, :]
                # 删除batch维度, 因为每次batch都为1
                last_cls_token_encoder = last_cls_token_encoder.squeeze(dim=0)
                # 将tensor转为ndarray
                return last_cls_token_encoder.data.cpu().numpy()

            return None


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    parser = argparse.ArgumentParser("bert 进行text to vector参数设置")

    parser.add_argument("--model_type", default="bert", type=str,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path",
                        default="./transformers/jddc_output/bert/",
                        type=str,
                        help="Path to pre-trained model or shortcut name selected.")
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--output_hidden_states", default=True, type=bool,
                        help="是否输出bert的隐层")
    parser.add_argument("--multi_layer_mean", default=True, type=bool,
                        help="是否使用多层hidden平均值作为句子的表征")

    args = parser.parse_args()

    main = BertTextToVector(args)

    test_sentence = "你能不能不要跟我车轱辘话来回说有意思吗嗯你说这是不是你们的问题吗这和我有什么关系" \
                    "让退款也是你们说发不了货也是你们发货的也是你们然后咋说都是你们说娜娜我来我作为消费" \
                    "者我的权益呢嗯你们就这么对待顾客的是吧 那申请退货有什么意义呢，" \
                    "者我的权益呢嗯你们就这么对待顾客的是吧 那申请退货有什么意义呢"
    _logger.info("sentence length: {}".format(len(test_sentence)))

    sentence_embedding = main.get_sentence_embedding(test_sentence)

    _logger.info("text: {}, embedding: {}".format(test_sentence, sentence_embedding))
