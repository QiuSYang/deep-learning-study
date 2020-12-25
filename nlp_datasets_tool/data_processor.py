"""
# 数据预处理过程
"""
import os
import logging
import datasets
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


class DataProcessor(object):
    def __init__(self,
                 tokenizer: PreTrainedTokenizer,
                 max_encode_length=512,
                 max_decode_length=32):
        self.tokenizer = tokenizer
        self.max_encode_length = max_encode_length
        self.max_decode_length = max_decode_length

    def process(self, dataset: datasets.Dataset):
        raise NotImplementedError()

    def _convert_to_features(self, **kwargs):
        raise NotImplementedError()
