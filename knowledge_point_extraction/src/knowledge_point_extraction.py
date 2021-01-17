"""
# 知识点抽取模型, 参考FastHan库分词模型
"""
import os
import logging
import torch
from transformers import (
    PreTrainedModel,
    BertModel,
    BertConfig
)
from fastNLP.modules import (
    MLP,
    ConditionalRandomField,
    allowed_transitions
)

logger = logging.getLogger(__name__)


class KnowledgePointExtractionModel(PreTrainedModel):
    """知识抽取---参照序列标注模型
        1. Embedding - 8 layer以下bert model,
        2. multi layer MLP 线性变换
        3. CRF layer 修正"""
    def __init__(self, config: BertConfig):
        super(KnowledgePointExtractionModel, self).__init__(config=config)

        self.embedding = BertModel(config=config)
        # MLP输入输出向量size
        self.kpe_mlp = MLP(size_layer=config.mlp_layer_sizes,
                           activation='relu',
                           output_activation=None)
        trans = allowed_transitions(tag_vocab=len(config.crf_labels), include_start_end=True)
        self.kpe_crf = ConditionalRandomField(num_tags=len(config.crf_labels),
                                              include_start_end_trans=True,
                                              allowed_transitions=trans)

    def forward(self,
                input_ids=None,
                attention_mask=None):
        """前向传播"""
        embedding_outputs = self.embedding(input_ids, attention_mask=attention_mask)
        mlp_outputs = self.kpe_mlp()
        crf_outputs = self.kpe_crf()
        pass
