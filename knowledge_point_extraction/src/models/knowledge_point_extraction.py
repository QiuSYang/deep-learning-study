"""
# 知识点抽取模型, 参考FastHan库分词模型
"""
import os
import logging
import torch
import torch.nn.functional as F
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
        # MLP输入输出向量size, mlp_layer_sizes: [hidden_size, middle_size1, middle_size2, len(config.crf_labels)]
        self.kpe_mlp = MLP(size_layer=config.mlp_layer_sizes,
                           activation='relu',
                           output_activation=None)
        # crf_labels = {0: "S", 1: "B", 2: "M", 3: "E", 4: "<pad>"}
        trans = allowed_transitions(tag_vocab=config.crf_labels, include_start_end=True)
        self.kpe_crf = ConditionalRandomField(num_tags=len(config.crf_labels),
                                              include_start_end_trans=True,
                                              allowed_transitions=trans)

    def forward(self,
                input_ids,
                labels,
                attention_mask=None):
        """前向传播"""
        bert_outputs = self.embedding(input_ids, attention_mask=attention_mask, return_dict=True)
        embedding_output = bert_outputs.last_hidden_state

        mlp_outputs = self.kpe_mlp(embedding_output)
        logits = F.softmax(mlp_outputs, dim=-1)

        if attention_mask is None:
            attention_mask = input_ids.ne(0)
        crf_outputs = self.kpe_crf(logits, labels, mask=attention_mask)

        return {"loss": crf_outputs}
