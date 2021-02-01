"""
# 复旦 fastHan库使用
"""
from fastHan import FastHan


def fasthan_cws():
    model = FastHan()
    sentence = "郭靖是金庸笔下的一名男主。"
    result = model(sentence, 'CWS')
    print(result)


def kpe():
    from transformers import BertConfig
    from src.models.knowledge_point_extraction import KnowledgePointExtractionModel

    config = BertConfig.from_pretrained("hfl/chinese-bert-wwm")
    config.crf_labels = {0: "S", 1: "B", 2: "M", 3: "E", 4: "<pad>"}
    config.mlp_layer_sizes = [config.hidden_size,
                              config.hidden_size//2,
                              config.hidden_size//2//2,
                              config.hidden_size//2//2//2,
                              len(config.crf_labels)]

    model = KnowledgePointExtractionModel(config=config)
    pass


if __name__ == "__main__":
    kpe()
