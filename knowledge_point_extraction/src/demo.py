"""
# 复旦 fastHan库使用
"""
from fastHan import FastHan


def fasthan_cws():
    model = FastHan()
    sentence = "郭靖是金庸笔下的一名男主。"
    result = model(sentence, 'CWS', use_dict=False)
    print(result)


def kpe():
    from transformers import BertConfig
    from src.models.knowledge_point_extraction import KnowledgePointExtractionModel

    config = BertConfig.from_pretrained("hfl/chinese-bert-wwm")
    config.crf_labels = {0: "S", 1: "B", 2: "M", 3: "E", 4: "<pad>"}
    # config.mlp_layer_sizes = [config.hidden_size,
    #                           config.hidden_size//2,
    #                           config.hidden_size//2//2,
    #                           config.hidden_size//2//2//2,
    #                           len(config.crf_labels)]
    mlp_hidden_num = 5
    mlp_layer_sizes = []
    for i in range(mlp_hidden_num):
        if i == mlp_hidden_num - 1:
            mlp_layer_sizes.append(len(config.crf_labels))
        else:
            mlp_layer_sizes.append(config.hidden_size//(2**i))
    config.mlp_layer_sizes = mlp_layer_sizes

    model = KnowledgePointExtractionModel(config=config)
    pass


def transformer_output():
    import torch
    from transformers.models.bert.modeling_bert import BertForPreTrainingOutput

    output = BertForPreTrainingOutput(
        loss=10,
        prediction_logits=torch.tensor([0.1, 0.2, 0.7])
    )

    if isinstance(output, dict):
        print(output.get("loss"))


if __name__ == "__main__":
    # kpe()
    # fasthan_cws()
    transformer_output()
