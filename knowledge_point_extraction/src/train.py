"""
# 训练文件
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import logging
from transformers import (
    BertTokenizer,
    BertConfig,
    Trainer,
    TrainingArguments
)
from src.models.knowledge_point_extraction import KnowledgePointExtractionModel
from src.datasets.tianchi_zhongyi_answer_extraction_datasets import ChineseMedicalAnswerExtractionDataset

logger = logging.getLogger(__name__)
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    id2label = {0: "<pad>", 1: "S", 2: "B", 3: "M", 4: "E"}  # label
    label2id = {}
    for key, value in id2label.items():
        label2id[value] = int(key)
    logger.info("Set model config.")
    config = BertConfig.from_pretrained("hfl/chinese-bert-wwm")
    config.crf_labels = id2label
    mlp_hidden_num = 5
    mlp_layer_sizes = []
    for i in range(mlp_hidden_num):
        if i == mlp_hidden_num - 1:
            mlp_layer_sizes.append(len(config.crf_labels))
        else:
            mlp_layer_sizes.append(config.hidden_size // (2 ** i))
    config.mlp_layer_sizes = mlp_layer_sizes
    config.num_hidden_layers = 4  # word2vec hidden layer size

    logger.info("Load pre-training model.")
    tokenizer = BertTokenizer.from_pretrained("hfl/chinese-bert-wwm")
    model = KnowledgePointExtractionModel.from_pretrained("hfl/chinese-bert-wwm", config=config)
    # for name, weight in zip(model.named_parameters(), model.parameters()):
    #     print("name: {} --- weight: {}".format(name, weight))

    logger.info("Loading dataset.")
    train_data_path = os.path.join(root, "data/round1_train_0907.json")
    valid_data_path = os.path.join(root, "data/round1_test_0907.json")
    train_dataset = ChineseMedicalAnswerExtractionDataset(train_data_path, tokenizer,
                                                          label2id=label2id, max_enc_len=512)
    valid_dataset = ChineseMedicalAnswerExtractionDataset(valid_data_path, tokenizer,
                                                          label2id=label2id, max_enc_len=512)

    logger.info("Initialize Trainer.")
    train_args = TrainingArguments(
        output_dir=os.path.join(root, "checkpoints"),
        logging_dir=os.path.join(root, "logs"),
        per_device_train_batch_size=96,
        per_device_eval_batch_size=32,
        max_steps=5000,
        eval_steps=1000,
        save_steps=1000,
        # num_train_epochs=10,
        evaluation_strategy="steps",
        warmup_steps=500,
        logging_steps=50
    )
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
    )

    logger.info("Train model.")
    trainer.train()
    trainer.save_model()
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(train_args.output_dir)


if __name__ == '__main__':
    logging.basicConfig(format='[%(asctime)s %(filename)s:%(lineno)s] %(message)s',
                        level=logging.INFO,
                        filename=None,
                        filemode='a')
    logger.info("root dir: {}".format(root))
    main()
