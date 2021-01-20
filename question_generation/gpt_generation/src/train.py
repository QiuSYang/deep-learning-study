"""
# GPT generation train codes
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import logging
import random
import numpy as np
import torch
from transformers import (
    BertTokenizer,
    Trainer,
    TrainingArguments,
    HfArgumentParser
)

from src.models.custom_gpt import CustomGPTGeneration
from src.datasets.chinese_medical_dataset import ChineseMedicalDataset
from src.hyper_parameters import DataArguments, ModelArguments, HyperParametersConfig

logger = logging.getLogger(__name__)
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def set_os_environ():
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "7001"


def main():
    config = HyperParametersConfig()
    # set_os_environ()
    # if config.do_train and torch.cuda.device_count() > 1:
    #     # 分布式初始化
    #     torch.distributed.init_process_group(backend="nccl", rank=0, world_size=1,
    #                                          init_method='tcp://localhost:7002')
    # set_seed(config.seed)  # Trainer内部已经包含
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    # config_dict = HyperParametersConfig().__dict__
    # print(config_dict)
    model_args, data_args, training_args = parser.parse_dict(config.__dict__)

    logger.info("Load pre-training model.")
    tokenizer = BertTokenizer.from_pretrained(model_args.model_name_or_path)
    model = CustomGPTGeneration.from_pretrained(model_args.model_name_or_path)

    # Get datasets
    logger.info("Loading dataset.")
    data = torch.load(data_args.dataset_path)
    train_dataset = ChineseMedicalDataset(data=data["train"],
                                          tokenizer=tokenizer,
                                          max_sequence_len=data_args.max_sequence_len,
                                          max_condition_len=data_args.max_condition_len,
                                          max_target_len=data_args.max_target_len,
                                          is_right_pad=data_args.is_right_pad,
                                          is_condition_first=data_args.is_condition_first,
                                          is_unilm_mask=data_args.is_unilm_mask) if training_args.do_train else None
    valid_dataset = ChineseMedicalDataset(data=data["valid"],
                                          tokenizer=tokenizer,
                                          max_sequence_len=data_args.max_sequence_len,
                                          max_condition_len=data_args.max_condition_len,
                                          max_target_len=data_args.max_target_len,
                                          is_right_pad=data_args.is_right_pad,
                                          is_condition_first=data_args.is_condition_first,
                                          is_unilm_mask=data_args.is_unilm_mask) if training_args.do_eval else None

    logger.info("Initialize Trainer.")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
    )

    logger.info("Training start.")
    if training_args.do_train:
        trainer.train(model_path=model_args.model_name_or_path
                      if os.path.isdir(model_args.model_name_or_path) else None)
        trainer.save_model()
        if trainer.is_world_master():
            tokenizer.save_pretrained(training_args.output_dir)

        # Evaluation
        results = {}
        if training_args.do_eval and training_args.local_rank in [-1, 0]:
            logger.info("*** Evaluate ***")

            eval_output = trainer.evaluate()

            output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
            with open(output_eval_file, "w") as writer:
                logger.info("***** Evaluate results *****")
                for key in sorted(eval_output.keys()):
                    logger.info("{} = {}".format(key, str(eval_output[key])))
                    writer.write("{} = {}\n".format(key, str(eval_output[key])))

            results.update(eval_output)

        return results


if __name__ == "__main__":
    logging.basicConfig(format='[%(asctime)s %(filename)s:%(lineno)s] %(message)s', level=logging.INFO,
                        filename=None, filemode='a')
    logger.info("root dir: {}".format(root))
    results = main()
    logger.info("Train results: {}".format(results))
