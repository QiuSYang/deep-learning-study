"""
# 主程序
"""
import os
import logging
import platform
import datasets
import torch
from transformers import BertTokenizer

from utils import *
from drmc.drmc_data_processor import DRMCDataProcessor

logger = logging.getLogger(__name__)
project_path = os.path.dirname(__file__)
if platform.system() == "Windows":
    project_path = project_path.replace('/', '\\')


def drmc_data_structuralization():
    script_path = 'drmc\\drmc.py'
    dataset_dir = "D:\\BaiduNetdiskDownload\\Tianchi2020ChineseMedicineQAG\\DataSet\\MultiTask"
    cache_dir = ".\\data_cache\\"
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    dataset = datasets.load_dataset(
                    script_path, cache_dir=cache_dir,
                    data_files={
                        datasets.Split.TRAIN: [
                            os.path.join(dataset_dir, "DRCD\\DRCD_training.json"),
                            os.path.join(dataset_dir, "CMRC\\cmrc2018_train.json"),
                            os.path.join(dataset_dir, "DRCD\\DRCD_dev.json"),
                            os.path.join(dataset_dir, "CMRC\\cmrc2018_dev.json")
                        ],
                        datasets.Split.VALIDATION: [
                            os.path.join(dataset_dir, "DRCD\\DRCD_test.json"),
                            os.path.join(dataset_dir, "CMRC\\cmrc2018_trial.json")
                        ]
                    })

    print(dataset['train'][0])
    print(dataset['validation'][0])

    tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext-large")

    processor = DRMCDataProcessor(
        tokenizer
    )

    dataset = processor.process(dataset)

    print(dataset['train'][0])
    print(dataset['validation'][0])

    columns = ["input_ids", "input_mask", "input_seg", "decode_input", "decode_target"]  # 指定训练需要使用的数据字段
    dataset.set_format(type='torch', columns=columns)

    dataset_save_path = os.path.join(project_path, "data_cache/drmc_dataset.pt")
    torch.save(dataset, dataset_save_path)


if __name__ == "__main__":
    log_to_init()
    logger.info("工作路径: {}".format(project_path))
    drmc_data_structuralization()
