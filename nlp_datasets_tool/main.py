"""
# 主程序
"""
import os
import logging
import platform
import datasets

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

    print(dataset['train'][0], dataset['validation'][0])


if __name__ == "__main__":
    log_to_init()
    logger.info("工作路径: {}".format(project_path))
    drmc_data_structuralization()
