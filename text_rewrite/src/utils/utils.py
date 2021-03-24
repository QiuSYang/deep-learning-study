"""
# 检索共有函数
"""
import logging
import json
import pickle

logger = logging.getLogger(__name__)


def load_json_data(json_file):
    with open(json_file, mode="r", encoding="utf-8") as fp:
        return json.load(fp)


def save_data_to_json(data_obj, json_file):
    with open(json_file, mode="w", encoding="utf-8") as fw:
        json.dump(data_obj, fw, ensure_ascii=False, indent=2)


def load_pickle_data(pkl_file):
    with open(pkl_file, mode='rb') as fp:
        return pickle.load(fp)


def save_data_to_pkl(obj, pickle_file_path):
    """将数据保存到pickle压缩文件中"""
    with open(pickle_file_path, mode='wb') as fw:
        pickle.dump(obj, fw)

