"""
# 共有函数集
"""
import os
import logging
import json

logger = logging.getLogger(__name__)


def init_logs(log_file=None):
    """初始化日志"""
    logging.basicConfig(format="[%(asctime)s - %(filename)s - %(lineno)s]: %(levelname)s - %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",
                        level=logging.INFO,
                        filemode="a",
                        filename=log_file)
    logger.info("初始化日志配置.")


def load_json_obj(json_file):
    """获取json文件的数据"""
    with open(json_file, mode="r", encoding="utf-8") as fp:
        return json.load(fp)


def save_dict_obj(dict_obj, save_json_file):
    """将数据输出到json文件"""
    with open(save_json_file, mode="w", encoding="utf-8") as fw:
        json.dump(dict_obj, fw, ensure_ascii=False, indent=2)


def build_dict(corpus):
    """构建词表"""
    word_freq_dict = dict()
    for sentence in corpus:
        for word in sentence:
            if word not in word_freq_dict:
                word_freq_dict[word] = 0
            word_freq_dict[word] += 1

    word_freq_dict = sorted(word_freq_dict.items(), key=lambda x: x[1], reverse=True)  # 使用词频排序

    word2id_dict = dict()
    word2id_freq = dict()
    id2word_dict = dict()

    # 一般来说，我们把oov和pad放在词典前面，给他们一个比较小的id，这样比较方便记忆，并且易于后续扩展词表
    word2id_dict["[pad]"] = 0
    id2word_dict[0] = "[pad]"
    word2id_freq[0] = 1e10

    word2id_dict["[oov]"] = 1
    id2word_dict[1] = "[oov]"
    word2id_freq[1] = 1e10

    for word, freq in word_freq_dict:
        current_id = len(word2id_dict)
        word2id_dict[word] = current_id
        id2word_dict[current_id] = word
        assert current_id == word2id_dict[word]
        word2id_freq[word2id_dict[word]] = freq

    return word2id_freq, word2id_dict, id2word_dict
