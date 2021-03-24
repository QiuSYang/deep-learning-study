"""
# 构建QA索引库
    使用天池中医问题生成数据构建QA知识库
"""
import os
import logging
from tqdm import tqdm
import jieba
import numpy as np
from src.retrieval_module.word2vec.word2vec_model import CustomWord2Vec
from src.retrieval_module.indexers.faiss_indexers import (
    DenseIndexer,
    DenseFlatIndexer,
    DenseHNSWFlatIndexer
)
from src.utils.utils import *

logger = logging.getLogger(__name__)
work_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INDEXER = {
    "flat": DenseFlatIndexer,
    "hnsw_flat": DenseHNSWFlatIndexer
}


class BuildQAIndexes(object):
    def __init__(self, word2vector_model: CustomWord2Vec, index_obj: DenseIndexer):
        self.w2v = word2vector_model  # 词转向量模型
        self.indexer = index_obj  # 构建索引库模型

    def build_cm_indexes(self, data_file, cm_qa_file, cm_qa_indexes_file):
        """构建中医QA数据索引库"""
        cm_data = load_json_data(data_file)
        # 收集数据
        cm_qas = []
        buffer = []
        index, error_num = 0, 0
        for id, single in enumerate(tqdm(cm_data)):
            try:
                for qa in single["annotations"]:
                    cm_qas.append(qa)  # 将qa对存入数据库中
                    ques_vec = self._query2vector(qa["Q"])
                    # temp = np.linalg.norm(ques_vec)
                    buffer.append((index, ques_vec))
                    index += 1
                    if len(buffer) == 50000:
                        self.indexer.index_data(buffer)  # 将索引入库
                        buffer = []  # 清空缓存
            except KeyError:
                error_num += 1
                continue

        logger.info("Valid data nums: {}".format(index))
        logger.info("Error nums: {}".format(error_num))

        self.indexer.index_data(buffer)  # 最后剩余索引入库

        self.indexer.serialize(cm_qa_indexes_file)  # 将索引库保存到文件

        save_data_to_json(cm_qas, cm_qa_file)  # 将原始QA对数据保存到文件中

    def _query2vector(self, query, is_normalization=True):
        """query向量化"""
        query_tokens = list(jieba.cut(query, cut_all=False))
        vector = self.w2v.get_sentence_vector(query_tokens, is_normalization=is_normalization)
        return vector


if __name__ == '__main__':
    logging.basicConfig(format="[%(asctime)s %(filename)s: %(lineno)s] %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",
                        level=logging.INFO,
                        filename=None,
                        filemode="a")  # set logging

    logger.info("Get word2vector model.")
    words_file = os.path.join(work_root, "models/tencent_words/1000000-small.words")
    features_file = os.path.join(work_root, "models/tencent_words/1000000-small.npy")
    w2v_model = CustomWord2Vec()
    w2v_model.load(word_file=words_file, feature_file=features_file)

    logger.info("Set indexer")
    indexer = INDEXER["flat"](vector_sz=200)

    logger.info("Build Chinese Medical Indexes")
    build_qa_obj = BuildQAIndexes(w2v_model, indexer)
    data_file = os.path.join(work_root, "data/tianchi_chinese_medical.json")
    cm_qa_file = os.path.join(work_root, "data/tianchi_chinese_medical_qas.json")
    cm_qa_indexes_file = os.path.join(work_root, "data/tianchi_chinese_medical_qas")
    build_qa_obj.build_cm_indexes(data_file, cm_qa_file=cm_qa_file, cm_qa_indexes_file=cm_qa_indexes_file)
