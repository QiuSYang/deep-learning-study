"""
# 相似度检索inference模块
"""
import os
import logging
import jieba
import numpy as np
from src.retrieval_module.indexers.faiss_indexers import (
    DenseIndexer,
    DenseFlatIndexer,
    DenseHNSWFlatIndexer
)
from src.retrieval_module.word2vec.word2vec_model import CustomWord2Vec
from src.utils.utils import *

logger = logging.getLogger(__name__)
work_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INDEXER = {
    "flat": DenseFlatIndexer,
    "hnsw_flat": DenseHNSWFlatIndexer
}


def load_indexer_w2v_qas(words_file, features_file, qas_file,
                         indexes_file, indexer_type="flat", vector_size=200):
    """导入索引检索器和文本向量化模型以及QAs对文档库"""
    indexer = INDEXER[indexer_type](vector_size)
    indexer.deserialize_from(indexes_file)

    w2v_model = CustomWord2Vec()
    w2v_model.load(word_file=words_file, feature_file=features_file)

    qas_document = load_json_data(qas_file)

    return indexer, w2v_model, qas_document


def get_retrieval_results(query: str, indexer: DenseIndexer, w2v: CustomWord2Vec, qas_docs: list, top_k: int = 10):
    """检索与query相似top_k向量"""
    query_tokens = list(jieba.cut(query, cut_all=False))
    # query_tokens_ = list(jieba.cut(query, cut_all=False))
    query_vector = w2v.get_sentence_vector(query_tokens, is_normalization=True)
    # query_vector_ = w2v.get_sentence_vector(query_tokens)
    query_vector = query_vector[np.newaxis, :]  # 扩围

    search_results = indexer.search_knn(query_vector, top_docs=top_k)
    I, D = search_results[0]
    cand_docs = list()
    for idx in I:
        cand_docs.append(qas_docs[idx])

    return cand_docs


if __name__ == '__main__':
    logging.basicConfig(format="[%(asctime)s %(filename)s: %(lineno)s] %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",
                        level=logging.INFO,
                        filename=None,
                        filemode="a")  # set logging

    words_file = os.path.join(work_root, "models/tencent_words/1000000-small.words")
    features_file = os.path.join(work_root, "models/tencent_words/1000000-small.npy")
    cm_qa_file = os.path.join(work_root, "data/tianchi_chinese_medical_qas.json")
    cm_qa_indexes_file = os.path.join(work_root, "data/tianchi_chinese_medical_qas")

    logger.info("Initialization Model")
    indexer, w2v_model, qas_document = load_indexer_w2v_qas(words_file, features_file,
                                                            qas_file=cm_qa_file,
                                                            indexes_file=cm_qa_indexes_file,
                                                            indexer_type="flat")
    logger.info("Get candidate documents")
    query = "什么类型的胆囊结石可不作治疗？"
    cand_docs = get_retrieval_results(query, indexer, w2v_model, qas_document)
    logger.info("candidate docs: {}".format(cand_docs))
