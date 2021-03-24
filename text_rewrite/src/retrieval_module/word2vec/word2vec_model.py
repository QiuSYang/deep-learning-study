"""
# word 2 vector
  1. 将word2vec词向量表读取, 使用key - value 方式存储, 通过numpy的 .npy存储, load时候就会被加速,
    load_word2vec_format load非常慢
"""
import os
import logging
import numpy as np
from gensim.models import KeyedVectors

logger = logging.getLogger(__name__)
work_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class CustomWord2Vec(object):
    """自定义 word 2 vector"""
    def __init__(self):
        self.keys = []
        self.features = None
        self.key_feature_map = {}
        self.FEATURE_SIZE = 200

    def load(self, word_file, feature_file):
        """从npy文件中获取feature, 加速模型load"""
        with open(word_file, mode='r') as fp:
            keys = []
            for word in fp.readlines():
                word = word.strip()
                keys.append(word)
        features = np.load(feature_file)
        if features.shape[0] != len(keys):
            raise Exception("Words not match features")

        self.keys = keys
        self.features = features
        for idx, key in enumerate(self.keys):
            self.key_feature_map[key] = self.features[idx]

    def keyed_vectors_load_model(self, word2vec_file):
        """gensim load word2vec model"""
        model = KeyedVectors.load_word2vec_format(word2vec_file)
        keys = []
        features = []
        for key in model.index2word:
            feature = model.get_vector(key)
            # 归一化
            # feature = feature / np.linalg.norm(feature)
            keys.append(key)
            features.append(feature)

        self.keys = keys
        self.features = features
        for idx, key in enumerate(self.keys):
            self.key_feature_map[key] = self.features[idx]

    def __contains__(self, key):
        return key in self.key_feature_map

    def save(self, word_file, feature_file):
        with open(word_file, "w") as f:
            content = '\n'.join(self.keys)
            f.write(content)
        np.save(feature_file, self.features)

    def word_vector(self, word, use_norm=False):
        return self.key_feature_map[word]

    def get_vector(self, word):
        return self.word_vector(word)

    def normalization(self, arr: np.ndarray):
        min_value, max_value = np.min(arr, axis=-1), np.max(arr, axis=-1)
        range = max_value - min_value
        normalization = (arr - min_value) / range
        return normalization

    def get_sentence_vector(self, tokens, is_normalization=False):
        """获取句子的embedding向量"""
        z = np.zeros(200, dtype=np.float32)
        count = 0

        for token in tokens:
            if self.__contains__(token):
                a = self.get_vector(token)
                z = a + z
                count = count + 1

        if count == 0:
            count = 1
        vector = (z / count).astype('float32')

        if not is_normalization:
            return vector
        else:
            return vector / np.linalg.norm(vector)


if __name__ == "__main__":
    logging.basicConfig(format="[%(asctime)s %(filename)s: %(lineno)s] %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",
                        level=logging.INFO,
                        filename=None,
                        filemode="a")  # set logging
    logger.info("Work root: {}".format(work_root))

    # 腾讯训练好的word2vector的词表文件
    tencent_word2vec_file = os.path.join(work_root, "models/tencent_words/1000000-small.txt")
    model_obj = CustomWord2Vec()
    logger.info("Load origin words")
    model_obj.keyed_vectors_load_model(tencent_word2vec_file)

    words_file = os.path.join(work_root, "models/tencent_words/1000000-small.words")
    features_file = os.path.join(work_root, "models/tencent_words/1000000-small.npy")
    logger.info("Save key value to list and .npy")
    model_obj.save(words_file, features_file)
