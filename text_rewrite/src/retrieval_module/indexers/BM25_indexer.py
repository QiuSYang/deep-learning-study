#!/usr/bin/env python3

import jieba
import faiss
import logging
import json
import math
from tqdm import tqdm
from gensim.models import KeyedVectors

from src.retrieval_module.indexers.faiss_indexers import *

logger = logging.getLogger(__name__)


def tokenize_spt(text):

    sp_token = ['<img>', '<url>', '<sos>', '<eos>', '<num>']

    resp_list = list()
    tmp_list = jieba.cut(text, cut_all=False)

    seg_list = list(tmp_list)
    i = 0

    while i < len(seg_list):
        if ''.join(seg_list[i:i + 3]) in sp_token:
            resp_list.append(''.join(seg_list[i:i + 3]))
            i = i + 3
        else:
            resp_list.append(''.join(seg_list[i]))
            i = i + 1

    return resp_list


class BM25Indexer(object):
    
    def __init__(self, w2v_model, index_top_n=10, index_threshold=0.95):
        self.index = None
        self.w2v_model = w2v_model
        self.word_search_cache = {}
        self.docs = []
        self.doc_scores = []
        self.word_doc_dict = {}
        self.words = []
        self.f = []
        self.df = {}
        self.idf = {}
        self.k1 = 1.5
        self.b = 0.75
        self.index_top_n = index_top_n
        self.index_threshold = index_threshold
        if self.w2v_model:
            self.FEATURE_SIZE = self.w2v_model.FEATURE_SIZE

        signals = "！？｡＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】" \
                  "〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏.!#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~"
        self.signals = set(tokenize_spt(signals))

    def add_docs(self, docs):
        for doc in docs:
            doc = [word for word in doc if word not in self.signals]
            self.docs.append(doc)

    def create_index(self):
        print("create index for BM25")
        for index, doc in enumerate(self.docs):
            if self.w2v_model:
                for word in doc:                    
                    if word not in self.word_doc_dict:
                        self.word_doc_dict[word] = set([index])
                        if word in self.w2v_model:
                            self.words.append(word)
                        else:
                            pass
                            # print(word)
                    else:
                        self.word_doc_dict[word].add(index)
            else:
                for word in doc:                    
                    if word not in self.word_doc_dict:
                            self.word_doc_dict[word] = set([index])
                    else:
                        self.word_doc_dict[word].add(index)

            word_dict = {}
            for word in doc:
                if word not in word_dict:
                    word_dict[word] = 0
                word_dict[word] += 1
            self.f.append(word_dict)
            for word, count in word_dict.items():
                if word not in self.df:
                    self.df[word] = 0
                self.df[word] += 1

        self.D = len(self.docs)
        self.avgdl = sum([len(doc) + 0.0 for doc in self.docs]) /self.D
        for word, count in self.df.items():
            self.idf[word] = math.log(self.D - count + 0.5) - math.log(count + 0.5)
        print("words count:", len(self.words))

        if self.w2v_model:
            self.index = faiss.IndexFlatIP(self.FEATURE_SIZE)
            for index, word in tqdm(enumerate(self.words)):
                if word in self.w2v_model:
                    v = self.w2v_model.word_vector(word, use_norm=True)
                    self.index.add(v.reshape((1, self.FEATURE_SIZE)))
                else:
                    print(word)

        for doc in self.docs:
            self.doc_scores.append(self.cal_base_score(doc))

    def get_similar_words(self):
        import re
        prog = re.compile("[0-9一二三四五六七八九十零两]")
        ws = set()
        for word in self.words:
            flag = prog.search(word)
            sims = [word]
            if word in ws:
                continue
            ws.add(word)
            v = self.w2v_model.word_vector(word, use_norm=True)            
            D, I = self.index.search(np.array([v]), 100)
            for d, i in zip(D[0], I[0]):
                if d > self.index_threshold and word != self.words[i]:
                    if flag and prog.search(self.words[i]):
                        continue
                    ws.add(self.words[i])
                    sims.append((d, self.words[i]))
            if len(sims) > 1:
                print(sims)

    def cal_base_score(self, doc):
        doc_dict = {}
        for word in doc:
            if word not in doc_dict:
                doc_dict[word] = 0
            doc_dict[word] += 1
        base_score = 0
        d = len(doc)
        for word, count in doc_dict.items():
            if word not in self.idf:
                continue
            base_score += (self.idf[word] * count * (self.k1 + 1)
                                      / (count + self.k1 * (1 - self.b + self.b * d / self.avgdl)))
        return base_score

    def sim(self, doc, index):
        base_score = self.cal_base_score(doc)
        score = 0
        for word in doc:
            if word not in self.f[index]:
                continue
            d = len(self.docs[index])
            score += (self.idf[word] * self.f[index][word] * (self.k1 + 1)
                      / (self.f[index][word] + self.k1 * (1 - self.b + self.b * d
                                                          / self.avgdl)))
        return score / max(base_score, self.doc_scores[index])

    def search_word_knn(self, word):
        if word in self.word_search_cache:
            return self.word_search_cache[word]
        if word not in self.w2v_model:
            if word in self.word_doc_dict:
                result = [[1.0, word]]
                self.word_search_cache[word] = result
                return result
            return []
        v = self.w2v_model.word_vector(word, use_norm=True).reshape((1, self.FEATURE_SIZE))
        D, I = self.index.search(v, self.index_top_n)
        result = []
        for d, i in zip(D[0], I[0]):
            if d > self.index_threshold:
                result.append([d, self.words[i]])
        self.word_search_cache[word] = result
        return result

    def search_knn(self, doc, top_n):
        base_score = self.cal_base_score(doc)
        word_scores = {}
        vecs = []
        for word in doc:
            result = self.search_word_knn(word)
            for (score, word) in result:
                if word not in word_scores or word_scores[word] < score:
                    word_scores[word] = score
        if not word_scores:
            print("not valid question")
            return None

        # print(doc, base_score, word_scores)
        scores = {}
        for word, simlarity in word_scores.items():
            if word not in self.word_doc_dict:
                continue
            for doc_index in self.word_doc_dict[word]:
                if doc_index not in scores:
                    scores[doc_index] = 0
                d = len(self.docs[doc_index])
                scores[doc_index] += simlarity * (self.idf[word] * self.f[doc_index][word] * (self.k1 + 1) /
                                     (self.f[doc_index][word] + self.k1 * (1 - self.b + self.b * d / self.avgdl)))
        scores = {key: value / max(base_score, self.doc_scores[key]) for key, value in scores.items()}
        scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # for doc_index, score in scores[:top_n]:
        #     print(score, "Q:" + "".join(self.docs[doc_index]), "score:", self.doc_scores[doc_index], "doc:", self.f[doc_index])

        return [[score, doc_index] for doc_index, score in scores[:top_n]]
