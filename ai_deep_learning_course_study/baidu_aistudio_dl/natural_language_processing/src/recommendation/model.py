"""
# 推荐系统模型
"""
import os
import logging
import math
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

logger = logging.getLogger(__name__)


class MovieRecommendModel(nn.Layer):
    def __init__(self, use_poster, use_mov_title, use_mov_cat, use_age_job,
                 vocab_sizes={}, fc_sizes=[]):
        super(MovieRecommendModel, self).__init__()

        # 将传入的name信息和bool型参数添加到模型类中
        self.use_mov_poster = use_poster  # 计算电影信息特征时是否使用海报, 现模型不使用
        self.use_mov_title = use_mov_title
        self.use_usr_age_job = use_age_job
        self.use_mov_cat = use_mov_cat
        self.fc_sizes = fc_sizes

        usr_embedding_dim = 32
        gender_embeding_dim = 16
        age_embedding_dim = 16

        job_embedding_dim = 16
        mov_embedding_dim = 16
        category_embedding_dim = 16
        title_embedding_dim = 32

        logger.info("define network layer for embedding usr info")
        USR_ID_NUM = vocab_sizes["max_usr_id"] + 1
        # 对用户ID做映射，并紧接着一个Linear层
        self.usr_emb = nn.Embedding(num_embeddings=USR_ID_NUM, embedding_dim=usr_embedding_dim, sparse=False)
        self.usr_fc = nn.Linear(in_features=usr_embedding_dim, out_features=32)

        # 对用户性别信息做映射，并紧接着一个Linear层
        USR_GENDER_DICT_SIZE = 2
        self.usr_gender_emb = nn.Embedding(num_embeddings=USR_GENDER_DICT_SIZE, embedding_dim=gender_embeding_dim)
        self.usr_gender_fc = nn.Linear(in_features=gender_embeding_dim, out_features=16)

        # 对用户年龄信息做映射，并紧接着一个Linear层
        USR_AGE_DICT_SIZE = vocab_sizes["max_usr_age"] + 1  # +1 考虑0自然数
        self.usr_age_emb = nn.Embedding(num_embeddings=USR_AGE_DICT_SIZE, embedding_dim=age_embedding_dim)
        self.usr_age_fc = nn.Linear(in_features=age_embedding_dim, out_features=16)

        # 对用户职业信息做映射，并紧接着一个Linear层
        USR_JOB_DICT_SIZE = vocab_sizes["max_usr_job"] + 1
        self.usr_job_emb = nn.Embedding(num_embeddings=USR_JOB_DICT_SIZE, embedding_dim=job_embedding_dim)
        self.usr_job_fc = nn.Linear(in_features=job_embedding_dim, out_features=16)

        # 新建一个Linear层，用于整合用户数据信息
        self.usr_combined = nn.Linear(in_features=80, out_features=200)

        logger.info("define network layer for embedding usr info")
        MOV_DICT_SIZE = vocab_sizes["max_mov_id"] + 1
        self.mov_emb = nn.Embedding(num_embeddings=MOV_DICT_SIZE, embedding_dim=mov_embedding_dim)
        self.mov_fc = nn.Linear(in_features=mov_embedding_dim, out_features=32)

        # 对电影类别做映射
        CATEGORY_DICT_SIZE = vocab_sizes["movie_category_size"] + 1  # +1 考虑添加0类别
        self.mov_cat_emb = nn.Embedding(num_embeddings=CATEGORY_DICT_SIZE,
                                        embedding_dim=category_embedding_dim,
                                        sparse=False)
        self.mov_cat_fc = nn.Linear(in_features=category_embedding_dim, out_features=32)

        # 对电影名称做映射
        MOV_TITLE_DICT_SIZE = vocab_sizes["movie_title_size"] + 1
        self.mov_title_emb = nn.Embedding(num_embeddings=MOV_TITLE_DICT_SIZE,
                                          embedding_dim=title_embedding_dim,
                                          sparse=False)
        self.mov_title_conv = nn.Conv2D(in_channels=1, out_channels=1,
                                        kernel_size=(3, 1), stride=(2, 1), padding=0)
        self.mov_title_conv2 = nn.Conv2D(in_channels=1, out_channels=1,
                                         kernel_size=(3, 1), stride=1, padding=0)

        # 新建一个Linear层，用于整合电影特征
        self.mov_concat_embed = nn.Linear(in_features=96, out_features=200)

        # 增加全连接层深度提取特征
        user_sizes = [200] + self.fc_sizes
        acts = ["relu" for _ in range(len(self.fc_sizes))]
        self._user_layers = []
        for i in range(len(self.fc_sizes)):
            linear = nn.Linear(
                in_features=user_sizes[i],
                out_features=user_sizes[i + 1],
                weight_attr=paddle.ParamAttr(
                    initializer=paddle.nn.initializer.Normal(
                        std=1.0 / math.sqrt(user_sizes[i]))))
            self.add_sublayer("linear_user_%d" % i, linear)
            self._user_layers.append(linear)
            if acts[i] == "relu":
                act = nn.ReLU()
                self.add_sublayer("user_act_%d" % i, act)
                self._user_layers.append(act)

        # 电影特征和用户特征使用了不同的全连接层，不共享参数
        movie_sizes = [200] + self.fc_sizes
        acts = ["relu" for _ in range(len(self.fc_sizes))]
        self._movie_layers = []
        for i in range(len(self.fc_sizes)):
            linear = nn.Linear(
                in_features=movie_sizes[i],
                out_features=movie_sizes[i + 1],
                weight_attr=paddle.ParamAttr(
                    initializer=paddle.nn.initializer.Normal(
                        std=1.0 / math.sqrt(movie_sizes[i]))))
            self.add_sublayer("linear_movie_%d" % i, linear)
            self._movie_layers.append(linear)
            if acts[i] == "relu":
                act = nn.ReLU()
                self.add_sublayer("movie_act_%d" % i, act)
                self._movie_layers.append(act)

    def get_usr_feature(self, usr_var):
        """定义计算用户特征的前向运算过程"""
        # 获取到用户数据
        usr_id, usr_gender, usr_age, usr_job = usr_var
        # 将用户的ID数据经过embedding和Linear计算，得到的特征保存在feats_collect中
        feats_collect = []

        usr_id_feat = F.relu(self.usr_fc(self.usr_emb(usr_id)))
        feats_collect.append(usr_id_feat)

        # 计算用户的性别特征，并保存在feats_collect中
        usr_gender_feat = F.relu(self.usr_gender_fc(self.usr_gender_emb(usr_gender)))
        feats_collect.append(usr_gender_feat)

        # 选择是否使用用户的年龄-职业特征
        if self.use_usr_age_job:
            usr_age_feat = F.relu(self.usr_age_fc(self.usr_age_emb(usr_age)))
            feats_collect.append(usr_age_feat)

            usr_job_feat = F.relu(self.usr_job_fc(self.usr_job_emb(usr_job)))
            feats_collect.append(usr_job_feat)

        # 将用户的特征级联，并通过Linear层得到最终的用户特征
        usr_feat = paddle.concat(feats_collect, axis=-1)
        user_features = F.tanh(self.usr_combined(usr_feat))

        # 通过3层全链接层，获得用于计算相似度的用户特征和电影特征
        for n_layer in self._user_layers:
            user_features = n_layer(user_features)

        return user_features

    def get_movie_feature(self, mov_var):
        """定义电影特征的前向计算过程"""
        # 获得电影数据
        mov_id, mov_cat, mov_title, mov_poster = mov_var
        feats_collect = []
        # 获得batchsize的大小
        batch_size = mov_id.shape[0]
        # 计算电影ID的特征，并存在feats_collect中
        mov_id_feat = F.relu(self.mov_fc(self.mov_emb(mov_id)))
        feats_collect.append(mov_id_feat)

        # 如果使用电影的种类数据，计算电影种类特征的映射
        if self.use_mov_cat:
            # 计算电影种类的特征映射，对多个种类的特征求和得到最终特征
            mov_cat_feat = self.mov_cat_emb(mov_cat)
            mov_cat_feat = paddle.sum(mov_cat_feat, axis=1, keepdim=False)  # shape: [btc, feature_size]
            mov_cat_feat = F.relu(self.mov_cat_fc(mov_cat_feat))
            feats_collect.append(mov_cat_feat)

        if self.use_mov_title:
            mov_title_feat = self.mov_title_emb(mov_title)
            mov_title_feat = self.mov_title_conv2(F.relu(self.mov_title_conv(mov_title_feat)))
            mov_title_feat = paddle.sum(mov_title_feat, axis=2, keepdim=False)  # shape: [bat, 1, feature_size]
            mov_title_feat = F.relu(mov_title_feat)
            mov_title_feat = paddle.reshape(mov_title_feat, [batch_size, -1])
            feats_collect.append(mov_title_feat)

        # 使用一个全连接层，整合所有电影特征，映射为一个200维的特征向量
        mov_feat = paddle.concat(feats_collect, axis=-1)
        mov_features = F.tanh(self.mov_concat_embed(mov_feat))
        # 继续使用深度网络提取特征
        for n_layer in self._movie_layers:
            mov_features = n_layer(mov_features)

        return mov_features

    def forward(self, usr_var, mov_var):
        """定义个性化推荐算法的前向计算"""
        # 计算用户特征和电影特征
        user_features = self.get_usr_feature(usr_var)
        mov_features = self.get_movie_feature(mov_var)

        # 根据计算的特征计算相似度, reshape方便loss计算
        sim = F.common.cosine_similarity(user_features, mov_features).reshape([-1, 1])
        # 使用余弦相似度算子，计算用户和电影的相似程度
        # sim = F.cosine_similarity(user_features, mov_features, axis=1).reshape([-1, 1])
        # 将相似度扩大范围到和电影评分相同数据范围
        res = paddle.scale(sim, scale=5)

        return user_features, mov_features, res
