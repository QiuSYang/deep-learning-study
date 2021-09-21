"""
# 电影推荐系统inference流程
    1.根据一个电影推荐其相似的电影。--- 计算电影之间的相似度
    2.根据用户的喜好，推荐其可能喜欢的电影。--- 计算用户与电影之间的相似度
    3.给指定用户推荐与其喜好相似的用户喜欢的电影。--- 计算用户与用户之间的相似度

# 根据用户喜好推荐电影
    根据用户对电影的喜好（评分高低）作为训练指标完成训练。神经网络有两个输入，用户数据和电影数据，
    通过神经网络提取用户特征和电影特征，并计算特征之间的相似度，相似度的大小和用户对该电影的评分存在对应关系。
    即如果用户对这个电影感兴趣，那么对这个电影的评分也是偏高的，最终神经网络输出的相似度就更大一些。
    完成训练后，我们就可以开始给用户推荐电影了。

    根据用户喜好推荐电影，是通过计算用户特征和电影特征之间的相似性，并排序选取相似度最大的结果来进行推荐，流程如下：
        计算用户A的特征 --|   相似度计算
                         | -----------> 构建相似度矩阵 ------> 对相似度排序 -----> 选取相似度最大的电影作为推荐结果
        读取保存的特征 --|
    从计算相似度到完成推荐的过程, 步骤包括：
        1. 读取保存的特征，根据一个给定的用户ID、电影ID，我们可以索引到对应的特征向量。
        2. 通过计算用户特征和其他电影特征向量的相似度，构建相似度矩阵。
        3. 对这些相似度排序后，选取相似度最大的几个特征向量，找到对应的电影ID，即得到推荐清单。
        4. 加入随机选择因素，从相似度最大的top_k结果中随机选取pick_num个推荐结果，其中pick_num必须小于top_k。
"""
import os
import sys
import logging
import pickle
from tqdm import tqdm
import numpy as np
import paddle
import paddle.nn.functional as F

logger = logging.getLogger(__name__)
work_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(work_root)  # 添加工作路径

from src.utils import init_logs


def get_features(feature_file):
    """获取特征向量"""
    with open(feature_file, mode="rb") as fp:
        return pickle.load(fp)


def get_movie_info(data_file):
    """获取电影信息"""
    mov_info = {}
    # 打开电影数据文件，根据电影ID索引到电影信息
    with open(data_file, "r", encoding="ISO-8859-1") as f:
        data = f.readlines()
        for item in data:
            item = item.strip().split("::")
            mov_info[str(item[0])] = item

    return mov_info


@paddle.no_grad()
def recommend_movie_for_usr(usr_id, usr_features, mov_features, mov_info,
                            top_k=10, pick_num=5):
    """根据用户喜好推荐电影
        usr_features: 库中用户特征向量
        mov_features： 库中电影特征向量
    """
    usr_feat = usr_features[str(usr_id)]  # 通过搜索查询, 实际可以通过模型计算(提供用户相应的属性数据)

    cos_sims = []
    for idx, key in enumerate(mov_features.keys()):
        mov_feat = mov_features[key]
        usr_feat = paddle.to_tensor(usr_feat)
        mov_feat = paddle.to_tensor(mov_feat)
        # 计算余弦相似度
        sim = F.common.cosine_similarity(usr_feat, mov_feat)
        cos_sims.append(sim.numpy()[0])

    # 对相似度排序, 获取距离最近的top_k
    index = np.argsort(cos_sims)[-top_k:]
    for i in range(top_k):
        logger.info("top-{} 用户与电影的距离: {}".format(i+top_k, cos_sims[index[i]]))

    # 加入随机选择因素，确保每次推荐的都不一样(即从top_k中随机选取num个数据作为召回)
    res = []
    while len(res) < pick_num:
        val = np.random.choice(len(index), 1)[0]
        idx = index[val]
        mov_id = list(mov_features.keys())[idx]
        if mov_id not in res:
            res.append(mov_id)

    for idx in res:
        logger.info("可能喜欢的电影id: {}, info: {}".format(idx, mov_info[str(idx)]))


def get_usr_highest_rated_movie(usr_id, movie_info, top_k=10):
    """获取用户评分最高的top_k个电影"""
    # 获得ID为usr_a的用户评分过的电影及对应评分
    rating_file = os.path.join(work_root, "data/ml-1m/ratings.dat")
    # 打开文件，ratings_data
    with open(rating_file, "r") as f:
        ratings_data = f.readlines()

    usr_rating_info = {}
    for item in ratings_data:
        item = item.strip().split("::")
        # 处理每行数据，分别得到用户ID，电影ID，和评分
        usr_id, movie_id, score = item[0], item[1], item[2]
        if usr_id == str(usr_id):
            usr_rating_info[movie_id] = float(score)

    # 获得评分过的电影ID
    movie_ids = list(usr_rating_info.keys())
    logger.info("ID为 {} 的用户，评分过的电影数量是: {}".format(usr_id, len(movie_ids)))

    # 选出ID为usr_a评分最高的前topk个电影
    ratings_top_k = sorted(usr_rating_info.items(), key=lambda item: item[1])[-top_k:]

    for k, score in ratings_top_k:
        logger.info("电影ID: {}，评分是: {}, 电影信息: {}".format(k, score, movie_info[k]))


def main():
    """主函数"""
    logger.info("读取特征向量")
    usr_feature_file = os.path.join(work_root, "data/usr_feat.pkl")
    mov_feature_file = os.path.join(work_root, "data/mov_feat.pkl")
    usr_features = get_features(usr_feature_file)
    mov_features = get_features(mov_feature_file)

    movie_info = get_movie_info(os.path.join(work_root, "data/ml-1m/movies.dat"))
    usr_id = 2
    logger.info("获取用户 {} 喜欢的电影".format(usr_id))
    recommend_movie_for_usr(usr_id, usr_features, mov_features, mov_info=movie_info,
                            top_k=10, pick_num=5)
    get_usr_highest_rated_movie(usr_id, movie_info, top_k=10)


if __name__ == '__main__':
    init_logs()
    main()
