"""
# 电影数据处理文件
"""
import os
import logging
import random
from tqdm import tqdm
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)
work_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class MovieDataset(object):
    def __init__(self, use_poster=False):
        self.use_poster = use_poster  # 是否使用海报信息
        # 声明每个数据文件的路径
        usr_info_path = os.path.join(work_root, "data/ml-1m/users.dat")
        if use_poster:
            rating_path = os.path.join(work_root, "data/ml-1m/new_rating.txt")
        else:
            rating_path = os.path.join(work_root, "data/ml-1m/ratings.dat")

        movie_info_path = os.path.join(work_root, "data/ml-1m/movies.dat")
        self.poster_path = os.path.join(work_root, "data/ml-1m/posters/")
        # 获取电影数据
        self.movie_info, self.movie_cat, self.movie_title = self.get_movie_info(movie_info_path)
        # 记录电影的最大ID
        self.max_mov_cat = np.max([self.movie_cat[k] for k in self.movie_cat])
        self.max_mov_tit = np.max([self.movie_title[k] for k in self.movie_title])
        self.max_mov_id = np.max(list(map(int, self.movie_info.keys())))
        # 记录用户数据的最大ID
        self.max_usr_id = 0
        self.max_usr_age = 0
        self.max_usr_job = 0
        # 得到用户数据
        self.usr_info = self.get_usr_info(usr_info_path)
        # 得到评分数据
        self.rating_info = self.get_rating_info(rating_path)
        # 构建数据集
        self.dataset = self.get_dataset(usr_info=self.usr_info,
                                        rating_info=self.rating_info,
                                        movie_info=self.movie_info)
        # 划分数据集，获得数据加载器
        train_data_len = int(len(self.dataset) * 0.9)
        self.train_dataset = self.dataset[:train_data_len]
        self.valid_dataset = self.dataset[train_data_len:]
        logger.info("##Total dataset instances: {}".format(len(self.dataset)))
        logger.info("##MovieLens dataset information: \nusr num: {}\n"
              "movies num: {}".format(len(self.usr_info), len(self.movie_info)))

    def get_movie_info(self, path):
        """获取电影数据"""
        # 打开文件，编码方式选择ISO-8859-1，读取所有数据到data中
        with open(path, 'r', encoding="ISO-8859-1") as f:
            data = f.readlines()
        # 建立三个字典，分别用户存放电影所有信息，电影的名字信息、类别信息
        movie_info, movie_titles, movie_cats = {}, {}, {}
        # 对电影名字、类别中不同的单词计数
        t_count, c_count = 1, 1

        count_tit = {}
        # 按行读取数据并处理
        for item in tqdm(data):
            item = item.strip().split("::")
            v_id = item[0]  # 电影id
            v_title = item[1][:-7]  # 电影名称, 剔除年份(年份不对年份有影响)
            cats = item[2].split("|")  # 电影类别
            v_year = item[1][-5:-1]  # 年份

            titles = v_title.split()  # 拆分为单个词
            # 统计电影名字的单词，并给每个单词一个序号，放在movie_titles中
            for t in titles:
                if t not in movie_titles:
                    movie_titles[t] = t_count
                    t_count += 1
            # 统计电影类别单词，并给每个单词一个序号，放在movie_cat中
            for cat in cats:
                if cat not in movie_cats:
                    movie_cats[cat] = c_count
                    c_count += 1
            # 补0使电影名称对应的列表长度为15
            v_tit = [movie_titles[k] for k in titles]
            while len(v_tit) < 15:
                v_tit.append(0)  # 电影名称最长15个单词

            # 补0使电影种类对应的列表长度为6
            v_cat = [movie_cats[k] for k in cats]
            while len(v_cat) < 6:
                v_cat.append(0)  # 电影最多六个类别
            # 保存电影数据到movie_info中
            movie_info[v_id] = {
                "mov_id": int(v_id),
                "title": v_tit,
                "category": v_cat,
                "years": int(v_year)
            }

        return movie_info, movie_cats, movie_titles

    def get_usr_info(self, path):
        """获取用户信息"""
        def gender2num(gender):
            """性别转换函数, M-0, F-1"""
            return 1 if gender == "F" else 1

        # 打开文件，读取所有行到data中
        with open(path, 'r') as f:
            data = f.readlines()
        # 建立用户信息的字典
        use_info = {}

        # 按行索引数据
        for item in tqdm(data):
            # 去除每一行中和数据无关的部分
            item = item.strip().split("::")
            usr_id = item[0]
            # 将字符数据转成数字并保存在字典中
            use_info[usr_id] = {
                "usr_id": int(usr_id),
                "gender": gender2num(item[1]),
                "age": int(item[2]),
                "job": int(item[3])
            }
            self.max_usr_id = max(self.max_usr_id, int(usr_id))
            self.max_usr_age = max(self.max_usr_age, int(item[2]))
            self.max_usr_job = max(self.max_usr_job, int(item[3]))

        return use_info

    def get_rating_info(self, path):
        """获取用户对电影的评分数据"""
        # 读取文件里的数据
        with open(path, 'r') as f:
            data = f.readlines()
        # 将数据保存在字典中并返回
        rating_info = {}
        for item in tqdm(data):
            item = item.strip().split("::")
            usr_id, movie_id, score = item[0], item[1], item[2]
            if usr_id not in rating_info.keys():
                rating_info[usr_id] = {movie_id: float(score)}
            else:
                rating_info[usr_id][movie_id] = float(score)

        return rating_info

    def get_dataset(self, usr_info, rating_info, movie_info):
        """构建数据集"""
        trainset = []
        for usr_id in rating_info.keys():
            usr_ratings = rating_info[usr_id]
            for movie_id in usr_ratings.keys():
                trainset.append({
                    "usr_info": usr_info[usr_id],
                    "mov_info": movie_info[movie_id],
                    "scores": usr_ratings[movie_id]
                })

        return trainset

    def load_data(self, dataset=None, mode="train", use_poster=False):
        """获取训练数据"""
        # 定义数据迭代Batch大小
        BATCHSIZE = 256

        data_length = len(dataset)
        index_list = list(range(data_length))

        def data_generator():
            """ 定义数据迭代加载器"""
            # 训练模式下，打乱训练数据
            if mode == 'train':
                random.shuffle(index_list)
            # 声明每个特征的列表
            usr_id_list, usr_gender_list, usr_age_list, usr_job_list = [], [], [], []
            mov_id_list, mov_tit_list, mov_cat_list, mov_poster_list = [], [], [], []
            score_list = []
            # 索引遍历输入数据集
            for idx, i in enumerate(index_list):
                # 获得特征数据保存到对应特征列表中
                usr_id_list.append(dataset[i]['usr_info']['usr_id'])
                usr_gender_list.append(dataset[i]['usr_info']['gender'])
                usr_age_list.append(dataset[i]['usr_info']['age'])
                usr_job_list.append(dataset[i]['usr_info']['job'])

                mov_id_list.append(dataset[i]['mov_info']['mov_id'])
                mov_tit_list.append(dataset[i]['mov_info']['title'])
                mov_cat_list.append(dataset[i]['mov_info']['category'])
                mov_id = dataset[i]['mov_info']['mov_id']
                if use_poster:
                    # 不使用图像特征时，不读取图像数据，加快数据读取速度
                    poster = Image.open(os.path.join(self.poster_path, "mov_id{}.jpg".format(str(mov_id[0]))))
                    poster = poster.resize([64, 64])
                    if len(poster.size) <= 2:
                        poster = poster.convert("RGB")

                    mov_poster_list.append(np.array(poster))

                score_list.append(int(dataset[i]['scores']))

                # 如果读取的数据量达到当前的batch大小，就返回当前批次
                if len(usr_id_list) == BATCHSIZE:
                    # 转换列表数据为数组形式，reshape到固定形状
                    usr_id_arr = np.array(usr_id_list)
                    usr_gender_arr = np.array(usr_gender_list)
                    usr_age_arr = np.array(usr_age_list)
                    usr_job_arr = np.array(usr_job_list)

                    mov_id_arr = np.array(mov_id_list)
                    mov_cat_arr = np.reshape(np.array(mov_cat_list), [BATCHSIZE, 6]).astype(np.int64)
                    # 转为三维向量, 就想图像通道为1
                    mov_tit_arr = np.reshape(np.array(mov_tit_list), [BATCHSIZE, 1, 15]).astype(np.int64)

                    if use_poster:
                        # 数据范围归到[-1, 1]
                        mov_poster_arr = np.reshape(np.array(mov_poster_list) / 127.5 - 1,
                                                    [BATCHSIZE, 3, 64, 64]).astype(np.float32)
                    else:
                        mov_poster_arr = np.array([0.])

                    scores_arr = np.reshape(np.array(score_list), [-1, 1]).astype(np.float32)  # [btc, 1]方便损失计算

                    # 放回当前批次数据
                    yield [usr_id_arr, usr_gender_arr, usr_age_arr, usr_job_arr], \
                          [mov_id_arr, mov_cat_arr, mov_tit_arr, mov_poster_arr], scores_arr

                    # 清空数据
                    usr_id_list, usr_gender_list, usr_age_list, usr_job_list = [], [], [], []
                    mov_id_list, mov_tit_list, mov_cat_list, score_list = [], [], [], []
                    mov_poster_list = []

        return data_generator


if __name__ == '__main__':
    pass
