# -*- coding: utf-8 -*-
"""
# 原始对话文本清理：
    1. 清理对话轮数过多的session
    2. 清理单次回答句子过长的session
    3. 清理句子中的表情符
"""
import os
import logging
import copy
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import logging
import json
from tqdm import tqdm

logging.basicConfig(format="%(asctime)s - %(filename)s - %(levelname)s - %(name)s -   %(message)s",
                    datefmt="%m/%d/%Y %H:%M:%S",
                    level=logging.INFO,)
_logger = logging.getLogger(__name__)


class OriginDataClean(object):
    """原始数据清理"""
    def __init__(self, args=None):
        self.args = args
        self.origin_data = None
        self.session_length_clean_data = None
        self.session_sentence_length_clean_data = None
        self.sessions_set = set()
        self.sessions_length = {}
        self.sessions_sentences_length = {}

    def read_origin_data_to_df(self, input_file):
        """获取原始数据"""
        try:
            self.origin_data = pd.read_csv(input_file, sep="\t", engine="python", index_col=False,
                                           warn_bad_lines=True, error_bad_lines=False, encoding="UTF-8", header=None)
            _logger.info("原始数据行数: {}".format(self.origin_data.shape[0]))
        except IOError:
            raise IOError('pandas read data file error.')

    def compute_total_length(self, sessions_length):
        """计算所有长度之和"""
        sum = 0
        for value in sessions_length.values():
            sum += value

        return sum

    def statistics_session_length(self, input_data_df):
        """统计每个对话的长度"""
        _logger.info("统计每个session的对话长度.")
        session_id = None
        sessions_ctx = {}
        for i in tqdm(range(input_data_df.shape[0]), ncols=80):
            if not input_data_df.iat[i, 0] in self.sessions_set:
                # 新的session号
                if i != 0:
                    # 第一行不进行session长度统计
                    self.sessions_length[session_id] = cnt_session_length
                    # 将一个session的所有上下文加入字典
                    sessions_ctx[session_id] = single_session_ctx

                single_session_ctx = []
                single_session_ctx.append(input_data_df.iloc[i].values.tolist())
                cnt_session_length = 1  # 记录每个session的长度
                session_id = input_data_df.iat[i, 0]
                self.sessions_set.add(session_id)
            else:
                # 当前session长度+1
                cnt_session_length += 1
                single_session_ctx.append(input_data_df.iloc[i].values.tolist())
        # 添加最后一个session的长度
        self.sessions_length[session_id] = cnt_session_length
        # 将最后session的所有上下文加入字典
        sessions_ctx[session_id] = single_session_ctx
        # 判断二者是否相等
        assert len(self.sessions_set) == len(self.sessions_length) == len(sessions_ctx)
        assert len(input_data_df) == self.compute_total_length(self.sessions_length)

        _logger.info("session长度阈值分析.")
        sessions_length_np = np.array(list(self.sessions_length.values()))
        if self.args.show:
            _logger.info("session 长度分布图.")
            self.show_distribution(sessions_length_np, "session length distribution")
        # 对session长度做统计分析, 确定边界阈值
        Q1 = np.percentile(sessions_length_np, 25)  # 前25%的最大长度, 也是下边界
        Q3 = np.percentile(sessions_length_np, 75)  # 前75%的最大长度, 也是上边界
        IQR = Q3 - Q1  # 合理范围值
        outlier_step = self.args.dialogue_length_factor * IQR  # 上下边界的差值
        _logger.info("25%: {}, 75%: {}, 合理范围: {}, outlier step: {}".format(Q1, Q3, IQR, outlier_step))

        session_length_clean_data = []
        new_sessions_length = {}
        _logger.info("删除超出阈值长度的session.")
        _logger.info("未清理之前data行数: {}".format(len(input_data_df)))
        _logger.info("未清理之前session数: {}".format(len(self.sessions_length)))
        for key, value in tqdm(self.sessions_length.items()):
            if value < (Q3 + outlier_step) and value > (Q1 - outlier_step):
                # session 长度符合要求的保留
                session_length_clean_data.extend(sessions_ctx.get(key))
                new_sessions_length[key] = value

        _logger.info("清理过长session之后data行数: {}".format(len(session_length_clean_data)))
        _logger.info("清理过长session之后session数: {}".format(len(new_sessions_length)))

        # 清除无用数据
        sessions_ctx.clear()

        # 将数据转为df
        return pd.DataFrame(data=session_length_clean_data, columns=input_data_df.columns)

    def combine_session_sentences_length(self, sessions_sentences_length):
        """合并所有session sentence length"""
        sentences_length = []
        for value in sessions_sentences_length.values():
            # 字典每个元素的值合并
            sentences_length.extend(value)

        return np.array(sentences_length)

    def statistics_sentence_length(self, input_data_df):
        """统计单个句子长度"""
        _logger.info("统计单个句子长度.")
        # self.sessions_set.clear()  # 清空set()
        self.sessions_set = set()
        sessions_ctx = {}
        for i in tqdm(range(input_data_df.shape[0]), ncols=80):
            if not input_data_df.iat[i, 0] in self.sessions_set:
                # 新的session号
                if i != 0:
                    # 第一行不进行session长度统计, 统计每个session每个句子的长度
                    self.sessions_sentences_length[session_id] = session_sentences_length
                    # 将一个session的所有上下文加入字典
                    sessions_ctx[session_id] = single_session_ctx

                # 每个session开始初始化列表
                session_sentences_length = []
                # 添加每个session第一行的长度
                session_sentences_length.append(len(input_data_df.iat[i, 3]))
                # 单行数据加入列表
                single_session_ctx = []
                single_session_ctx.append(input_data_df.iloc[i].values.tolist())
                session_id = input_data_df.iat[i, 0]
                self.sessions_set.add(session_id)
            else:
                # 当前session每个句子长度添加进入列表
                session_sentences_length.append(len(input_data_df.iat[i, 3]))
                # 单行数据加入列表
                single_session_ctx.append(input_data_df.iloc[i].values.tolist())
        # 添加最后一个session的每个句子的长度
        self.sessions_sentences_length[session_id] = session_sentences_length
        # 将最后session的所有上下文加入字典
        sessions_ctx[session_id] = single_session_ctx

        _logger.info("sentence长度阈值分析.")
        assert len(self.sessions_set) == len(self.sessions_sentences_length)
        sentences_length_np = self.combine_session_sentences_length(self.sessions_sentences_length)
        # 数据统计出来之后大小是否正确
        assert len(sentences_length_np) == len(input_data_df)
        if self.args.show:
            _logger.info("sentence 长度统计分布图.")
            self.show_distribution(sentences_length_np, "sentences length distribution")
        # 对sentences长度做统计分析, 确定边界阈值
        Q1 = np.percentile(sentences_length_np, 25)  # 前25%的最大长度, 也是下边界
        Q3 = np.percentile(sentences_length_np, 75)  # 前75%的最大长度, 也是上边界
        IQR = Q3 - Q1  # 合理范围值
        outlier_step = self.args.sentence_length_factor * IQR  # 上下边界的差值
        _logger.info("25%: {}, 75%: {}, 合理范围: {}, outlier step: {}".format(Q1, Q3, IQR, outlier_step))

        sessions_sentences_length_clean_data = []
        new_sessions_length = copy.deepcopy(self.sessions_sentences_length)
        _logger.info("删除超出阈值长度的session.")
        _logger.info("未清理之前data行数: {}".format(len(input_data_df)))
        _logger.info("未清理之前session数: {}".format(len(self.sessions_sentences_length)))
        for key, value in tqdm(self.sessions_sentences_length.items()):
            session_contain_over_length_sentence = False
            for sentence_length in value:
                if sentence_length > (Q3 + outlier_step) or sentence_length < (Q1 - outlier_step):
                    # 删除超出阈值的数据行(只要session包含超出阈值的句子直接删除怎么session)
                    session_contain_over_length_sentence = True
                    new_sessions_length.pop(key)
                    # 随便找到一个就是直接跳出真个session, 因为此处已经删除了整个session
                    break
            if session_contain_over_length_sentence == False:
                # 将符合要求的session的所有句子加入列表
                sessions_sentences_length_clean_data.extend(sessions_ctx.get(key))

        _logger.info("清理过长session之后data行数: {}".format(len(sessions_sentences_length_clean_data)))
        _logger.info("清理过长session之后session数: {}".format(len(new_sessions_length)))
        _logger.info("数据清除率: {}(session).".format(
                    float(len(new_sessions_length))/len(self.sessions_length)))

        # 清除无用数据
        sessions_ctx.clear()

        # return sessions_sentences_length_clean_data

        # 将数据转为df
        return pd.DataFrame(data=sessions_sentences_length_clean_data, columns=input_data_df.columns)

    def processing(self):
        """过程函数"""
        # 1. 读取数据
        self.read_origin_data_to_df(self.args.input_file)
        # 2. 删除超出最大对话轮数的session
        self.session_length_clean_data = self.statistics_session_length(self.origin_data)
        # 3. 删除单句过长句子的整个session, 输入数据集删除过长session之后的数据
        self.session_sentence_length_clean_data = self.statistics_sentence_length(self.session_length_clean_data)

        return self.session_sentence_length_clean_data

    def save_clean_data(self, data_df, data_file_path):
        """保存清理数据到文件"""
        data_df.to_csv(data_file_path, sep='\t', header=None,
                       index=False, mode='w', encoding='utf-8')

    def show_distribution(self, length_data, distribution_name):
        """绘制长度分布"""
        _logger.info("{} describe: \n{}".format(distribution_name,
                                                pd.Series(length_data).describe()))
        length_fred = np.bincount(length_data)
        _logger.info("{} 众数: {}".format(distribution_name, np.argmax(length_fred)))
        plt.title(distribution_name)
        plt.xlabel('length')
        plt.ylabel('fred')
        plt.plot(range(len(length_fred)), length_fred)
        plt.legend()
        plt.show()


if __name__ == "__main__":
    parse = argparse.ArgumentParser(description="设置原始数据清理的基本参数设置")
    parse.add_argument('-i', '--input_file', type=str, default='./data/data_dev.txt',
                       help='需要处理的原始数据文件.')
    parse.add_argument('-o', '--output_file', type=str, default='./data/data_dev_clean.txt',
                       help='进行清理之后数据的保存文件.')
    parse.add_argument('-dlf', '--dialogue_length_factor', type=float, default=3,
                       help='删除最大对话轮数session系数因子, length 25%% 75%% percentile统计确定合理对话轮数')
    parse.add_argument('-slf', '--sentence_length_factor', type=float, default=20,
                       help='删除最长句子session系数因子, length 25%% 75%% percentile统计确定合理单轮最长回复')
    parse.add_argument('--show', action="store_true", default=True,
                       help='显示分布图')

    args = parse.parse_args()

    main = OriginDataClean(args=args)
    session_sentence_length_clean_data = main.processing()
    _logger.info("保存清理之后的数据")
    main.save_clean_data(session_sentence_length_clean_data, args.output_file)
