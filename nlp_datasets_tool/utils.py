"""
# 常用工具函数
"""
import os
import re
import logging

logger = logging.getLogger(__name__)


def get_highlight_context_sentences(sentences: list, highlight_index: int, max_len: int = 508):
    """滑动获取训练数据- 保持最大长度左右"""
    left_pos, right_pos = highlight_index - 1, highlight_index + 1
    left_len, right_len, sum_len = 0, 0, len(sentences[highlight_index])
    while True:
        flag = False
        if left_pos >= 0:
            if left_len <= right_len or right_pos >= len(sentences):
                # 左边长度小于右边 or 右边句子已经被遍历完
                tmp_len = len(sentences[left_pos])
                if sum_len + tmp_len < max_len:
                    left_pos -= 1
                    left_len += tmp_len
                    sum_len += tmp_len
                    flag = True
        if right_pos < len(sentences):
            if right_len <= left_len or left_pos < 0:
                # 右边长度小于左边 or 左边句子已经被遍历完
                tmp_len = len(sentences[right_pos])
                if sum_len + tmp_len < max_len:
                    right_pos += 1
                    right_len += tmp_len
                    sum_len += tmp_len
                    flag = True
        if not flag:
            break

    new_sentences = sentences[left_pos+1: right_pos]
    new_highlight_index = highlight_index - left_pos - 1
    assert new_sentences[new_highlight_index] == sentences[highlight_index]

    return new_sentences, new_highlight_index


def split_sentence(paragraph: str, flag: str = "all", limit: int = 510):
    """
    Args:
        paragraph:
        flag: Type:str, "all" 中英文标点分句，"zh" 中文标点分句，"en" 英文标点分句
        limit: 默认单句最大长度为510个字符
    Returns: Type:list
    """
    sent_list = []
    try:
        if flag == "zh":
            paragraph = re.sub('(?P<quotation_mark>([。？！…](?![”’"\'])))', r'\g<quotation_mark>\n',
                              paragraph)  # 单字符断句符
            paragraph = re.sub('(?P<quotation_mark>([。？！]|…{1,2})[”’"\'])', r'\g<quotation_mark>\n',
                              paragraph)  # 特殊引号
        elif flag == "en":
            paragraph = re.sub('(?P<quotation_mark>([.?!](?![”’"\'])))', r'\g<quotation_mark>\n',
                              paragraph)  # 英文单字符断句符
            paragraph = re.sub('(?P<quotation_mark>([?!.]["\']))', r'\g<quotation_mark>\n', paragraph)  # 特殊引号
        else:
            paragraph = re.sub('(?P<quotation_mark>([。？！….?!](?![”’"\'])))', r'\g<quotation_mark>\n',
                              paragraph)  # 单字符断句符
            paragraph = re.sub('(?P<quotation_mark>(([。？！.!?]|…{1,2})[”’"\']))', r'\g<quotation_mark>\n',
                              paragraph)  # 特殊引号

        sent_list_ori = paragraph.splitlines()
        for sent in sent_list_ori:
            sent = sent.strip()
            if not sent:
                continue
            else:
                while len(sent) > limit:
                    temp = sent[0:limit]
                    sent_list.append(temp)
                    sent = sent[limit:]
                sent_list.append(sent)
    except:
        sent_list.clear()
        sent_list.append(paragraph)
    return sent_list


def log_to_init(logs_file_path=None):
    """将日志输出到文件初始化"""
    LOG_FORMAT = "%(asctime)s-%(name)s-%(levelname)s-%(pathname)s: %(message)s"  # 配置输出日志格式
    DATE_FORMAT = '%Y-%m-%d  %H:%M:%S'  # 配置输出时间的格式, 注意月份和天数不要搞乱了
    logging.basicConfig(filename=logs_file_path,
                        level=logging.DEBUG,
                        format=LOG_FORMAT,
                        datefmt=DATE_FORMAT)