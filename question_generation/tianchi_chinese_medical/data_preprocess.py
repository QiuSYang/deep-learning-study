"""
# 原始数据预处理
"""
import os
import logging
import json
import pickle
from random import shuffle
import numpy as np
from tqdm import tqdm
from langconv import *

obj = Converter('zh-hans')
logger = logging.getLogger(__name__)


class DatasetPreprocess(object):
    def __init__(self):
        self.output = {
            "train_items": None,
            "test_items": None,
            "valid_items": None,
            "dureader_train_items": [],
            "cmrc_train_items": [],
            "drcd_train_items": [],
            "multi_task_epoch": 6
        }
        self.a2q = False

    def dureader_json(self, data):
        """# 最后处理DuReader(完全是用来粗调的==> 粒度太碎)"""
        for item in data:
            if item["question_type"] == "YES_NO":
                continue
            context = ""
            for doc_item in item["documents"]:
                if doc_item["is_selected"]:
                    context += " ".join(doc_item["paragraphs"])
                    if len(context) >= 600:
                        break
            context = context[:600]
            answers = item["answers"]
            for atem in answers:
                self.output["dureader_train_items"].append({
                        "context": context,
                        "query": item["question"],
                        "answer": atem
                    })

    def deal_dureader_dataset(self):
        """"""
        for file in os.listdir("DataSet/MultiTask/DuReader/devset"):
            with open("DataSet/MultiTask/DuReader/devset/" + file, "r", encoding="UTF-8") as f:
                self.dureader_json([json.loads(s) for s in tqdm(f.readlines())])
        print("===完成DuReader Dev数据处理===")

        for file in os.listdir("DataSet/MultiTask/DuReader/trainset"):
            with open("DataSet/MultiTask/DuReader/trainset/" + file, "r", encoding="UTF-8") as f:
                self.dureader_json([json.loads(s) for s in tqdm(f.readlines())])
        print("===完成DuReader Train数据处理===")

    def drcd_json(self, data):
        """其次处理DRCD数据"""
        for dtem in data:
            paragraphs = dtem["paragraphs"]
            for ptem in paragraphs:
                context = obj.convert(ptem["context"][:600])
                qas = ptem["qas"]
                for qtem in qas:
                    query = obj.convert(qtem["question"])
                    answer = obj.convert(qtem["answers"][0]["text"])
                    self.output["drcd_train_items"].append({
                        "context": context, "query": query, "answer": answer
                    })

    def deal_drcd_dataset(self):
        for file in os.listdir("/home/yckj2453/nlp_space/kbqa/DataSet/MultiTask/DRCD/"):
            with open("/home/yckj2453/nlp_space/kbqa/DataSet/MultiTask/DRCD/" + file, "r", encoding="UTF-8") as f:
                self.drcd_json(json.load(f)["data"])

        print("===完成DRCD数据处理===")

    def cloudwalk_json(self, data_path, output_data_path):
        """处理自定义数据"""
        data_output = {"train_items": []}
        with open(data_path, mode='r', encoding='utf-8') as fp:
            origin_data = json.load(fp)
            for art_idx, single_artical in enumerate(tqdm(origin_data)):
                for pgh_idx, paragraph in enumerate(single_artical['paragraphs']):
                    for qa_ids, qas in enumerate(paragraph['annotations']):
                        if self.a2q:
                            data_output['train_items'].append({
                                "context": paragraph["text"],
                                "query": qas["Q"],
                                "answer": qas["A"]
                            })
                        else:
                            data_output['train_items'].append({
                                "context": paragraph["text"],
                                "query": qas["A"],
                                "answer": qas["Q"]
                            })

        for i in range(3):
            shuffle(data_output['train_items'])  # 数据随机化
        print("cloudwalk用于训练数据一共有{}条".format(len(data_output['train_items'])))

        with open(output_data_path, mode='wb') as f:
            pickle.dump(data_output, f)

    def cloudwalk_json_slide(self, data_path, output_data_path):
        """为数据上下文添加化窗"""
        data_output = {"train_items": []}
        with open(data_path, mode='r', encoding='utf-8') as fp:
            origin_data = json.load(fp)
            for art_idx, single_artical in enumerate(tqdm(origin_data)):
                paragraph_str = str()
                qas = []
                for pgh_idx, paragraph in enumerate(single_artical['paragraphs']):
                    text = paragraph['text']  # 是否需要对末尾标点进行判断处理
                    paragraph_str += text
                    qas.extend(paragraph['annotations'])

                if paragraph_str and qas:
                    artical_data = self.slide_paragraph(paragraph_str, qas, max_len=512, step=3)
                    if artical_data:
                        data_output['train_items'].extend(artical_data)

        for i in range(3):
            shuffle(data_output['train_items'])  # 数据随机化
        print("cloudwalk用于训练数据一共有{}条".format(len(data_output['train_items'])))

        with open(output_data_path, mode='wb') as f:
            pickle.dump(data_output, f)

    def slide_paragraph(self, doc, answers, max_len=512, step=3):
        """
        :param doc:
        :param answers:
        :param max_len:
        :param step:
        :return:
        """
        sentences = self.split_sentence(doc, flag='zh')

        output_data = []
        start_id = 0
        while True:
            flag = True  # 跳出循环标志
            paragraph = str()
            for idx, sentence in enumerate(sentences[start_id:]):
                if len(paragraph) + len(sentence) > max_len:
                    flag = False
                    break
                else:
                    paragraph += sentence
            if paragraph:
                # 查找context对应的QA对
                for qa in answers:
                    if paragraph.find(qa['A']) >= 0:
                        if self.a2q:
                            output_data.append({
                                "context": paragraph,
                                "query": qa["Q"],
                                "answer": qa["A"]
                            })
                        else:
                            output_data.append({
                                "context": paragraph,
                                "query": qa["A"],
                                "answer": qa["Q"]
                            })
            if flag:
                break
            start_id += step

        return output_data

    def split_sentence(self, paragraph: str, flag: str = "all", limit: int = 510):
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


if __name__ == "__main__":
    data_obj = DatasetPreprocess()
    # print("处理DuReader数据.")
    # data_obj.deal_dureader_dataset()
    #
    # for i in range(3):
    #     shuffle(data_obj.output["dureader_train_items"])
    #     # shuffle(data_obj.output["drcd_train_items"])
    #     # shuffle(data_obj.output["cmrc_train_items"])
    # print("DuReader用于训练的数据一共有%d条" % len(data_obj.output["dureader_train_items"]))
    # print("CMRC2018用于训练的数据一共有%d条" % len(data_obj.output["cmrc_train_items"]))
    # print("DRCD用于训练的数据一共有%d条" % len(data_obj.output["drcd_train_items"]))
    # with open("DataSet/multi_task.pkl", "wb") as f:
    #     pickle.dump(data_obj.output, f)

    # print("处理DRCD数据.")
    # data_obj.deal_drcd_dataset()

    print("处理CloudWalk数据.")
    data_path = "./DataSet/cloudwalk_train_enhancement.json"
    output_data_path = "./DataSet/cloudwalk_dataset_enhancement_q2a.pkl"
    data_obj.cloudwalk_json(data_path, output_data_path)
    # data_obj.cloudwalk_json_slide(data_path, output_data_path)
