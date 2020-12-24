# -*- coding: utf-8 -*-
import os
import time
import json
import random
import numpy as np
import argparse
import re
import jieba
from tqdm import tqdm

from vocab import Vocab

pattern_pun = '！，；：？、。"!,;:?."\''
pattern_jpg = re.compile(r'[A-Za-z0-9]+\.jpg')

patterns = [
    r".*?导购客服多多",
    r".*?机器是出现了什么问题",
    r".*?购物过程中有什么问题",
    r".*?人工客服",
    r".*?在的在的",
    r".*?您好.*?旗舰",
    r".*?您好.*?咨询",
    r".*?您好.*?帮助",
    r".*?有什么可以帮",
    r".*很高兴为您服务",
    r".*很荣幸为您服务",
    r".*为您",
    r".*帮您",
]
patterns = [re.compile(pattern) for pattern in patterns]

MAX_LEN = 160
MIN_LEN = 20


def save_info_to_json(file_path, object):
    """将数据object保存到json文件"""
    with open(file_path, mode='w', encoding='utf-8') as fw:
        json.dump(object, fw, ensure_ascii=False, indent=2)


def match_robot(text, first_ans):
    # if len(text) < MIN_LEN:
    #     return False
    # if len(text) > MAX_LEN:
    #     return True
    if not first_ans:
        return False
    for pattern in patterns:
        result = pattern.match(text)
        if result:
            return True
    return False


def is_usefull_ans(ans):
    if ans == "<url>" or ans.find('action?') >= 0:
        return False
    if len(ans) > 4:
        t = re.sub(r'#E-s\d{2,}', '', ans).strip()
        if len(t) <= 1:
            return False
        return True
    meanless_patterns = ["是的", "好的", "在的", "你好", "您好", "嗯", "恩", "呢", "哦", "哈"]
    for pattern in meanless_patterns:
        ans = re.sub(r'{}'.format(pattern), '', ans)
    if len(ans) <= 1:
        return False
    return True


ans_dict = {}  # 存储train answer出现的次数


def random_delete(ans):
    if ans not in ans_dict:
        ans_dict[ans] = 1
    else:
        ans_dict[ans] += 1
    c = ans_dict[ans]
    if c > 10000:
        if random.random() > 0.3:
            # 10000次之上, 之后概率大于0.3数据集全删除
            return True
        else:
            return False
    elif c > 1000:
        if random.random() > 0.5:
            return True
        else:
            return False
        pass
    elif c > 100:
        if random.random() > 0.8:
            return True
        else:
            return False
    else:
        return True


def clean_text(text):

    # text = re.sub(r'[{}]+'.format(r'\d+\*\*'), ' <num> ', text)
    # text = re.sub(r'[{}]+'.format(r'\d+'), ' <num> ', text)
    text = re.sub(r'\d{4,}\*\*', '<num>', text)
    text = re.sub(r'\d{4,}', '<num>', text)

    # text = clean_punctuation(text)
    return text


def clean_punctuation(text):
    text = re.sub(r'[{}]+'.format(pattern_pun), '', text)
    return text.strip().lower()


def tokenize_spt(text):
    sp_token = ['<img>', '<url>', '<sos>',
                '<eos>', '<num>', '</s>',
                '<unk>', '<speaker1>', '<speaker2>']

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


class DataIterm(object):
    def __init__(self, sid, ques, ans, ctx, pid):
        self.sid = sid
        self.ques = ques
        self.ans = ans
        self.ctx = ctx
        self.pid = pid


class ProductInfo(object):
    def __init__(self, kb_f):
        self.kb_f = kb_f
        self.product_infos = {}
        self.load()

    def load(self):
        import json
        with open(self.kb_f) as f:
            infos = json.load(f)
        for index, info in enumerate(infos):
            # content = [" ".join(str(index))]
            pid = info["pid"]
            # for k, v in info.items():
            #     if k == "pid":
            #         continue
            #     content.append(clean_text(k)+":"+clean_text(v))
            # self.product_infos[pid] = content
            self.product_infos[pid] = info

    def get_info_by_pid(self, pid=None):
        if not pid:
            return "<unk>"
        if pid not in self.product_infos:
            return "<unk>"
        # return ";".join(self.product_infos[pid])
        # kb_info 分类
        kb_info_class = self.product_infos[pid].get("分类")
        if kb_info_class:
            return kb_info_class
        else:
            return "<unk>"


def do_preprocess(args):
    """
    :param args: directory: 官方数据存放路径
                sess_turns: context中保存的历史上下文的对话轮数(此处采取保存全部历史轮数据)
    :return: train_items, dev_items
             用于训练的train和dev数据，其中每条数据记录由以下几部分原始信息组成
             sid, 对话原始的session信息，后续按照需要可以根据该信息查询对话相关的知识库，本实例中未使用
             question, 该条训练数据所对应的用户的问题
             answer, 该条训练数据对应的客服的回答
             context, 该对话发生的上下文信息，该信息最大信息长度不超过sess_turn所定义的轮数
    """
    sess_len = args.sess_turns * 2
    train_items = list()
    dev_items = list()

    for file, item_list in [(args.train_data_name, train_items), (args.dev_data_name, dev_items)]:
        with open(os.path.join(args.directory, file), mode='r', encoding='utf-8') as f:
            lines = f.readlines()
        data_list = list()
        sess_pid = dict()
        pre_sid = None
        first_ques = False
        first_ans = False
        text_empty_num = 0
        for line in tqdm(lines, ncols=80):
            word = line.strip().split('\t')
            sid = word[0]
            shop = word[1]
            pid = word[2]
            text = word[3]
            waiter = word[4]

            if not text:
                # 删除空文本
                text_empty_num += 1
                continue

            # 清理过长空格
            text = ' '.join(text.strip().split())

            if sid != pre_sid:
                pre_sid = sid
                first_ques = True
                first_ans = False

            if pid:
                sess_pid[sid] = pid

            if waiter == '1':
                first_ans = first_ques
                text = 'A:' + text
            else:
                first_ques = first_ques and (not first_ans)
                text = 'Q:' + text

            # if match_robot(text, first_ans):
            #     # 删除首轮回复的[您好欢迎光临]
            #     continue

            if len(text) > MAX_LEN:
                text = text[:MAX_LEN]

            data_list.append((sid, text))
        print("{} empty text number: {}".format(file, text_empty_num))

        data_len = len(data_list)
        i = 0

        tmp_data_list = list()
        single_tmp_data_list = []  # 保存single conversation QA data

        # 将原始数据按照session和问题、回答类型，
        # 用'|||'连接不同回车发送的内容
        while i < data_len:

            i_head = data_list[i][1][0]
            i_text = data_list[i][1]
            i_sid = data_list[i][0]

            j = i+1
            if j >= data_len:
                if len(single_tmp_data_list) > 0:
                    # 将最后一个conversation入列
                    single_tmp_data_list.append((i_sid, i_text))
                    tmp_data_list.append(single_tmp_data_list)
                break

            j_head = data_list[j][1][0]
            j_text = data_list[j][1]
            j_sid = data_list[j][0]

            add = 0
            while i_head == j_head and i_sid == j_sid:
                # 去除需要连接文本左右两侧的‘|’字符
                i_text = i_text + '|||' + j_text[2:].strip('|')
                # i_text = i_text + '|||' + j_text[2:]
                add = add + 1
                j = j + 1

                if j >= data_len:
                    break

                j_head = data_list[j][1][0]
                j_text = data_list[j][1]
                j_sid = data_list[j][0]

            i = i + add + 1
            # 单条数据的获取
            single_tmp_data_list.append((i_sid, i_text))
            # 判断是否还在同一session ID
            if i_sid != j_sid:
                # session ID 不相同, 进入下一个conversation
                if len(single_tmp_data_list) > 1:
                    # 不成对数据丢弃
                    tmp_data_list.append(single_tmp_data_list)
                single_tmp_data_list = []  # 另起一个conversation

        # 遍历全部（session, Q:xxx） (session, A:xxx),
        # 构建训练输入文件，Q，A，Context，
        # 其中'@@@'间隔Context里面不同的Q或者A
        empty_num = 0
        data_num = 0
        for conversation_id, conversation_data in enumerate(tqdm(tmp_data_list, ncols=80)):
            single_conversation = []
            session_id = None
            for idx, item in enumerate(conversation_data):
                if idx == 0:
                    session_id = item[0]
                sid = item[0]
                text = item[1]

                assert session_id == sid  # 判断一个conversation内的session号是否都相同

                if text.startswith('A'):
                    continue

                question = text.replace('Q:', '').strip()

                if question == '':
                    continue

                if idx+1 >= len(conversation_data):
                    continue

                n_item = conversation_data[idx+1]
                n_sid = n_item[0]

                if sid != n_sid:
                    continue

                n_text = n_item[1]

                answer = n_text.replace('A:', '').strip()

                if answer == '':
                    continue

                # 获取历史内容
                # cand_data_list = conversation_data[:idx]  # 获取每个conversation全部的历史轮
                if idx > sess_len:
                    # 获取前n轮历史
                    cand_data_list = conversation_data[idx - sess_len:idx]
                else:
                    cand_data_list = conversation_data[:idx]

                contxt_list = list()
                for cand_item in cand_data_list:
                    cand_sid = cand_item[0]
                    cand_text = cand_item[1]
                    if not cand_text[2:]:
                        empty_num += 1

                    if cand_sid != sid:
                        continue
                    contxt_list.append(cand_text)

                # 上下文的句子使用@@@连接
                context = '@@@'.join(contxt_list)
                pid = sess_pid[sid] if sid in sess_pid else ""
                single_conversation.append(DataIterm(sid, question, answer, context, pid))
            item_list.append(single_conversation)
            data_num += len(single_conversation)
        print('the {}, empty data numbers: {}'.format(file, empty_num))
        print('all data sample number: {}'.format(data_num))
    return train_items, dev_items


def gen_train_dev_set(args, train_items, dev_items, kb_f=None):
    datasets = {}
    voc_data = {}
    if kb_f is not None:
        product_info = ProductInfo(kb_f)

    for type in ['train', 'dev']:
        if type == 'train':
            items = train_items
        elif type == 'dev':
            items = dev_items

        conversations = []
        candidates_answers = []
        conversations_voc_list = []
        target_single_empty_num = 0
        target_empty_num = 0
        for conversation_item in tqdm(items, ncols=80):
            conversation = {}
            utterances = []
            session_pid = None
            voc_list = []
            for idx_, item in enumerate(conversation_item):
                single_data = {}
                img_list = list()
                src_str = ''
                qus_str = ''
                trg_str = ''

                ques = item.ques.strip()
                ans = item.ans.strip()
                ctx = item.ctx
                # 保存 conversation session id
                session_id = item.sid
                session_pid = item.pid

                ctx_list = ctx.split('@@@')
                ctx_str_list = []
                for idx, sent_i in enumerate(ctx_list):
                    if sent_i == '':
                        continue
                    sent_i_type = sent_i[0]  # 说话人角色
                    if (len(ctx_list) % 2 == 1
                            and sent_i_type == "A"
                            and idx == 0):
                        # 历史轮数为奇数, 且第一轮为bot表达
                        continue

                    sent_i = sent_i[2:].strip()
                    sent_i_list = sent_i.split('|||')
                    single_ctx_str = str()
                    for sent_j in sent_i_list:

                        if sent_j.endswith('.jpg'):
                            img_list.append(sent_j.strip('|'))
                            sent_j = '<img>'
                        else:
                            img_list.append('NULL')
                            sent_j = clean_text(sent_j)

                        if args.seg_voc:
                            single_seg_ctx = tokenize_spt(sent_j.strip())
                            voc_list.append(single_seg_ctx)
                            sent_seg_ctx = ' '.join(single_seg_ctx)
                        else:
                            # 预训练模型不进行拆词
                            sent_seg_ctx = sent_j.strip()

                        if sent_seg_ctx:
                            # 多句拼接
                            src_str = src_str + sent_seg_ctx + '</s>'
                            single_ctx_str = single_ctx_str + sent_seg_ctx + ' '
                        else:
                            img_list.pop(-1)
                    # 将单轮历史记录添加至列表
                    ctx_str_list.append(single_ctx_str)
                if len(ctx_str_list) == 0:
                    # 上下文为空, 直接删除此单条数据
                    continue
                assert len(ctx_str_list) % 2 == 0, "非2的倍数"

                ques_list = ques.split('|||')
                for sent_q in ques_list:
                    if sent_q.endswith('.jpg'):
                        img_list.append(sent_q.strip('|'))
                        sent_q = '<img>'
                    else:
                        img_list.append('NULL')
                        sent_q = clean_text(sent_q)

                    # sent_q = sent_q.strip()
                    if sent_q:
                        if args.seg_voc:
                            # 分词
                            single_seg_q = tokenize_spt(sent_q.strip())
                            voc_list.append(single_seg_q)
                            sent_seg_q = ' '.join(single_seg_q)
                        else:
                            sent_seg_q = sent_q.strip()
                        src_str = src_str + sent_seg_q + '</s>'
                        qus_str = qus_str + sent_seg_q + ' '
                    else:
                        img_list.pop(-1)
                if not qus_str:
                    # 问题为空, 直接删除此单条数据
                    continue
                # 将当前数据的Q加入到历史列表的最后
                ctx_str_list.append(qus_str)

                candidates = []  # 存储单条数据的target
                ans_list = ans.split('|||')
                for sent_a in ans_list:
                    if sent_a.endswith('jpg'):
                        sent = '<img>'
                        # 去除answer中的<img>字符
                        continue
                    else:
                        sent_a = clean_text(sent_a)
                    # if (len(sent_a) <= 2
                    #         or sent_a == '<url>'
                    #         or sent_a.find('action?') >= 0):
                    #     # 删除单句过短的句子
                    #     target_single_empty_num += 1
                    #     continue
                    if not is_usefull_ans(sent_a):
                        target_single_empty_num += 1
                        continue

                    if sent_a:
                        # 单行依然使用</s>分割符分开
                        if args.seg_voc:
                            single_seg_a = tokenize_spt(sent_a.strip())
                            voc_list.append(single_seg_a)
                            trg_str = trg_str + ' '.join(single_seg_a) + ' '
                        else:
                            trg_str = trg_str + sent_a.strip() + ' '
                if not trg_str:
                    # 删除answer为空conversation
                    target_empty_num += 1
                    continue
                candidates.append(trg_str)

                if src_str.endswith("</s>"):
                    src_str = src_str[:-4]
                # # 上下文+问题内容中所有包含图片的句子
                # img_str = ' '.join(img_list)

                src_list = src_str.split('</s>')
                assert len(src_list) == len(img_list)
                if '<img>' not in trg_str and trg_str:
                    single_data['candidates'] = candidates
                    single_data['history'] = ctx_str_list
                    single_data['img_list'] = img_list
                    single_data['sid'] = session_id
                    # 将单条数据的答案添加进入答案列表
                    if len(candidates[-1]) <= 128:
                        # 候选答案长度必须小于阈值
                        candidates_answers.extend(candidates)
                    # 单个conversation数据列表集合
                    utterances.append(single_data)

            # 添加单个conversation水平的数据
            pid_str = None
            if kb_f:
                # 获取PID info
                pid_str = product_info.get_info_by_pid(session_pid)
                if args.seg_voc:
                    # 分词
                    seg_pid_str = pid_str
                    single_seg_pid = tokenize_spt(seg_pid_str.strip())
                    voc_list.append(single_seg_pid)
                    pid_str = str()  # 置空重新赋值
                    pid_str = ' '.join(single_seg_pid)

            conversation['personality'] = pid_str
            conversation['utterances'] = utterances
            # 将conversation数据存入列表
            conversations.append(conversation)

            # conversation voc list 存入 conversations_voc_list
            conversations_voc_list.append(voc_list)

        print("the {}, target single empty num: {}, target empty num: {}".format(type,
                                                                                 target_single_empty_num,
                                                                                 target_empty_num))
        # 添加每个type词汇
        voc_data[type] = conversations_voc_list
        datasets[type] = conversations

        if args.distractor_nums > 0:
            # 仅仅在进行multi-task时才需要random candidate set
            # randomly sampled utterances from all answers as distractors
            candidates_answers = np.array(list(set(candidates_answers)))
            candidates_answers_len = len(candidates_answers)
            for i in tqdm(range(len(datasets.get(type))), ncols=80):
                utterances_len = len(datasets.get(type)[i].get('utterances'))
                # 生成一个随机数据矩阵
                idx_arr = np.random.randint(0, candidates_answers_len, (utterances_len, args.distractor_nums))
                for j in range(utterances_len):
                    # distractors = random.sample(candidates_answers, args.distractor_nums)
                    distractors = list(candidates_answers[idx_arr[j]])
                    # 更新每个样本数据的候选集, 增加混淆项
                    datasets[type][i]['utterances'][j]['candidates'] = \
                        distractors + datasets.get(type)[i].get('utterances')[j].get('candidates')

    # 保存数据集
    save_info_to_json(os.path.join(args.directory, args.out_json_name), datasets)

    return voc_data


def generate_vocabulary(args, voc_data):
    """保存词汇表"""
    vocab = Vocab(lang='zh')
    for key, value in voc_data.items():
        print('Save Vocabulary...')
        vocab.add_dataframe(value)
        vocab.update(max_size=args.max_vocab_size, min_freq=args.min_vocab_frequency)

        print('Vocabulary size: ', len(vocab))
        vocab.pickle(os.path.join(args.directory, 'word2id.pkl'),
                     os.path.join(args.directory, 'id2word.pkl'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="tool to process raw data")

    parser.add_argument('-d', '--directory', default='./data/')
    parser.add_argument('-train', '--train_data_name', default='data_train.txt')
    parser.add_argument('-dev', '--dev_data_name', default='data_dev.txt')
    parser.add_argument('-out', '--out_json_name', default='datasets_ctx.json')
    parser.add_argument('-s', '--sess_turns', default=2)
    parser.add_argument('-d_nums', '--distractor_nums', default=0)
    parser.add_argument('-f', '--kb_f', default='./data/kb_info.json')
    parser.add_argument('--seg_voc', action='store_true', default=True, help='是否分词')
    parser.add_argument('--save_voc', action='store_true', default=True, help='是否存储词汇表')
    # Vocabulary
    parser.add_argument('--max_vocab_size', type=int, default=50000)
    parser.add_argument('--min_vocab_frequency', type=int, default=3)

    args = parser.parse_args()

    train_items, dev_items = do_preprocess(args)

    conversations_vocab_data = gen_train_dev_set(args, train_items, dev_items, kb_f=args.kb_f)

    if args.seg_voc and args.save_voc:
        generate_vocabulary(args, conversations_vocab_data)
