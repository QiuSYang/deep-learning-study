# -*- coding: utf-8 -*-

import json
import argparse
from raw_data_preprocess_transfertransfo import *
from pathlib import Path

# set default path for data and test data
project_dir = Path(__file__).resolve().parent
datasets_dir = project_dir.joinpath('./data/')
images_test_dir = project_dir.joinpath('./online_test_data/images_test/')


def preprocessing_ctx(ctx_list):
    """预处理上下文text, 删除空行"""
    clean_ctx_list = []
    flag = False  # 标志上一轮是否为空(Q或者A)
    for single_ctx in ctx_list:
        if not single_ctx[2:]:
            # 当前轮为空
            flag = True
            continue
        if flag:
            # 直接将当前轮数据与clean_ctx_list[-1]拼接(他们是同一个人说的)
            clean_ctx_list[-1] = clean_ctx_list[-1] + "|||" + single_ctx[2:]
            flag = False
        else:
            # 原始数据不为空，直接不做任何处理
            clean_ctx_list.append(single_ctx)

    if len(clean_ctx_list) == 1 and clean_ctx_list[0][0] == 'Q':
        clean_ctx_list.append("A:~")  # 人为的多＋一个波浪线answer(主要是为了不丢弃这条Q)

    return clean_ctx_list


def gen_test_set(args):
    sess_length = args.sess_turns * 2
    with open(os.path.join(args.directory + 'test_questions.json'), mode='r', encoding='utf-8') as f:
        items = json.load(f)
        
    if args.kb_f:
        product_info = ProductInfo(args.kb_f)

    datasets = {}
    # all conversations
    conversations = []
    for item in items:
        # single conversation 所有数据
        conversation = {}
        utterances = []
        single_data = {}

        img_list = list()
        src_str = ''
        # ctx_str = ''
        # qus_str = ''

        ques_id = item['Id']
        session_id = item['SessionId']
        p_id = item["ProductId"]
        # ctx_list = item['Context']
        origin_ctx_list = item['Context']
        ques = item['Question']

        ctx_list = preprocessing_ctx(origin_ctx_list)
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
                # 清理过长空格
                sent_j = ' '.join(sent_j.strip().split())
                # fir_ans = (idx <= 1 and sent_i_type == "A" and len(item['Context']) == len(ctx_list))
                # if match_robot(sent_j, fir_ans):
                #     continue
                if len(sent_j) > MAX_LEN:
                    # 超长截取
                    sent_j = sent_j[:MAX_LEN]

                if sent_j.endswith('.jpg'):
                    img_list.append(sent_j)
                    sent_j = '<img>'
                else:
                    img_list.append('NULL')
                    sent_j = clean_text(sent_j)
                if args.seg_voc:
                    sent_seg = ' '.join(tokenize_spt(sent_j.strip()))
                else:
                    # bert预训练模型不需要分词
                    sent_seg = sent_j.strip()

                if sent_seg:
                    src_str = src_str + sent_seg + '</s>'
                    single_ctx_str = single_ctx_str + sent_seg + ' '
                else:
                    img_list.pop(-1)
            ctx_str_list.append(single_ctx_str)

        assert len(ctx_str_list) % 2 == 0, "非2的倍数"
        ques_list = ques.split('|||')
        qus_str = str()
        for sent in ques_list:
            # 清理过长空格
            sent = ' '.join(sent.strip().split())
            if len(sent) > MAX_LEN:
                # 超长截取
                sent = sent[:MAX_LEN]

            if sent.endswith('.jpg'):
                img_list.append(sent)
                sent = '<img>'
            else:
                img_list.append('NULL')
                sent = clean_text(sent)

            if sent:
                if args.seg_voc:
                    sent_seg = ' '.join(tokenize_spt(sent.strip()))
                else:
                    sent_seg = sent.strip()
                src_str = src_str + sent_seg + '</s>'
                qus_str = qus_str + sent_seg + ' '  # 行与行之间使用空格隔开
            else:
                img_list.pop(-1)
        ctx_str_list.append(qus_str)

        if src_str.endswith("</s>"):
            src_str = src_str[:-4]
        img_str = ' '.join(img_list)
        src_list = src_str.split('</s>')
        assert len(src_list) == len(img_list)
        # lazy to reuse train dataloader code
        candidates = []
        trg_str = 'NULL'
        candidates.append(trg_str)
        # 收集单条数据
        single_data['candidates'] = candidates
        single_data['history'] = ctx_str_list
        single_data['img_list'] = img_list
        single_data['sid'] = session_id
        # 单条数据的集合
        utterances.append(single_data)

        pid_str = None
        if args.kb_f:
            # 获取PID info
            pid_str = product_info.get_info_by_pid(p_id)
            if args.seg_voc:
                # 分词
                seg_pid_str = pid_str
                single_seg_pid = tokenize_spt(seg_pid_str.strip())
                pid_str = str()  # 置空重新赋值
                pid_str = ' '.join(single_seg_pid)

        conversation['personality'] = pid_str
        conversation['utterances'] = utterances
        # 将conversation数据存入列表
        conversations.append(conversation)
    # 将测试数据加入datasets
    datasets['test'] = conversations

    save_info_to_json(os.path.join(datasets_dir, args.out_json_name), datasets)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="tool to process online test data")

    parser.add_argument('-d', '--directory', default='online_test_data/')
    parser.add_argument('-t', '--sess_turns', default=2)
    parser.add_argument('-out', '--out_json_name', default='test_datasets.json')
    parser.add_argument('-f', '--kb_f', default='./data/kb_info.json')
    parser.add_argument('--seg_voc', action='store_true', default=True, help='是否分词')

    args = parser.parse_args()

    # get the online test file to generate internal data format
    gen_test_set(args)
