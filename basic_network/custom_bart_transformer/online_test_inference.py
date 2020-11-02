import os
import re
import torch
from bart_solver import BartSolver
from utils import get_data_loader, get_dataset
from configs import get_config
from vocab import Vocab
import pickle
from pathlib import Path

# set default path for data and test data
project_dir = Path(__file__).resolve().parent
datasets_dir = project_dir.joinpath('./data/')
images_test_dir = project_dir.joinpath('./online_test_data/images_test/')
# images_test_dir = project_dir.joinpath('./data/images_dev/')


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def set_distributed(config):
    """进行分布式基本配置"""
    if config.local_rank == -1:
        # DataParallel形式的分布式, 负载不平衡
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("{}: {}".format(config.local_rank, device))
        # 测试永远使用0号卡
        config.gpu_nums = 1 if torch.cuda.is_available() else 0
    else:
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(config.local_rank)
        device = torch.device('cuda', config.local_rank)
        print("{}: {}.".format(config.local_rank, device))
        torch.distributed.init_process_group(backend="nccl")
        config.gpu_nums = 1
        config.distributed = True
    config.device = device


if __name__ == '__main__':
    config = get_config(mode='test')
    # 设置分布式模式
    config.distributed = False  # 初始不支持分布训练
    set_distributed(config)

    print('Loading Vocabulary...')
    vocab = Vocab()
    vocab.load(config.word2id_path, config.id2word_path)
    print(f'Vocabulary size: {vocab.vocab_size}')
    config.vocab_size = vocab.vocab_size

    # print('Loading Vocabulary...')
    # vocab = CustomBertVocab(lang='zh')
    # vocab.load(os.path.join(config.bert_pre_training, 'vocab.txt'))
    # config.vocab_size = vocab.vocab_size
    # print(f'Vocabulary size: {vocab.vocab_size}')

    # 获取test数据集
    test_ids_json = get_dataset(tokenizer=vocab,
                                dataset_path=config.test_ids_path,
                                dataset_cache=None,
                                single_sentence_token_max_num=config.max_sentence_token_num,
                                seg_voc=config.seg_voc,
                                inference=config.inference)

    test_data_loader = get_data_loader(config, vocab, test_ids_json.get('test'),
                                       image_dir=images_test_dir,
                                       batch_size=config.per_gpu_test_batch_size,
                                       data_type='val',
                                       multi_task=config.multi_task)

    main = BartSolver(config=config,
                      vocab=vocab,
                      train_data_loader=None,
                      eval_data_loader=test_data_loader,
                      is_train=False)
    main.build_graph()
    # 生成结果
    main.generate_for_evaluation()

