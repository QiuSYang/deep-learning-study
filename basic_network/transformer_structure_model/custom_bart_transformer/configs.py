import os
import argparse
from datetime import datetime
from collections import defaultdict
from pathlib import Path
import pprint
from torch import optim
import torch.nn as nn

project_dir = Path(__file__).resolve().parent
data_dir = project_dir.joinpath('./data/')
optimizer_dict = {'RMSprop': optim.RMSprop, 'Adam': optim.Adam}
rnn_dict = {'lstm': nn.LSTM, 'gru': nn.GRU}
save_dir = project_dir.joinpath('./ckpt/')
pred_dir = project_dir.joinpath('./pred/')


def str2bool(v):
    """string to boolean"""
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class Config(object):
    def __init__(self, **kwargs):
        """Configuration Class: set kwargs as class attributes with setattr"""
        if kwargs is not None:
            for key, value in kwargs.items():
                if key == 'optimizer':
                    value = optimizer_dict[value]
                if key == 'rnn':
                    value = rnn_dict[value]
                setattr(self, key, value)

        # Dataset directory: ex) ./datasets/cornell/
        self.dataset_dir = project_dir.joinpath(self.data.lower())

        # Data Split ex) 'train', 'valid', 'test'
        self.data_dir = self.dataset_dir.joinpath(self.mode)
        # Pickled Vocabulary
        self.word2id_path = self.dataset_dir.joinpath('word2id.pkl')
        self.id2word_path = self.dataset_dir.joinpath('id2word.pkl')

        # ids 数据集文件地址(train and dev 共同组成datasets)
        self.train_ids_path = self.dataset_dir.joinpath('datasets_ctx.json')
        # self.val_ids_path = self.dataset_dir.joinpath('dev_ids.json')
        self.train_cache_path = self.dataset_dir.joinpath('datasets_ctx.pkl')
        self.test_ids_path = self.dataset_dir.joinpath('test_datasets.json')

        os.makedirs(pred_dir, exist_ok=True)
        self.pred_path = pred_dir.joinpath('res.txt')

        # Save path
        if self.mode == 'train' and self.checkpoint is None:
            time_now = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
            self.save_path = save_dir.joinpath(self.data, self.model, time_now)
            self.logdir = self.save_path
            os.makedirs(self.save_path, exist_ok=True)
        elif self.checkpoint is not None:
            assert os.path.exists(self.checkpoint)
            self.save_path = os.path.dirname(self.checkpoint)
            self.logdir = self.save_path

        # Save BLEU
        self.good_qa_threshold = 0.08
        self.predict_json_path = project_dir.joinpath('./offline_dev_data/dev_answers_predict.json')
        self.target_json_path = project_dir.joinpath('./offline_dev_data/dev_answers_target.json')
        self.question_json_path = project_dir.joinpath('./offline_dev_data/dev_questions.json')
        self.good_question_json_path = project_dir.joinpath('./offline_dev_data/good_dev_questions.json')
        self.bad_question_json_path = project_dir.joinpath('./offline_dev_data/bad_dev_questions.json')

    def __str__(self):
        """Pretty-print configurations in alphabetical order"""
        config_str = 'Configurations\n'
        config_str += pprint.pformat(self.__dict__)
        return config_str


def get_config(parse=True, **optional_kwargs):
    """
    Get configurations as attributes of class
    1. Parse configurations with argparse.
    2. Create Config class initilized with parsed kwargs.
    3. Return Config class.
    """
    parser = argparse.ArgumentParser()

    # Mode
    parser.add_argument('--mode', type=str, default='train')

    parser.add_argument('--ctx_length', type=int, default=132)
    parser.add_argument('--ans_length', type=int, default=100)
    parser.add_argument('--max_sentence_token_num', type=int, default=128)

    # Train
    parser.add_argument('--per_gpu_train_batch_size', type=int, default=128)
    parser.add_argument('--per_gpu_eval_batch_size', type=int, default=16)
    parser.add_argument('--per_gpu_test_batch_size', type=int, default=128)
    parser.add_argument('--n_epoch', type=int, default=30)
    parser.add_argument('--checkpoint', type=str,
                        # default=None,
                        # default='ckpt/data/BART/2020-09-05_13:32:12/26.pkl',
                        default='ckpt/data/BART/2020-09-10_05:44:08/7.pkl',
                        help="预测模型路径.")
    parser.add_argument('--print_every', type=int, default=100)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--save_every_epoch', type=int, default=1)

    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--num_warmup_steps", default=500, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--num_training_steps", default=-1, type=int,
                        help="set total number of training steps to perform")

    # Data
    parser.add_argument('--data', type=str, default='./data/')
    parser.add_argument('--seg_voc', type=str2bool, default=True,
                        help="是否进行分词.")
    parser.add_argument("--num_candidates", type=int, default=1,
                        help="候选集数量.")
    parser.add_argument("--multi_task", type=str2bool, default=False,
                        help="是否多任务.")
    parser.add_argument("--lm_coef", type=float, default=1.0, help="LM loss coefficient")
    parser.add_argument("--mc_coef", type=float, default=1.0, help="Multiple-choice loss coefficient")

    # Transformer
    parser.add_argument('--model', type=str, default='BART',
                        help='model type, the default one is BART')
    parser.add_argument('--model_type', type=str, default='TransformerModel')
    parser.add_argument('--activate_func', type=str, default='gelu_new')
    parser.add_argument('--n_layers', type=int, default=6)
    parser.add_argument('--n_head', type=int, default=8)
    parser.add_argument('--embed_size', type=int, default=512)
    parser.add_argument('--ffn_dim', type=int, default=2048)
    parser.add_argument('--num_labels', type=int, default=3)  # 分类任务(bart默认配置为3)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--bart_pre_training', type=str,
                        default=None,  # project_dir.joinpath('bart_per_training_model'),
                        help='bart per-training model.')
    parser.add_argument('--gradient_accumulation_step', type=int, default=1,
                        required=False, help='梯度积累')
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--inference", type=str2bool, default=True,
                        help="random seed for initialization")
    parser.add_argument("--image_para_freeze", type=str2bool, default=True,
                        help="图像编码是否冻结参数.")
    parser.add_argument("--model_val", type=str2bool, default=False,
                        help="random seed for initialization")

    parser.add_argument('--max_generate_length', type=int, default=100)
    parser.add_argument('--beam_size', type=int, default=3)

    # Parse arguments
    if parse:
        kwargs = parser.parse_args()
    else:
        kwargs = parser.parse_known_args()[0]

    # Namespace => Dictionary
    kwargs = vars(kwargs)
    kwargs.update(optional_kwargs)

    return Config(**kwargs)
