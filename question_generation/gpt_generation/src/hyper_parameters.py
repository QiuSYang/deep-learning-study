"""
# 超参数
"""
import logging
from dataclasses import dataclass, field
from typing import Optional
import torch

logger = logging.getLogger(__name__)


@dataclass
class DataArguments:
    """数据预处理超参数"""
    dataset_path: str = field(
        metadata={"help": "原始数据路径."}
    )
    max_sequence_len: int = field(
        default=512, metadata={"help": "序列最大长度"}
    )
    max_condition_len: int = field(
        default=100, metadata={"help": "条件序列最大长度"}
    )
    max_target_len: int = field(
        default=50, metadata={"help": "目标序列最大长度"}
    )
    is_right_pad: bool = field(
        default=True, metadata={"help": "Padding补齐方式"}
    )
    is_condition_first: bool = field(
        default=False, metadata={"help": "条件文本是否放在最前面"}
    )
    is_unilm_mask: bool = field(
        default=False, metadata={"help": "是否构造unilm attention mask"}
    )


@dataclass
class ModelArguments:
    """模型超参数"""
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    use_fast: bool = field(default=False, metadata={"help": "Set this flag to use fast tokenization."})
    # If you want to tweak more attributes on your tokenizer, you should do it in a distinct script,
    # or just modify its tokenizer_config.json.
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )


class HyperParametersConfig(object):
    def __init__(self):
        # decoder data
        self.max_sequence_len = 512  # context_len + condition_len = target_len
        self.max_condition_len = 100
        self.max_target_len = 50
        self.is_right_pad = True
        self.is_condition_first = True
        self.is_unilm_mask = False
        self.dataset_path = "/home/yckj2453/nlp_space/kbqa/gpt_generation/data/chinese_medical_dataset.pt"

        # train
        self.seed = 2021
        self.model_name_or_path = "thu-coai/CDial-GPT2_LCCC-base"
        self.output_dir = "/home/yckj2453/nlp_space/kbqa/gpt_generation/models/2021-01-20"
        self.logging_dir = "/home/yckj2453/nlp_space/kbqa/gpt_generation/logs/2021-01-20"

        gpu_nums = torch.cuda.device_count()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.per_device_train_batch_size = 24
        self.per_device_eval_batch_size = 32
        self.gradient_accumulation_steps = 1
        self.max_steps = 15000
        self.warmup_steps = 1500
        self.save_steps = 1500
        self.eval_steps = 1500
        self.logging_steps = 150

        self.local_rank = -1

        self.do_train = True
        self.do_eval = True
        self.evaluation_strategy = "steps"
        # self.evaluate_during_training = True
        self.prediction_loss_only = True
