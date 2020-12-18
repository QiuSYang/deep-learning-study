"""
# 答案提取共有函数
"""
import torch
from dataclasses import dataclass
from typing import List, Optional, Tuple
from transformers.modeling_outputs import TokenClassifierOutput

# label
FLAG_INVALID = -100
FLAG_NONE = 0
FLAG_START = 1
FLAG_CONTENT = 2
FLAG_CLS = 3
FLAG_SEP = 4


@dataclass
class CustomCrfTokenClassifierOutput(TokenClassifierOutput):
    """Base class for outputs of token classification models.(last logits add crf model)"""
    loss: Optional[torch.FloatTensor] = None
    target_sequence: Optional[List[int]] = None


def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))