"""
# 共用functions
"""
import torch


def comput_unilm_attention_mask(token_type_ids, attention_mask):
    """ 进行生产数据时直接单条制作2d attention mask, 使其适用于unilm的训练
       token_type_id shape: [sequence_length, ],
       attention_mask shape: [sequence_length, ]
       return shape [sequence_length, sequence_length]"""
    assert token_type_ids.shape == attention_mask.shape
    device = attention_mask.device

    seq_len = token_type_ids.shape[0]
    # 1,num_heads,seq,seq
    ones = torch.ones(seq_len, seq_len, device=device)

    # 下三角矩阵
    a_mask = torch.tril(ones)
    s_ex12 = token_type_ids.unsqueeze(0)
    s_ex13 = token_type_ids.unsqueeze(1)

    # [batch, num_heads, seq, seq]
    a_mask = (1 - s_ex13) * (1 - s_ex12) + a_mask * s_ex13

    attention_mask = attention_mask.unsqueeze(0).expand(seq_len, -1)  # 添加padding

    return a_mask * attention_mask
