"""
# transformer结构的一些层
"""
import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention, self-attention计算层"""
    def __init__(self, temperature, attn_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        """q shape: [batch, n_head, sequence_len, hidden_size]
           k shape: same q, 因此转置的是2,3维度
           v shape: same q
           Attention(Q,K,V) = softmax(QK^T / sqrt(d_k))*V
        """
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            # F.softmax(-1e9) = 0
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention module"""
    def __init__(self):
        pass


class PositionwiseFeedForward(nn.Module):
    """A two-feed-forward-layer module"""
    def __init__(self, d_in, d_hid, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)  # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """FFN(x) = max(0, xW_1 + b_1)W_1 + b_2, max(0, y)<=>relu(y)"""
        residual = x  # 残差

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual  # 残差block

        x = self.layer_norm(x)

        return x
