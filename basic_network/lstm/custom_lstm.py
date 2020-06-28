"""
# 使用 pytorch 自定义 LSTM 网络
"""
import os
import math
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

_logger = logging.getLogger(__name__)


class NaiveCustomLSTM(nn.Module):
    def __init__(self, input_sz: int, hidden_sz: int):
        super().__init__()
        # 一般数据输入为[bs, seq_size, feature_size]
        # 输入每个tokenize的size(即每个词向量编码长度)
        self.input_size = input_sz
        # 隐层每个tokenize的size
        self.hidden_size = hidden_sz

        # i_t
        self.W_i = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.U_i = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_i = nn.Parameter(torch.Tensor(hidden_sz))

        # f_t
        self.W_f = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.U_f = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_f = nn.Parameter(torch.Tensor(hidden_sz))

        # c_t
        self.W_c = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.U_c = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_c = nn.Parameter(torch.Tensor(hidden_sz))

        # o_t
        self.W_o = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.U_o = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_o = nn.Parameter(torch.Tensor(hidden_sz))

        self.init_weights()

    def init_weights(self):
        """ 权重与偏置初始化
        :return:
        """
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x, init_states=None):
        """assumes x.shape represents (batch_size, sequence_size, input_size)
        input: 前馈操作接收init_states参数，该参数是上面方程的（h_t, c_t）参数的元组，
                如果不引入，则设置为零。然后，我们对每个保留（h_t, c_t）的序列元素执行LSTM方程的前馈，
                并将其作为序列下一个元素的状态引入.
        return: 预测和最后一个状态元组
        """
        bs, seq_sz, _ = x.size()
        hidden_seq = []

        if init_states is None:
            # 初始化历史状态
            h_t, c_t = (
                torch.zeros(bs, self.hidden_size).to(x.device),
                torch.zeros(bs, self.hidden_size).to(x.device),
            )
        else:
            h_t, c_t = init_states

        for t in range(seq_sz):
            # 获取单个词的feature 向量
            x_t = x[:, t, :]

            i_t = torch.sigmoid(x_t @ self.W_i + h_t @ self.U_i + self.b_i)
            f_t = torch.sigmoid(x_t @ self.W_f + h_t @ self.U_f + self.b_f)
            g_t = torch.tanh(x_t @ self.W_c + h_t @ self.U_c + self.b_c)
            o_t = torch.sigmoid(x_t @ self.W_o + h_t @ self.U_o + self.b_o)
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)

            hidden_seq.append(h_t.unsqueeze(0))

        # reshape hidden_seq p/ retornar
        hidden_seq = torch.cat(hidden_seq, dim=0)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()

        return hidden_seq, (h_t, c_t)


class CustomLSTM(nn.Module):
    def __init__(self, input_sz, hidden_sz):
        super().__init__()
        # 一般数据输入为[bs, seq_size, feature_size]
        # 输入每个tokenize的size(即每个词向量编码长度)
        self.input_size = input_sz
        # 隐层每个tokenize的size
        self.hidden_size = hidden_sz

        self.W = nn.Parameter(torch.Tensor(input_sz, hidden_sz * 4))
        self.U = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz * 4))
        self.bias = nn.Parameter(torch.Tensor(hidden_sz * 4))

        self.init_weights()

    def init_weights(self):
        """ 权重与偏置初始化
        :return:
        """
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x, init_states=None):
        """assumes x.shape represents (batch_size, sequence_size, input_size)
        input: 前馈操作接收init_states参数，该参数是上面方程的（h_t, c_t）参数的元组，
                如果不引入，则设置为零。然后，我们对每个保留（h_t, c_t）的序列元素执行LSTM方程的前馈，
                并将其作为序列下一个元素的状态引入.
        return: 预测和最后一个状态元组
        """
        bs, seq_sz, _ = x.size()
        hidden_seq = []

        if init_states is None:
            # 初始化历史状态
            h_t, c_t = (
                torch.zeros(bs, self.hidden_size).to(x.device),
                torch.zeros(bs, self.hidden_size).to(x.device),
            )
        else:
            h_t, c_t = init_states

        HS = self.hidden_size
        for t in range(seq_sz):
            # 获取单个词的feature 向量
            x_t = x[:, t, :]

            # batch the computations into a single matrix multiplication
            gate = x_t @ self.W + h_t @ self.U + self.bias
            i_t, f_t, g_t, o_t = (
                torch.sigmoid(gate[:, :HS]),  # input
                torch.sigmoid(gate[:, HS:HS*2]),  # forget
                torch.tanh(gate[:, HS*2:HS*3]),
                torch.sigmoid(gate[:, HS*3:]),  # output
            )

            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)

            hidden_seq.append(h_t.unsqueeze(0))

        # reshape hidden_seq p/ retornar
        hidden_seq = torch.cat(hidden_seq, dim=0)
        # reshape from shape (sequence, batch, feature) to (batch, sequence, feature)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()

        return hidden_seq, (h_t, c_t)


class CustomLSTMPeephole(nn.Module):
    def __init__(self, input_sz, hidden_sz, peephole=False):
        super().__init__()
        # 一般数据输入为[bs, seq_size, feature_size]
        # 输入每个tokenize的size(即每个词向量编码长度)
        self.input_size = input_sz
        # 隐层每个tokenize的size
        self.hidden_size = hidden_sz
        self.peephole = peephole

        self.W = nn.Parameter(torch.Tensor(input_sz, hidden_sz * 4))
        self.U = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz * 4))
        self.bias = nn.Parameter(torch.Tensor(hidden_sz * 4))

        self.init_weights()

    def init_weights(self):
        """ 权重与偏置初始化
        :return:
        """
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x, init_states=None):
        """assumes x.shape represents (batch_size, sequence_size, input_size)
        input: 前馈操作接收init_states参数，该参数是上面方程的（h_t, c_t）参数的元组，
                如果不引入，则设置为零。然后，我们对每个保留（h_t, c_t）的序列元素执行LSTM方程的前馈，
                并将其作为序列下一个元素的状态引入.
        return: 预测和最后一个状态元组
        """
        bs, seq_sz, _ = x.size()
        hidden_seq = []

        if init_states is None:
            # 初始化历史状态
            h_t, c_t = (
                torch.zeros(bs, self.hidden_size).to(x.device),
                torch.zeros(bs, self.hidden_size).to(x.device),
            )
        else:
            h_t, c_t = init_states

        HS = self.hidden_size
        for t in range(seq_sz):
            # 获取单个词的feature 向量
            x_t = x[:, t, :]

            # batch the computations into a single matrix multiplication
            if self.peephole:
                gate = x_t @ self.W + h_t @ self.U + self.bias
            else:
                gate = x_t @ self.W + h_t @ self.U + self.bias
                g_t = torch.tanh(gate[:, HS*2:HS*3])

            i_t, f_t, o_t = (
                torch.sigmoid(gate[:, :HS]),  # input
                torch.sigmoid(gate[:, HS:HS*2]),  # forget
                torch.sigmoid(gate[:, HS*3:]),  # output
            )

            if self.peephole:
                c_t = f_t * c_t + i_t * torch.sigmoid(x_t @ self.W + self.bias)[:, HS*2:HS*3]
                h_t = torch.tanh(o_t * c_t)
            else:
                c_t = f_t * c_t + i_t * g_t
                h_t = o_t * torch.tanh(c_t)

            hidden_seq.append(h_t.unsqueeze(0))

        # reshape hidden_seq p/ retornar
        hidden_seq = torch.cat(hidden_seq, dim=0)
        # reshape from shape (sequence, batch, feature) to (batch, sequence, feature)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()

        return hidden_seq, (h_t, c_t)
