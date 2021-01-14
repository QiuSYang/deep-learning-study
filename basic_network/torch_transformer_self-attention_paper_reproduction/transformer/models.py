"""
# define the transformer model
"""
import os
import logging
import numpy as np
import torch
import torch.nn as nn

from transformer.configs import TransformerConfig
from transformer.layers import EncoderLayer, DecoderLayer
from utils import *

logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, d_hid, n_position=512):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer("position_table", self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        """
            PE(pos, 2i) = sin(pos/(10000^(2i/d_model)))
            PE(pos, 2i+1) = cos(pos/(10000^(2i/d_model))), i代表单词的维度
        """
        # TODO: make it with torch instead of numpy
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.position_table[:, :x.size(1)].clone().detach()


class Encoder(nn.Module):
    """ A encoder model with self attention mechanism. """
    def __init__(self, config: TransformerConfig):
        super(Encoder, self).__init__()

        self.config = config
        self.word_emb = nn.Embedding(self.config.vocab_size, self.config.word_vec_size,
                                     padding_idx=self.config.pad_idx)
        self.position_encoder = PositionalEncoding(self.config.word_vec_size, n_position=self.config.n_position)
        self.dropout = nn.Dropout(p=self.config.dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model=self.config.d_model, d_inner=self.config.d_inner, n_head=self.config.n_head,
                         d_k=self.config.d_k, d_v=self.config.d_v, dropout=self.config.dropout)
            for _ in range(self.config.encoder_n_layers)])  # note: d_model == word_vec_size, 即token向量化之中的大小
        self.layer_norm = nn.LayerNorm(self.config.d_model, eps=1e-6)

    def forward(self, input_ids, attention_mask, return_attentions=False):
        encoder_self_attention_list = []

        encoder_output = self.dropout(self.position_encoder(self.word_emb(input_ids)))
        encoder_output = self.layer_norm(encoder_output)
        for encoder_layer in self.layer_stack:
            encoder_output, encoder_self_attention = encoder_layer(encoder_output,
                                                                   self_attention_mask=attention_mask)
            encoder_self_attention_list += [encoder_self_attention] if return_attentions else []

        if return_attentions:
            return encoder_output, encoder_self_attention_list

        return encoder_output


class Decoder(nn.Module):
    """ A decoder model with self attention mechanism. """
    def __init__(self, config: TransformerConfig):
        super(Decoder, self).__init__()

        self.config = config
        self.word_emb = nn.Embedding(self.config.vocab_size, self.config.word_vec_size,
                                     padding_idx=self.config.pad_idx)
        self.position_encoder = PositionalEncoding(self.config.word_vec_size, n_position=self.config.n_position)
        self.dropout = nn.Dropout(p=self.config.dropout)
        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model=self.config.d_model, d_inner=self.config.d_inner, n_head=self.config.n_head,
                         d_k=self.config.d_k, d_v=self.config.d_v, dropout=self.config.dropout)
            for _ in range(self.decoder_n_layers)])
        self.layer_norm = nn.LayerNorm(self.config.d_model, eps=1e-6)

    def forward(self, decoder_input_ids, decoder_attention_mask,
                encoder_output, encoder_attention_mask, return_attentions=False):
        decoder_self_attention_list, decoder_encoder_attention_list = [], []

        decoder_output = self.dropout(self.position_encoder(self.word_emb(decoder_input_ids)))
        decoder_output = self.layer_norm(decoder_output)

        for decoder_layer in self.layer_stack:
            decoder_output, decoder_self_attention, decoder_encoder_attention = decoder_layer(
                                    decoder_output, encoder_output,
                                    self_attention_mask=decoder_attention_mask,
                                    cross_attention_mask=encoder_attention_mask)
            decoder_self_attention_list += [decoder_self_attention] if return_attentions else []
            decoder_encoder_attention_list += [decoder_encoder_attention] if return_attentions else []

        if return_attentions:
            return decoder_output, decoder_self_attention_list, decoder_encoder_attention_list

        return decoder_output


class Transformer(nn.Module):
    """A sequence to sequence model with attention mechanism."""
    def __init__(self, config: TransformerConfig):
        super(Transformer, self).__init__()
        self.config = config

        self.encoder = Encoder(config=self.config)
        self.decoder = Decoder(config=self.config)

        self.trg_word_prj = nn.Linear(self.config.d_model, self.config.vocab_size, bias=False)

        self.x_logit_scale = 1.0

        self._init_weights()  # 权重初始化

    def _init_weights(self):
        """权重初始化"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)
        assert self.config.d_model == self.config.word_vec_size, \
            'To facilitate the residual connections, \
             the dimensions of all module outputs shall be the same.'
        if self.configtrg_emb_prj_weight_sharing:
            # Share the weight between target word embedding & last dense layer
            self.trg_word_prj.weight = self.decoder.word_emb.weight
            self.x_logit_scale = (self.config.d_model ** -0.5)

        if self.config.emb_src_trg_weight_sharing:
            self.encoder.word_emb.weight = self.decoder.word_emb.weight

    def forward(self, input_ids, decoder_input_ids):
        encoder_attention_mask = get_pad_mask(input_ids, self.config.pad_idx)
        decoder_attention_mask = (get_pad_mask(decoder_input_ids, self.config.pad_idx) &
                                  get_subsequent_mask(decoder_input_ids))

        encoder_output, *_ = self.decoder(input_ids, encoder_attention_mask)
        decoder_output, *_ = self.decoder(decoder_input_ids, decoder_attention_mask,
                                          encoder_output=encoder_output,
                                          encoder_attention_mask=encoder_attention_mask)
        sequence_logit = self.trg_word_prj(decoder_output) * self.x_logit_scale

        return sequence_logit.view(-1, sequence_logit.size(2))
