"""
# Torch 实现transformer结构
"""
import math
import torch
import torch.nn as nn
# from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
from layers.encoder import CustomEmbedding
import layers


class PositionalEncoding(nn.Module):
    """
    Args:
        d_model: the number of expected features in the input (required).
        dropout: the dropout value (default=0.1).
    """
    def __init__(self, d_model, dropout=0.1, max_len=512):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    """
    Args:
        vocab_size: # the size of vocabulary.
        embed_size: embedding dimension, the number of expected features in the input
        nhead: the number of heads in the multiheadattention models.
        dim_feedforward: the dimension of the feedforward network model in nn.TransformerEncoder.
        nlayers: the number of sub-decoder-layers in the decoder (default=6).
        dropout: the dropout value (default=0.1).
    """
    def __init__(self,  config):
        super(TransformerModel, self).__init__()
        self.config = config
        self.pos_encoder = PositionalEncoding(config.embed_size, config.dropout, config.max_len)
        self.pos_decoder = PositionalEncoding(config.embed_size, config.dropout, config.max_len)
        self.src_embedding = CustomEmbedding(config.vocab_size, config.embed_size)
        self.tgt_embedding = nn.Embedding(config.vocab_size, config.embed_size)
        # encode images
        self.image_encoder = layers.ImageEncoder(config.embedding_size)
        # encoder
        encoder_layers = nn.TransformerEncoderLayer(config.embed_size, config.nhead,
                                                    config.dim_feedforward, config.dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, config.nlayers)
        # decoder
        decoder_layers = nn.TransformerDecoderLayer(config.embed_size, config.nhead,
                                                    config.dim_feedforward, config.dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, config.nlayers)
        self.linear = nn.Linear(config.embed_size, config.vocab_size)

        self.init_weights()

    def _attn_padding_mask(self, seq):
        """ seq_q: [batch_size, seq_len]
            seq_k: [batch_size, seq_len]
            seq_len could be src_len or it could be tgt_len
            seq_len in seq_q and seq_len in seq_k maybe not equal
        """
        # eq(zero) is PAD token
        return seq.data.eq(0)  # [batch_size, 1, len_k], True is masked
        # return pad_attn_mask.expand(batch_size, len_q, len_k)  # [batch_size, len_q, len_k]

    def _sequence_mask(self, seq):
        """
        Along with the input sequence, a square attention mask is required because the self-attention layers in nn.
        TransformerEncoder are only allowed to attend the earlier positions in the sequence. For the language modeling
        task, any tokens on the future positions should be masked. To have the actual words, the output of nn.
        TransformerEncoder model is sent to the final Linear layer, which is followed by a log-Softmax function.
        """
        batch_size, seq_len = seq.size()
        mask = (torch.triu(torch.ones(seq_len, seq_len)) == 1).transpose(0, 1).to(seq.device)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.src_embedding.embedding.weight.data.uniform_(-initrange, initrange)
        self.tgt_embedding.weight.data.uniform_(-initrange, initrange)

    def forward(self, src,
                input_sentence_length,
                input_conversation_length,
                tgt,
                input_images,
                input_images_length,
                input_image_indexes):
        # encode images
        img_encoder_outputs = self.image_encoder(input_images)

        # encoder
        src_padding_mask = self._attn_padding_mask(src)
        src_embed = self.src_embedding(src,
                                       img_encoder_outputs,
                                       input_images_length,
                                       input_image_indexes) * math.sqrt(self.config.embed_size)
        # Shape must be [Len, Batch, Embed] for nn.TransformerEncoderLayer.
        src_embed = self.pos_encoder(src_embed).transpose(0, 1)
        memory = self.transformer_encoder(src=src_embed, src_key_padding_mask=src_padding_mask)

        # decoder
        tgt_padding_mask = self._attn_padding_mask(tgt)
        tgt_sequence_mask = self._sequence_mask(tgt)
        tgt_embed = self.tgt_embedding(tgt) * math.sqrt(self.config.embed_size)
        tgt_embed = self.pos_decoder(tgt_embed).transpose(0, 1)
        output = self.transformer_decoder(tgt=tgt_embed,
                                          memory=memory,
                                          tgt_mask=tgt_sequence_mask,
                                          tgt_key_padding_mask=tgt_padding_mask,
                                          memory_key_padding_mask=src_padding_mask)

        output = self.linear(output)

        return output.transpose(0, 1).contiguous()
