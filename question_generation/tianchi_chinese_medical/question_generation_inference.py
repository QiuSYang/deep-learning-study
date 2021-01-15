"""
# robert + transformers-xl 问题生产inference
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '7'
import sys
import logging
from tqdm import tqdm
from typing import Any
from transformers import BertTokenizer, BertModel, BertConfig
import torch
from torch import nn
import pickle
import json
import re
from copy import deepcopy
from configs import *

args = qg_configs
# 获取可用设备
if args.get("device"):
    device = args["device"]
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
# set logs
logger = logging.getLogger(__name__)


class QGUtils(object):
    @staticmethod
    def text_encode(tokenizer, context, answer):
        process_context = context.replace("\n", " ").replace("\t", " ").replace("\\", "")
        context_tokens = tokenizer.tokenize(process_context)
        answer_tokens = tokenizer.tokenize(answer)[:args["max_answer_len"]]
        c = ["[CLS]"] + answer_tokens + ["[SEP]"] + context_tokens
        if len(c) > args["max_enc_len"] - 1:
            c = c[:args["max_enc_len"] - 1]
        c += ["[SEP]"]
        input_ids = tokenizer.convert_tokens_to_ids(c)
        input_mask = [1.0] * len(input_ids)
        input_seg = [0] * (len(answer_tokens) + 2) + [1] * (len(input_ids) - 2 - len(answer_tokens))
        extra = args["max_enc_len"] - len(input_ids)
        if extra > 0:
            input_ids += [0] * extra
            input_mask += [0.0] * extra
            input_seg += [1] * extra
        return {
            "input_ids": torch.tensor(input_ids).long().unsqueeze(dim=0).to(device),
            "input_mask": torch.tensor(input_mask).float().unsqueeze(dim=0).to(device),
            "input_seg": torch.tensor(input_seg).long().unsqueeze(dim=0).to(device)
        }

    @staticmethod
    def ids_to_string(tokenizer, y, text, answer):
        """ 获取预测字符串
        :param tokenizer: 原始分词器
        :param y: 预测获取的ids
        :param text: 原始文本
        :param answer: 答案文本
        :return:
        """
        s = []
        for k in y:
            if int(k) == args["end_token_id"]:
                break
            else:
                s.append(int(k))
        s = "".join(tokenizer.convert_ids_to_tokens(s))
        s = s.replace("，", "").replace("[UNK]", "").replace("#", "")
        char_list = []
        for c in s:
            if c not in char_list:
                char_list.append(c)
        for c in char_list:
            try:
                p = re.compile("(%s){2,}" % c)
                s = re.sub(p, c, s)
            except Exception as e:
                continue
        # 针对英文的一些修正
        t_text = text.lower()
        p = re.compile("([A-Za-z]+)")
        m = re.finditer(p, s)
        for i_match in m:
            start, end, i_str = i_match.start(), i_match.end(), i_match.group()
            if i_str in t_text:
                i_index = t_text.index(i_str)
                s = s[:start] + text[i_index: i_index + (end - start)] + s[end:]
        if len(s) < 2:
            s = answer  # 生成文本过短, 直接将答案作为问题

        return s


class XLRelPosEmb(nn.Module):
    def _forward_unimplemented(self, *input: Any) -> None:
        pass

    def __init__(self, d_embed):
        super().__init__()

        self.d_embed = d_embed
        inv_freq = 1 / (10000 ** (torch.arange(0.0, self.d_embed, 2.0) / self.d_embed))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, pos_seq):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)
        return pos_emb


class PositionwiseFFN(nn.Module):
    def _forward_unimplemented(self, *input: Any) -> None:
        pass

    def __init__(self, d_model, d_inner, layer_norm_epsilon=1e-5):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_inner
        self.CoreNet = nn.Sequential(
            nn.Linear(d_model, d_inner),
            nn.GELU(),
            nn.Dropout(p=args["dropout"]),
            nn.Linear(d_inner, d_model),
            nn.Dropout(p=args["dropout"])
        )
        self.layer_norm = nn.LayerNorm(d_model, eps=layer_norm_epsilon)

    def forward(self, inp):
        core_out = self.CoreNet(inp)
        output = self.layer_norm(inp + core_out)
        return output


class RelPartialLearnableMultiHeadAttn(torch.nn.Module):

    def _forward_unimplemented(self, *input: Any) -> None:
        pass

    def __init__(self, n_heads, d_model, layer_norm_epsilon=1e-5):
        super().__init__()

        self.n_heads = n_heads
        self.d_model = d_model
        self.d_head = d_model // n_heads

        self.mask_attn_qkv_net = nn.Linear(d_model, 3 * d_model, bias=False)
        self.mask_attn_o_net = nn.Linear(d_model, d_model, bias=False)

        self.interaction_kv_net = nn.Linear(d_model, 2 * d_model, bias=False)
        self.interaction_q_net = nn.Linear(d_model, d_model, bias=False)
        self.interaction_o_net = nn.Linear(d_model, d_model, bias=False)

        self.layer_norm_mask_attn = nn.LayerNorm(d_model, eps=layer_norm_epsilon)
        self.layer_norm_interaction = nn.LayerNorm(d_model, eps=layer_norm_epsilon)
        self.scale = 1 / (self.d_head ** 0.5)

        self.r_r_bias = nn.Parameter(torch.FloatTensor(self.n_heads, self.d_head))
        self.r_w_bias = nn.Parameter(torch.FloatTensor(self.n_heads, self.d_head))

        self.r_net = nn.Linear(d_model, d_model, bias=False)

        self.drop = nn.Dropout(p=args["dropout"])

    @staticmethod
    def _rel_shift(x):
        zero_pad_shape = (x.size(0), 1) + x.size()[2:]
        zero_pad = torch.zeros(zero_pad_shape, device=device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=1)

        x_padded_shape = (x.size(1) + 1, x.size(0)) + x.size()[2:]
        x_padded = x_padded.view(*x_padded_shape)

        x = x_padded[1:].view_as(x)

        return x

    def forward(self, w, r, enc_context, attn_mask, padding_mask):
        # attn_mask用于Masked-Attn Mechanism(decode自身部分)
        # padding_mask用于Norm Multi-Attn, Decode与Encode Contextual Rep交互
        dec_len, bsz, enc_len = w.size(0), w.size(1), enc_context.size(0)
        w_heads = self.mask_attn_qkv_net(w)
        r_head_k = self.r_net(r)
        w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

        w_head_q = w_head_q.view(dec_len, bsz, self.n_heads, self.d_head)  # dec_len x bsz x n_head x d_head
        w_head_k = w_head_k.view(dec_len, bsz, self.n_heads, self.d_head)  # dec_len x bsz x n_head x d_head
        w_head_v = w_head_v.view(dec_len, bsz, self.n_heads, self.d_head)  # dec_len x bsz x n_head x d_head

        r_head_k = r_head_k.view(dec_len, self.n_heads, self.d_head)  # dec_len x n_head x d_head
        rw_head_q = w_head_q + self.r_w_bias  # dec_len x bsz x n_head x d_head
        AC = torch.einsum("ibnd,jbnd->ijbn", rw_head_q, w_head_k)  # dec_len x dec_len x bsz x n_head
        rr_head_q = w_head_q + self.r_r_bias
        BD = torch.einsum("ibnd,jnd->ijbn", rr_head_q, r_head_k)  # dec_len x dec_len x bsz x n_head
        BD = self._rel_shift(BD)

        attn_score = AC + BD
        attn_score.mul_(self.scale)

        # causal masking mechanism
        attn_mask = attn_mask == 0  # switch to bool
        attn_score = attn_score.float().masked_fill(attn_mask, -1e30).type_as(attn_score)
        attn_prob = torch.softmax(attn_score, dim=1)
        attn_prob = self.drop(attn_prob)

        attn_vec = torch.einsum("ijbn,jbnd->ibnd", attn_prob, w_head_v)
        attn_vec = attn_vec.contiguous().view(dec_len, bsz, self.d_model)

        attn_out = self.mask_attn_o_net(attn_vec)
        attn_out = self.drop(attn_out)

        mask_attn_output = self.layer_norm_mask_attn(w + attn_out)

        # 与编码器交互部分
        inter_k, inter_v = torch.chunk(self.interaction_kv_net(enc_context), 2, dim=-1)
        inter_q = self.interaction_q_net(mask_attn_output)
        inter_q = inter_q.view(dec_len, bsz, self.n_heads, self.d_head)
        inter_k = inter_k.view(enc_len, bsz, self.n_heads, self.d_head)
        inter_v = inter_v.view(enc_len, bsz, self.n_heads, self.d_head)

        attn_score = torch.einsum("qbnd,kbnd->qkbn", inter_q, inter_k)
        attn_score.mul_(self.scale)

        # use padding_mask to mask input [PAD] token
        padding_mask = padding_mask[None, :, :, None].repeat(dec_len, 1, 1, 1)
        attn_score = attn_score + (1 - padding_mask) * (-1e30)
        attn_prob = torch.softmax(attn_score, dim=1)
        attn_prob = self.drop(attn_prob)
        attn_vec = torch.einsum("ijbn,jbnd->ibnd", attn_prob, inter_v)
        attn_vec = attn_vec.contiguous().view(dec_len, bsz, self.d_model)

        attn_out = self.interaction_o_net(attn_vec)
        attn_out = self.drop(attn_out)

        interaction_output = self.layer_norm_interaction(attn_out + mask_attn_output)
        return interaction_output


class RelPartialLearnableDecoderLayer(torch.nn.Module):

    def _forward_unimplemented(self, *input: Any) -> None:
        pass

    def __init__(self, n_heads, d_model, d_inner):
        super().__init__()

        self.dec_attn = RelPartialLearnableMultiHeadAttn(n_heads=n_heads, d_model=d_model)
        self.ffn_layer = PositionwiseFFN(d_model=d_model, d_inner=d_inner)

    def forward(self, dec_inp, r, enc_inp, dec_mask, enc_mask):
        attn_output = self.dec_attn(w=dec_inp, r=r, enc_context=enc_inp, attn_mask=dec_mask, padding_mask=enc_mask)
        ffn_out = self.ffn_layer(attn_output)
        return ffn_out


class XLDecoder(torch.nn.Module):

    def _forward_unimplemented(self, *input: Any) -> None:
        pass

    def __init__(self, dim, embedding_matrix: nn.Embedding, seq_len):
        super().__init__()
        self.d_model = dim
        self.word_emb = embedding_matrix
        self.seq_len = seq_len
        self.n_layers = args["decoder_layers"]
        self.n_heads = 16
        self.ffn = 4 * dim
        self.epsilon = 1e-6

        self.drop = nn.Dropout(p=args["dropout"])
        self.pos_emb = XLRelPosEmb(d_embed=dim)
        self.layers = nn.ModuleList()

        self.layers = nn.ModuleList()
        for i in range(self.n_layers):
            self.layers.append(
                RelPartialLearnableDecoderLayer(
                    n_heads=self.n_heads, d_model=self.d_model, d_inner=self.ffn
                )
            )
        self.output = nn.Linear(in_features=dim, out_features=dim)
        self.copy_output = nn.Linear(in_features=dim, out_features=dim)
        # 自适应的解码概率结合
        self.mode_select = nn.Sequential(
            nn.Linear(in_features=dim, out_features=1),
            nn.Sigmoid()
        )

    def forward(self, input_ids, encoder_rep, input_mask, decode_input, decode_target, use_beam_search, beam_width):
        bsz = input_ids.size(0)
        if decode_input is not None:  # 代表训练模式
            input_ids = input_ids[:, None, :].repeat(1, self.seq_len, 1)
            decode_embed = self.drop(self.word_emb(decode_input))
            all_ones = decode_embed.new_ones((self.seq_len, self.seq_len), dtype=torch.uint8)
            dec_attn_mask = torch.tril(all_ones, diagonal=0)[:, :, None, None]
            pos_seq = torch.arange(self.seq_len - 1, -1, -1.0, device=device, dtype=decode_embed.dtype)
            pos_embed = self.drop(self.pos_emb(pos_seq))
            core_out = decode_embed.transpose(0, 1).contiguous()
            enc_rep_t = encoder_rep.transpose(0, 1).contiguous()
            enc_mask_t = input_mask.transpose(0, 1).contiguous()
            for layer in self.layers:
                core_out = layer(
                    dec_inp=core_out, r=pos_embed, enc_inp=enc_rep_t,
                    dec_mask=dec_attn_mask, enc_mask=enc_mask_t
                )
            core_out = self.drop(core_out.transpose(0, 1).contiguous())  # (bsz, dec_len, dim)
            output = self.output(core_out)
            vocab_logits = torch.nn.functional.linear(input=output, weight=self.word_emb.weight)
            vocab_prob = torch.softmax(vocab_logits, dim=-1)
            input_logits = torch.einsum("bid,bjd->bij", self.copy_output(core_out), encoder_rep)  # (bsz, dec_len, enc_len)
            input_logits = input_logits + (1.0 - input_mask[:, None, :].repeat(1, self.seq_len, 1)) * (-1e30)
            input_prob = torch.softmax(input_logits, dim=-1)  # (bsz, dec_len, enc_len)
            mode_sig = self.mode_select(core_out)  # (bsz, dec_len, 1)
            vocab_prob = vocab_prob * mode_sig
            vocab_prob = torch.scatter_add(vocab_prob, dim=2, index=input_ids, src=(1 - mode_sig) * input_prob)
            vocab_prob = vocab_prob.view(-1, args["vocab_size"])
            decode_target = decode_target.view(-1)
            predict = torch.gather(vocab_prob, dim=1, index=decode_target[:, None]).squeeze(dim=-1)
            init_loss = -torch.log(predict + self.epsilon)
            init_loss *= (decode_target != 0).float()
            loss = torch.sum(init_loss) / torch.nonzero(decode_target != 0, as_tuple=False).size(0)
            # 为了并行化设计, 将loss变成(bsz,)
            return loss[None].repeat(bsz)
        else:  # 代表验证或者测试解码模式 ==> 比较耗时
            if not use_beam_search:  # 使用贪心搜索 ==> 验证集
                dec_list = []
                decode_ids = torch.full(size=(bsz, 1), fill_value=args["start_token_id"], dtype=torch.int32).long().to(device)
                for i in range(1, self.seq_len + 1):
                    if i > 1:
                        decode_ids = torch.cat([decode_ids, dec_list[i - 2]], dim=-1)
                    decode_embed = self.word_emb(decode_ids)
                    all_ones = decode_embed.new_ones((i, i), dtype=torch.uint8)
                    dec_attn_mask = torch.tril(all_ones, diagonal=0)[:, :, None, None]
                    pos_seq = torch.arange(i - 1, -1, -1.0, device=device, dtype=decode_embed.dtype)
                    pos_embed = self.pos_emb(pos_seq)
                    core_out = decode_embed.transpose(0, 1).contiguous()
                    enc_rep_t = encoder_rep.transpose(0, 1).contiguous()
                    enc_mask_t = input_mask.transpose(0, 1).contiguous()
                    for layer in self.layers:
                        core_out = layer(
                            dec_inp=core_out, r=pos_embed, enc_inp=enc_rep_t,
                            dec_mask=dec_attn_mask, enc_mask=enc_mask_t
                        )
                    core_out = core_out.transpose(0, 1).contiguous()[:, -1, :]
                    output = self.output(core_out)
                    vocab_logits = torch.nn.functional.linear(input=output, weight=self.word_emb.weight)
                    vocab_prob = torch.softmax(vocab_logits, dim=-1)
                    input_logits = torch.einsum("bd,bjd->bj", self.copy_output(core_out), encoder_rep)  # (bsz, enc_len)
                    input_logits = input_logits + (1.0 - input_mask) * (-1e30)
                    input_prob = torch.softmax(input_logits, dim=-1)  # (bsz, enc_len)
                    mode_sig = self.mode_select(core_out)  # (bsz, 1)
                    vocab_prob = vocab_prob * mode_sig
                    vocab_prob = torch.scatter_add(vocab_prob, dim=1, index=input_ids, src=(1 - mode_sig) * input_prob)
                    dec_list.append(torch.argmax(vocab_prob, dim=-1)[:, None])
                return torch.cat(dec_list, dim=-1)
            else:  # 使用集束搜索
                # 扩展成beam_width * bsz
                """
                需要注意: 1. trigram-block的使用 ==> 出现重复直接加上-1e9(需要考虑end_token边界=>只在边界范围内使用)
                2. 长度惩罚, 考虑end_token边界
                """
                decode_ids = torch.full(size=(bsz * beam_width, 1),
                                        fill_value=args["start_token_id"],
                                        dtype=torch.int32).long().to(device)
                input_ids = input_ids.repeat((beam_width, 1))
                encoder_rep = encoder_rep.repeat((beam_width, 1, 1))
                input_mask = input_mask.repeat((beam_width, 1))
                dec_topK_log_probs = [0] * (beam_width * bsz)  # (bsz*beam)  每个序列的当前log概率和
                dec_topK_sequences = [[] for _ in range(beam_width * bsz)]  # (bsz*beam, seq_len) 解码id序列
                dec_topK_seq_lens = [1] * (beam_width * bsz)  # 解码序列长度 ==> 加上一个偏置项1, 防止进行长度惩罚时出现div 0的情况
                for i in range(1, self.seq_len + 1):
                    if i > 1:
                        input_decode_ids = torch.cat([decode_ids, torch.tensor(dec_topK_sequences).long().to(device)],
                                                     dim=-1)
                    else:
                        input_decode_ids = decode_ids
                    decode_embed = self.word_emb(input_decode_ids)
                    all_ones = decode_embed.new_ones((i, i), dtype=torch.uint8)
                    dec_attn_mask = torch.tril(all_ones, diagonal=0)[:, :, None, None]
                    pos_seq = torch.arange(i - 1, -1, -1.0, device=device, dtype=decode_embed.dtype)
                    pos_embed = self.pos_emb(pos_seq)
                    core_out = decode_embed.transpose(0, 1).contiguous()
                    enc_rep_t = encoder_rep.transpose(0, 1).contiguous()
                    enc_mask_t = input_mask.transpose(0, 1).contiguous()
                    for layer in self.layers:
                        core_out = layer(
                            dec_inp=core_out, r=pos_embed, enc_inp=enc_rep_t,
                            dec_mask=dec_attn_mask, enc_mask=enc_mask_t
                        )
                    core_out = core_out.transpose(0, 1).contiguous()[:, -1, :]
                    output = self.output(core_out)
                    vocab_logits = torch.nn.functional.linear(input=output, weight=self.word_emb.weight)
                    vocab_prob = torch.softmax(vocab_logits, dim=-1)
                    input_logits = torch.einsum("bd,bjd->bj", self.copy_output(core_out), encoder_rep)  # (bsz*beam, enc_len)
                    input_logits = input_logits + (1.0 - input_mask) * (-1e30)
                    input_prob = torch.softmax(input_logits, dim=-1)  # (bsz*beam, enc_len)
                    mode_sig = self.mode_select(core_out)  # (bsz*beam, 1)
                    vocab_prob = vocab_prob * mode_sig
                    vocab_prob = torch.scatter_add(vocab_prob, dim=1, index=input_ids, src=(1 - mode_sig) * input_prob)  # (bsz*beam, vocab)
                    vocab_logp = torch.log(vocab_prob + self.epsilon)  # 取对数， 加eps
                    """ step1: 检查是否存在trigram blocking重叠, 只需要检查最后一项和之前项是否存在重叠即可 """
                    if i > 4:  # 当序列长度大于等于4时才有意义, 或者当前解码时刻大于4时才有检查的必要
                        for j in range(beam_width * bsz):
                            trigram_blocks = []
                            for k in range(3, i):
                                if dec_topK_sequences[j][k-1] == args["end_token_id"]:
                                    break
                                trigram_blocks.append(dec_topK_sequences[j][k-3:k])
                            if len(trigram_blocks) > 1 and trigram_blocks[-1] in trigram_blocks[:-1]:
                                dec_topK_log_probs[j] += -1e9
                    """ step2: 为每个样本, 选择topK个序列 ==> 类似于重构dec_topK_sequences"""
                    for j in range(bsz):
                        topK_vocab_logp = vocab_logp[j::bsz]  # (k, vocab)
                        candidate_list = []
                        """ 容易出错的地方, i=1的时候不需要为每个K生成K个候选,否则beam search将完全沦为greedy search """
                        for k in range(beam_width):
                            ind = j + k * bsz
                            if args["end_token_id"] in dec_topK_sequences[ind]:
                                candidate_list.append({
                                    "add_logit": 0, "add_seq_len": 0, "affiliate_k": k, "add_token_id": args["end_token_id"],
                                    "sort_logits": dec_topK_log_probs[ind] / (dec_topK_seq_lens[ind] ** args["beam_length_penalty"])
                                })
                            else:
                                k_logps, k_indices = topK_vocab_logp[k].topk(dim=0, k=beam_width)
                                k_logps, k_indices = k_logps.cpu().numpy(), k_indices.cpu().numpy()
                                for l in range(beam_width):
                                    aff = l if i == 1 else k
                                    candidate_list.append({
                                        "add_logit": k_logps[l], "add_seq_len": 1, "affiliate_k": aff, "add_token_id": k_indices[l],
                                        "sort_logits": (dec_topK_log_probs[ind] + k_logps[l]) / ((dec_topK_seq_lens[ind] + 1) ** args["beam_length_penalty"])
                                    })
                            if i == 1:  ## 当解码第一个词的时候只能考虑一个
                                break
                        candidate_list.sort(key=lambda x: x["sort_logits"], reverse=True)
                        candidate_list = candidate_list[:beam_width]
                        """ 序列修正, 更新topK """
                        c_dec_topK_sequences, c_dec_topK_log_probs, c_dec_topK_seq_lens = \
                            deepcopy(dec_topK_sequences), deepcopy(dec_topK_log_probs), deepcopy(dec_topK_seq_lens)
                        for k in range(beam_width):
                            ind = bsz * candidate_list[k]["affiliate_k"] + j
                            r_ind = bsz * k + j
                            father_seq, father_logits, father_len = c_dec_topK_sequences[ind], c_dec_topK_log_probs[ind], c_dec_topK_seq_lens[ind]
                            dec_topK_sequences[r_ind] = father_seq + [candidate_list[k]["add_token_id"]]
                            dec_topK_log_probs[r_ind] = father_logits + candidate_list[k]["add_logit"]
                            dec_topK_seq_lens[r_ind] = father_len + candidate_list[k]["add_seq_len"]
                return torch.tensor(dec_topK_sequences[:bsz]).long().to(device)


class QuestionGeneration(torch.nn.Module):
    def _forward_unimplemented(self, *input: Any) -> None:
        pass

    def __init__(self, pre_train_dir: str):
        super().__init__()
        if os.path.isdir(pre_train_dir):
            self.roberta_encoder = BertModel(
                config=BertConfig.from_json_file(os.path.join(pre_train_dir, "config.json")))
        else:
            self.roberta_encoder = BertModel(
                config=BertConfig.from_pretrained(pre_train_dir))
        self.decoder_layer = XLDecoder(dim=args["dimension"],
                                       embedding_matrix=self.roberta_encoder.get_input_embeddings(),
                                       seq_len=args["max_dec_len"])

    def forward(self, input_ids, input_mask, input_seg, decode_input=None, decode_target=None):
        encoder_rep = self.roberta_encoder(input_ids, input_mask, input_seg)[0]
        # print("encoder shape: {}, encoder vector: {}".format(encoder_rep.shape, encoder_rep))
        return self.decoder_layer(input_ids, encoder_rep, input_mask, decode_input, decode_target,
                                  args["use_beam_search"],
                                  args["beam_width"])

    @classmethod
    def from_pretrained(cls, pretrained_model_path=None):
        """ load model
        :param pretrained_model_path: 模型文件绝对路径
        :return:
        """
        model = cls(pre_train_dir=args["pre_train_dir"])
        if pretrained_model_path:
            model_path = pretrained_model_path
        elif args["save_path"]:
            model_path = args["save_path"]
        else:
            raise Exception("Please input model file.")
        model.load_state_dict(torch.load(model_path, map_location=device), strict=False)

        return model


class QuestionGenerationInfer(object):
    def __init__(self, model=None, tokenizer=None, test_items=None):
        # 1. 初始化分词器
        if tokenizer:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = BertTokenizer.from_pretrained(args["pre_train_dir"])
        if args["vocab_size"] != len(self.tokenizer):
            args["vocab_size"] = len(self.tokenizer)

        self.test_items = test_items
        # 2. 构建模型
        if model:
            self.model = model
        else:
            self.model = QuestionGeneration.from_pretrained(args["save_path"])
            self.model.to(device=device)

    def test(self):
        self.model.eval()
        output = self.test_items
        with torch.no_grad():
            for i in tqdm(range(len(output))):
                text = output[i]["text"]
                annotations = output[i]["annotations"]
                tmp_enc_ids, tmp_enc_mask, tmp_enc_seg = [], [], []
                for j in range(len(annotations)):
                    y = QGUtils.text_encode(self.tokenizer, text, annotations[j]["A"])
                    tmp_enc_ids.append(y["input_ids"])
                    tmp_enc_mask.append(y["input_mask"])
                    tmp_enc_seg.append(y["input_seg"])
                # 模型预测
                dec_seq = self.model(
                    input_ids=torch.cat(tmp_enc_ids, dim=0),
                    input_mask=torch.cat(tmp_enc_mask, dim=0),
                    input_seg=torch.cat(tmp_enc_seg, dim=0)
                )
                dec_seq = dec_seq.cpu().numpy()
                for j in range(len(dec_seq)):
                    y = dec_seq[j]
                    s = QGUtils.ids_to_string(self.tokenizer, y, text=text, answer=annotations[j]["A"])
                    annotations[j]["Q"] = s
                if i % 50 == 0 and i > 0:
                    print("The program has completed %s predictions" % i)
        # 保存测试结果
        with open("submit_test.json", "w", encoding="UTF-8") as fw:
            json.dump(output, fw, ensure_ascii=False, indent=2)
            print("The program has completed all predictions")

    def predict_single(self, text, answer):
        """ 模型预测
        :param text: 上下文
        :param answer: 答案
        :return:
        """
        self.model.eval()
        with torch.no_grad():
            tmp_enc_ids, tmp_enc_mask, tmp_enc_seg = [], [], []
            y = QGUtils.text_encode(self.tokenizer, text, answer)
            tmp_enc_ids.append(y["input_ids"])
            tmp_enc_mask.append(y["input_mask"])
            tmp_enc_seg.append(y["input_seg"])
            # 模型预测
            dec_seq = self.model(
                input_ids=torch.cat(tmp_enc_ids, dim=0),
                input_mask=torch.cat(tmp_enc_mask, dim=0),
                input_seg=torch.cat(tmp_enc_seg, dim=0)
            )
            y = dec_seq.cpu().numpy()[0]  # 获取第0个result
            s = QGUtils.ids_to_string(self.tokenizer, y, text=text, answer=answer)

            return s

    def predict(self, text, answers):
        """ inference
        :param text: 段落文档
        :param answers: 答案list
        :return:
        """
        self.model.eval()
        with torch.no_grad():
            tmp_enc_ids, tmp_enc_mask, tmp_enc_seg = [], [], []
            for answer in answers:
                y = QGUtils.text_encode(self.tokenizer, text, answer)
                tmp_enc_ids.append(y["input_ids"])
                tmp_enc_mask.append(y["input_mask"])
                tmp_enc_seg.append(y["input_seg"])
            # 模型预测
            dec_seq = self.model(
                input_ids=torch.cat(tmp_enc_ids, dim=0),
                input_mask=torch.cat(tmp_enc_mask, dim=0),
                input_seg=torch.cat(tmp_enc_seg, dim=0)
            )
            dec_logits = dec_seq.cpu().numpy()
            questions = []
            for i in range(len(dec_logits)):
                y = dec_logits[i]
                result = QGUtils.ids_to_string(self.tokenizer, y, text=text, answer=answer)
                questions.append(result)

            return questions


if __name__ == "__main__":
    args["mode"] = sys.argv[1]
    if args["mode"] == "test":
        with open("DataSet/multi_task.pkl", "rb") as f:
            x = pickle.load(f)

        args["use_beam_search"] = True
        m = QuestionGenerationInfer(test_items=x["test_items"])
        m.test()
    else:
        args["use_beam_search"] = True
        m = QuestionGenerationInfer()
        context = "黄帝说：我愿意听你讲讲三阴三阳的离合情况。岐伯说：圣人面向南方站立，前方名叫广明，后方名叫太冲，行于太冲部位的经脉，叫做少阴。" \
                  "在少阴经上面的经脉，名叫太阳，太阳经的下端起于足小趾外侧的至阴穴，其上端结于晴明穴，因太阳为少阴之表，故称为阴中之阳。" \
                  "再以人身上下而言，上半身属于阳，称为广明，广明之下称为太阴，太阴前面的经脉，名叫阳明，阳明经的下端起于足大趾侧次趾之端的历兑穴，因阴阳是太阴之表，故称为阴中之阳。" \
                  "厥阴为里，少阳为表，故厥阴精之表，为少阳经，少阳经下端起于窍阴穴，因少阳居厥阴之表，故称为阴中之少阳。" \
                  "因此，三阳经的离合，分开来说，太阳主表为开，阴明主里为阖，少阳介于表里之间为枢。但三者之间，不是各自为政，而是相互紧密联系着的，所以合起来称为一阳。"
        answers = ["三阳经的离合，分开来说，太阳主表为开，阴明主里为阖，少阳介于表里之间为枢。但三者之间，不是各自为政，而是相互紧密联系着的，所以合起来称为一阳。",
                   "太阳经的下端起于足小趾外侧的至阴穴，其上端结于晴明穴，因太阳为少阴之表，故称为阴中之阳。"]

        context = "2020年11月4日，云从科技创始人周曦博士出席“2020企业创新生态圈大会”，" \
                  "与全球知名机器人企业——波士顿动力的创始人马克·雷波特进行对话，开启一场关于人工智能的中西巅峰探讨。" \
                  "云从与波士顿动力，都是由学者创立的企业，都持续致力于智能技术的落地，也都代表着所在国家智能技术发展的前沿水平，" \
                  "然而两家企业的发展路径看起来迥然不同：云从致力于延伸人的“大脑”,通过人机协同让顶尖的专家智慧广泛普及，" \
                  "让更多人受益，提升全社会的效能；波士顿动力专注于让机器人像人类一样灵活地在各类场景行动，成为人类四肢的延伸。" \
                  "本次，周曦与雷波特围绕人工智能与人类、人工智能的应用、以及行业未来规范等三大主题进行深入探讨。" \
                  "周曦反复强调，人工智能的发展离不开人类智慧，云从要做的，是让人机协同把人类智慧延展出去。" \
                  "这是云从在人机协同战略背后的深刻思考，也是对于人工智能的价值重塑与航道选择：如今的云从，早已不是视觉识别企业，" \
                  "而是已经成为拥有完整的 “感知——认知——决策”技术闭环、为各个行业提供人机协同操作系统与全面智能解决方案的先行者。"
        answers = ["2020年11月4日", "周曦与雷波特围绕人工智能与人类、人工智能的应用、以及行业未来规范等三大主题进行深入探讨", "拥有完整的 “感知——认知——决策”技术闭环"]

        # context = "人工智能主要有两个发展路径：第一种是工具化，由机器人完成机械化工作；第二种，让人工智能按照人的逻辑层层递进，变成人的良师益友。" \
        #           "“让人工智能按照人的逻辑层层递进，变成人的良师益友”——正是云从科技贯彻的人机协同战略，一条与工具型路线完全不同的路径。" \
        #           "周曦分析道，面对复杂的、新的情况，大数据训练出来的人工智能并不能解决根本问题。" \
        #           "同时，各种抢教育、医疗资源的社会现象，正说明了优质资源的稀缺。“这个时候需要另外一条路，我们叫专家知识。" \
        #           "我们要相信人的力量，把人工智能和人结合。人能够在很复杂的环境、很小的样本的情况下，做出创造性的决定。”" \
        #           "这也正是云从从根本上与“工具性人工智能”思路截然不同的价值观：人机协同从技术层面支持专家能力的大规模复制，" \
        #           "可千百倍地拓展人的能力边界，让更多人享受到优质的资源；同时能够让人的体力、时间得到释放，帮助人类从事更有创造性的事。" \
        #           "人机协同，不仅是云从持续深耕的企业战略，也是人工智能进化的必然方向、时代发展的必然要求。我国经济正面临从规模化向高质量发展转型的挑战，" \
        #           "提升生产力水平与效率的任务迫在眉睫。人机协同正是切中高质量发展的题中之义，扩大顶尖资源的覆盖面，帮助人们增强专业能力，" \
        #           "从根本上解决效率瓶颈，从而实现数量级的社会生产效率与品质的提升。"
        # answers = ["第一种是工具化，由机器人完成机械化工作；第二种，让人工智能按照人的逻辑层层递进，变成人的良师益友",
        #            "扩大顶尖资源的覆盖面，帮助人们增强专业能力，从根本上解决效率瓶颈，从而实现数量级的社会生产效率与品质的提升。"]

        # question = m.predict_single(text=context, answer=answer)
        questions = m.predict(text=context, answers=answers)

        print("generate question: {}".format(questions))
