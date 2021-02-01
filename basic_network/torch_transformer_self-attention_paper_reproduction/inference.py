"""
# 模型inference
"""
import os
import json
import logging
import torch
import torch.nn.functional as F
from transformers import BertTokenizer

from transformer.models import Transformer
from transformer.configs import TransformerConfig
from utils import get_pad_mask, get_subsequent_mask

logger = logging.getLogger(__name__)
root = os.path.dirname(os.path.abspath(__file__))


def top_filtering(logits, top_k=0., top_p=0.9, threshold=-float('Inf'), filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
            top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                whose total probability mass is greater than or equal to the threshold top_p.
                In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                the threshold top_p.
            threshold: a minimal threshold to keep logits
    """
    assert logits.dim() == 1  # Only work for batch size 1 for now - could update but it would obfuscate a bit the code
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        # Remove all tokens with a probability less than the last token in the top-k tokens
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # Compute cumulative probabilities of sorted tokens
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probabilities > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Back to unsorted indices and set them to -infinity
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    indices_to_remove = logits < threshold
    logits[indices_to_remove] = filter_value

    return logits


def encoder(encoder_model, encode_input_ids, pad_idx):
    encode_attention_mask = get_pad_mask(encode_input_ids, pad_idx=pad_idx)
    encode_outputs, *_ = encoder_model(encode_input_ids, encode_attention_mask)

    return encode_outputs, encode_attention_mask


def decoder(decoder_model, model_cls, decode_input_ids,
            encode_outputs, encode_attention_mask):
    decode_attention_mask = get_subsequent_mask(decode_input_ids)
    decode_outputs, *_ = decoder_model(decode_input_ids, decode_attention_mask,
                                       encoder_output=encode_outputs,
                                       encoder_attention_mask=encode_attention_mask)

    logits = model_cls(decode_outputs)

    return logits


def main():
    import argparse
    parse = argparse.ArgumentParser(description="设置基本参数")
    parse.add_argument("--para_path", type=str,
                       default=os.path.join(root, "data/para.json"),
                       help="所有配置参数")
    parse.add_argument("--model_path", type=str,
                       default=os.path.join(root, "model/transformer_0127/checkpoint_5.pt"),
                       help="所有配置参数")
    parse.add_argument("--no_sample", action='store_true', default=False,
                       help="Set to use greedy decoding instead of sampling")
    parse.add_argument("--repetition_penalty", type=float, default=0.01,
                       help="重复惩罚项")
    parse.add_argument("--temperature", type=float, default=0.7,
                       help="Sampling softmax temperature")
    parse.add_argument("--top_k", type=int, default=0,
                       help="Filter top-k tokens before sampling (<=0: no filtering)")
    parse.add_argument("--top_p", type=float, default=0.9,
                       help="Nucleus filtering (top-p) before sampling (<=0.0: no filtering)")
    args = parse.parse_args()

    with open(args.para_path, mode='r', encoding='utf-8') as fp:
        para_dict = json.load(fp)

    config = TransformerConfig(**para_dict)

    tokenizer = BertTokenizer(vocab_file=config.vocab_path)
    bos_token_id = tokenizer._convert_token_to_id("[CLS]")
    eos_token_id = tokenizer._convert_token_to_id("[SEP]")
    pad_token_id = tokenizer._convert_token_to_id("[PAD]")

    logger.info("Load model.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 标准写法
    model = Transformer(config=config)
    model.load_state_dict(torch.load(args.model_path, map_location="cpu"), strict=False)
    for name, weights in zip(model.named_parameters(), model.parameters()):
        logger.info("{} --- {}".format(name, weights))
    model.to(device)

    history_tokens = []
    while True:
        user_text = input("User-->>")
        while not user_text:
            logger.info('Prompt should not be empty!')
            user_text = input("User-->>")
        tokens = tokenizer.tokenize(user_text)
        history_tokens.append(tokens)

        # 获取输入tokens
        context_tokens = ["[SEP]"]
        for turn in history_tokens[::-1]:  # 逆序访问
            if len(context_tokens) + len(turn) < config.max_encode_len:
                context_tokens = turn + context_tokens
                context_tokens = ["[SEP]"] + context_tokens
            else:
                break
        context_tokens[0] = "[CLS]"  # 将头部[SEP] token替换为[CLS] token

        # 编码部分
        encode_input_ids = tokenizer.convert_tokens_to_ids(context_tokens)
        encode_input_ids = torch.tensor(encode_input_ids).long().unsqueeze(dim=0).to(device)
        encode_outputs, encode_attention_mask = encoder(model.encoder, encode_input_ids, pad_idx=pad_token_id)

        # 解码部分, 生成文本
        index = 1
        generate_sequence_ids = [bos_token_id]
        while index <= config.max_decode_len:
            # decode_input_ids = torch.LongTensor([generate_sequence_ids])  # 扩充为二维向量
            decode_input_ids = torch.tensor(generate_sequence_ids).long().unsqueeze(dim=0).to(device)
            logits = decoder(model.decoder, model.trg_word_prj, decode_input_ids,
                             encode_outputs=encode_outputs, encode_attention_mask=encode_attention_mask)
            next_token_logit = logits[0][-1, :]  # 获取最后一个token的Logit
            for id in set(generate_sequence_ids):
                next_token_logit[id] /= args.repetition_penalty
            next_token_logit = top_filtering(next_token_logit, top_k=args.top_k, top_p=args.top_p)
            probs = F.softmax(next_token_logit, dim=-1)

            temp_token_id = torch.topk(probs, 1)
            next_token_id = torch.topk(probs, 1)[1] if args.no_sample else torch.multinomial(probs, 1)
            next_token_id = next_token_id.item()

            if next_token_id == eos_token_id:
                generate_sequence_ids.append(next_token_id)
                break

            generate_sequence_ids.append(next_token_id)
            index += 1

        system_tokens = tokenizer.convert_ids_to_tokens(generate_sequence_ids)
        print("System-->>{}".format("".join(system_tokens[1:-1])))
        history_tokens.append(system_tokens[1:-1])  # 删除首尾[CLS] 与 [SEP] token


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(filename)s - %(levelname)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S')
    main()
