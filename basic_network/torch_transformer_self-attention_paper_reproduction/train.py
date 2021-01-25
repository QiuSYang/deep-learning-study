"""
# transformer train functions
"""
import os
import logging
import torch
import torch.nn.functional as F
import torch.optim as optim
from transformers import BertTokenizer

from transformer.models import Transformer
from transformer.configs import TransformerConfig

logger = logging.getLogger(__name__)
root = os.path.dirname(os.path.abspath(__file__))


def cal_loss(pred, gold, trg_pad_idx, smoothing=False):
    """Calculate cross entropy loss, apply label smoothing if needed. """

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.1
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = gold.ne(trg_pad_idx)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum()  # average later
    else:
        loss = F.cross_entropy(pred, gold, ignore_index=trg_pad_idx, reduction='sum')
    return loss


def cal_performance(pred, gold, trg_pad_idx, smoothing=False):
    """Apply label smoothing if needed """

    loss = cal_loss(pred, gold, trg_pad_idx, smoothing=smoothing)

    pred = pred.max(1)[1]
    gold = gold.contiguous().view(-1)
    non_pad_mask = gold.ne(trg_pad_idx)
    n_correct = pred.eq(gold).masked_select(non_pad_mask).sum().item()
    n_word = non_pad_mask.sum().item()

    return loss, n_correct, n_word


def train():
    """tran functions"""
    pass


def main():
    import argparse
    parse = argparse.ArgumentParser(description="设置基本参数")
    # model parameter
    parse.add_argument("--vocab_size", type=int, default=1000, help="字典大小")
    parse.add_argument("--n_position", type=int, default=256, help="位置数量序列最大长度")
    parse.add_argument("--word_vec_size", type=int, default=512, help="embedding输出大小")
    parse.add_argument("--d_model", type=int, default=512, help="隐层大小")
    parse.add_argument("--d_inner", type=int, default=1024, help="隐层中间层大小")
    parse.add_argument("--n_head", type=int, default=8, help="自注意力头的数量")
    parse.add_argument("--d_k", type=int, default=64, help="d_model/n_head每个头隐层的大小")
    parse.add_argument("--d_v", type=int, default=64, help="d_model/n_head每个头隐层的大小")
    parse.add_argument("--encoder_n_layers", type=int, default=6, help="编码的层数")
    parse.add_argument("--decoder_n_layers", type=int, default=6, help="解码的层数")
    parse.add_argument("--dropout", type=float, default=0.1, help="dropout概率")
    parse.add_argument("--pad_idx", type=int, default=-1, help="padding index")
    parse.add_argument("--trg_emb_prj_weight_sharing", action="store_true",
                       default=True)
    parse.add_argument("--emb_src_trg_weight_sharing", action="store_true",
                       default=True)

    # data parameter
    parse.add_argument("--vocab_path", type=str,
                       default=os.path.join(root, "vocabulary/vocab.txt"),
                       help="词汇表路径")
    parse.add_argument("--train_data_path", type=str,
                       default=os.path.join(root, "data/train_small.txt"),
                       help="训练数据路径")
    parse.add_argument("--evaluate_data_path", type=str, default=None,
                       help="评估数据路径")
    args = parse.parse_args()

    tokenizer = BertTokenizer(vocab_file=args.vocab_path)
    args.vocab_size = tokenizer.vocab_size
    args.pad_idx = tokenizer._convert_token_to_id("[PAD]")
    args_dict = vars(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = TransformerConfig(**args_dict)
    model = Transformer(config=config)
    model.to(device)
    pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(filename)s - %(levelname)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S')
    logging.info("root path: {}".format(root))
    main()
