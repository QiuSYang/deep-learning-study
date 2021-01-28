"""
# transformer train functions
"""
import os
import logging
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from transformer.models import Transformer
from transformer.configs import TransformerConfig
from chat_data_loader import ChatDataset
from transformer.optim import ScheduledOptim

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


def evaluate():
    """evaluate function"""
    pass


def train(config: Transformer, model: Transformer, optimizer: ScheduledOptim,
          train_loader: DataLoader, eval_loader: DataLoader = None, device='cpu'):
    """train function"""
    model.train()
    for epoch in range(config.epochs):
        logger.info("Epoch: {}".format(epoch))
        total_loss, n_word_total, n_word_correct = 0, 0, 0
        for ids, sample in enumerate(tqdm(train_loader)):
            for k, v in sample.items():
                sample[k] = v.to(device)
            input_ids, decoder_input_ids, decoder_target_ids = (sample['input_ids'],
                                                                sample['decode_input_ids'],
                                                                sample['decode_label_ids'])
            optimizer.zero_grad()
            logits = model(input_ids, decoder_input_ids)
            loss, n_correct, n_word = cal_performance(logits,
                                                      gold=decoder_target_ids,
                                                      trg_pad_idx=config.pad_idx,
                                                      smoothing=config.label_smoothing)
            loss.backward()
            optimizer.step_and_update_lr()

            # note keeping
            n_word_total += n_word
            n_word_correct += n_correct
            total_loss += loss.item()
        loss_per_word = total_loss / n_word_total
        accuracy = n_word_correct / n_word_total
        logger.info("The {} epoch train loss: {}, train accuray: {}".format(epoch, loss_per_word, accuracy))

        if eval_loader is not None:
            evaluate()

        if epoch % config.save_epoch == 0:
            model_save = model.module if hasattr(model, "module") else model
            model_file = os.path.join(config.save_dir, "checkpoint_{}.pt".format(epoch))
            torch.save(model_save.state_dict(), f=model_file)

    return model


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
    parse.add_argument("--max_encode_len", type=int, default=192,
                       help="最大编码序列长度")
    parse.add_argument("--max_decode_len", type=int, default=64,
                       help="最大解码序列长度")
    parse.add_argument("--history_turns", type=int, default=3,
                       help="历史对话轮数")
    parse.add_argument("--max_lines", type=int, default=525106,
                       help="最多处理数据量")
    parse.add_argument("--batch_size", type=int, default=32,
                       help="batch size 大小")

    # train parameter
    parse.add_argument("--epochs", type=int, default=20, help="训练epoch数量")
    parse.add_argument("--save_epoch", type=int, default=5, help="每训练多少epoch保存一次模型")
    parse.add_argument("--save_dir", type=str,
                       default=os.path.join(root, "model/transformer_0127"),
                       help="模型保存路径")
    parse.add_argument("--init_lr", type=float, default=1.0, help="初始学习率")
    parse.add_argument("--n_warmup_steps", type=int, default=100, help="热身步长")
    parse.add_argument("--label_smoothing", action="store_true",
                       default=False)

    args = parse.parse_args()

    tokenizer = BertTokenizer(vocab_file=args.vocab_path)
    args.vocab_size = tokenizer.vocab_size
    args.pad_idx = tokenizer._convert_token_to_id("[PAD]")

    args_dict = vars(args)
    config = TransformerConfig(**args_dict)

    if not os.path.exists(config.save_dir):
        os.makedirs(config.save_dir)  # 创建模型保存路径

    logger.info("Load dataset.")
    train_dataset = ChatDataset(config.train_data_path,
                                tokenizer=tokenizer,
                                max_encode_len=config.max_encode_len,
                                max_decode_len=config.max_decode_len,
                                history_turns=config.history_turns,
                                max_lines=config.max_lines)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    if config.evaluate_data_path is not None:
        eval_dataset = ChatDataset(config.evaluate_data_path,
                                   tokenizer=tokenizer,
                                   max_encode_len=config.max_encode_len,
                                   max_decode_len=config.max_decode_len,
                                   history_turns=config.history_turns,
                                   max_lines=config.max_lines)
        eval_loader = DataLoader(eval_dataset, batch_size=config.batch_size, shuffle=False)
    else:
        eval_loader = False

    logger.info("Load model.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 标准写法
    model = Transformer(config=config)
    model.to(device)

    logger.info("Load optimizer.")
    optimizer = ScheduledOptim(
        optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-09),
        config.init_lr, config.d_model, config.n_warmup_steps)

    logger.info("Save all config parameter.")
    config.save_para_to_json_file(os.path.join(root, "data/para.json"))

    logger.info("Training model.")
    train(config, model, optimizer, train_loader=train_loader, eval_loader=eval_loader, device=device)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(filename)s - %(levelname)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S')
    logging.info("root path: {}".format(root))
    main()
