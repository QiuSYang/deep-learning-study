"""
# 工具类
"""
import torch


def get_pad_mask(seq, pad_idx):
    # [bts, 1, sqln]
    return (seq != pad_idx).unsqueeze(-2)  # 扩充到每个token attention都是一样的(encoder每个token的attention mask都一样)


def get_subsequent_mask(seq):
    """For masking out the subsequent info. """
    sz_b, len_s = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    # [bts, sqln, sqln]
    return subsequent_mask


if __name__ == "__main__":
    arr_ids = torch.tensor([[12, 25, 45, 67, 23, 2, 0, 0, 0]])
    inputs = get_pad_mask(arr_ids, 0)
    print(inputs)
    att_score = torch.ones((1, 1, 9, 9))
    att_score_ = att_score.masked_fill(inputs.unsqueeze(1) == 0, -1e-9)

    x = get_subsequent_mask(arr_ids).unsqueeze(1)
    print(x)
