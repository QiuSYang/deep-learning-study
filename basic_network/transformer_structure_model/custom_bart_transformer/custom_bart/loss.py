import torch
from torch.nn import functional as F
import torch.nn as nn
from convert import to_var


def sequence_mask(sequence_length, max_len=None):
    """
    Args:
        sequence_length (Variable, LongTensor) [batch_size]
            - list of sequence length of each batch
        max_len (int)
    Return:
        masks (bool): [batch_size, max_len]
            - True if current sequence is valid (not padded), False otherwise

    Ex.
    sequence length: [3, 2, 1]

    seq_length_expand
    [[3, 3, 3],
     [2, 2, 2]
     [1, 1, 1]]

    seq_range_expand
    [[0, 1, 2]
     [0, 1, 2],
     [0, 1, 2]]

    masks
    [[True, True, True],
     [True, True, False],
     [True, False, False]]
    """
    if max_len is None:
        max_len = sequence_length.max()
    batch_size = sequence_length.size(0)

    # [max_len]
    seq_range = torch.arange(0, max_len).long()  # [0, 1, ... max_len-1]

    # [batch_size, max_len]
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_range_expand = to_var(seq_range_expand)

    # [batch_size, max_len]
    seq_length_expand = sequence_length.unsqueeze(1).expand_as(seq_range_expand)

    # [batch_size, max_len]
    masks = seq_range_expand < seq_length_expand

    return masks


# https://gist.github.com/jihunchoi/f1434a77df9db1bb337417854b398df1
def masked_cross_entropy(logits, target, length, per_example=False):
    """
    Args:
        logits (Variable, FloatTensor): [batch, max_len, num_classes]
            - unnormalized probability for each class
        target (Variable, LongTensor): [batch, max_len]
            - index of true class for each corresponding step
        length (Variable, LongTensor): [batch]
            - length of each data in a batch
    Returns:
        loss (Variable): []
            - An average loss value masked by the length
    """
    batch_size, max_len, num_classes = logits.size()

    # [batch_size * max_len, num_classes]
    logits_flat = logits.view(-1, num_classes)

    # [batch_size * max_len, num_classes]
    log_probs = F.log_softmax(logits_flat, dim=1)

    # [batch_size * max_len, 1]
    target_flat = target.contiguous().view(-1, 1)

    # Negative Log-likelihood: -sum {  1* log P(target)  + 0 log P(non-target)} = -sum( log P(target) )
    # [batch_size * max_len, 1]
    nll_loss = -torch.gather(log_probs, dim=1, index=target_flat)

    nll_loss = nll_loss.squeeze(1)

    # [batch_size, max_len]
    # nll_loss = nll_loss.view(batch_size, max_len)

    smooth_loss = -log_probs.mean(dim=-1)

    # 损失平滑
    losses = (1 - 0.1) * nll_loss + 0.1 * smooth_loss

    losses = losses.view(batch_size, max_len)

    # [batch_size, max_len]
    mask = sequence_mask(sequence_length=length, max_len=max_len)

    # Apply masking on loss
    losses = losses * mask.float()

    # word-wise cross entropy
    # loss = losses.sum() / length.float().sum()

    if per_example:
        # loss: [batch_size]
        return losses.sum(1)
    else:
        loss = losses.sum()
        return loss, length.float().sum()


def masked_cross_entropy_bk(logits, target, length, per_example=False):
    """
    Args:
        logits (Variable, FloatTensor): [batch, max_len, num_classes]
            - unnormalized probability for each class
        target (Variable, LongTensor): [batch, max_len]
            - index of true class for each corresponding step
        length (Variable, LongTensor): [batch]
            - length of each data in a batch
    Returns:
        loss (Variable): []
            - An average loss value masked by the length
    """
    batch_size, max_len, num_classes = logits.size()

    # [batch_size * max_len, num_classes]
    logits_flat = logits.view(-1, num_classes)

    # [batch_size * max_len, num_classes]
    log_probs_flat = F.log_softmax(logits_flat, dim=1)

    # [batch_size * max_len, 1]
    target_flat = target.view(-1, 1)

    # Negative Log-likelihood: -sum {  1* log P(target)  + 0 log P(non-target)} = -sum( log P(target) )
    # [batch_size * max_len, 1]
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)

    # [batch_size, max_len]
    losses = losses_flat.view(batch_size, max_len)

    # [batch_size, max_len]
    mask = sequence_mask(sequence_length=length, max_len=max_len)

    # Apply masking on loss
    losses = losses * mask.float()

    # word-wise cross entropy
    # loss = losses.sum() / length.float().sum()

    if per_example:
        # loss: [batch_size]
        return losses.sum(1)
    else:
        loss = losses.sum()
        return loss, length.float().sum()
