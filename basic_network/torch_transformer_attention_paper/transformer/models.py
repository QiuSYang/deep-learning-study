"""
# define the transformer model
"""
import os
import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class Transformer(nn.Module):
    """A sequence to sequence model with attention mechanism."""
    def __init__(self):
        super(Transformer, self).__init__()
