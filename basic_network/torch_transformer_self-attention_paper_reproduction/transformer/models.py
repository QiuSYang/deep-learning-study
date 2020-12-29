"""
# define the transformer model
"""
import os
import logging
import torch
import torch.nn as nn

from transformer.configs import TransformerConfig

logger = logging.getLogger(__name__)


class Transformer(nn.Module):
    """A sequence to sequence model with attention mechanism."""
    def __init__(self, config: TransformerConfig):
        super(Transformer, self).__init__()
        self.config = config

        self.encoder = None
        self.decoder = None
