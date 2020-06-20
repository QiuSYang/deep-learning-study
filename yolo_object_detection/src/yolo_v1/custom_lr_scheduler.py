"""
# 自定义 lr_scheduler类，主要功能是在训练过程中动态调整learning rate
"""
import os
import logging
from typing import List
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR

_logger = logging.getLogger(__name__)


class WarmUpMultiStepLR(MultiStepLR):
    def __init__(self,
                 optimizer: optim.Optimizer,
                 milestones: List[int],
                 gamma: float = 0.1,
                 warm_up_factor: float = 0.1,
                 warm_up_iters: int = 500,
                 last_epoch: int = -1):
        self.factor = warm_up_factor
        self.warm_up_iters = warm_up_iters
        super().__init__(optimizer, milestones, gamma, last_epoch)

    def get_lr(self) -> List[float]:
        if self.last_epoch < self.warm_up_iters:
            alpha = self.last_epoch / self.warm_up_iters
            factor = (1 - self.factor) * alpha + self.factor
        else:
            factor = 1

        return [lr * factor for lr in super().get_lr()]


if __name__ == "__main__":
    last_epoch = 2
    for iter in range(1, 1000):
        factor = 0.1
        alpha = iter / 1000
        factor = (1 - factor) * alpha + factor
        print(f'factor:{factor}')
