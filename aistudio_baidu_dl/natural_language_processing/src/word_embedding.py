"""
使用飞桨实现Skip-gram --- A kind of the word embedding model
接下来我们将学习使用飞桨实现Skip-gram模型的方法。在飞桨中，不同深度学习模型的训练过程基本一致，流程如下：
    1. 数据处理：选择需要使用的数据，并做好必要的预处理工作。
    2. 网络定义：使用飞桨定义好网络结构，包括输入层，中间层，输出层，损失函数和优化算法。
    3. 网络训练：将准备好的数据送入神经网络进行学习，并观察学习的过程是否正常，如损失函数值是否在降低，也可以打印一些中间步骤的结果出来等。
    4. 网络评估：使用测试集合测试训练好的神经网络，看看训练效果如何。
"""
import io
import os
import logging
import sys
import requests
from collections import OrderedDict
import math
import random
import numpy as np
import paddle
from paddle.nn import Embedding
import paddle.nn.functional as F

logger = logging.getLogger(__name__)


if __name__ == '__main__':
    pass
