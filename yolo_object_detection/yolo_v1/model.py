"""
# yolo v1 主干网络构建, darknet-19
"""
import os
import logging
import torch
from torch import nn

_logger = logging.getLogger(__name__)


class ConvBNLeakyRelu(nn.Module):
    """conv 单元结构"""
    def __init__(self, input_channels, output_channels,
                 kernel_size, stride=1, padding=0, dilation=1):
        super(ConvBNLeakyRelu, self).__init__()
        self.conv = nn.Conv2d(in_channels=input_channels,
                              out_channels=output_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=dilation)
        self.bn = nn.BatchNorm2d(num_features=output_channels)
        self.activate = nn.LeakyReLU(negative_slope=0.1,
                                     inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activate(x)

        return x


class BasicBlock(nn.Module):
    def __init__(self, input_channels,
                 output_channels_middle, output_channels,
                 kernel_size1, kernel_size2,
                 stride1=1, stride2=1,
                 padding1=0, padding2=0,
                 units=1):
        self.conv_1 = ConvBNLeakyRelu(input_channels, output_channels_middle,
                                      kernel_size=kernel_size1, stride=stride1, padding=padding1)
        self.conv_2 = ConvBNLeakyRelu(output_channels_middle, output_channels,
                                      kernel_size=kernel_size2, stride=stride2, padding=padding2)
        self.units = units

    def forward(self, x):
        # n 个单元串联
        for i in range(self.units):
            x = self.conv_1(x)
            x = self.conv_2(x)

        return x


class YoloV1Net(nn.Module):
    def __init__(self, num_classes=1000):
        super(YoloV1Net, self).__init__()

        # backbone network: darknet-19
        # 1-output : stride = 4, c = 64
        self.conv_1 = ConvBNLeakyRelu(3, 64, 7, stride=2, padding=3)
        self.pool_1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        # 2-output: stride = 2, c =192
        self.conv_2 = ConvBNLeakyRelu(64, 192, 3, stride=1, padding=1)
        self.pool_2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        # 3-output: stride = 2, c = 512
        self.conv_3 = BasicBlock(192, 128, 256,
                                 kernel_size1=1, kernel_size2=3,
                                 padding1=0, padding2=1)
        self.conv_4 = BasicBlock(256, 256, 512,
                                 kernel_size1=1, kernel_size2=3,
                                 padding1=0, padding2=1)
        self.pool_3 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        # 4-output: stride = 2, c = 1024
        self.conv_5 = BasicBlock(512, 256, 512,
                                 kernel_size1=1, kernel_size2=3,
                                 padding1=0, padding2=1, units=4)
        self.conv_6 = BasicBlock(512, 512, 1024,
                                 kernel_size1=1, kernel_size2=3,
                                 padding1=0, padding2=1)
        self.pool_4 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        # 5-output: stride = 2, c = 1024
        self.conv_7 = BasicBlock(1024, 512, 1024,
                                 kernel_size1=1, kernel_size2=3,
                                 padding1=0, padding2=1, units=2)
        self.conv_8 = BasicBlock(1024, 1024, 1024,
                                 kernel_size1=3, kernel_size2=3,
                                 stride1=1, stride2=2,
                                 padding1=1, padding2=1)

        # 6-output: stride=1, c =1024
        self.conv_9 = BasicBlock(1024, 1024, 1024,
                                 kernel_size1=3, kernel_size2=3,
                                 padding1=1, padding2=1)

        self.fc1 = nn.Linear(1024*7*7, 4096)

        self.fc2 = nn.Linear(4096, 30*7*7)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.pool_1(x)

        x = self.conv_2(x)
        x = self.pool_2(x)

        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.pool_3(x)

        x = self.conv_5(x)
        x = self.conv_6(x)
        x = self.pool_4(x)

        x = self.conv_7(x)
        x = self.conv_8(x)

        x = self.conv_9(x)

        x = x.view(-1, self.num_flat_features(x))
        x = self.fc1(x)
        x = nn.ReLU(inplace=True)(x)
        x = nn.Dropout()(x)

        x = self.fc2(x)
        # 归一化到0-1
        x = torch.sigmoid(x)
        x = x.view(-1, 7, 7, 30)

        return x

    def num_flat_features(self, x):
        # all dimensions except the batch dimension
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s

        return num_features
