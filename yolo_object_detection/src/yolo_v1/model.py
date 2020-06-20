"""
# yolo v1 主干网络构建, darknet-19
"""
import os
import logging
import torch
import torch.nn as nn

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
    def __init__(self, s=7, b=2, num_classes=1000):
        super(YoloV1Net, self).__init__()

        self.s = s  # 正方形网格数
        self.b = b  # 每个格的预测框数
        self.num_classes = num_classes  # 类别数

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

        self.fc2 = nn.Linear(4096, (5*self.b+self.num_classes)*self.s*self.s)

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
        # 最后输出形状是[batch_size, H, W, channel],
        # 而输入形状是[batch_size, channel, H, W](是第一个全连接层之后被打乱的)
        x = x.view(-1, 7, 7, 30)

        return x

    def num_flat_features(self, x):
        # all dimensions except the batch dimension
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s

        return num_features

    def save_model(self, model_dir_path, model_file_name,
                   step=None, optimizer=None, lr_scheduler=None):
        """保存模型
        :param model_dir_path: 模型保存路径
        :param model_file_name: 模型文件名
        :param step:
        :param optimizer: 训练优化器state dict 参数
        :param lr_scheduler: scheduler state dict 参数
        :return:
        """
        # 安全保存模型参数
        self.save_safely(state_dict=self.state_dict(),
                         dir_path=model_dir_path,
                         file_name='{}.pkl'.format(model_file_name))
        _logger.info("model weights saved successfully at {}.".format(os.path.join(model_dir_path,
                                                                                   '{}.pkl'.format(model_file_name))))

        if optimizer and lr_scheduler and step is not None:
            # 安全保存辅助参数
            temp = {'step': step,
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'optimizer': optimizer.state_dict()}
            self.save_safely(state_dict=temp,
                             dir_path=model_dir_path,
                             file_name='{}_para.pkl'.format(model_file_name))
            _logger.info("auxiliary part saved successfully at {}.".format(os.path.join(model_dir_path,
                                                      '{}_para.pkl'.format(model_file_name))))

    def load_model(self, model_dir_path, model_file_name,
                   optimizer=None, lr_scheduler=None):
        """加载模型
        :param model_dir_path: 模型保存路径
        :param model_file_name: 模型文件名
        :param optimizer: 训练优化器state dict 参数
        :param lr_scheduler: scheduler state dict 参数
        :return:
        """
        # 1. 加载模型结构参数
        try:
            saved_model = torch.load(os.path.join(model_dir_path, '{}.pkl'.format(model_file_name)),
                                     map_location='cpu')
            self.load_state_dict(saved_model)
            _logger.info("loading model weight successfully.")
        except Exception:
            _logger.info("loading model weight fail.")

        if optimizer and lr_scheduler is not None:
            # 加载辅助参数
            try:
                temp = torch.load(os.path.join(model_dir_path, '{}_para.pkl'.format(model_file_name)),
                                  map_location='cpu')
                lr_scheduler.load_state_dict(temp.get('lr_scheduler'))
                optimizer.load_state_dict(temp.get('optimizer'))
                step = temp.get('step')
                _logger.info("loading optimizer, lr_scheduler and step successfully.")

                return step
            except Exception:
                _logger.info("loading optimizer, lr_scheduler and step fail.")

                return 0

    @staticmethod
    def save_safely(state_dict, dir_path, file_name):
        """save the file safely, if detect the file name conflict,
            save the new file first and remove the old file
        :param state_dict: 模型参数字典或者辅助参数optimizer、lr_scheduler字典
        :param dir_path: 模型保存路径
        :param file_name: 模型名字
        :return:
        """
        if not os.path.exists(dir_path):
            # 模型保存路径不存在那就创建
            os.mkdir(dir_path)
            _logger.info("dir not exit, create one.")
        save_path = os.path.join(dir_path, file_name)
        if os.path.exists(save_path):
            # 文件已经存在, 先保存到临时文件,
            # 之后先删除之前的文件, 再将临时文件重命名为目标文件
            temp_name = '{}.temp'.format(save_path)
            torch.save(state_dict, temp_name)
            os.remove(save_path)
            os.rename(temp_name, save_path)
            _logger.info("find the file conflict while saving, saved safely.")
        else:
            torch.save(state_dict, save_path)


if __name__ == "__main__":
    pass
