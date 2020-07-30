"""
# 使用resnet做图片分类
"""
import os
import json
import logging
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import PIL
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torchvision
from sklearn.model_selection import train_test_split
from image_analysis.data_preprocessing import (images_train_dir,
                                               images_test_dir,
                                               data_transforms,
                                               get_image_classes,
                                               DialogueImageDataset)
# 日志记录
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(filename)s - %(levelname)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S')
_logger = logging.getLogger(__name__)
# data file dir
data_dir = './'
model_save_dir = './'


class ImageClassification(nn.Module):
    def __init__(self, feature_size, num_classes, feature_extracting=False, resnet_method='resnet18'):
        """Image Encoder"""
        super(ImageClassification, self).__init__()

        self.input_size = feature_size
        self.num_classes = num_classes
        self.resnet_method = resnet_method
        self.feature_extracting = feature_extracting

        self.model = self.init_resnet18(self.input_size)
        # 分类层
        self.classification = nn.Linear(feature_size, num_classes)

    def init_resnet18(self, feature_size):
        def set_parameter_requires_grad(model, feature_extracting):
            if feature_extracting:
                for param in model.parameters():
                    param.requires_grad = False

        # model_ft = torchvision.models.resnet18(pretrained=True)
        model_method = eval("torchvision.models." + self.resnet_method)
        model_ft = model_method(pretrained=True)
        # 设置参数是否需要更新
        set_parameter_requires_grad(model_ft, self.feature_extracting)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, feature_size)

        return model_ft

    def forward(self, inputs):

        """
        Args:
            inputs (Variable, LongTensor): [num_setences, 3, 224, 224]
            input_length (Variable, LongTensor): [num_sentences]
        Return:
            outputs (Variable): [max_source_length, batch_size, hidden_size]
                - list of all hidden states
            hidden ((tuple of) Variable): [num_layers*num_directions, batch_size, hidden_size]
                - last hidden state
                - (h, c) or h
        """
        x = self.model(inputs)
        x = self.classification(x)

        return x


class Main(object):
    def __init__(self, images_info=None, classes2id=None, args=None):
        self.args = args
        self.model = ImageClassification(self.args.feature_size, self.args.num_classes,
                                         feature_extracting=self.args.feature_extracting,
                                         resnet_method=self.args.resnet_method)
        self.epoch_i = 0
        # 设置GPU
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.args.seed:
            self.set_random_seed()

        # 将模型拷贝到固定设备
        self.model.to(self.device)
        if self.args.checkpoint:
            # load init model weight
            self.model_load(self.args.checkpoint)

        if self.args.is_train:
            self.criterion = nn.CrossEntropyLoss()
            self.optimizer = None
            self.train_images_info, self.val_images_info = train_test_split(images_info, test_size=0.2)
            self.classes2id = classes2id

    def data_loader(self, images_info, images_dir=images_test_dir, data_type='val'):
        """数据生产器"""
        dataset = DialogueImageDataset(images_info, self.classes2id, images_dir=images_dir, data_type=data_type)

        data_loader = DataLoader(
            dataset=dataset,
            batch_size=self.args.batch_size,
            # num_workers=3,
            # pin_memory=True,
            shuffle=True)

        return data_loader

    def train(self):
        """训练函数"""
        # 1. 加载数据
        train_data_loader = self.data_loader(self.train_images_info,
                                             images_dir=images_train_dir, data_type='train')
        # train_data_num = len(train_data_loader)

        # 2. 配置训练基本
        # Observe that all parameters are being optimized
        optimizer_ft = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        self.optimizer = optimizer_ft

        # Decay LR by a factor of 0.1 every 7 epochs
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

        self.model.train()

        _logger.info("Start training.")
        best_acc = 0.0
        for epoch in range(self.epoch_i, self.args.epoch):
            _logger.info("Epoch {}/{}".format(epoch, self.args.epoch-1))

            # 每轮参数的记录
            running_loss = 0.0
            running_corrects = 0
            # Iterate over data
            for batch_id, (inputs_image, labels) in enumerate(tqdm(train_data_loader, ncols=80)):
                inputs_image = inputs_image.to(self.device)
                labels = labels.to(self.device)

                # zero the parameter gradients
                optimizer_ft.zero_grad()
                # forward
                outputs = self.model(inputs_image)
                # 计算预测标签
                _, preds = torch.max(outputs, dim=1)
                # 计算损失
                loss = self.criterion(outputs, labels)

                # backward + optimize only if in training phase
                loss.backward()
                optimizer_ft.step()

                # statistics
                running_loss += loss.item() * inputs_image.size(0)
                running_corrects += torch.sum(preds == labels.data)

            # 参数更新
            exp_lr_scheduler.step()

            epoch_loss = running_loss / len(self.train_images_info)
            epoch_acc = running_corrects.double() / len(self.train_images_info)

            _logger.info('Train Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

            # 进行模型评测
            _logger.info("The {} epoch model evaluate.".format(epoch))
            val_loss, val_acc = self.evaluate()
            _logger.info('Evaluate Loss: {:.4f} Acc: {:.4f}'.format(val_loss, val_acc))
            if val_acc > best_acc:
                # 保存最佳模型
                _logger.info("Save beat model.")
                self.model_save(model_save_dir, 'best', epoch)
                # 最佳精度更新
                best_acc = val_acc

        return self.model

    def evaluate(self):
        """评估函数"""
        # 1. 加载数据
        val_data_loader = self.data_loader(self.val_images_info,
                                           images_dir=images_train_dir, data_type='val')

        self.model.eval()
        # 每轮参数的记录
        running_loss = 0.0
        running_corrects = 0
        with torch.no_grad():
            for batch_id, (inputs_image, labels) in enumerate(tqdm(val_data_loader, ncols=80)):
                inputs_image = inputs_image.to(self.device)
                labels = labels.to(self.device)

                # forward
                outputs = self.model(inputs_image)
                # 计算预测标签
                _, preds = torch.max(outputs, dim=1)
                # 计算损失
                loss = self.criterion(outputs, labels)

                # statistics
                running_loss += loss.item() * inputs_image.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(self.val_images_info)
            epoch_acc = running_corrects.double() / len(self.val_images_info)

        return epoch_loss, epoch_acc

    def predict(self, images, id2classes):
        """模型预测
        id2classes:
        """
        images_class = []
        self.model.eval()
        with torch.no_grad():
            # forward
            images = images.to(self.device)
            outputs = self.model(images)
            # 计算预测标签
            _, preds = torch.max(outputs, dim=1)

            for single_image_label_index in preds:
                images_class.append(id2classes.get(str(single_image_label_index.item())))

        return images_class, preds

    def model_save(self, model_dir, model_name, epoch=0):
        """保存模型"""
        ckpt_path = os.path.join(model_dir, f'{model_name}.pkl')
        model_state = {'model': self.model.state_dict(),
                       'optimizer': self.optimizer.state_dict(),
                       'epoch': epoch}
        print(f'Save parameters to {ckpt_path}')
        torch.save(model_state, ckpt_path)

    def model_load(self, model_path):
        """加载模型"""
        model_weight = torch.load(model_path)
        self.epoch_i = model_weight.get('epoch')
        # load model weights
        self.model.load_state_dict(model_weight.get('model'))

    def set_random_seed(self):
        """
        设置训练的随机种子
        """
        torch.manual_seed(self.args.seed)
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.args.seed)
            torch.cuda.manual_seed_all(self.args.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def imshow(self, inp, title=None):
        """Imshow for Tensor."""
        inp = inp.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        plt.imshow(inp)
        if title is not None:
            plt.title(title)
        # pause a bit so that plots are updated
        plt.pause(0.001)


def save_id2classes(classes2id):
    """将id转为class"""
    id2classes = {}
    for key, value in classes2id.items():
        id2classes[value] = key
    # 字典写入json 文件
    with open(os.path.join(data_dir, 'id2classes.json'), mode='w', encoding='utf-8') as fw:
        json.dump(id2classes, fw)


def get_single_image_and_transforms(image_path):
    """图像数据转换"""
    image_data = torch.zeros(3, 224, 224)
    try:
        img_tmp = PIL.Image.open(image_path)
        # 数据转换
        image_data = data_transforms['val'](img_tmp)
    except:
        print("can't open image file: ", image_path)

    return image_data


def get_single_image_data():
    """获取单张图片"""
    test_images_dir = os.path.join(data_dir, 'images_test')
    if os.path.exists(test_images_dir):
        files = os.listdir(test_images_dir)
        if files:
            single_image_path = os.path.join(test_images_dir,
                                             files[random.randint(0, len(files))])
            image_data = get_single_image_and_transforms(single_image_path)

            return image_data

    return None


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    import argparse
    parser = argparse.ArgumentParser(description="image classification hyper-parameter.")

    parser.add_argument('--is_train', type=bool, default=False)
    parser.add_argument('--feature_extracting', type=bool, default=False)
    parser.add_argument('--resnet_method', type=str, default='resnet18')
    parser.add_argument('--feature_size', type=int, default=256)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--checkpoint', type=str, default='./best.pkl')
    parser.add_argument('--id2classes', type=str, default=os.path.join(data_dir, 'id2classes.json'))
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    args = parser.parse_args()

    if args.is_train:
        # 训练
        images_info, image_classes, classes2id = get_image_classes()
        # id2classes save to file
        save_id2classes(classes2id)
        args.num_classes = len(classes2id.keys())

        main = Main(args=args, images_info=images_info, classes2id=classes2id)
        model = main.train()
    else:
        # 预测
        if not args.checkpoint:
            raise FileNotFoundError("model weights file not input.")
        with open(args.id2classes, mode='r', encoding='utf-8') as fp:
            id2classes = json.load(fp)
        args.num_classes = len(id2classes.keys())
        main = Main(args=args)
        images = get_single_image_data()
        if len(images.shape) < 4:
            images = images.unsqueeze(dim=0)
        else:
            if not isinstance(type(images), torch.Tensor):
                # images 是一个图片列表
                images = torch.stack(images, dim=0)
        for image in images:
            main.imshow(image)

        # 进行预测
        images_class, images_class_id = main.predict(images=images, id2classes=id2classes)
        print("class name: {}, class id: {}.".format(images_class, images_class_id))
