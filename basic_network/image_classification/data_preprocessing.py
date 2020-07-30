"""
# 对图片数据进行预测处理
"""
import os
import json
from pathlib import Path
import PIL
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split

data_in_project = False

# temporarily use resent18 image statistics
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

if data_in_project:
    # set default path for data and test data
    project_dir = Path(__file__).resolve().parent
    datasets_dir = project_dir.joinpath('./data/')
    images_train_dir = project_dir.joinpath('./data/images_train/')
    images_valid_dir = project_dir.joinpath('./data/images_dev/')
    images_test_dir = project_dir.joinpath('./online_test_data/images_test/')
else:
    # 设置数据的绝对路径
    datasets_dir = "/home/yckj2453/nlp_space/jd_multimodal_dialogue/attention-custom-embedding"
    images_train_dir = os.path.join(datasets_dir, 'data/images_train/')
    images_valid_dir = os.path.join(datasets_dir, 'data/images_dev/')
    images_test_dir = os.path.join(datasets_dir, 'online_test_data/images_test/')


def get_image_classes(NO_BG=True):
    images_classification_json_path = os.path.join(datasets_dir, 'data/images_classification.json')
    with open(images_classification_json_path, mode='r', encoding='utf-8') as fp:
        images_info = json.load(fp)

    image_classes = []
    # temp_image_classes = []
    if NO_BG:
        classes2id = {}
        index = 0
    else:
        # 包含背景
        classes2id = {'背景': 0}
        index = 1
    for single_image_info in images_info:
        if single_image_info.get('type') not in image_classes:
            # 将图片类别添加至类别库中
            image_classes.append(single_image_info.get('type'))
            classes2id[single_image_info.get('type')] = index
            index += 1
        # temp_image_classes.append(single_image_info.get('type'))

    # temp_image_classes_ = list(set(temp_image_classes))

    return images_info, image_classes, classes2id


class DialogueImageDataset(Dataset):
    def __init__(self, images_info, classes2id,
                 images_dir=images_test_dir, data_type='val'):
        self.images_info = images_info
        self.classes2id = classes2id

        self.images_dir = images_dir
        self.data_type = data_type

    def __getitem__(self, index):
        image_path = os.path.join(self.images_dir,
                                  self.images_info[index].get('name'))
        image_data, BG = self.image_transform(image_path=image_path)

        if BG:
            class_id = self.classes2id.get('背景')
        else:
            class_id = self.classes2id.get(self.images_info[index].get('type'))

        return image_data, class_id

    def __len__(self):
        return len(self.images_info)

    def image_transform(self, image_path):
        BG = False
        image_data = torch.zeros(3, 224, 224)
        try:
            img_tmp = PIL.Image.open(image_path)
            image_data = data_transforms[self.data_type](img_tmp)
        except:
            print("can't open image file: ", image_path)
            # 图片读取失败将其设置为背景类
            BG = True

        return image_data, BG


if __name__ == "__main__":
    images_info, image_classes, classes2id = get_image_classes()

    # 8:2的拆分方式
    train_images_info, test_images_info = train_test_split(images_info, test_size=0.2)

    dataset = DialogueImageDataset(train_images_info, classes2id, images_dir=images_train_dir, data_type='train')

    temp = dataset[0]
