"""
# pascalvoc dataset loader
"""
import os
import logging
import json
import cv2
import torch
import torch.utils.data as data
import xml.etree.ElementTree as ET

_logger = logging.getLogger(__name__)


class PascalVocDataset(data.Dataset):
    """VOC Detection Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """
    def __init__(self,
                 data_path_root,  # /Vocdevkit ?
                 image_sets=(('2007', 'trainval'), ('2012', 'trainval')),
                 transform=None,
                 dataset_name='VOC0712'):
        self.root = data_path_root
        self.image_sets = image_sets
        self.transform = transform
        self.name = dataset_name
        # 标记文本的位置
        self._annopath = os.path.join('%s', 'Annotations', '%s.xml')
        # 图片的位置
        self._imgpath = os.path.join('%s', 'JPEGImages', '%s.jpg')
        self.ids = list()
        for (year, name) in image_sets:
            # ./root/VOC2007
            rootpath = os.path.join(self.root, 'VOC' + year)
            # /root/VOC2007/ImageSets/Main/trainval.txt
            for line in open(os.path.join(rootpath, 'ImageSets', 'Main', name + '.txt')):
                # (./root/VOC2007, Image_ID)
                self.ids.append((rootpath, line.strip()))
        pass

    def __getitem__(self, item):
        pass

    def __len__(self):
        pass
