"""
# pascalvoc dataset loader
"""
import os
import logging
import json
import cv2
import numpy as np
import torch
import torch.utils.data as data
import xml.etree.ElementTree as ET

_logger = logging.getLogger(__name__)

VOC_CLASSES = (  # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor'
)


class VOCAnnotationTransform(object):
    # 将VOC的标注转换为 (x,y,w,h,class), class为上面VOC_CLASSES的序号
    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, class_to_ind=None, keep_difficult=False):
        self.class_to_ind = class_to_ind or dict(
            # 将物体名称与0~class数量绑定
            zip(VOC_CLASSES, range(len(VOC_CLASSES))))
        self.keep_difficult = keep_difficult

    # 可调用对象
    def __call__(self, target, width, height):
        """Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        res = []
        # 对ET.Element 里面名字为'object'的对象进行遍历
        # 具体用法：https://www.cnblogs.com/ifantastic/archive/2013/04/12/3017110.html
        for obj in target.iter('object'):
            # difficult VOC文档里的含义，标为 1 表示难以辨认
            # ‘difficult’: an object marked as ‘difficult’ indicates that the object is considered
            # difficult to recognize, for example an object which is clearly visible but unidentifiable
            # without substantial use of context. Objects marked as difficult are currently ignored
            # in the evaluation of the challenge.
            difficult = int(obj.find('difficult').text) == 1
            # 检测目标为难以检测而且self.keep_difficult标记为1才继续进行操作
            if not self.keep_difficult and difficult:
                continue

            # 用法解释：
            # str = "00000003210Runoob01230000000";
            # print str.strip( '0' );  # 去除首尾字符 0
            name = obj.find('name').text.lower().strip()
            # 数据格式：
            # <bndbox>
            # 			<xmin>174</xmin>
            # 			<ymin>101</ymin>
            # 			<xmax>349</xmax>
            # 			<ymax>351</ymax>
            # 		</bndbox>
            bbox = obj.find('bndbox')
            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            # 得到一组0~1.0范围的值
            for i, pt in enumerate(pts):
                # bbox 数值为像素点的位置，从1开始取所以要减去1？
                cur_pt = int(bbox.find(pt).text) - 1
                # scale height or width(变换之后再归一化)
                # cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                bndbox.append(cur_pt)
            label_idx = self.class_to_ind[name]
            # 查找name类别对应的标号
            bndbox.append(label_idx)
            res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind]

        # res: tensor[ ,5] i.e. [xmin, ymin, xmax, ymax, label_ind], ... ]
        return res


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
                 dataset_name='VOC0712',
                 class_to_index=None,
                 keep_difficult=False):
        self.root = data_path_root
        self.image_sets = image_sets
        self.transform = transform
        self.target_transform = VOCAnnotationTransform(class_to_ind=class_to_index,
                                                       keep_difficult=keep_difficult)
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

    def __getitem__(self, item):
        # im为图片，gt=get_target
        im, gt, h, w = self.pull_item(item)
        # i.e. tensor[c,h,w],[[xmin, ymin, xmax, ymax, label_idx], ... ]
        return im, gt

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        # img_id=(./VOCdevkit/VOC2007, Image_ID)
        img_id = self.ids[index]
        # '%s/Annotations/%s.xml'.format((./VOCdevkit/VOC2007, Img_ID))
        # ===>./root/VOC2007/Annotations/Image_ID.xml'
        # target 为解析后的.xml 文件根节点。
        # xml_file = self._annopath % img_id
        target = ET.parse(self._annopath % img_id).getroot()
        # ===>./root/VOC2007/Annotations/Image_ID.jpg'
        img = cv2.imread(self._imgpath % img_id)
        # 得到图片的宽高
        height, width, channels = img.shape

        # 对标注格式进行转换，默认为上文的VOCAnnotationTransform()
        # 输入一个ET.parse().getroot()的element，得到[[xmin, ymin, xmax, ymax, label_ind], ... ]
        if self.target_transform is not None:
            target = self.target_transform(target, width, height)

        if self.transform is not None:
            # 将list转化为np.ndarray
            target = np.array(target)

            # # 调试使用，显示原始图像
            # self.image_show(img, target)

            # img为cv图片
            # boxes=[xmin, ymin, xmax, ymax]\in[0,1],
            # abels=类名对应的序号,i.e.[idx]
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            # to rgb：[h,w,c], 其中c 为 BGR
            # i.e. img = img.transpose(2, 0, 1)
            img = img[:, :, (2, 1, 0)]

            # hstack, 在最低的维度进行连接，这不还原成了上面的target？
            # [[xmin, ymin, xmax, ymax, label_idx], ... ]
            target = np.hstack((boxes, np.expand_dims(labels.astype(int), axis=1)))

            # # 调试使用，显示变换后的图像
            # self.image_show(img, target)

        # scale height or width([xmin, ymin, xmax, ymax, label_idx])-归一化
        for i in range(len(target[:4])):
            if i % 2 == 0:
                target[:, i] = target[:, i]/float(img.shape[1])
            else:
                target[:, i] = target[:, i] / float(img.shape[0])

        # tensor[c,h,w], np.array[[xmin, ymin, xmax, ymax, label_ind], ... ]
        # 返回的转置的数组, target, 原始图像(即没有变换之前的)高、宽
        return torch.from_numpy(img).permute(2, 0, 1), target, height, width
        # return torch.from_numpy(img), target, height, width

    def pull_image(self, index):
        """Returns the original image object at index in PIL form（返回原始的PIL图片）

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            PIL img
        """
        img_id = self.ids[index]
        # cv.IMREAD_COLOR = 1 : 将图像转为彩色读取

        return cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)

    def pull_anno(self, index):
        """Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        """
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        gt = self.target_transform(anno, 1, 1)

        return img_id[1], gt

    def pull_tensor(self, index):
        """Returns the original image at an index in tensor form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            tensorized version of img, squeezed
        """

        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)

    def image_show(self, image, target_list):
        """
        :param image: 图像数据
        :param target: bbox和标签[xmin, ymin, xmax, ymax, label_index]
        :return:
        """
        image_draw = image.copy()
        for target in target_list:
            bbox, label_index = target[:4], target[4]
            label_class = VOC_CLASSES[int(label_index)]
            image_draw = cv2.rectangle(image_draw,
                                       tuple(bbox[:2].astype(int)), tuple(bbox[2:].astype(int)),
                                       (0, 255, 0), 2)
            image_draw = cv2.putText(image_draw,
                                     text=label_class,
                                     org=tuple(bbox[:2].astype(int)),
                                     fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                     fontScale=1.2,
                                     color=(0, 0, 255),
                                     thickness=2)

        cv2.imshow("image", image_draw)
        cv2.waitKey()


def gt_data_encoder(boxes, labels, grid_num=7):
    """ yolo-v1 gt bbox 编码，变为[[30 length tensor]]
    boxes (tensor) [[x1,y1,x2,y2],[]]
    labels (tensor) [...]
    return SxSx30
    30: B1[:4], Obj1[4], B2[5:9], Obj[9], C[9:]
    """
    # 初始化gt tensor(与yolo-v1最后一层网络的输出相同)
    target = torch.zeros((grid_num, grid_num, 30))
    # 将网格归一化(计算单位网格的宽度)
    cell_size = 1.0 / grid_num
    # boxes的(w, h) - （右下角坐标 - 右上角坐标）
    wh = boxes[:, 2:] - boxes[:, :2]
    # center(x, y)
    cxcy = (boxes[:, 2:] + boxes[:, :2]) / 2.0

    # 扫描每个box
    for i in range(cxcy.size()[0]):
        cxcy_sample = cxcy[i]
        # 计算属于格子的第几行第几列(向上取整但还要减一，实际就是1.5对应1-向上为2，再减1到1)
        ij = (cxcy_sample / cell_size).ceil() - 1
        # B1、B2、C 标记为1
        target[int(ij[1]), int(ij[0]), 4] = 1
        target[int(ij[1]), int(ij[0]), 9] = 1
        # int(labels[i]) + 10
        target[int(ij[1]), int(ij[0]), int(labels[i]) + 10] = 1
        # 匹配到的网格的左上角相对坐标
        xy = ij * cell_size
        # 真框相对于格子坐上角的偏移量
        # (box的中心坐标存储是相当目标中心所在格点的左上角的偏移量)
        delta_xy = (cxcy_sample - xy) / cell_size
        target[int(ij[1]), int(ij[0]), 2:4] = wh[i]
        target[int(ij[1]), int(ij[0]), :2] = delta_xy
        target[int(ij[1]), int(ij[0]), 7:9] = wh[i]
        target[int(ij[1]), int(ij[0]), 5:7] = delta_xy

    return target


def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).-如何把多个sample打包成batch的函数

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        imgs: tensor [batch_size, 3, 448, 448]
        boxes: list of tensor:[, 4] for (x1, y1, x2, y2)
        labels: list of LongTensor:[,1]
        gt_outs: the ground truth outputs of model
    """
    imgs = []
    boxes, labels, gt_outs = [], [], []
    for sample in batch:
        # sample[0]:[3,h,w], sample[1]:[, 5]
        imgs.append(sample[0])
        # print(sample[1].shape, sample[1])
        # 下面两组都可以使用torch.tensor(data, dtype=None, device=None, requires_grad=False)替换
        # box = torch.tensor(data=[i[:4] for i in sample[1]], dtype=torch.float32)
        box = torch.FloatTensor([i[:4] for i in sample[1]])
        # label = torch.LongTensor(data=[i[4] for i in sample[1]], dtype=torch.long)
        label = torch.LongTensor([i[4] for i in sample[1]])

        boxes.append(box)
        labels.append(label)
        # 对boxes和label在网格上编码(即将每个网格赋值)
        gt_outs.append(gt_data_encoder(box, label))

    # print(f'boxes:{boxes}\n, labels:{labels}')

    return torch.stack(imgs, 0), boxes, labels, torch.stack(gt_outs, 0)


if __name__ == "__main__":
    from src.tool.image_augmentations import CustomCompose, DistortionLessResize
    transforms_composed = CustomCompose([DistortionLessResize(max_width=448)])
    data_root = "../../data/pascalvoc/VOCdevkit"
    dataset = PascalVocDataset(data_path_root=data_root, transform=transforms_composed)
    test = dataset[2]

    cv2.destroyAllWindows()
    pass
