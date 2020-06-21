"""
# 模型预测模块，主要包括predict(), nms(), decoder()等function
"""
import os
import logging
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms

from src.datasets.pascalvoc_dataset import VOC_CLASSES
from src.yolo_v1.model import YoloV1Net
from src.tool.image_augmentations import CustomImageNormalize
from src.yolo_v1.config import YoloV1Config

_logger = logging.getLogger(__name__)


def draw_box(img_np, boxes_np, tags_np,
             scores_np=None, relative_coord=False, save_path=None):
    """绘制结果"""
    if scores_np is None:
        scores_np = [1.0 for i in tags_np]
    # img = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
    h, w, _ = img_np.shape
    if relative_coord and len(boxes_np) > 0:
        boxes_np = np.array([
            boxes_np[:, 0] * w,
            boxes_np[:, 1] * h,
            boxes_np[:, 2] * w,
            boxes_np[:, 3] * h,
        ]).T
    plt.figure(figsize=(10, 10))
    colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
    currentAxis = plt.gca()
    for box, tag, score in zip(boxes_np, tags_np, scores_np):
        LABLES = VOC_CLASSES
        tag = int(tag)
        label_name = LABLES[tag]
        display_txt = '%s: %.2f' % (label_name, score)
        coords = (box[0], box[1]), box[2] - box[0] + 1, box[3] - box[1] + 1
        color = colors[tag]
        currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
        currentAxis.text(box[0], box[1], display_txt, bbox={'facecolor': color, 'alpha': 0.5})
    plt.imshow(img_np)
    if save_path is not None:
        # fig, ax = plt.subplots()
        fig = plt.gcf()
        fig.savefig(save_path)
        plt.cla()
        plt.clf()
        plt.close('all')
    else:
        plt.show()


class YoloV1Predict(object):
    def __init__(self, size=448):
        self.size = size
        self.LABELS = VOC_CLASSES
        self.config = YoloV1Config(dataset_classes_num=len(VOC_CLASSES))

    def decoder(self, pred, obj_thres=0.1):
        """
            :param pred: the output of the yolov1 model, should be tensor of [1, grid_num, grid_num, 30]
            :param obj_thres: the threshold of objectness
            :return: list of [c, [boxes, labels]], boxes is [:4], labels is [4]
            """
        pred = pred.cpu()
        assert pred.shape[0] == 1
        # i for W, j for H
        res = [[] for i in range(len(VOC_CLASSES))]

        for h in range(self.config.GRID_NUM):
            for w in range(self.config.GRID_NUM):
                better_box = pred[0, h, w, :5] if pred[0, h, w, 4] > pred[0, h, w, 9] else pred[0, h, w, 5:10]
                if better_box[4] < obj_thres:
                    continue
                better_box_xyxy = torch.FloatTensor(better_box.size())

                better_box_xyxy[:2] = better_box[:2] / float(self.config.GRID_NUM) - 0.5 * better_box[2:4]
                better_box_xyxy[2:4] = better_box[:2] / float(self.config.GRID_NUM) + 0.5 * better_box[2:4]
                better_box_xyxy[0:4:2] += (w / float(self.config.GRID_NUM))
                better_box_xyxy[1:4:2] += (h / float(self.config.GRID_NUM))
                better_box_xyxy = better_box_xyxy.clamp(max=1.0, min=0.0)
                score, cls = pred[0, h, w, 10:].max(dim=0)
                _logger.info("score:{}\tcls:{}\ttag:{}".format(score, cls, self.LABELS[cls]))

                better_box_xyxy[4] = score * better_box[4]
                res[cls].append(better_box_xyxy)

        for i in range(len(VOC_CLASSES)):
            if len(res[i]) > 0:
                # res[i] = [box.unsqueeze(0) for box in res[i]]
                res[i] = torch.stack(res[i], 0)
            else:
                res[i] = torch.tensor([])

        return res

    def _nms(self, boxes, scores, overlap=0.5, top_k=None):
        """ Apply non-maximum suppression at test time to avoid detecting too many
        overlapping bounding boxes for a given object.
        Args:
            boxes: (tensor) The location preds for the img, Shape: [num_priors,4].
            scores: (tensor) The class predscores for the img, Shape:[num_priors].
            overlap: (float) The overlap thresh for suppressing unnecessary boxes.
            top_k: (int) The Maximum number of box preds to consider.
        Return:
            The indices of the kept boxes with respect to num_priors.
        """
        # boxes = boxes.detach()
        # keep shape [num_prior] type: Long
        keep = scores.new(scores.size(0)).zero_().long()

        # tensor.numel()用于计算tensor里面包含元素的总数，i.e. shape[0]*shape[1]...
        if boxes.numel() == 0:
            return keep
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        _logger.info("x1: {}\ty1: {}\tx2: {}\ty2: {}".format(x1, y1, x2, y2))
        # area shape[prior_num], 代表每个prior框的面积
        area = torch.mul(x2 - x1, y2 - y1)
        v, idx = scores.sort(0)  # sort in ascending order

        # I = I[v >= 0.01]
        if top_k is not None:
            # indices of the top-k largest vals
            idx = idx[-top_k:]
        # keep = torch.Tensor()
        count = 0
        # Returns the total number of elements in the input tensor.
        while idx.numel() > 0:
            i = idx[-1]  # index of current largest val
            # keep.append(i)
            keep[count] = i
            count += 1
            if idx.size(0) == 1:
                break
            idx = idx[:-1]  # remove kept element from view
            # load bboxes of next highest vals
            # torch.index_select(input, dim, index, out=None)
            # 将input里面dim维度上序号为idx的元素放到out里面去
            # >>> x
            # tensor([[1, 2, 3],
            #         [3, 4, 5]])
            # >>> z=torch.index_select(x,0,torch.tensor([1,0]))
            # >>> z
            # tensor([[3, 4, 5],
            #         [1, 2, 3]])
            xx1 = x1[idx]
            # torch.index_select(x1, 0, idx, out=xx1)
            yy1 = y1[idx]
            # torch.index_select(y1, 0, idx, out=yy1)
            xx2 = x2[idx]
            # torch.index_select(x2, 0, idx, out=xx2)
            yy2 = y2[idx]
            # torch.index_select(y2, 0, idx, out=yy2)

            # store element-wise max with next highest score
            # 将除置信度最高的prior框外的所有框进行clip以计算inter大小
            _logger.info("xx1 shape: {}".format(xx1.shape))
            xx1 = torch.clamp(xx1, min=float(x1[i]))
            yy1 = torch.clamp(yy1, min=float(y1[i]))
            xx2 = torch.clamp(xx2, max=float(x2[i]))
            yy2 = torch.clamp(yy2, max=float(y2[i]))
            # w.resize_as_(xx2)
            # h.resize_as_(yy2)
            w = xx2 - xx1
            h = yy2 - yy1
            # check sizes of xx1 and xx2.. after each iteration
            w = torch.clamp(w, min=0.0)
            h = torch.clamp(h, min=0.0)
            inter = w * h
            # IoU = i / (area(a) + area(b) - i)
            rem_areas = torch.index_select(area, 0, idx)  # load remaining areas)
            union = (rem_areas - inter) + area[i]
            IoU = inter / union  # store result in iou
            # keep only elements with an IoU <= overlap
            # torch.le===>less and equal to
            idx = idx[IoU.le(overlap)]

        # keep 包含置信度从大到小的prior框的indices，count表示数量
        return keep, count

    def predict(self, image_tensor, model):
        """图像预测"""
        # 设置模型模式
        model.eval()
        img_tensor, model = (image_tensor.to(self.config.DEVICE),
                             model.to(self.config.DEVICE))
        with torch.no_grad():
            out = model(img_tensor)
            # out:list[tensor[, 5]]
            out = self.decoder(out, obj_thres=0.3)
            boxes, tags, scores = [], [], []
            for cls, pred_target in enumerate(out):
                if pred_target.shape[0] > 0:
                    b = pred_target[:, :4]
                    p = pred_target[:, 4]

                    keep_idx, count = self._nms(b, p, overlap=0.5)
                    # keep:[, 5]
                    keep = pred_target[keep_idx]
                    for box in keep[..., :4]:
                        boxes.append(box)
                    for tag in range(count):
                        tags.append(torch.LongTensor([cls]))
                    for score in keep[..., 4]:
                        scores.append(score)
            _logger.info("boxes: {}\ttags:{}\tscores: {}".format(boxes, tags, scores))

            if len(boxes) > 0:
                boxes = torch.stack(boxes, 0).numpy()  # .squeeze(dim=0)
                tags = torch.stack(tags, 0).numpy()  # .squeeze(dim=0)
                scores = torch.stack(scores, 0).numpy()  # .squeeze(dim=0)
            else:
                boxes = torch.FloatTensor([])
                tags = torch.LongTensor([])  # .squeeze(dim=0)
                scores = torch.FloatTensor([])  # .squeeze(dim=0)

            # img, boxes, tags, scores = np.array(img), np.array(boxes), np.array(tags), np.array(scores)
            return boxes, tags, scores

    def resize_image(self, image, size=448):
        """ 图像等比例缩放，边界0补充
        :param image: 原始图像数据
        :param size:
        :return:
        """
        image_type = image.dtype
        h, w = image.shape[:2]
        image_max = max(w, h)

        scale = size / float(image_max)

        if scale is not 1.0:
            image = cv2.resize(image, dsize=(round(w * scale), round(h * scale)))

        h, w = image.shape[:2]
        top_pad = int((size - h) // 2)
        bottom_pad = int(size - h - top_pad)
        left_pad = int((size - w) // 2)
        right_pad = int(size - w - left_pad)

        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        image = np.pad(image, padding, mode='constant', constant_values=0)

        return image.astype(image_type)

    def image_to_tensor_batch(self, image_path):
        image = cv2.imread(image_path)
        image_resize = self.resize_image(image, size=self.size)
        normalize = CustomImageNormalize()
        image_normalize = normalize(image_resize)
        image_tensor = torch.from_numpy(image_normalize).permute(2, 0, 1)

        return image_tensor, image

    def predict_one_image(self, image_path, model):
        image_tensor, image = self.image_to_tensor_batch(image_path=image_path)
        boxes, tags, scores = self.predict(image_tensor=image_tensor, model=model)
        _logger.info("draw result box in origin image.")
        draw_box(img_np=image, boxes_np=boxes,
                 scores_np=scores, tags_np=tags, relative_coord=True)


if __name__ == "__main__":
    pass
