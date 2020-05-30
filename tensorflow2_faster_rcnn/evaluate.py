"""
# 模型评估
"""

import os
import json
import tensorflow as tf
import numpy as np
from pycocotools.cocoeval import COCOeval

from detectron.datasets import coco, data_generator
from detectron.models.detectors import faster_rcnn


def get_detection_result_to_json(model, val_dataset,
                                 save_result_path="coco_val2017_detection_result.json"):
    dataset_results = []
    imgIds = []
    for idx in range(len(val_dataset)):
        if idx % 10 == 0:
            print(idx)

        img, img_meta, _, _ = val_dataset[idx]

        proposals = model.simple_test_rpn(img, img_meta)
        res = model.simple_test_bboxes(img, img_meta, proposals)

        image_id = val_dataset.img_ids[idx]
        imgIds.append(image_id)

        # 将检测结果解析为coco API可以评测的格式
        for pos in range(res['class_ids'].shape[0]):
            results = dict()
            results['score'] = float(res['scores'][pos])
            results['category_id'] = val_dataset.label2cat[int(res['class_ids'][pos])]
            y1, x1, y2, x2 = [float(num) for num in list(res['rois'][pos])]
            results['bbox'] = [x1, y1, x2 - x1 + 1, y2 - y1 + 1]
            results['image_id'] = image_id
            dataset_results.append(results)

    with open(save_result_path, mode='w') as f:
        f.write(json.dumps(dataset_results))

    return imgIds


def detection_result_evaluate(val_dataset, imgIds, save_result_path="coco_val2017_detection_result.json"):
    # 将检测结果加载进入COCO
    coco_dt = val_dataset.coco.loadRes(resFile=save_result_path)

    cocoEval = COCOeval(val_dataset.coco, coco_dt, 'bbox')
    cocoEval.params.imgIds = imgIds

    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()


if __name__ == "__main__":
    img_mean = (123.675, 116.28, 103.53)
    # img_std = (58.395, 57.12, 57.375)
    img_std = (1., 1., 1.)

    val_dataset = coco.CocoDataSet('./COCO2017/', 'val',
                                   flip_ratio=0,
                                   pad_mode='fixed',
                                   mean=img_mean,
                                   std=img_std,
                                   scale=(800, 1344))
    print(len(val_dataset))

    model = faster_rcnn.FasterRCNN(
        num_classes=len(val_dataset.get_categories()))

    img, img_meta, bboxes, labels = val_dataset[0]
    batch_imgs = tf.Variable(np.expand_dims(img, 0))
    batch_metas = tf.Variable(np.expand_dims(img_meta, 0))

    _ = model((batch_imgs, batch_metas), training=False)

    model.load_weights('weights/faster_rcnn.h5', by_name=True)

    save_result_path = "./datasets/coco_val2017_detection_result.json"
    # 加载模型检测数据
    imgIds = get_detection_result_to_json(model, val_dataset, save_result_path)
    # 加载检测的结果，使用COCO API进行评测
    detection_result_evaluate(val_dataset, imgIds, save_result_path)
