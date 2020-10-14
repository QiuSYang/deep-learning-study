# Only support version: python3.x
"""
# 公用函数
"""

import numpy as np
import tensorflow as tf
from tqdm import tqdm

BASE_VALUE = 128
SCALE = 255


def cal_pass_rate(dis_pos, dis_neg, mis_rate):
    dis_neg = sorted(dis_neg)
    dis_pos = sorted(dis_pos)
    print(dis_pos[:100])
    print(dis_neg[:100])

    mis_num = int(len(dis_neg) * mis_rate)
    if mis_num < 1:
        mis_num = 1

    threshold = dis_neg[mis_num - 1]

    pass_count = 0
    for dis in dis_pos:
        if dis < threshold:
            pass_count += 1

    return pass_count / len(dis_pos)


def get_pos_dis(x, y):
    # print(1 - np.multiply(x, y).sum(axis=1))
    return 1 - np.multiply(x, y).sum(axis=1)


def get_neg_dis(x, y):
    dis_matric = 1 - np.dot(y, x.T)
    # print(dis_matric)
    return np.amin(dis_matric, axis=1)


def norm(x):
    x = x.astype(np.float32)
    m = np.sqrt(np.multiply(x, x).sum(axis=1))
    m = np.reshape(m, (np.shape(m)[0], -1))
    x = x / m
    return x


def data_pre_process(x):
    return (x.astype(np.float32) + BASE_VALUE) / float(SCALE)


def data_post_process(x):
    return (x * SCALE - BASE_VALUE).astype(np.int8)


# def data_pre_process(x):
#     return (x.astype(np.float32) + 0) / 255
#
#
# def data_post_process(x):
#     return convert_to_int8(x)


def evaluate(x, y_pos, y_neg, mis_rate):
    x, y_pos, y_neg = norm(x), norm(y_pos), norm(y_neg)
    dis_pos = get_pos_dis(x, y_pos)
    dis_neg = get_neg_dis(x, y_neg)
    return cal_pass_rate(dis_pos, dis_neg, mis_rate)


def convert_to_int8(x):
    EPS = 1e-30
    MAX_INT8 = 127
    step = (np.amax(np.abs(x), axis=1) - EPS) / MAX_INT8
    step = np.reshape(step, (np.shape(step)[0], -1))
    x = x / step
    return x.astype(np.int8)


def get_predict_value(model, features):
    """获取预测特征向量"""
    predict_dataset = tf.data.Dataset.from_tensor_slices(features).batch(256)
    predict_result = None
    for batch_id, (transform_features) in enumerate(tqdm(predict_dataset, ncols=80)):
        batch_result = model.predict(transform_features)
        if batch_id == 0:
            predict_result = batch_result
        else:
            # numpy array 拼接
            predict_result = np.concatenate((predict_result, batch_result), axis=0)

    return predict_result


def model_predict(model, file_name):
    print(model_predict)
    feature = np.load(file_name)
    print("origin:", feature)
    feature = data_pre_process(feature)
    print("pre process:", feature)
    # result = model.predict(feature)
    result = get_predict_value(model, feature)
    print("predict", result)
    result = data_post_process(result)
    print("post process", result)
    return result


def model_evaluate(model):
    root = "./datasets/"
    file_id = root + "jr_1w_ID_1210.npy"
    file_snap = root + "jr_1w_snap_1210.npy"
    file_neg = root + "jr_1w_neg_1210.npy"
    feature_id, feature_snap, feature_neg = model_predict(model, file_id), model_predict(model, file_snap), model_predict(model, file_neg)
    pass_rate = evaluate(feature_id, feature_snap, feature_neg, 0.001)
    pass_rate2 = evaluate(feature_id, feature_snap, feature_neg, 0.0001)
    print("千一:", pass_rate, "万一:", pass_rate2)
