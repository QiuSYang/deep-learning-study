# Only support version: python3.x
"""
# 数据集下载
"""
# !/usr/bin/env python
# # -*- coding: utf-8 -*-

import pymysql
import numpy as np
import logging

datefmt = '%Y-%m-%d %H:%M:%S'
fmt = '%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=fmt)

logger = logging.getLogger(__file__)

db_config = dict(
    host="10.128.61.6",
    port=13307,
    user="pc",
    passwd="pc123456",
    db="clustering_int8",
    charset="utf8",
    sql_mode="REAL_AS_FLOAT",
    init_command="SET max_join_size=DEFAULT",
    connect_timeout=360
)


def get_data_from_table(db, table_name, limit=None):
    logger.info(F"get data from table:{table_name}, limit:{limit}")
    batch_size = 1000
    query = F'SELECT `user_id`, `feature` FROM `{table_name}`'
    if limit:
        query = query + F' limit {limit}'
    logger.info(F"query:{query}")
    data = {}
    with db.cursor() as cursor:
        cursor.execute(query)
        while True:
            result = cursor.fetchmany(batch_size)
            if not result:
                break
            for record in result:
                data[record[0]] = record[1]
    return data


def bytes_to_feature(bytes_list, origin_dim, dim, dtype=np.int8, with_decode=False):
    if with_decode:
        import base64
        bytes_list = [base64.b64decode(b) for b in bytes_list]
    contents = b''.join(bytes_list)
    features = np.frombuffer(contents, dtype=dtype)
    features = np.reshape(features, (int(np.shape(features)[0] / origin_dim), origin_dim))[:, :dim]
    return features


def main():
    logger.info("create db connect")
    db = pymysql.connect(**db_config)

    table_1, table_2 = "faces_1210", "faces_0220_2000w"
    limit = 20000000
    data_1 = get_data_from_table(db, table_1, limit)
    data_2 = get_data_from_table(db, table_2, limit)

    features_1, features_2 = [], []
    for key, value in data_1.items():
        if key not in data_2:
            continue
        features_1.append(value)
        features_2.append(data_2[key])
    features_1 = bytes_to_feature(features_1, 384, 384)
    features_2 = bytes_to_feature(features_2, 448, 384)

    np.save("features_1.npy", features_1)
    np.save("features_2.npy", features_2)

    logger.info("finish")


def download_evaluate_data():
    logger.info("create db connect")
    import copy
    config = copy.deepcopy(db_config)
    config["db"] = "jrtest"
    db = pymysql.connect(**config)

    table = "jr_1w_pic_1210"
    limit = 20000
    data = get_data_from_table(db, table, limit)

    features_1, features_2 = [], []
    for key, value in data.items():
        if not key.endswith("_1"):
            continue
        key_2 = key[:-1] + "2"
        if key_2 not in data:
            continue
        features_1.append(value)
        features_2.append(data[key_2])
    features_1 = bytes_to_feature(features_1, 448, 384, with_decode=True)
    features_2 = bytes_to_feature(features_2, 448, 384, with_decode=True)

    np.save("jr_1w_ID_1210.npy", features_1)
    np.save("jr_1w_snap_1210.npy", features_2)

    logger.info("finish")


def test():
    root = "./"
    file_name = root + "features_1.npy"
    features = np.load(file_name)[-100000:]
    np.save("jr_1w_neg_1210.npy", features)


if __name__ == "__main__":
    # main()
    download_evaluate_data()
    test()
