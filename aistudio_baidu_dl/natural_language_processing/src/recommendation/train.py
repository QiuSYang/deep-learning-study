"""
# 电影推荐模型训练
"""
import os
import sys
import logging
import pickle
from tqdm import tqdm
import numpy as np
from math import sqrt
import paddle
import paddle.nn.functional as F

logger = logging.getLogger(__name__)
work_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(work_root)  # 添加工作路径

from src.recommendation.model import MovieRecommendModel
from src.recommendation.dataset import MovieDataset
from src.utils import init_logs, save_dict_obj


def create_model_opt(vocab_sizes={}):
    """创建模型以及优化器"""
    lr = 0.001
    fc_sizes = [128, 64, 32]
    use_poster, use_mov_title, use_mov_cat, use_age_job = False, True, True, True
    model = MovieRecommendModel(use_poster, use_mov_title, use_mov_cat, use_age_job,
                                vocab_sizes=vocab_sizes, fc_sizes=fc_sizes)
    # 使用adam优化器，学习率使用0.01
    opt = paddle.optimizer.Adam(learning_rate=lr, parameters=model.parameters())

    return model, opt


@paddle.no_grad()
def evaluate(model, data_loader):
    """模型评估"""
    model.eval()

    acc_set = []
    avg_loss_set = []
    squaredError = []
    for idx, data in enumerate(tqdm(data_loader())):
        usr, mov, score_label = data
        usr_v = [paddle.to_tensor(var) for var in usr]
        mov_v = [paddle.to_tensor(var) for var in mov]

        _, _, scores_predict = model(usr_v, mov_v)

        pred_scores = scores_predict.numpy()

        avg_loss_set.append(np.mean(np.abs(pred_scores - score_label)))
        squaredError.extend(np.abs(pred_scores - score_label) ** 2)

        diff = np.abs(pred_scores - score_label)
        diff[diff > 0.5] = 1
        acc = 1 - np.mean(diff)
        acc_set.append(acc)

    RMSE = sqrt(np.sum(squaredError) / len(squaredError))
    logger.info("RMSE = {}".format(sqrt(np.sum(squaredError) / len(squaredError))))  # 均方根误差RMSE

    return np.mean(acc_set), np.mean(avg_loss_set), RMSE


def train(model, opt, data_loader, eval_data_loader=None, epochs=3):
    """训练函数"""
    model.train()
    best_rmse = float("inf")
    for epoch in range(epochs):
        for idx, data in enumerate(tqdm(data_loader())):
            # 获得数据，并转为tensor格式
            usr, mov, score = data
            usr_v = [paddle.to_tensor(var) for var in usr]
            mov_v = [paddle.to_tensor(var) for var in mov]
            scores_label = paddle.to_tensor(score)
            # 前向计算结果
            _, _, scores_predict = model(usr_v, mov_v)
            # 计算损失
            loss = F.square_error_cost(scores_predict, scores_label)  # 均方误差
            avg_loss = paddle.mean(loss)

            if idx % 500 == 0:
                logger.info("epoch: {}, batch_id: {}, loss is: {}".format(epoch, idx, avg_loss.numpy()))

            avg_loss.backward()
            opt.step()
            opt.clear_grad()

        if eval_data_loader:
            _, _, rmse = evaluate(model, data_loader=eval_data_loader)
            if rmse < best_rmse:
                best_rmse = rmse
                # 保存最佳模型
                paddle.save(model.state_dict(),
                            os.path.join(work_root, "models/movie_recommend/best.pdparams"))

    # 保存最后模型
    paddle.save(model.state_dict(),
                os.path.join(work_root, "models/movie_recommend/{}.pdparams".format(epoch)))


@paddle.no_grad()
def get_usr_mov_features(model, dataset):
    """获取用户和电影特征"""
    model.eval()
    usr_pkl = {}
    mov_pkl = {}

    # 定义将list中每个元素转成tensor的函数
    def list2tensor(inputs, shape):
        inputs = np.reshape(np.array(inputs).astype(np.int64), shape)
        return paddle.to_tensor(inputs)

    for i in range(len(dataset)):
        # 获得用户数据，电影数据，评分数据
        # 本案例只转换所有在样本中出现过的user和movie，实际中可以使用业务系统中的全量数据
        usr_info, mov_info, score = dataset[i]['usr_info'], dataset[i]['mov_info'], dataset[i]['scores']
        usrid = str(usr_info['usr_id'])
        movid = str(mov_info['mov_id'])

        # 获得用户数据，计算得到用户特征，保存在usr_pkl字典中
        if usrid not in usr_pkl.keys():
            usr_id_v = list2tensor(usr_info['usr_id'], [1])
            usr_age_v = list2tensor(usr_info['age'], [1])
            usr_gender_v = list2tensor(usr_info['gender'], [1])
            usr_job_v = list2tensor(usr_info['job'], [1])

            usr_in = [usr_id_v, usr_gender_v, usr_age_v, usr_job_v]
            usr_feat = model.get_usr_feature(usr_in)

            usr_pkl[usrid] = usr_feat.numpy()

        # 获得电影数据，计算得到电影特征，保存在mov_pkl字典中
        if movid not in mov_pkl.keys():
            mov_id_v = list2tensor(mov_info['mov_id'], [1])
            mov_tit_v = list2tensor(mov_info['title'], [1, 1, 15])
            mov_cat_v = list2tensor(mov_info['category'], [1, 6])

            mov_in = [mov_id_v, mov_cat_v, mov_tit_v, None]
            mov_feat = model.get_movie_feature(mov_in)

            mov_pkl[movid] = mov_feat.numpy()

    logger.info(len(mov_pkl.keys()))
    # 保存特征到本地
    pickle.dump(usr_pkl, open(os.path.join(work_root, 'data/usr_feat.pkl'), 'wb'))
    pickle.dump(mov_pkl, open(os.path.join(work_root, 'data/mov_feat.pkl'), 'wb'))
    logger.info("usr / mov features saved!!!")


def main():
    """主函数"""
    paddle.set_device("gpu:1")
    logger.info("构建数据集")
    dataset = MovieDataset(use_poster=False)
    vocab_sizes = {
        "max_usr_id": int(dataset.max_usr_id),
        "max_usr_age": int(dataset.max_usr_age),
        "max_usr_job": int(dataset.max_usr_job),
        "max_mov_id": int(dataset.max_mov_id),
        "movie_category_size": len(dataset.movie_cat),
        "movie_title_size": len(dataset.movie_title)
    }
    save_dict_obj(vocab_sizes, os.path.join(work_root, "data/movie_vocab_size.json"))
    train_data_loader = dataset.load_data(dataset.train_dataset, mode="train", use_poster=False)
    eval_data_loader = dataset.load_data(dataset.valid_dataset, mode="eval", use_poster=False)

    logger.info("构建模型")
    model, opt = create_model_opt(vocab_sizes)

    logger.info("训练模型")
    EPOCHS = 10
    train(model, opt, data_loader=train_data_loader, eval_data_loader=eval_data_loader, epochs=EPOCHS)

    logger.info("构建特征信息")
    get_usr_mov_features(model, dataset.dataset)


if __name__ == '__main__':
    init_logs()
    main()
