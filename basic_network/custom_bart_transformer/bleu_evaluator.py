"""
# 使用nltk库实现BLEU评估算法
"""
import os
import logging
import json
import argparse
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
_logger = logging.getLogger(__name__)


class BLEUEvaluator(object):
    def __init__(self,
                 predict_json_path='../offline_dev_data/dev_answers_predict.json',
                 target_json_path='../offline_dev_data/dev_answers_target.json',
                 question_json_path=None,
                 good_qa_threshold=None):
        with open(predict_json_path, mode='r', encoding='utf-8') as file_predict:
            self.predict_content = json.load(file_predict)
        with open(target_json_path, mode='r', encoding='utf-8') as file_target:
            self.target_content = json.load(file_target)

        self.good_qa_threshold = None
        if question_json_path and good_qa_threshold:
            # 二者条件都满足才进行此项操作
            self.good_qa_threshold = good_qa_threshold
            with open(question_json_path, mode='r', encoding='utf-8') as file_question:
                # 获取问题列表
                self.questions_dict = self.get_question_dict(json.load(file_question))

        # 记录bleu分比较高的回答, 追溯其问题以及上下文
        self.good_test_questions = []
        self.bad_test_questions = []

    def eval(self):
        """评估函数"""
        # predict_list, target_list = self.get_result_sentence_list()
        #
        # return self.compute_bleu(predict_list, target_list)

        predict_answers_dict = self.get_answer_dict(self.predict_content)
        target_answers_dict = self.get_answer_dict(self.target_content)

        return self.compute_bleu(predict_answers_dict, target_answers_dict)

    def compute_bleu(self, predict_dict, target_dict):
        """
        Args:
            predict_dict: 预测字典列表
            target_dict:

        Returns:

        """
        n_sum = 0
        smooth = SmoothingFunction()

        predict_data_length = len(predict_dict.keys())
        _logger.info("all predict data size: {}.".format(predict_data_length))
        for single_key in predict_dict.keys():
            if not target_dict.get(single_key):
                # 跳过查不大不到目标的数据
                predict_data_length -= 1
                continue

            target_list_three = target_dict.get(single_key).split("<sep>")
            predict_sentence = predict_dict.get(single_key)
            n_eval_result = sentence_bleu(target_list_three, predict_sentence, smoothing_function=smooth.method1)

            if self.good_qa_threshold:
                if n_eval_result > self.good_qa_threshold:
                    res = self.single_qa_track(single_key, predict_dict.get(single_key), n_eval_result)
                    self.good_test_questions.append(res)
                else:
                    res = self.single_qa_track(single_key, predict_dict.get(single_key), n_eval_result)
                    self.bad_test_questions.append(res)

            n_sum += n_eval_result

        _logger.info("resize predict data size: {}.".format(predict_data_length))

        return float(n_sum) / predict_data_length

    def single_qa_track(self, predict_single_key, predict_single_value, bleu_score):
        # 1. 对应ID的question context
        single_question = self.questions_dict.get(predict_single_key)
        # 2. 添加预测结果
        single_question['PredictAnswer'] = predict_single_value
        single_question['BLEU'] = bleu_score

        return single_question

    def get_answer_dict(self, answers_dict_list):
        """将dict list 转为dict, Id为key, answer为value"""
        answers_dict = {}
        for single_dict in answers_dict_list:
            answers_dict[single_dict.get('Id')] = single_dict.get('Answer')

        return answers_dict

    def get_question_dict(self, questions_dict_list):
        """将dict list 转为dict, Id为key, 整体dict为value"""
        questions_dict = {}
        for single_dict in questions_dict_list:
            questions_dict[single_dict.get('Id')] = single_dict

        return questions_dict

    def save_bleu(self, contents, file_path):
        """保存qa的question"""
        with open(file_path, mode='w', encoding='utf-8') as fw:
            json.dump(contents,  fw, ensure_ascii=False, indent=4)

    # def compute_bleu(self, predict_list, target_list):
    #     """接受一一对应的list列表"""
    #     n_sum = 0
    #
    #     smooth = SmoothingFunction()
    #
    #     for i in range(len(predict_list)):
    #         target_list_three = target_list[i].split("<sep>")
    #
    #         n_eval_result = sentence_bleu(target_list_three, predict_list[i],
    #                                       smoothing_function=smooth.method1)
    #
    #         n_sum += n_eval_result
    #
    #     return float(n_sum) / len(predict_list)

    # def get_result_sentence_list(self):
    #     predict_list = []
    #     target_list = []
    #     assert len(self.predict_content) == len(self.target_content), '数据大小不一致.'
    #     for single_predict in self.predict_content:
    #         for single_target in self.target_content:
    #             if single_predict.get('Id') == single_target.get('Id'):
    #                 predict_list.append(single_predict.get('Answer'))
    #                 target_list.append(single_target.get('Answer'))
    #                 # 已经找到与之对应的结果，跳出内存循环
    #                 break
    #
    #     return predict_list, target_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation of the sentence generation effect.")

    parser.add_argument('-p', '--predict_json_path',
                        default='./online_test_data/test_answers_dev.json',
                        type=str,
                        help='The json file for the predict results.')
    parser.add_argument('-t', '--target_json_path',
                        default='./online_test_data/test_answers_target.json',
                        type=str,
                        help='The json file for the target results.')
    parser.add_argument('-q', '--question_json_path',
                        default='./online_test_data/test_questions_dev.json',
                        type=str,
                        help='The json file for the question contents.')
    parser.add_argument('-gq', '--good_question_json_path',
                        default='./online_test_data/good_sample.json',
                        type=str,
                        help='The json file for the save good ga track question contents.')
    parser.add_argument('-bq', '--bad_question_json_path',
                        default='./online_test_data/bad_sample.json',
                        type=str,
                        help='The json file for the save bad ba track question contents.')
    parser.add_argument('-trd', '--good_qa_threshold',
                        default=None,
                        type=float,
                        help='the judge threshold for the predict and question is good qa track.')

    args = parser.parse_args()

    evaluator = BLEUEvaluator(predict_json_path=args.predict_json_path,
                              target_json_path=args.target_json_path,
                              question_json_path=args.question_json_path,
                              good_qa_threshold=args.good_qa_threshold)

    eval_result = evaluator.eval()

    if args.good_qa_threshold and args.question_json_path:
        _logger.info("save good qa track question contents.")
        evaluator.save_bleu(evaluator.good_test_questions, args.good_question_json_path)
        _logger.info("save bad qa track question contents.")
        evaluator.save_bleu(evaluator.bad_test_questions, args.bad_question_json_path)

    _logger.info(f'Select threshold: {args.good_qa_threshold}')
    _logger.info(f'Good: {len(evaluator.good_test_questions)}, Bad: {len(evaluator.bad_test_questions)}')
    _logger.info(f'Total samples: {len(evaluator.good_test_questions) + len(evaluator.bad_test_questions)}, Total average: {eval_result}')

    # result = evaluator.compute_bleu(["好的"],
    #                                 ["非常抱歉给您带来不便"])
    # print(result)
