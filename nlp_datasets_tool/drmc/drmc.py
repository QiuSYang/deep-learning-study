"""
# 台湾DRCM数据 + 大陆CMRC数据集结构化处理
"""
import os
import json
import logging
import datasets

# 确保所有模块被copy
from .langconv import *
from .zh_wiki import zh2Hant, zh2Hans
from utils import *

logger = logging.getLogger(__name__)
obj = Converter('zh-hans')

_CITATION = """
@article{
        author = {YangQS}, 
        title = "{drcm dataset + cmrc dataset of the chinese}", 
        year = 2020, 
}"""

_DESCRIPTION = """
DRCM dataset and CMRC dataset.
"""


class DRMCConfig(datasets.BuilderConfig):
    """BuilderConfig for DRMC."""
    def __init__(self, **kwargs):
        """
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(DRMCConfig, self).__init__(**kwargs)


class DRMC(datasets.GeneratorBasedBuilder):
    """drcm + cmrc dataset"""
    BUILDER_CONFIGS = [
        DRMCConfig(
            name='drcm+cmrc',
            version=datasets.Version("1.0.0", "New split API"),
            description="Plain text",
        ),
    ]  # 用于自定义配置

    CMRC_DATA_TYPE = "cmrc"
    DRCD_DATA_TYPE = "DRCD"

    MAX_CONTEXT_LENGTH = 508

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "context": datasets.Features(
                        {
                            "sentences": datasets.Sequence(datasets.Value("string")),  # 将段落拆为多句保存
                            "highlight_idx": datasets.Value("int32"),  # 记录答案位于哪句话
                        }
                    ),
                    # "context": datasets.Sequence(datasets.Value("string")),
                    # "highlight_idx": datasets.Value("int32"),  # 记录答案位于哪句话
                    "question": datasets.Sequence(datasets.Value("string")),  # datasets.Value("string"),  # 段落问题文本
                    "answer": datasets.Sequence(datasets.Value("string")),  # datasets.Value("string"),  # 段落答案文本
                    "data_name": datasets.Value("string"),  # 数据集名称
                }
            ),
            # No default supervised_keys (as we have to pass both question
            # and context as input).
            supervised_keys=None,
            homepage=None,  # Local dataset
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """We handle string, list and dicts in datafiles"""
        if not self.config.data_files:
            raise ValueError(f"At least one data file must be specified, but got data_files={self.config.data_files}")
        data_files = dl_manager.download_and_extract(self.config.data_files)
        if isinstance(data_files, (str, list, tuple)):
            files = data_files
            if isinstance(files, str):
                files = [files]
            return [datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"files": files})]
        splits = []
        for split_name, files in data_files.items():
            if isinstance(files, str):
                files = [files]
            splits.append(datasets.SplitGenerator(name=split_name, gen_kwargs={"files": files}))
        return splits

    def _generate_examples(self, files):
        count = 0
        for file_path in files:
            logging.info("generating examples from = {}.".format(file_path))
            is_drcd_data = self.DRCD_DATA_TYPE in file_path.replace(os.sep, "/").split('/')[-1]
            with open(file_path, mode='r', encoding='utf-8') as f:
                datas = json.load(f).get('data')
                for data in datas:
                    for paragraph in data.get("paragraphs"):
                        context = paragraph.get('context')
                        context_sentences = self._split_sentence(context)  # 句子拆分

                        # 先做句子拆分, 再进行繁体->简体转换, 不会产生索引问题
                        answer_extraction_samples = self._get_highlight_sentence_data_sample(
                            context_sentences, qas=paragraph.get('qas'), is_drcd_data=is_drcd_data)
                        for sample in answer_extraction_samples:
                            yield count, sample
                            count += 1

    def _get_highlight_sentence_data_sample(self, context_sentences, qas, is_drcd_data=False):
        """将对应答案句子添加高亮存入数据集"""
        samples = []
        sentence_left_id, sentence_right_id = 0, 0
        for idx, sentence in enumerate(context_sentences):
            sentence_left_id = sentence_right_id
            sentence_right_id = sentence_left_id + len(sentence)
            answers, questions = [], []
            highlight_idx = None
            for qa_idx, qa in enumerate(qas):
                answer_text = qa.get('answers')[0].get('text')
                question_text = qa.get('question')

                answer_start_id = qa.get('answers')[0].get('answer_start')
                answer_end_id = answer_start_id + len(answer_text)
                if (answer_text in sentence
                        and answer_start_id >= sentence_left_id
                        and answer_end_id <= sentence_right_id):
                    # 文本在句子之中, 并且位置对应
                    highlight_idx = idx
                    answers.append(answer_text)
                    questions.append(question_text)

            if answers and highlight_idx is not None:
                valid_sentences, new_highlight_idx = self._get_highlight_context_sentences(
                    context_sentences, highlight_idx, max_len=self.MAX_CONTEXT_LENGTH)
                # 存储数据
                if is_drcd_data:
                    # 繁体转简体
                    for i in range(len(valid_sentences)):
                        valid_sentences[i] = obj.convert(valid_sentences[i])
                    for i in range(len(answers)):
                        answers[i] = obj.convert(answers[i])
                    for i in range(len(questions)):
                        questions[i] = obj.convert(questions[i])
                    data_name = self.DRCD_DATA_TYPE
                    samples.append({
                        "context": {
                            "sentences": valid_sentences,  # 将段落拆为多句保存
                            "highlight_idx": new_highlight_idx,  # 记录答案位于哪句话
                        },
                        # "context": valid_sentences,  # 将段落拆为多句保存
                        # "highlight_idx": new_highlight_idx,  # 记录答案位于哪句话
                        "question": questions,  # 段落问题文本
                        "answer": answers,  # 段落答案文本
                        "data_name": data_name  # 数据集名称
                    })
                else:
                    data_name = self.CMRC_DATA_TYPE
                    samples.append({
                        "context": {
                            "sentences": valid_sentences,  # 将段落拆为多句保存
                            "highlight_idx": new_highlight_idx,  # 记录答案位于哪句话
                        },
                        # "context": valid_sentences,  # 将段落拆为多句保存
                        # "highlight_idx": new_highlight_idx,  # 记录答案位于哪句话
                        "question": questions,  # 段落问题文本
                        "answer": answers,  # 段落答案文本
                        "data_name": data_name  # 数据集名称
                    })

        return samples

    def _split_sentence(self, paragraph: str, flag: str = "all", limit: int = 510):
        """
        Args:
            paragraph:
            flag: Type:str, "all" 中英文标点分句，"zh" 中文标点分句，"en" 英文标点分句
            limit: 默认单句最大长度为510个字符
        Returns: Type:list
        """
        sent_list = []
        try:
            if flag == "zh":
                paragraph = re.sub('(?P<quotation_mark>([。？！…](?![”’"\'])))', r'\g<quotation_mark>\n',
                                  paragraph)  # 单字符断句符
                paragraph = re.sub('(?P<quotation_mark>([。？！]|…{1,2})[”’"\'])', r'\g<quotation_mark>\n',
                                  paragraph)  # 特殊引号
            elif flag == "en":
                paragraph = re.sub('(?P<quotation_mark>([.?!](?![”’"\'])))', r'\g<quotation_mark>\n',
                                  paragraph)  # 英文单字符断句符
                paragraph = re.sub('(?P<quotation_mark>([?!.]["\']))', r'\g<quotation_mark>\n', paragraph)  # 特殊引号
            else:
                paragraph = re.sub('(?P<quotation_mark>([。？！….?!](?![”’"\'])))', r'\g<quotation_mark>\n',
                                  paragraph)  # 单字符断句符
                paragraph = re.sub('(?P<quotation_mark>(([。？！.!?]|…{1,2})[”’"\']))', r'\g<quotation_mark>\n',
                                  paragraph)  # 特殊引号

            sent_list_ori = paragraph.splitlines()
            for sent in sent_list_ori:
                # sent = sent.strip()
                if not sent:
                    continue
                else:
                    while len(sent) > limit:
                        temp = sent[0:limit]
                        sent_list.append(temp)
                        sent = sent[limit:]
                    sent_list.append(sent)
        except:
            sent_list.clear()
            sent_list.append(paragraph)

        return sent_list

    def _get_highlight_context_sentences(self, sentences: list, highlight_index: int, max_len: int = 508):
        """滑动获取训练数据- 保持最大长度左右"""
        left_pos, right_pos = highlight_index - 1, highlight_index + 1
        left_len, right_len, sum_len = 0, 0, len(sentences[highlight_index])
        while True:
            flag = False
            if left_pos >= 0:
                if left_len <= right_len or right_pos >= len(sentences):
                    # 左边长度小于右边 or 右边句子已经被遍历完
                    tmp_len = len(sentences[left_pos])
                    if sum_len + tmp_len < max_len:
                        left_pos -= 1
                        left_len += tmp_len
                        sum_len += tmp_len
                        flag = True
            if right_pos < len(sentences):
                if right_len <= left_len or left_pos < 0:
                    # 右边长度小于左边 or 左边句子已经被遍历完
                    tmp_len = len(sentences[right_pos])
                    if sum_len + tmp_len < max_len:
                        right_pos += 1
                        right_len += tmp_len
                        sum_len += tmp_len
                        flag = True
            if not flag:
                break

        new_sentences = sentences[left_pos+1: right_pos]
        new_highlight_index = highlight_index - left_pos - 1
        assert new_sentences[new_highlight_index] == sentences[highlight_index]

        return new_sentences, new_highlight_index
