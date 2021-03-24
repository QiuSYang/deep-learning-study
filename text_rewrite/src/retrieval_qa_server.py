"""
# 基于text rewrite model + retrieval 实现QA问答
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import argparse
import logging
from official.utils.flags import core as flags_core

from flask import Flask, json, jsonify, request  # server

from src.models import model_params
from src.inference import load_model_tokenizer, inference
from src.retrieval_module.inference import load_indexer_w2v_qas, get_retrieval_results


logger = logging.getLogger(__name__)
work_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARAMS_MAP = {
    'tiny': model_params.TINY_PARAMS,
    'base': model_params.BASE_PARAMS,
    'big': model_params.BIG_PARAMS,
}

sever_app = Flask(__name__)  # flask server


def set_parameters():
    """设置参数"""
    parse = argparse.ArgumentParser(description="设置基本参数")
    parse.add_argument("--vocab_file", type=str,
                       default=os.path.join(work_root, "resource/vocab.txt"),
                       help="tokenizer 词汇表位置.")
    parse.add_argument("--model_dir", type=str,
                       default=os.path.join(work_root, "models/tiny_x"),
                       help="已训练模型保存路径.")
    parse.add_argument("--dtype", type=str, default="fp32",
                       help="运算过程中数据类型.")
    parse.add_argument("--param_set", type=str, default="tiny",
                       help="模型结构配置参数.")
    parse.add_argument("--words_file", type=str,
                       default=os.path.join(work_root, "models/tencent_words/1000000-small.words"),
                       help="词汇表.")
    parse.add_argument("--features_file", type=str,
                       default=os.path.join(work_root, "models/tencent_words/1000000-small.npy"),
                       help="词汇表对应每个词的向量表征.")
    parse.add_argument("--docs_file", type=str,
                       default=os.path.join(work_root, "data/tianchi_chinese_medical_qas.json"),
                       help="文档库文件.")
    parse.add_argument("--indexes_file", type=str,
                       default=os.path.join(work_root, "data/tianchi_chinese_medical_qas"),
                       help="文档库对应的索引库.")
    args = parse.parse_args()

    params = PARAMS_MAP[args.param_set].copy()

    params["vocab_file"] = args.vocab_file
    params["model_dir"] = args.model_dir
    params["dtype"] = flags_core.get_tf_dtype(args)
    params["words_file"] = args.words_file
    params["features_file"] = args.features_file
    params["docs_file"] = args.docs_file
    params["indexes_file"] = args.indexes_file

    return params


@sever_app.route("/qa", methods=["POST"])
def retrieval_qa():
    """ 通过检索技术实现QA问答
        1. 借助文本重写生成技术---根据对话进行上文生成更加精确query, 避免指代(通过第三人称代词替代上文关键词)，信息缺失的问题,
        这个模型的关键就是解决上面两个问题, 对query进行补全;
        2. 使用检索技术查询与query相似query, query对应的answer就是结果"""
    if request.method == "POST":
        data = json.loads(request.data)
        results = inference(model, subtokenizer, params, contexts=data["contexts"])

        cond_docs = []
        for query in results:
            cond_doc = get_retrieval_results(query, indexer, w2v_model, qas_document, top_k=data["top_k"])
            cond_docs.append(cond_doc)
        return jsonify(results=results,
                       cond_docs=cond_docs)
    else:
        raise


if __name__ == '__main__':
    logging.basicConfig(format="[%(asctime)s %(filename)s: %(lineno)s] %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",
                        level=logging.INFO,
                        filename=None,  # os.path.join(work_root, "data/logs.txt"),  # 日志文件
                        filemode="a")  # set logging
    logger.info("Load model")
    params = set_parameters()
    model, subtokenizer = load_model_tokenizer(params)
    indexer, w2v_model, qas_document = load_indexer_w2v_qas(words_file=params["words_file"],
                                                            features_file=params["features_file"],
                                                            qas_file=params["docs_file"],
                                                            indexes_file=params["indexes_file"],
                                                            indexer_type="flat")
    sever_app.run(host="0.0.0.0", port="8280")
