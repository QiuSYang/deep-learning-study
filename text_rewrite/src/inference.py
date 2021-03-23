"""
# inference codes
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import argparse
import logging
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from official.utils.flags import core as flags_core

from src.models import misc
from src.models import model_params
from src.models import transformer
from src.utils import tokenizer
from src.utils.dataset import load_vocab, convert_tokens_to_ids

from flask import Flask, json, jsonify, request  # server

logger = logging.getLogger(__name__)
work_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARAMS_MAP = {
    'tiny': model_params.TINY_PARAMS,
    'base': model_params.BASE_PARAMS,
    'big': model_params.BIG_PARAMS,
}

sever_app = Flask(__name__)  # flask server


def set_parameters():
    """设置参数"""
    parse = argparse.ArgumentParser(description="设置基本参数")
    parse.add_argument("--vocab_file", type=str, required=True,
                       help="tokenizer 词汇表位置.")
    parse.add_argument("--model_dir", type=str, required=True,
                       help="已训练模型保存路径.")
    parse.add_argument("--dtype", type=str, default="fp32",
                       help="运算过程中数据类型.")
    parse.add_argument("--param_set", type=str, default="tiny",
                       help="模型结构配置参数")
    args = parse.parse_args()

    params = PARAMS_MAP[args.param_set].copy()

    params["vocab_file"] = args.vocab_file
    params["model_dir"] = args.model_dir
    params["dtype"] = flags_core.get_tf_dtype(args)

    return params


def _trim_and_decode(ids, subtokenizer):
    """Trim EOS and PAD tokens from ids, and decode to return a string."""
    try:
        index = list(ids).index(tokenizer.EOS_ID)
        return subtokenizer.decode(ids[:index])
    except ValueError:  # No EOS found in sequence
        return subtokenizer.decode(ids)


def _init_text_encode(contexts, vocab_file, max_length_source=256):
    """输入text数据编码"""
    vocab = load_vocab(vocab_file)

    inputs = np.zeros(shape=(1, max_length_source), dtype='int32')
    segments = np.zeros(shape=(1, max_length_source), dtype='int32')
    masks = np.zeros(shape=(1, max_length_source), dtype='int32')

    CLS_ID = vocab["[CLS]"]
    SEP_ID = vocab["[SEP]"]
    EOS_ID = vocab["[EOS]"]

    query = list(contexts[-1].replace(' ', '').lower())  # 倒数第二句为要改写的句子
    content, content_num = [], len(contexts[:-1])
    for id, item in enumerate(contexts[:-1]):
        if id == content_num - 1:
            content += list(item.lower())
        else:
            content += list(item.lower()) + ["[SEP]"]

    # convert into ids
    query_ids = convert_tokens_to_ids(vocab, query)
    # query_ids_ = token_obj.encode(query)
    content_ids = convert_tokens_to_ids(vocab, content)

    # add `EOS_ID` for copy the last word
    inputs_ids = [CLS_ID] + [EOS_ID] + query_ids + [SEP_ID] + content_ids
    query_len = len(query_ids) + 3
    # segment
    # segment = [1] * query_len
    # for id in range(content_num):
    #     if id == content_num - 1:
    #         segment += [id + 2] * len(contexts[id])
    #     else:
    #         segment += [id + 2] * (len(contexts[id]) + 1)
    segment = [1] * query_len + [2] * len(content_ids)
    assert len(inputs_ids) == len(segment)
    inputs_ids = inputs_ids[:max_length_source]
    segment = segment[:max_length_source]
    mask = ([1] * query_len)[:max_length_source]  # only query visible

    inputs[0, :len(inputs_ids)] = inputs_ids
    segments[0, :len(segment)] = segment
    masks[0, :len(mask)] = mask

    dataset = tf.data.Dataset.from_tensor_slices(
        {
            "inputs": inputs,
            "segments": segments,
            "masks": masks,
        }
    )

    return dataset


def inference(model, subtokenizer, params, contexts: list):
    """文本重写"""
    def parse_example(serialized_example):
        """Return inputs and targets Tensors from a serialized tf.Example."""
        inputs = serialized_example["inputs"]
        segments = serialized_example["segments"]
        masks = serialized_example["masks"]

        return (inputs, segments, masks)

    def input_generator():
        """Yield encoded strings from sorted_inputs."""
        ds = _init_text_encode(contexts, params["vocab_file"], max_length_source=params["max_length_source"])
        ds = ds.map(parse_example, num_parallel_calls=None)
        ds = ds.batch(1)

        return ds

    results = []
    for id, inputs in enumerate(tqdm(input_generator())):
        inputs, segments, masks = inputs
        if params["is_beam_search"]:
            val_outputs, _ = model([inputs, segments, masks], training=False)
        else:
            val_outputs = model([inputs, segments, masks], training=False)
        length = len(val_outputs)
        for j in range(length):
            result = _trim_and_decode(val_outputs[j].numpy(), subtokenizer)
            results.append(result)

    return results


def load_model_tokenizer(params):
    logger.info("Restore Model")
    model = transformer.Transformer(params, name="transformer_v2")
    inputs, segments, masks = tf.zeros((1, 10), dtype=tf.int32), \
                              tf.zeros((1, 10), dtype=tf.int32), \
                              tf.zeros((1, 10), dtype=tf.int32)
    # Call the model once to create the weights.
    _ = model([inputs, segments, masks], training=False)
    model.summary()

    latest_checkpoint = tf.train.latest_checkpoint(params["model_dir"])
    if latest_checkpoint:
        model.load_weights(latest_checkpoint)
        logger.info("Loaded checkpoint {}".format(latest_checkpoint))
    else:
        raise ModuleNotFoundError("Load model failure.")

    subtokenizer = tokenizer.Subtokenizer(params["vocab_file"])

    return model, subtokenizer


def main(contexts: list):
    """主函数"""
    params = set_parameters()
    logger.info("Load model and tokenizer")
    model, tokenizer = load_model_tokenizer(params)

    logger.info("Inference start")
    results = inference(model, tokenizer, params, contexts=contexts)
    for result in results:
        logger.info("result: {}".format(result))


@sever_app.route("/rewrite", methods=["POST"])
def text_rewrite():
    if request.method == "POST":
        data = json.loads(request.data)
        results = inference(model, subtokenizer, params, contexts=data["contexts"])
        return jsonify(results=results)
    else:
        raise


if __name__ == '__main__':
    logging.basicConfig(format="[%(asctime)s %(filename)s: %(lineno)s] %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",
                        level=logging.INFO,
                        filename=None,
                        filemode="a")  # set logging
    is_server = True
    if not is_server:
        contexts = ["你知道板泉井水吗", "知道", "她是歌手"]
        main(contexts)
    else:
        params = set_parameters()
        model, subtokenizer = load_model_tokenizer(params)
        sever_app.run(host="0.0.0.0", port="8280")
