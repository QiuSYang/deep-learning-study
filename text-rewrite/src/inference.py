"""
# inference codes
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import numpy as np
import tensorflow as tf
from absl import (
    logging,
    app,
    flags
)
from tqdm import tqdm
from official.utils.flags import core as flags_core

from src.models import misc
from src.models import transformer
from src.utils import tokenizer
from src.utils.dataset import load_vocab, convert_tokens_to_ids

work_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def set_parameters(flags_obj):
    """设置参数"""
    num_gpus = flags_core.get_num_gpus(flags_obj)
    params = misc.get_model_params(flags_obj.param_set, num_gpus)

    params["vocab_file"] = flags_obj.vocab_file
    params["num_gpus"] = num_gpus
    params["model_dir"] = flags_obj.model_dir
    params["dtype"] = flags_core.get_tf_dtype(flags_obj)

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
    #         segment += [id + 1] * len(contexts[id])
    #     else:
    #         segment += [id + 1] * (len(contexts[id]) + 1)
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


def inference(params, contexts: list):
    """文本重写"""
    logging.info("Restore Model")
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
        logging.info("Loaded checkpoint {}".format(latest_checkpoint))
    else:
        raise ModuleNotFoundError("Load model failure.")

    subtokenizer = tokenizer.Subtokenizer(params["vocab_file"])

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


def main(_):
    """主函数"""
    flags_obj = flags.FLAGS
    params = set_parameters(flags_obj)

    logging.info("Inference start")
    contexts = ["你知道板泉井水吗", "知道", "她是歌手"]
    results = inference(params, contexts=contexts)
    for result in results:
        logging.info("result: {}".format(result))


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    misc.define_transformer_flags()
    app.run(main)
