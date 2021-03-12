"""
# 自定义训练代码
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import tempfile

# Import libraries
from tqdm import tqdm
from absl import app
from absl import flags
from absl import logging
import tensorflow as tf
from official.common import distribute_utils
from official.modeling import performance
from official.utils.flags import core as flags_core
from official.utils.misc import keras_utils

from src.models import compute_bleu
from src.models import metrics
from src.models import misc
from src.models import optimizer
from src.models import transformer
from src.models import translate
from src.utils import tokenizer
from src.utils import dataset

INF = int(1e9)
BLEU_DIR = "bleu"
_SINGLE_SAMPLE = 1
work_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _get_sorted_inputs(filename):
  """Read and sort lines from the file sorted by decreasing length.

  Args:
    filename: String name of file to read inputs from.
  Returns:
    Sorted list of inputs, and dictionary mapping original index->sorted index
    of each element.
  """
  with tf.compat.v1.gfile.GFile(filename) as f:
    records = f.read().split("\n")
    inputs = [record.strip() for record in records]
    if not inputs[-1]:
      inputs.pop()

  sorted_keys = list(range(len(inputs)))

  return inputs, sorted_keys


def translate_file(model,
                   params,
                   subtokenizer,
                   input_file,
                   output_file=None,
                   print_all_translations=True,
                   vocab_file=None):
    """Translate lines in file, and save to output file if specified.

    Args:
    model: A Keras model, used to generate the translations.
    params: A dictionary, containing the translation related parameters.
    subtokenizer: A subtokenizer object, used for encoding and decoding source
      and translated lines.
    input_file: A file containing lines to translate.
    output_file: A file that stores the generated translations.
    print_all_translations: A bool. If true, all translations are printed to
      stdout.
    vocab_file: 词汇表.

    Raises:
    ValueError: if output file is invalid.
    """
    batch_size = params["decode_batch_size"]

    # Read and sort inputs by length. Keep dictionary (original index-->new index
    # in sorted list) to write translations in the original order.
    sorted_inputs, sorted_keys = _get_sorted_inputs(input_file)
    total_samples = len(sorted_inputs)

    def input_generator():
        """Yield encoded strings from sorted_inputs."""
        ds = dataset.init_dataset_from_text_file(input_file, vocab_file,
                                                 max_length_source=params["max_length_source"],
                                                 max_length_target=params["max_length_target"])
        ds = ds.map(dataset._parse_example, num_parallel_calls=None)
        ds = ds.batch(batch_size)

        return ds

    translations = []
    for id, inputs in enumerate(tqdm(input_generator())):
        inputs, segments, masks, targets = inputs[0]
        # results = model([inputs, segments, masks], training=False)
        # val_outputs = results["outputs"]
        val_outputs, _ = model([inputs, segments, masks], training=False)

        length = len(val_outputs)
        for j in range(length):
            if j + id * batch_size < total_samples:
                translation = _trim_and_decode(val_outputs[j].numpy(), subtokenizer)
                translations.append(translation)
            if print_all_translations:
                logging.info("Translating:\n\tInput: %s\n\tOutput: %s",
                           sorted_inputs[j + id * batch_size], translation)

    # Write translations in the order they appeared in the original file.
    if output_file is not None:
        if tf.io.gfile.isdir(output_file):
            raise ValueError("File output is a directory, will not save outputs to "
                           "file.")
        logging.info("Writing to file %s", output_file)
        with tf.io.gfile.GFile(output_file, "w") as f:
            for i in sorted_keys:
                f.write("%s\n" % translations[i])


def _trim_and_decode(ids, subtokenizer):
    """Trim EOS and PAD tokens from ids, and decode to return a string."""
    try:
        index = list(ids).index(tokenizer.EOS_ID)
        return subtokenizer.decode(ids[:index])
    except ValueError:  # No EOS found in sequence
        return subtokenizer.decode(ids)


def translate_and_compute_bleu(model,
                               params,
                               subtokenizer,
                               bleu_source,
                               bleu_ref):
    """Translate file and report the cased and uncased bleu scores.

    Args:
    model: A Keras model, used to generate the translations.
    params: A dictionary, containing the translation related parameters.
    subtokenizer: A subtokenizer object, used for encoding and decoding source
      and translated lines.
    bleu_source: A file containing source sentences for translation.
    bleu_ref: A file containing the reference for the translated sentences.

    Returns:
    uncased_score: A float, the case insensitive BLEU score.
    cased_score: A float, the case sensitive BLEU score.
    """
    # Create temporary file to store translation.
    tmp = tempfile.NamedTemporaryFile(delete=False)
    tmp_filename = tmp.name

    translate_file(
        model,
        params,
        subtokenizer,
        bleu_source,
        output_file=tmp_filename,
        vocab_file=params["vocab_file"],
        print_all_translations=True)

    # Compute uncased and cased bleu scores.
    uncased_score = compute_bleu.bleu_wrapper(bleu_ref, tmp_filename, False)
    cased_score = compute_bleu.bleu_wrapper(bleu_ref, tmp_filename, True)
    os.remove(tmp_filename)
    return uncased_score, cased_score


def evaluate_and_log_bleu(model,
                          params,
                          bleu_source,
                          bleu_ref,
                          vocab_file):
    """Calculate and record the BLEU score.

    Args:
    model: A Keras model, used to generate the translations.
    params: A dictionary, containing the translation related parameters.
    bleu_source: A file containing source sentences for translation.
    bleu_ref: A file containing the reference for the translated sentences.
    vocab_file: A file containing the vocabulary for translation.

    Returns:
    uncased_score: A float, the case insensitive BLEU score.
    cased_score: A float, the case sensitive BLEU score.
    """
    subtokenizer = tokenizer.Subtokenizer(vocab_file)

    uncased_score, cased_score = translate_and_compute_bleu(
            model, params, subtokenizer, bleu_source, bleu_ref)

    logging.info("Bleu score (uncased): %s", uncased_score)
    logging.info("Bleu score (cased): %s", cased_score)
    return uncased_score, cased_score


class TextRewriteTask(object):
    """Main entry of Transformer model."""

    def __init__(self, flags_obj):
        """Init function of TransformerMain.

        Args:
          flags_obj: Object containing parsed flag values, i.e., FLAGS.

        Raises:
          ValueError: if not using static batch for input data on TPU.
        """
        self.flags_obj = flags_obj
        self.predict_model = None

        # Add flag-defined parameters to params object
        num_gpus = flags_core.get_num_gpus(flags_obj)
        self.params = params = misc.get_model_params(flags_obj.param_set, num_gpus)

        params["vocab_file"] = flags_obj.vocab_file
        params["data_train"] = flags_obj.data_train
        params["data_dev"] = flags_obj.data_dev
        params["num_gpus"] = num_gpus
        params["use_ctl"] = flags_obj.use_ctl
        params["model_dir"] = flags_obj.model_dir
        params["static_batch"] = flags_obj.static_batch
        params["decode_batch_size"] = flags_obj.decode_batch_size
        params["max_io_parallelism"] = (
            flags_obj.num_parallel_calls or tf.data.experimental.AUTOTUNE)

        params["use_synthetic_data"] = flags_obj.use_synthetic_data
        params["batch_size"] = flags_obj.batch_size or params["default_batch_size"]
        params["dtype"] = flags_core.get_tf_dtype(flags_obj)
        params["enable_tensorboard"] = flags_obj.enable_tensorboard
        params["steps_between_evals"] = flags_obj.steps_between_evals
        params["enable_checkpointing"] = flags_obj.enable_checkpointing
        params["save_weights_only"] = flags_obj.save_weights_only
        params["bleu_source"] = flags_obj.bleu_source
        params["bleu_ref"] = flags_obj.bleu_ref

        # crate model and optimizer
        self.model = None  # transformer.Transformer(params, name="transformer_v2")
        self.opt = self._create_optimizer()

    def _create_model(self, params, is_train=False):
        """创建模型"""
        with tf.name_scope("model"):
            if is_train:
                inputs = tf.keras.layers.Input((None,), dtype="int32", name="inputs")
                segments = tf.keras.layers.Input((None,), dtype="int32", name="segments")
                masks = tf.keras.layers.Input((None,), dtype="int32", name="masks")
                targets = tf.keras.layers.Input((None,), dtype="int32", name="targets")

                internal_model = transformer.Transformer(params, name="transformer_v2")
                logits = internal_model([inputs, segments, masks, targets], training=is_train)

                logits = tf.keras.layers.Lambda(
                    lambda x: x, name="logits", dtype=tf.float32)(
                    logits)
                model = tf.keras.Model([inputs, segments, masks, targets], logits)

                return model

            else:
                inputs = tf.keras.layers.Input((None,), dtype="int32", name="inputs")
                segments = tf.keras.layers.Input((None,), dtype="int32", name="segments")
                masks = tf.keras.layers.Input((None,), dtype="int32", name="masks")

                internal_model = transformer.Transformer(params, name="transformer_v2")
                ret = internal_model([inputs, segments, masks], training=is_train)

                outputs, scores = ret["outputs"], ret["scores"]
                model = tf.keras.Model([inputs, segments, masks], [outputs, scores])

                return model

    def _create_optimizer(self):
        """Creates optimizer."""
        params = self.params
        lr_schedule = optimizer.LearningRateSchedule(
            params["learning_rate"], params["hidden_size"],
            params["learning_rate_warmup_steps"])
        opt = tf.keras.optimizers.Adam(
            lr_schedule,
            params["optimizer_adam_beta1"],
            params["optimizer_adam_beta2"],
            epsilon=params["optimizer_adam_epsilon"])

        return opt

    def train(self):
        """训练模型"""
        params = self.params
        # 创建模型
        self.model = self._create_model(params, is_train=True)
        self.model.summary()

        # 模型恢复训练
        # current_step = 0
        # checkpoint = tf.train.Checkpoint(model=self.model, optimizer=self.opt)
        latest_checkpoint = tf.train.latest_checkpoint(params["model_dir"])
        if latest_checkpoint:
            # checkpoint.restore(latest_checkpoint)
            # logging.info("Loaded checkpoint {}".format(latest_checkpoint))
            # current_step = self.opt.iterations.numpy()
            self.model.load_weights(latest_checkpoint)

        train_ds = dataset.train_input_fn(params)
        # logging.info("data size: {}".format(len(train_ds)))

        train_loss_metric = tf.keras.metrics.Mean("training_loss", dtype=tf.float32)
        if params["enable_tensorboard"]:
            summary_writer = tf.summary.create_file_writer(os.path.join(params["model_dir"], "summary"))

        with summary_writer.as_default():
            steps = 0
            for epoch in tf.range(1, 11):
                logging.info("Train epoch: {}".format(epoch))
                for batch_id, inputs in enumerate(tqdm(train_ds)):
                    train_loss_metric.reset_states()
                    loss = self.train_step(inputs, params)
                    # logging.info("loss: {}".format(loss))

                    train_loss_metric.update_state(loss)

                    if params["enable_tensorboard"]:
                        tf.summary.scalar(train_loss_metric.name, train_loss_metric.result(),
                                          steps)
                        summary_writer.flush()
                    steps += 1
                logging.info("the {} epoch loss: {}".format(epoch, train_loss_metric.result()))
                # save model
                # checkpoint.save(file_prefix=os.path.join(params["model_dir"], "{}.ckpt".format(epoch)))
                self.model.save_weights(filepath=os.path.join(params["model_dir"], "{}.ckpt".format(epoch)))

        summary_writer.close()

    def train_step(self, inputs, params):
        inputs, segments, masks, targets = inputs[0]
        with tf.GradientTape() as tape:
            logits = self.model([inputs, segments, masks, targets], training=True)
            # loss = metrics.transformer_loss(logits, targets,
            #                                 params["label_smoothing"], params["vocab_size"])
            xentropy, weights = metrics.custom_padded_cross_entropy_loss(
                logits, targets, params["vocab_size"])
            loss = tf.reduce_sum(xentropy) / tf.reduce_sum(weights)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))

        return loss

    def evaluate(self, params, is_restore=False):
        """模型评估"""
        if is_restore:
            # 模型恢复
            # 创建模型
            self.model = self._create_model(params, is_train=False)
            self.model.summary()

            # inputs, segments, masks = tf.zeros((1, 10), dtype=tf.int32), \
            #                           tf.zeros((1, 10), dtype=tf.int32), \
            #                           tf.zeros((1, 10), dtype=tf.int32)
            # _ = self.model([inputs, segments, masks], training=False)

            # checkpoint = tf.train.Checkpoint(model=self.model)
            latest_checkpoint = tf.train.latest_checkpoint(params["model_dir"])
            if latest_checkpoint:
                # checkpoint.restore(latest_checkpoint)
                self.model.load_weights(latest_checkpoint)
                logging.info("Loaded checkpoint {}".format(latest_checkpoint))

        uncased_score, cased_score = evaluate_and_log_bleu(self.model,
                                                           params,
                                                           bleu_source=params["bleu_source"],
                                                           bleu_ref=params["bleu_ref"],
                                                           vocab_file=params["vocab_file"])

        return uncased_score, cased_score


def main(_):
    """主函数"""
    flags_obj = flags.FLAGS
    task = TextRewriteTask(flags_obj)

    if flags_obj.mode == "train":
        task.train()
    elif flags_obj.mode == "eval":
        params = task.params
        task.evaluate(params, is_restore=True)
    else:
        pass


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    misc.define_transformer_flags()
    app.run(main)
