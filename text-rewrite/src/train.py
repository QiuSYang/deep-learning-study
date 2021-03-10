"""
# 自定义训练代码
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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


def translate_and_compute_bleu(model,
                               params,
                               subtokenizer,
                               bleu_source,
                               bleu_ref,
                               distribution_strategy=None):
    """Translate file and report the cased and uncased bleu scores.

    Args:
    model: A Keras model, used to generate the translations.
    params: A dictionary, containing the translation related parameters.
    subtokenizer: A subtokenizer object, used for encoding and decoding source
      and translated lines.
    bleu_source: A file containing source sentences for translation.
    bleu_ref: A file containing the reference for the translated sentences.
    distribution_strategy: A platform distribution strategy, used for TPU based
      translation.

    Returns:
    uncased_score: A float, the case insensitive BLEU score.
    cased_score: A float, the case sensitive BLEU score.
    """
    # Create temporary file to store translation.
    tmp = tempfile.NamedTemporaryFile(delete=False)
    tmp_filename = tmp.name

    translate.translate_file(
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
                          vocab_file,
                          distribution_strategy=None):
    """Calculate and record the BLEU score.

    Args:
    model: A Keras model, used to generate the translations.
    params: A dictionary, containing the translation related parameters.
    bleu_source: A file containing source sentences for translation.
    bleu_ref: A file containing the reference for the translated sentences.
    vocab_file: A file containing the vocabulary for translation.
    distribution_strategy: A platform distribution strategy, used for TPU based
      translation.

    Returns:
    uncased_score: A float, the case insensitive BLEU score.
    cased_score: A float, the case sensitive BLEU score.
    """
    subtokenizer = tokenizer.Subtokenizer(vocab_file)

    uncased_score, cased_score = translate_and_compute_bleu(
      model, params, subtokenizer, bleu_source, bleu_ref, distribution_strategy)

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
        params["repeat_dataset"] = None
        params["dtype"] = flags_core.get_tf_dtype(flags_obj)
        params["enable_tensorboard"] = flags_obj.enable_tensorboard
        params["enable_metrics_in_training"] = flags_obj.enable_metrics_in_training
        params["steps_between_evals"] = flags_obj.steps_between_evals
        params["enable_checkpointing"] = flags_obj.enable_checkpointing
        params["save_weights_only"] = flags_obj.save_weights_only

        # crate model and optimizer
        self.model = transformer.Transformer(params, name="transformer_v2")
        self.opt = self._create_optimizer()

    def train(self):
        """训练模型"""
        params = self.params

        # 模型恢复训练
        current_step = 0
        checkpoint = tf.train.Checkpoint(model=self.model, optimizer=self.opt)
        latest_checkpoint = tf.train.latest_checkpoint(params["model_dir"])
        if latest_checkpoint:
            checkpoint.restore(latest_checkpoint)
            logging.info("Loaded checkpoint {}".format(latest_checkpoint))
            current_step = self.opt.iterations.numpy()

        train_ds = dataset.train_input_fn(params)

        train_loss_metric = tf.keras.metrics.Mean("training_loss", dtype=tf.float32)
        if params["enable_tensorboard"]:
            summary_writer = tf.summary.create_file_writer(os.path.join(params["model_dir"], "summary"))

        train_metrics = [train_loss_metric]
        if params["enable_metrics_in_training"]:
            train_metrics = train_metrics + self.model.metrics

        steps = 0
        for epoch in tf.range(1, 11):
            logging.info("train epoch: {}".format(epoch))
            for batch_id, inputs in enumerate(tqdm(train_ds, ncols=80)):
                train_loss_metric.reset_states()
                loss = self.train_step(inputs, params)
                logging.info("loss: {}".format(loss))

                train_loss_metric.update_state(loss)

                if params["enable_tensorboard"]:
                    for metric_obj in train_metrics:
                        tf.summary.scalar(metric_obj.name, metric_obj.result(),
                                          steps)
                        summary_writer.flush()
                steps += 1
            logging.info("the {} epoch loss: {}".format(epoch, loss))

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


def main(_):
    """主函数"""
    flags_obj = flags.FLAGS
    task = TextRewriteTask(flags_obj)

    if flags_obj.mode == "train":
        task.train()


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    misc.define_transformer_flags()
    app.run(main)
