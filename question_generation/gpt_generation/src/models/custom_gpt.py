"""
# 自定义GPT model
"""
import os
import logging
import torch
from typing import Any, Dict, Iterable, List, Optional, Tuple
from transformers import (
    GPT2Model,
    GPT2LMHeadModel
)
from transformers.file_utils import ModelOutput

logger = logging.getLogger(__name__)


def load_tf_weights_in_gpt2(model, config, gpt2_checkpoint_path):
    """Load tf checkpoints in a pytorch model"""
    try:
        import re

        import tensorflow as tf
    except ImportError:
        logger.error(
            "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise
    tf_path = os.path.abspath(gpt2_checkpoint_path)
    logger.info("Converting TensorFlow checkpoint from {}".format(tf_path))
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        logger.info("Loading TF weight {} with shape {}".format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array.squeeze())

    for name, array in zip(names, arrays):
        # name = name[5:]  # skip "bert/"
        name = name[6:]  # skip "model/"
        name = name.split("/")
        pointer = model
        for m_name in name:
            if re.fullmatch(r"[A-Za-z]+\d+", m_name):
                scope_names = re.split(r"(\d+)", m_name)
            else:
                scope_names = [m_name]
            if scope_names[0] == "w" or scope_names[0] == "g":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "b":
                pointer = getattr(pointer, "bias")
            elif scope_names[0] == "wpe" or scope_names[0] == "wte":
                pointer = getattr(pointer, scope_names[0])
                pointer = getattr(pointer, "weight")
            else:
                pointer = getattr(pointer, scope_names[0])
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]
        try:
            assert (
                pointer.shape == array.shape
            ), f"Pointer shape {pointer.shape} and array shape {array.shape} mismatched"
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        logger.info("Initialize PyTorch weight {}".format(name))
        pointer.data = torch.from_numpy(array)

    return model


class CustomGPT(GPT2Model):
    """custom gpt model"""
    load_tf_weights = load_tf_weights_in_gpt2  # tf model load function

    def __init__(self, config):
        super(CustomGPT, self).__init__(config)


class CustomGPTGeneration(GPT2LMHeadModel):
    """custom gpt generation model"""
    def __init__(self, config):
        super(CustomGPTGeneration, self).__init__(config)

    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        """覆写父类方法"""
        # only last token for inputs_ids if past is defined in kwargs
        token_type_ids = kwargs.get("token_type_ids", None)  # 获取token type
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None

        return {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
        }

    @staticmethod
    def _update_model_kwargs_for_generation(
            outputs: ModelOutput,
            model_kwargs: Dict[str, Any],
            is_encoder_decoder: bool = False
    ) -> Dict[str, Any]:
        # update past
        if "past_key_values" in outputs:
            model_kwargs["past"] = outputs.past_key_values
        elif "mems" in outputs:
            model_kwargs["past"] = outputs.mems
        elif "past_buckets_states" in outputs:
            model_kwargs["past"] = outputs.past_buckets_states
        else:
            model_kwargs["past"] = None

        # update attention mask
        if not is_encoder_decoder:
            if "attention_mask" in model_kwargs:
                attention_mask = model_kwargs["attention_mask"]
                # shape: [batch_size, sequence_length]
                model_kwargs["attention_mask"] = torch.cat(
                    [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
                )

            # update token type
            if "token_type_ids" in model_kwargs:
                token_type_ids = model_kwargs.get("token_type_ids", None)
                if token_type_ids is not None:
                    target_type_id = model_kwargs.get("target_type_id", None)  # target text type id(生成token的type id)
                    if target_type_id is None:
                        target_type_id = token_type_ids[0][-1].item() + 1  # 获取type最后一个元素+1
                        model_kwargs["target_type_id"] = target_type_id  # update dict element value
                    model_kwargs["token_type_ids"] = torch.cat(
                        [token_type_ids, token_type_ids.new_ones((token_type_ids.shape[0], 1)) * target_type_id],
                        dim=-1
                    )

        return model_kwargs

    @staticmethod
    def _expand_inputs_for_generation(
            input_ids: torch.LongTensor,
            expand_size: int = 1,
            is_encoder_decoder: bool = False,
            attention_mask: torch.LongTensor = None,
            encoder_outputs: ModelOutput = None,
            **model_kwargs
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        expanded_return_idx = (
            torch.arange(input_ids.shape[0]).view(-1, 1).repeat(1, expand_size).view(-1).to(input_ids.device)
        )
        input_ids = input_ids.index_select(0, expanded_return_idx)

        if attention_mask is not None:
            model_kwargs["attention_mask"] = attention_mask.index_select(0, expanded_return_idx)

        if model_kwargs["token_type_ids"] is not None:
            model_kwargs["token_type_ids"] = model_kwargs["token_type_ids"].index_select(0, expanded_return_idx)

        if is_encoder_decoder:
            assert encoder_outputs is not None
            encoder_outputs["last_hidden_state"] = encoder_outputs.last_hidden_state.index_select(
                0, expanded_return_idx
            )
            model_kwargs["encoder_outputs"] = encoder_outputs

        return input_ids, model_kwargs
