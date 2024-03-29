# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Defines Transformer model parameters."""

from collections import defaultdict


BASE_PARAMS = defaultdict(
    lambda: None,  # Set default value to None.

    # Input params
    default_batch_size=64,  # Maximum number of tokens per batch of examples.
    default_batch_size_tpu=64,
    max_length_source=128,  # Maximum number of tokens per example.
    max_length_target=32,

    # Model params
    initializer_gain=1.0,  # Used in trainable variable initialization.
    vocab_size=21128,  # Number of tokens defined in the vocabulary file.
    hidden_size=256,  # Model dimension in the hidden layers.
    num_hidden_layers=6,  # Number of layers in the encoder and decoder stacks.
    num_heads=8,  # Number of heads to use in multi-headed attention.
    filter_size=1024,  # Inner layer dimension in the feedforward network.

    # Dropout values (only used when training)
    layer_postprocess_dropout=0.1,
    attention_dropout=0.1,
    relu_dropout=0.1,

    # Training params
    epochs=15,
    label_smoothing=0.1,
    learning_rate=0.5,
    learning_rate_decay_rate=1.0,
    learning_rate_warmup_steps=1000,

    # Optimizer params
    optimizer_adam_beta1=0.9,
    optimizer_adam_beta2=0.997,
    optimizer_adam_epsilon=1e-09,

    # Default prediction params
    is_beam_search=False,
    is_custom_beam_search=True,
    beam_size=5,
    alpha=0.6,  # used to calculate length normalization in beam search
    length_penalty=1.0,
    early_stopping=False,

    # do sample
    do_sample=False,
    top_k=0.0,
    top_p=0.9,
    temperature=0.7,
    repetition_penalty=1.0,

    # TPU specific parameters
    use_tpu=False,
    static_batch=False,
    allow_ffn_pad=True,

    # model
    use_keras_model=False,  # 使用tf.keras.Model模块封装pipeline
)

BIG_PARAMS = BASE_PARAMS.copy()
BIG_PARAMS.update(
    default_batch_size=4096,

    # default batch size is smaller than for BASE_PARAMS due to memory limits.
    default_batch_size_tpu=16384,

    hidden_size=1024,
    filter_size=4096,
    num_heads=16,
)

# Parameters for running the model in multi gpu. These should not change the
# params that modify the model shape (such as the hidden_size or num_heads).
BASE_MULTI_GPU_PARAMS = BASE_PARAMS.copy()
BASE_MULTI_GPU_PARAMS.update(
    learning_rate_warmup_steps=8000
)

BIG_MULTI_GPU_PARAMS = BIG_PARAMS.copy()
BIG_MULTI_GPU_PARAMS.update(
    layer_postprocess_dropout=0.3,
    learning_rate_warmup_steps=8000
)

# Parameters for testing the model
TINY_PARAMS = BASE_PARAMS.copy()
TINY_PARAMS.update(
    default_batch_size=64,
    default_batch_size_tpu=64,
    hidden_size=256,
    num_heads=8,
    filter_size=1024,
)
