# -*- coding: utf-8 -*-
"""
# 继承transformers Bart 搭建custom Bart
"""
import os
import logging
import torch
import torchvision
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import CrossEntropyLoss
from transformers import BartConfig
from .modeling_bart import BartForConditionalGeneration
from .modeling_utils import SequenceSummary

logger = logging.getLogger(__name__)
IMAGE_FEATURE_INPUT_SIZE = 512


class ImageEncoder(nn.Module):
    def __init__(self, config, feature_size):
        """Image Encoder"""
        super(ImageEncoder, self).__init__()

        self.config = config
        self.input_size = feature_size
        self.model = self.init_resnet18(self.input_size)

    def init_resnet18(self, feature_size):
        def set_parameter_requires_grad(model, feature_extracting):
            if feature_extracting:
                # 是否冻结网络参数
                for param in model.parameters():
                    param.requires_grad = False

        model_ft = torchvision.models.resnet18(pretrained=True)
        set_parameter_requires_grad(model_ft, self.config.image_para_freeze)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, feature_size)
        return model_ft

    def forward(self, inputs):

        """
        Args:
            inputs (Variable, LongTensor): [num_setences, 3, 224, 224]
            input_length (Variable, LongTensor): [num_sentences]
        Return:
            outputs (Variable): [max_source_length, batch_size, hidden_size]
                - list of all hidden states
            hidden ((tuple of) Variable): [num_layers*num_directions, batch_size, hidden_size]
                - last hidden state
                - (h, c) or h
        """

        return self.model(inputs)


class CustomBartGeneration(BartForConditionalGeneration):
    """自定义Bart条件生成模型"""
    def __init__(self, config: BartConfig):
        super(CustomBartGeneration, self).__init__(config)

        self.image_encoder = ImageEncoder(config=config, feature_size=config.d_model)
        self.fc = nn.Linear(IMAGE_FEATURE_INPUT_SIZE, config.d_model)  # 图像输入为已经编码的feature

    def get_image_embeds(self, sample_image_embed, batch_images, batch_images_id,
                         sentence_length, device='cpu'):
        """获取图像embedding编码"""
        input_images_embeds = []
        for idx, _single_images in enumerate(batch_images):
            # 将每个token位置image embed 赋予相同的初始值
            sentence_embeds = [sample_image_embed] * sentence_length
            # 删除最后一位样例图片, 图片数量与索引数量是否对应
            single_images_id = batch_images_id[idx]
            assert len(_single_images[:-1]) == len(single_images_id)
            if single_images_id and len(single_images_id) > 0:
                # 删除最后一张填充图像(包含图像才进行下面的操作)
                single_images = _single_images[:-1].to(device)
                single_images_embeds = self.image_encoder(single_images)
                # 计算每个token位置的image embed
                for idx_, image_embed in enumerate(single_images_embeds):
                    i = single_images_id[idx_]
                    sentence_embeds[i] = image_embed

            input_images_embeds.append(torch.stack(sentence_embeds))

        return torch.stack(input_images_embeds)

    def forward(
            self,
            input_ids,
            input_token_type_ids=None,
            attention_mask=None,  # 需要外部给出
            input_image_embeds=None,
            input_images=None,
            input_images_id=None,
            encoder_outputs=None,  # 不希望指定参数(即外部编码)
            decoder_input_ids=None,
            decoder_attention_mask=None,  # 内部帮忙算, 不希望给出
            decoder_cached_states=None,  # 内部帮忙算, 不希望给出
            lm_labels=None,
            use_cache=False,
            **unused
    ):
        """
        @param input_ids:
        @param input_token_type_ids:
        @param attention_mask:
        @param input_image_embeds:
        @param input_images:
        @param input_images_id:
        @param encoder_outputs:
        @param decoder_input_ids:
        @param decoder_attention_mask:
        @param decoder_cached_states:
        @param lm_labels:
        @param use_cache:
        @param unused:
        @return:
        """
        # batch_size = input_ids.size(0)
        # sequence_length = input_ids.size(1)
        # if input_images is not None:
        #     # 图像encoder, 生产image embedding
        #     input_image_embeds = self.image_encoder(input_images)
        #     input_image_embeds = input_image_embeds.view(batch_size, sequence_length, -1)
        # else:
        #     input_image_embeds = None

        if input_images is not None and input_images_id is not None and input_image_embeds is None:
            # 图像embedding encoder
            device = input_ids.device
            # image_sample = torch.zeros(3, 224, 224).unsqueeze(dim=0).to(device=device)
            # sample_image_embed = self.image_encoder(image_sample)[0]  # 取出第0张图片(每个batch初始化一次)
            sample_image_embed = torch.zeros(self.config.d_model).to(self.config.device)  # 全部初始化为0
            input_image_embeds = self.get_image_embeds(sample_image_embed,
                                                       input_images,
                                                       input_images_id,
                                                       input_ids.size(-1),
                                                       device=device)

        transformer_outputs = self.model(
            input_ids,
            input_token_type_ids=input_token_type_ids,
            attention_mask=attention_mask,
            input_image_embeds=input_image_embeds,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            decoder_cached_states=decoder_cached_states,
            use_cache=use_cache,
        )

        lm_logits = F.linear(transformer_outputs[0], self.model.shared.weight, bias=self.final_logits_bias)
        outputs = (lm_logits,) + transformer_outputs[1:]  # Add cache, hidden states and attention if they are here

        if lm_labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=0)
            # TODO(SS): do we need to ignore pad tokens in lm_labels?
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), lm_labels.view(-1))
            outputs = (masked_lm_loss,) + outputs

        return outputs


class CustomBartGenerationDoubleHeads(BartForConditionalGeneration):
    """自定义Bart多任务条件生成模型"""
    def __init__(self, config: BartConfig):
        super(CustomBartGenerationDoubleHeads, self).__init__(config)

        self.image_encoder = ImageEncoder(config=config, feature_size=config.d_model)
        self.fc = nn.Linear(IMAGE_FEATURE_INPUT_SIZE, config.d_model)  # 图像输入为已经编码的feature

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.multiple_choice_head = SequenceSummary(config)

    def forward(
            self,
            input_ids,
            input_token_type_ids=None,
            attention_mask=None,  # 需要外部给出
            input_image_embeds=None,
            encoder_outputs=None,  # 不希望指定参数(即外部编码)
            decoder_input_ids=None,
            decoder_attention_mask=None,  # 内部帮忙算, 不希望给出
            decoder_cached_states=None,  # 内部帮忙算, 不希望给出
            lm_labels=None,
            mc_token_ids=None,
            mc_labels=None,
            use_cache=False,
            **unused
    ):
        """
        @param input_ids:
        @param input_token_type_ids:
        @param attention_mask:
        @param input_image_embeds:
        @param encoder_outputs:
        @param decoder_input_ids:
        @param decoder_attention_mask:
        @param decoder_cached_states:
        @param lm_labels:
        @param mc_token_ids:
        @param mc_labels:
        @param use_cache:
        @param unused:
        @return:
        """
        # 自定义Bart为了兼容double head multi-task, 输入shape为[batch, num_candidate, sequence_length]
        input_shape = input_ids.size()
        # 转为之后可以处理的shape[batch*num_candidate, sequence_length]
        input_ids = input_ids.view(-1, input_shape[-1])
        if input_token_type_ids is not None:
            assert input_token_type_ids.size() == input_shape, "输入数据形状不统一"
            input_token_type_ids = input_token_type_ids.view(-1, input_shape[-1])
        if attention_mask is not None:
            assert attention_mask.size() == input_shape, "输入数据形状不统一"
            attention_mask = attention_mask.view(-1, input_shape[-1])
        if input_image_embeds is not None:
            # assert input_image_embeds.size[:2] == input_shape[:2]
            input_image_embeds = input_image_embeds.view(-1, input_shape[-1], input_image_embeds.size(-1))
        if decoder_input_ids is not None:
            # assert decoder_input_ids.size[:2] == input_shape[:2]
            decoder_input_ids = decoder_input_ids.view(-1, decoder_input_ids.size(-1))

        transformer_outputs = self.model(
            input_ids,
            input_token_type_ids=input_token_type_ids,
            attention_mask=attention_mask,
            input_image_embeds=input_image_embeds,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            decoder_cached_states=decoder_cached_states,
            use_cache=use_cache,
        )

        label_shape = lm_labels.size()
        output_shape = label_shape + (-1, )
        # assert label_shape == mc_labels.size() == mc_token_ids.size()

        hidden_states = transformer_outputs[0].view(*output_shape)

        lm_logits = self.lm_head(hidden_states)
        # 计算好像有问题, 每次计算都一样
        mc_logits = self.multiple_choice_head(hidden_states, mc_token_ids).squeeze(-1)  # 最后一维数据合并

        # transformer_outputs[1:] 索引1之后数据是否也要转换为原来的形状, inference不存在候选集
        outputs = (lm_logits, mc_logits) + transformer_outputs[1:]
        if mc_labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(mc_logits.view(-1, mc_logits.size(-1)), mc_labels.view(-1))
            outputs = (loss,) + outputs
        if lm_labels is not None:
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = lm_labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss(ignore_index=0)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (lm loss), (mc loss), lm logits, mc logits, presents, (all hidden_states), (attentions)

