# -*- coding: utf-8 -*-
"""
# custom Bart pipeline：
    1. 构建模型，保存模型
    2. 训练模型，模型验证评估
    3. inference
"""
import os
import logging
import copy
import json
import time
import gc
import PIL
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from transformers import BartConfig
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm

from custom_bart.model import CustomBartGeneration, CustomBartGenerationDoubleHeads
from custom_bart.loss import masked_cross_entropy
from vocab import BOS_ID, EOS_ID, PAD_ID

logger = logging.getLogger(__name__)
# temporarily use resent18 image statistics
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


class BartSolver(object):
    def __init__(self, config, vocab, train_data_loader, eval_data_loader, is_train=True, model=None):
        """
        @param config: 超参数
        @param vocab: 词汇表
        @param train_data_loader:
        @param eval_data_loader:
        """
        self.config = config
        self.vocab = vocab
        self.train_data_loader = train_data_loader
        self.eval_data_loader = eval_data_loader
        self.is_train = is_train
        self.model = model
        self.checkpoint_dict = None
        self.epoch_i = 0

    def build_graph(self):
        """构建模型"""
        if self.config.bart_pre_training:
            self.model = CustomBartGeneration.from_pretrained(self.config.bart_pre_training)
            if self.model.config.vocab_size != self.config.vocab_size:
                # 使用预训练模型时词汇表发生变化, 重置embedding表的大小
                self.model.resize_token_embeddings(self.config.vocab_size)
        else:
            bart_config = BartConfig()
            bart_config.activation_function = self.config.activate_func
            bart_config.vocab_size = self.config.vocab_size
            bart_config.d_model = self.config.embed_size
            bart_config.max_position_embeddings = self.config.embed_size
            bart_config.max_length = self.config.max_generate_length
            bart_config.num_labels = self.config.num_labels
            bart_config.image_para_freeze = self.config.image_para_freeze
            bart_config.encoder_layers = self.config.n_layers
            bart_config.decoder_layers = self.config.n_layers
            bart_config.encoder_attention_heads = self.config.n_head
            bart_config.decoder_attention_heads = self.config.n_head
            bart_config.encoder_ffn_dim = self.config.ffn_dim
            bart_config.decoder_ffn_dim = self.config.ffn_dim
            bart_config.pad_token_id = PAD_ID
            bart_config.bos_token_id = BOS_ID
            bart_config.eos_token_id = EOS_ID
            self.model = CustomBartGeneration(config=bart_config)

            # multi-task
            # bart_config.summary_use_proj = True
            # bart_config.summary_activation = None
            # bart_config.summary_first_dropout = True
            # bart_config.summary_proj_to_labels = 0.1
            # bart_config.summary_type = "cls_index"
            # self.model = CustomBartGenerationDoubleHeads(config=bart_config)

        if torch.cuda.is_available():
            self.model.to(self.config.device)

        if self.config.checkpoint:
            self.checkpoint_dict = self.load_model(self.config.checkpoint)

        if self.is_train:
            no_decay = ['bias', 'layer_norm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                 'weight_decay': self.config.weight_decay},
                {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                 'weight_decay': 0.0}
            ]
            self.optimizer = AdamW(optimizer_grouped_parameters,
                                   lr=self.config.learning_rate,
                                   eps=self.config.adam_epsilon)

            self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                             num_warmup_steps=self.config.num_warmup_steps,
                                                             num_training_steps=self.config.num_training_steps)
            if self.config.checkpoint and self.checkpoint_dict:
                self.optimizer.load_state_dict(self.checkpoint_dict["optimizer"])  # 加载优化器参数
                self.scheduler.load_state_dict(self.checkpoint_dict["lr_scheduler"])  # 加载lr_scheduler

    def save_model(self, epoch, best=None):
        """Save parameters to checkpoint"""
        if best:
            ckpt_path = os.path.join(self.config.save_path, f'{best}.pkl')
        else:
            ckpt_path = os.path.join(self.config.save_path, f'{epoch}.pkl')
        model_state = {'model': self.model.state_dict(),
                       'optimizer': self.optimizer.state_dict(),
                       'lr_scheduler': self.scheduler.state_dict(),
                       'epoch': epoch}
        print(f'Save parameters to {ckpt_path}')
        torch.save(model_state, ckpt_path)

    def load_model(self, checkpoint_path):
        """Load parameters from checkpoint"""
        print(f'Load parameters from {checkpoint_path}')
        checkpoint = torch.load(checkpoint_path)
        self.epoch_i = checkpoint.get('epoch')
        self.model.load_state_dict(checkpoint['model'])

        return checkpoint

    def write_summary(self, epoch_i):
        epoch_loss = getattr(self, 'epoch_loss', None)
        if epoch_loss is not None:
            self.writer.update_loss(
                loss=epoch_loss,
                step_i=epoch_i + 1,
                name='train_loss')

        epoch_recon_loss = getattr(self, 'epoch_recon_loss', None)
        if epoch_recon_loss is not None:
            self.writer.update_loss(
                loss=epoch_recon_loss,
                step_i=epoch_i + 1,
                name='train_recon_loss')

        epoch_kl_div = getattr(self, 'epoch_kl_div', None)
        if epoch_kl_div is not None:
            self.writer.update_loss(
                loss=epoch_kl_div,
                step_i=epoch_i + 1,
                name='train_kl_div')

        kl_mult = getattr(self, 'kl_mult', None)
        if kl_mult is not None:
            self.writer.update_loss(
                loss=kl_mult,
                step_i=epoch_i + 1,
                name='kl_mult')

        epoch_bow_loss = getattr(self, 'epoch_bow_loss', None)
        if epoch_bow_loss is not None:
            self.writer.update_loss(
                loss=epoch_bow_loss,
                step_i=epoch_i + 1,
                name='bow_loss')

        validation_loss = getattr(self, 'validation_loss', None)
        if validation_loss is not None:
            self.writer.update_loss(
                loss=validation_loss,
                step_i=epoch_i + 1,
                name='validation_loss')

    def get_image_embeds(self, batch_images, batch_images_id, sentence_length):
        """获取图像embedding编码"""
        input_images_embeds = []
        # 初始化一个样例image embedding, 不包含图片的token全部使用样例代替(每个batch初始化一次)
        # image_sample = torch.zeros(3, 224, 224).unsqueeze(dim=0).to(device=self.config.device)
        # sample_image_embed = self.model.image_encoder(image_sample)[0]  # 取出第0张图片
        sample_image_embed = torch.zeros(self.config.embed_size).to(self.config.device)  # 全部初始化为0
        for idx, _single_images in enumerate(batch_images):
            # 将每个token位置image embed 赋予相同的初始值
            sentence_embeds = [sample_image_embed] * sentence_length
            # 删除最后一位样例图片, 图片数量与索引数量是否对应
            single_images_id = batch_images_id[idx]
            # assert len(_single_images[:-1]) == len(single_images_id)
            assert len(_single_images) == len(single_images_id)
            if _single_images and len(single_images_id) > 0:
                # 删除最后一张填充图像(包含图像才进行下面的操作)
                # single_images = _single_images[:-1].to(self.config.device)
                single_images = torch.stack(_single_images).to(self.config.device)  # 输入没有图像直接是空list
                single_images_embeds = self.model.image_encoder(single_images)
                # 计算每个token位置的image embed
                for idx_, image_embed in enumerate(single_images_embeds):
                    i = single_images_id[idx_]
                    sentence_embeds[i] = image_embed

            if self.config.candidates_size > 1:
                for i in range(self.config.candidates_size):
                    # 连续添加多次, 因为同一组候选集图像都是一样的(这样一个[batch, sequence_length]之后不用reshape)
                    input_images_embeds.append(torch.stack(sentence_embeds))
            else:
                input_images_embeds.append(torch.stack(sentence_embeds))

        # temp = input_images_embeds[0][43]

        return torch.stack(input_images_embeds)

    def get_image_embeds_from_feature(self, batch_images, batch_images_id, sentence_length):
        """获取图像embedding编码"""
        input_images_embeds = []
        sample_image_embed = torch.zeros(self.config.embed_size).to(self.config.device)
        for idx, _single_images in enumerate(batch_images):
            # 将每个token位置image embed 赋予相同的初始值
            sentence_embeds = [sample_image_embed] * sentence_length
            single_images_id = batch_images_id[idx]
            assert len(_single_images) == len(single_images_id)
            if _single_images and len(single_images_id) > 0:
                # 需要搭建一个fc层, 输入512, 输出是embedding size
                fc_images_feature = self.model.fc(torch.stack(_single_images).to(self.config.device))
                for idx_, image_embed in enumerate(fc_images_feature):
                    i = single_images_id[idx_]
                    image_embed = torch.tensor(image_embed)
                    sentence_embeds[i] = image_embed

            if self.config.candidates_size > 1:
                for i in range(self.config.candidates_size):
                    # 连续添加多次, 因为同一组候选集图像都是一样的(这样一个[batch, sequence_length]之后不用reshape)
                    input_images_embeds.append(torch.stack(sentence_embeds))
            else:
                input_images_embeds.append(torch.stack(sentence_embeds))

        # temp = input_images_embeds[0][43]

        return torch.stack(input_images_embeds)

    def train(self):
        """训练函数"""
        self.config.candidates_size = self.config.train_num_candidates
        epoch_loss_history = []
        best_eval_loss = float('inf')  # 记录最佳损失

        # 设置并行计算
        if self.config.gpu_nums > 1:
            print("use torch.nn.DataParallel for the parallel operations.")
            self.model = nn.DataParallel(self.model)
        if self.config.local_rank != -1 and self.config.distributed:
            print("use torch.nn.parallel.DistributedDataParallel for the parallel operations.")
            self.model = nn.parallel.DistributedDataParallel(self.model,
                                                             device_ids=[self.config.local_rank],
                                                             output_device=self.config.local_rank,
                                                             find_unused_parameters=True)

        for epoch_i in range(self.epoch_i, self.config.n_epoch):
            batch_loss_history = []
            loss_history = []
            num_batch = 0
            self.model.train()
            n_total_words = 0

            # 每个batch开始之前, 先进行梯度清空
            # self.optimizer.zero_grad()
            self.model.zero_grad()  # 更加安全的清理梯度

            for batch_i, (input_ids, input_token_type_ids,
                          input_images, input_images_id,
                          lm_labels, mc_token_ids, mc_labels) in enumerate(tqdm(self.train_data_loader, ncols=80)):
                num_batch = batch_i
                # start_time_ = time.clock()
                if isinstance(self.model, CustomBartGeneration):
                    # 单头任务, 进行数据shape: [batch, sequence_length] mc_token_ids, mc_labels不使用
                    input_ids = torch.LongTensor(input_ids).to(self.config.device)
                    # input_token_type_ids = torch.LongTensor(input_token_type_ids).to(self.config.device)
                    input_token_type_ids = None
                    # 图像embedding编码
                    input_image_embeds = self.get_image_embeds(input_images, input_images_id, input_ids.size(-1))
                    # input_image_embeds = None

                    # decoder inputs
                    input_lm_labels = torch.LongTensor(lm_labels).to(self.config.device)
                    input_lm_labels_length = torch.LongTensor(mc_token_ids)
                    # 计算每个target sentence length
                    input_lm_labels_length = input_lm_labels_length.view(-1).to(self.config.device)

                    encoder_attention_mask = input_ids.ne(0).long()  # 对输入数据进行mask
                    # target输入, 去除最后一位, 本应该去除end符, 但实际计算end符loss不被计算
                    decoder_input_ids = input_lm_labels[:, :-1]
                    sentence_logits, _ = self.model(input_ids=input_ids,
                                                    input_token_type_ids=input_token_type_ids,
                                                    attention_mask=encoder_attention_mask,
                                                    input_image_embeds=input_image_embeds,
                                                    # input_images=input_images,
                                                    # input_images_id=input_images_id,
                                                    decoder_input_ids=decoder_input_ids)

                    decoder_target_label_ids = input_lm_labels[:, 1:]  # GPT解码Label, 去除首部的起始字符
                    # sentence_logits = outputs[0]  # 获取Bart的logits
                    batch_loss, n_words = masked_cross_entropy(
                        sentence_logits,
                        decoder_target_label_ids,
                        input_lm_labels_length)

                elif isinstance(self.model, CustomBartGenerationDoubleHeads):
                    # 多任务, shape: [batch_size, num_candidates, sequence_length],
                    # mc_token_ids, mc_labels-第二个任务评价对象
                    input_ids = torch.LongTensor(input_ids).to(self.config.device)
                    input_token_type_ids = torch.LongTensor(input_token_type_ids).to(self.config.device)
                    # 图像embedding编码, shape: [batch_size * num_candidates, sequence_length]
                    input_image_embeds = self.get_image_embeds(input_images, input_images_id, input_ids.size(-1))
                    assert (input_ids.size(0)*input_ids.size(1)) == input_image_embeds.size(0)  # 形状判定

                    # decoder inputs
                    input_lm_labels = torch.LongTensor(lm_labels).to(self.config.device)
                    input_mc_token_ids = torch.LongTensor(mc_token_ids).to(self.config.device)
                    input_mc_labels = torch.LongTensor(mc_labels).to(self.config.device)

                    def get_double_decoder_input_ids(input_target_ids):
                        for batch_idx, batch_data in enumerate(input_target_ids):
                            for idx in range(len(batch_data)-1):
                                # 最后一条为label input
                                input_target_ids[batch_idx][idx] = input_target_ids[batch_idx][-1]

                        return input_target_ids

                    encoder_attention_mask = input_ids.ne(0).long()  # 对输入数据进行mask
                    decoder_input_ids = get_double_decoder_input_ids(input_lm_labels.clone())
                    lm_loss, mc_loss, *_ = self.model(input_ids=input_ids,
                                                      input_token_type_ids=input_token_type_ids,
                                                      attention_mask=encoder_attention_mask,
                                                      input_image_embeds=input_image_embeds,
                                                      decoder_input_ids=decoder_input_ids,  # 不在去除最后一个字符
                                                      lm_labels=input_lm_labels,
                                                      mc_token_ids=input_mc_token_ids,
                                                      mc_labels=input_mc_labels)
                    batch_loss = lm_loss * self.config.lm_coef + mc_loss * self.config.mc_coef
                    target_length = input_mc_token_ids.clone()
                    n_words = target_length.view(-1).sum()

                if self.config.gpu_nums > 1:
                    # mean() to average on multi-gpu parallel (not distributed) training
                    batch_loss = batch_loss.mean()
                    n_words = n_words.mean()
                if self.config.gradient_accumulation_step > 1:
                    batch_loss = batch_loss / self.config.gradient_accumulation_step
                    n_words = n_words / self.config.gradient_accumulation_step

                # assert not isnan(batch_loss.item())
                batch_loss_history.append(batch_loss.item())
                n_total_words += n_words.item()
                # loss_history.append(loss)

                if batch_i % self.config.print_every == 0:
                    print(
                        f'Epoch: {epoch_i + 1}, iter {batch_i}: loss = {batch_loss.item() / n_words.item():.3f}')

                # Back-propagation
                # loss.backward()
                # batch_loss.backward(retain_graph=True)
                batch_loss.backward()

                # Gradient cliping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip)

                # 进行梯度累积
                if (batch_i + 1) % self.config.gradient_accumulation_step == 0:
                    # Run optimizer & scheduler
                    self.optimizer.step()
                    self.scheduler.step()
                    # self.optimizer.zero_grad()  # 清空梯度
                    self.model.zero_grad()

                # print("2", time.clock() - start_time_)

            torch.cuda.empty_cache()
            gc.collect()

            # epoch_loss = np.sum(loss_history) / (num_batch + 1)
            epoch_loss = np.sum(batch_loss_history) / n_total_words
            epoch_loss_history.append(epoch_loss)
            self.epoch_loss = epoch_loss

            print_str = f'Epoch {epoch_i + 1} loss average: {epoch_loss:.3f}'
            print(print_str)

            if epoch_i % self.config.save_every_epoch == 0:
                self.save_model(epoch_i + 1)

            # Only evaluate when single GPU otherwise metrics may not average well
            if self.config.local_rank == -1 and self.config.model_val:
                # print('\n<BLEU score>...')
                # self.calculate_bleu()

                print('\n<Validation>...')
                self.validation_loss = self.evaluate()

                # 保存最佳validation los model
                if self.validation_loss < best_eval_loss:
                    self.save_model(epoch_i, best='best_model')
                    # 更新最佳验证损失
                    best_eval_loss = self.validation_loss
            #
            # if epoch_i % self.config.plot_every_epoch == 0:
            #     self.write_summary(epoch_i)

        self.save_model(self.config.n_epoch)

        return epoch_loss_history

    def evaluate(self):
        """评估函数"""
        self.config.candidates_size = self.config.train_num_candidates
        self.model.eval()
        batch_loss_history = []
        n_total_words = 0

        for batch_i, (input_ids, input_token_type_ids,
                      input_images, input_images_id,
                      lm_labels, mc_token_ids, mc_labels) in enumerate(tqdm(self.train_data_loader, ncols=80)):
            num_batch = batch_i
            # start_time_ = time.clock()
            with torch.no_grad():
                if isinstance(self.model, CustomBartGeneration):
                    # 单头任务, 进行数据shape: [batch, sequence_length] mc_token_ids, mc_labels不使用
                    input_ids = torch.LongTensor(input_ids).to(self.config.device)
                    # input_token_type_ids = torch.LongTensor(input_token_type_ids).to(self.config.device)
                    input_token_type_ids = None
                    # 图像embedding编码
                    input_image_embeds = self.get_image_embeds(input_images, input_images_id, input_ids.size(-1))
                    # input_image_embeds = None

                    # target inputs
                    input_lm_labels = torch.LongTensor(lm_labels).to(self.config.device)
                    input_lm_labels_length = torch.LongTensor(mc_token_ids)
                    # 计算每个target sentence length
                    input_lm_labels_length = input_lm_labels_length.view(-1).to(self.config.device)

                    encoder_attention_mask = input_ids.ne(0).long()  # 对输入数据进行mask
                    # target输入, 去除最后一位, 本应该去除end符, 但实际计算end符loss不被计算
                    decoder_input_ids = input_lm_labels[:, :-1]
                    sentence_logits, _ = self.model(input_ids=input_ids,
                                                    input_token_type_ids=input_token_type_ids,
                                                    attention_mask=encoder_attention_mask,
                                                    input_image_embeds=input_image_embeds,
                                                    # input_images=input_images,
                                                    # input_images_id=input_images_id,
                                                    decoder_input_ids=decoder_input_ids)

                    decoder_target_label_ids = input_lm_labels[:, 1:]  # GPT解码Label, 去除首部的起始字符
                    # sentence_logits = outputs[0]  # 获取Bart的logits
                    batch_loss, n_words = masked_cross_entropy(
                        sentence_logits,
                        decoder_target_label_ids,
                        input_lm_labels_length)

                elif isinstance(self.model, CustomBartGenerationDoubleHeads):
                    # 多任务, shape: [batch_size, num_candidates, sequence_length],
                    # mc_token_ids, mc_labels-第二个任务评价对象
                    input_ids = torch.LongTensor(input_ids).to(self.config.device)
                    input_token_type_ids = torch.LongTensor(input_token_type_ids).to(self.config.device)
                    # 图像embedding编码, shape: [batch_size * num_candidates, sequence_length]
                    input_image_embeds = self.get_image_embeds(input_images, input_images_id, input_ids.size(-1))
                    assert (input_ids.size(0) * input_ids.size(1)) == input_image_embeds.size(0)  # 形状判定

                    # decoder inputs
                    input_lm_labels = torch.LongTensor(lm_labels).to(self.config.device)
                    input_mc_token_ids = torch.LongTensor(mc_token_ids).to(self.config.device)
                    input_mc_labels = torch.LongTensor(mc_labels).to(self.config.device)

                    def get_double_decoder_input_ids(input_target_ids):
                        for batch_idx, batch_data in enumerate(input_target_ids):
                            for idx in range(len(batch_data) - 1):
                                # 最后一条为label input
                                input_target_ids[batch_idx][idx] = input_target_ids[batch_idx][-1]

                        return input_target_ids

                    encoder_attention_mask = input_ids.ne(0).long()  # 对输入数据进行mask
                    decoder_input_ids = get_double_decoder_input_ids(input_lm_labels.clone())
                    lm_loss, mc_loss, *_ = self.model(input_ids=input_ids,
                                                      input_token_type_ids=input_token_type_ids,
                                                      attention_mask=encoder_attention_mask,
                                                      input_image_embeds=input_image_embeds,
                                                      decoder_input_ids=decoder_input_ids,  # 不在去除最后一个字符
                                                      lm_labels=input_lm_labels,
                                                      mc_token_ids=input_mc_token_ids,
                                                      mc_labels=input_mc_labels)
                    batch_loss = lm_loss * self.config.lm_coef + mc_loss * self.config.mc_coef
                    target_length = input_mc_token_ids.clone()
                    n_words = target_length.view(-1).sum()

                if self.config.gpu_nums > 1:
                    # mean() to average on multi-gpu parallel (not distributed) training
                    batch_loss = batch_loss.mean()
                    n_words = n_words.mean()
                if self.config.gradient_accumulation_step > 1:
                    batch_loss = batch_loss / self.config.gradient_accumulation_step
                    n_words = n_words / self.config.gradient_accumulation_step

                batch_loss_history.append(batch_loss.item())
                n_total_words += n_words.item()
        # 显存回收
        torch.cuda.empty_cache()
        gc.collect()

        epoch_loss = np.sum(batch_loss_history) / n_total_words
        # epoch_loss = np.sum(loss_history) / (num_batch + 1)
        print_str = f'Validation custom loss: {epoch_loss:.3f}\n'
        # print_str = f'Validation loss: {epoch_loss:.3f}\n'
        print(print_str)

        return epoch_loss

    def generate_for_evaluation(self):
        """模型inference"""
        self.config.candidates_size = self.config.val_num_candidates
        self.model.eval()
        n_sent = 0
        fo = open(self.config.pred_path, "w")
        for batch_i, (input_ids, input_token_type_ids,
                      input_images, input_images_id,
                      lm_labels, mc_token_ids, mc_labels) in enumerate(tqdm(self.eval_data_loader, ncols=80)):
            batch_size = len(input_ids)
            src_input_ids = copy.copy(input_ids)
            with torch.no_grad():
                # inference过程
                input_ids = torch.LongTensor(input_ids).to(self.config.device)
                # input_token_type_ids = torch.LongTensor(input_token_type_ids).to(self.config.device)
                input_token_type_ids = None
                # 图像embedding编码
                input_image_embeds = self.get_image_embeds(input_images, input_images_id, input_ids.size(-1))
                # input_image_embeds = None

                # encoder_attention_mask = input_ids.ne(0).long()  # 对输入数据进行mask
                encoder_attention_mask = None

                batch_samples = self.model.generate(input_ids=input_ids,
                                                    attention_mask=encoder_attention_mask,  # encoder mask
                                                    num_beams=self.config.beam_size,
                                                    max_length=self.config.max_generate_length,
                                                    early_stopping=True,
                                                    do_sample=True,
                                                    # temperature=0.7,
                                                    # top_k=0,
                                                    # top_p=0.9,
                                                    num_return_sequences=1,
                                                    input_token_type_ids=input_token_type_ids,
                                                    input_image_embeds=input_image_embeds)

                samples = batch_samples.data.cpu().numpy().tolist()
                for i in range(len(samples)):
                    sample = self.vocab.decode(samples[i])
                    ground_truth = self.vocab.decode(lm_labels[i])
                    context_str = self.vocab.decode(src_input_ids[i])

                    n_sent += 1
                    # fo.write(context_str + "\t" + ground_truth + "\t" + sample + "\n")
                    fo.write("context string." + "\t" + "ground truth string." + "\t" + sample + "\n")

        print('n_sentences: {}\n'.format(n_sent))
        fo.close()

