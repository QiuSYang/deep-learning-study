"""
# 设置配置参数
"""
import os


qg_configs = dict(
    save_path="ModelStorage/cloudwalk_qg_enhance.path",
    pre_train_dir="hfl/chinese-roberta-wwm-ext-large",
    start_token="[unused1]",
    end_token="[unused2]",
    start_token_id=1,
    end_token_id=2,
    dimension=1024,
    max_enc_len=512,
    max_dec_len=50,
    max_answer_len=100,
    use_beam_search=False,
    beam_width=5,
    beam_length_penalty=0.6,
    decoder_layers=3,
    dropout=0.1,
    vocab_size=21128,
    mode="test",
    device="cuda",
)
