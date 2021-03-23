transformer + pointer network 模型实现文本重写 --- tf2 实现

运行脚本

    python train.py --data_train=../data/train.txt --data_dev=../data/dev.txt 
    --model_dir=../models/tiny_0312 --vocab_file=../resource/vocab.txt 
    --param_set=tiny --batch_size=32 --decode_batch_size=32 
    --bleu_source=../data/dev.txt --bleu_ref=../data/BLEU_REF.txt 
    --num_gpus=1 --enable_time_history=false 
    --mode="train"(or "eval" or "predict") 
    --use_ctl=false --enable_tensorboard=true
    
评测结果: tf2 transformer + pointer network model

|model|bleu1|bleu2|bleu4|rouge_1|rouge_2|rouge_l|em|match|total|
|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
|greedy|0.854|0.817|0.776|0.899|0.820|0.888|0.477|954|2000|
|beam-5|0.869|0.827|0.788|0.909|0.826|0.889|0.485|970|2000|

参考链接: 
    
    1. https://github.com/liu-nlper/dialogue-utterance-rewriter(脚本运行参考此项目)
