set -eux

# conda deactivate
# conda activate paddle
export PYTHONPATH=../src

export BATCH_SIZE=80

export LR=2e-5
export EPOCH=10

unset CUDA_VISIBLE_DEVICES
python duie/run_duie.py \
        --device gpu:2 \
        --seed 42 \
        --do_train \
        --data_path /home/yckj2453/dataset/Info-Extract/duie \
        --max_seq_length 128 \
        --batch_size $BATCH_SIZE \
        --num_train_epochs $EPOCH \
        --learning_rate $LR \
        --warmup_ratio 0.06 \
        --output_dir ../checkpoints_roberta_large
