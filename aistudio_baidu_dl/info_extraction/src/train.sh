set -eux

# conda deactivate
# conda activate paddle

export BATCH_SIZE=256

export LR=2e-5
export EPOCH=10

unset CUDA_VISIBLE_DEVICES
python duie/run_duie.py \
        --device gpu:3 \
        --seed 42 \
        --do_train \
        --data_path /home/yckj2453/dataset/Info-Extract/duie \
        --max_seq_length 128 \
        --batch_size $BATCH_SIZE \
        --num_train_epochs $EPOCH \
        --learning_rate $LR \
        --warmup_ratio 0.06 \
        --output_dir ../checkpoints
