CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 --nnodes=4 --node_rank=0 --master_addr="0.0.0.0" --master_port=7001 train.py
