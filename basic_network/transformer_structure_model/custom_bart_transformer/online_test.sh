#!/bin/sh
echo "Welcom to JDDC 2020"
export TORCH_HOME=./.torch
export CUDA_VISIBLE_DEVICES=0
# pip3 install -r requirements.txt -i https://pypi.doubanio.com/simple
python3 online_test_data_preprocess.py
python3 online_test_inference.py --checkpoint ckpt/data/BART/2020-09-10_05:44:08/9.pkl
python3 online_test_data_postprocess.py

echo "Done!"
