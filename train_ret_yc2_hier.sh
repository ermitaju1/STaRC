#!/bin/bash

export CUDA_VISIBLE_DEVICES=$1

SAVE_DIR="./presave/youcook2"

mkdir -p ${SAVE_DIR}
python down_t5.py
python dvc_ret.py \
    --bank_type yc2 \
    --window_size 10 \
    --sim_match anchor_cos \
    --sampling origin \
    --save_dir=${SAVE_DIR} \
    --epochs=10 \
    --lr=1e-5 \
    --load ./ckpt/best_model/vid2seq_pretrained/vid2seq_youcook.pth \
    --combine_datasets youcook \
    --combine_datasets_val youcook \
    --batch_size=4 \
    --batch_size_val=4 \
    --schedule="cosine_with_warmup" \
    --soft_k 10 \
    --loss_lambda 6.0 \
    --asot_topk_for_retrieval 5 \
    --asot_K 8 \
    --asot_mu_salbias 0.1 \
    --asot_lambda_frames 0.3 \
    --window 8 \
    --use_saliency \
    --use_salip \
    --self_attn \
    --use_ret
