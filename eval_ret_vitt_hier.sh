#!/bin/bash
export CUDA_VISIBLE_DEVICES=$1

SAVE_DIR="./presave/vitt/vitt_lambda_${val}_${asot_K}_top${k}"
python down_t5.py
python dvc_ret.py \
    --bank_type ViTT \
    --window_size 10 \
    --sim_match anchor_cos \
    --sampling origin \
    --save_dir=${SAVE_DIR} \
    --load /data/shchoi/STaRC/ckpt/best_model/vitt_best_model.pth \
    --epochs=10 \
    --lr=1e-5 \
    --combine_datasets vitt \
    --combine_datasets_val vitt \
    --batch_size=4 \
    --batch_size_val=4 \
    --schedule="cosine_with_warmup" \
    --soft_k 10 \
    --alpha 1.5 \
    --asot_topk_for_retrieval 5 \
    --asot_K 8 \
    --asot_mu_salbias 0.1 \
    --asot_lambda_frames 0.3 \
    --stride 8  \
    --self_attn \
    --use_saliency \
    --use_salip \
    --use_ret \
    --eval





