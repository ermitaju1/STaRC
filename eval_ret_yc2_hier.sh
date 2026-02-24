# conda activate HiCM2
export CUDA_VISIBLE_DEVICES=$1

SAVE_DIR="/data/shchoi/STaRC/presave/presave/yc2__21"

python down_t5.py
python dvc_ret.py \
  --bank_type yc2 \
  --window_size 8 \
  --sim_match anchor_cos \
  --sampling origin \
  --save_dir="${SAVE_DIR}" \
  --load /data/shchoi/STaRC/ckpt/best_model/yc2_best_model.pth  \
  --lr=1e-5 \
  --combine_datasets youcook \
  --combine_datasets_val youcook \
  --batch_size=4 \
  --batch_size_val=4 \
  --schedule="cosine_with_warmup" \
  --soft_k 10 \
  --alpha 3.0 \
  --asot_topk_for_retrieval 5 \
  --asot_K 8 \
  --asot_mu_salbias 0.1 \
  --asot_lambda_frames 0.3 \
  --stride 8 \
  --eval \
  --use_saliency \
  --use_ret \
  --self_attn \
  --use_salip