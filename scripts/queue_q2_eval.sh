#!/usr/bin/env bash
# After Q2 finishes (epoch 20): generate per-class with THRESHOLD (split_sample
# off), render, and run nearest-CD eval vs im_5 GT. Detached.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
source /home/dataset-assist-0/miniconda/etc/profile.d/conda.sh; conda activate ydg-ocnn
echo "[q2eval] waiting for Q2 done.flag ... $(date)"
until [ -f logs/exp/Q2_scaled/done.flag ]; do sleep 120; done
ck=$(ls logs/exp/Q2_scaled/checkpoints/*.model.pth | sort | tail -1)
echo "[q2eval] Q2 done, using $ck at $(date)"
G=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | awk -F, '$2+0<12000{print $1; exit}')
export CUDA_VISIBLE_DEVICES="${G:-0}"
out=logs/exp/Q2_scaled/gen_thresh
python eval/gen_from_ckpt.py --config configs/exp/Q2_scaled.yaml --ckpt "$ck" \
    --out_dir "$out" --per_class 3 --split_sample 0
for f in "$out"/*.obj; do python render_obj.py "$f" "${f%.obj}.png" 2>/dev/null; done
python eval/eval_fractal.py --gen_dir "$out" \
    --ref_dir data/ShapeNet/datasets_256_test \
    --ref_filelist data/ShapeNet/filelist/test_im_5.txt \
    --out_csv "$out/eval_cd.csv" --n_points 2048 --max_ref 100
echo "[q2eval] done at $(date)"
