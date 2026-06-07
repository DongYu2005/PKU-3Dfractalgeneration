#!/usr/bin/env bash
# C1 class-conditional warm-start validation run.
# First launch: warm-starts from E0 baseline (class_embedding zero-init).
# Re-launch after a crash: auto-resumes from the latest checkpoint in ckpt_dir
# (leave SOLVER.ckpt empty -> thsolver picks the newest *.solver.tar).
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
logdir=logs/exp/C1_class_cond
mkdir -p "$logdir"

# Resume if a checkpoint already exists, else warm-start from baseline.
if ls "$logdir"/checkpoints/*.solver.tar >/dev/null 2>&1; then
    ckpt_arg=()   # empty -> thsolver auto-resumes from latest in ckpt_dir
    echo "[resume] found existing checkpoint, auto-resuming"
else
    ckpt_arg=(SOLVER.ckpt "$logdir/warmstart.model.pth")
    echo "[warmstart] no checkpoint, loading baseline warmstart"
fi

python main_fractal.py \
    --config configs/exp/C1_class_cond.yaml \
    "SOLVER.gpu" "(0,)" \
    "${ckpt_arg[@]}" \
    SOLVER.max_epoch 5 \
    SOLVER.test_every_epoch 1 \
    DATA.train.batch_size 16 \
    DATA.train.num_workers 8 \
    DATA.test.batch_size 8 \
    2>&1 | tee -a "$logdir/train.log"
