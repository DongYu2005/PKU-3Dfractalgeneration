#!/usr/bin/env bash
# Launch a conditioning experiment (C2/V1/...) on a single GPU, resilient to
# external process kills on the shared machine: retries from the latest
# checkpoint until the final epoch's checkpoint exists.
# Usage: bash scripts/run_cond_exp.sh <EXP> [GPU] [BS] [MAX_EPOCH] [TEST_EVERY]
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
EXP="${1:?usage: run_cond_exp.sh <EXP> [GPU] [BS] [MAX_EPOCH] [TEST_EVERY]}"
GPU="${2:-1}"; BS="${3:-16}"; MAXEP="${4:-20}"; TESTEVERY="${5:-5}"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
logdir="logs/exp/${EXP}"
mkdir -p "$logdir" "$logdir/checkpoints"

final_ckpt="$logdir/checkpoints/$(printf '%05d' "$MAXEP").solver.tar"

attempt=0
while true; do
    if [ -f "$final_ckpt" ]; then
        echo "[done] final checkpoint $final_ckpt exists"
        touch "$logdir/done.flag"
        break
    fi
    attempt=$((attempt + 1))
    echo "============================================"
    echo "[attempt $attempt] $(date)"

    if ls "$logdir"/checkpoints/*.solver.tar >/dev/null 2>&1; then
        ckpt_arg=()
        echo "[resume] auto-resuming from latest checkpoint"
    elif [ -f "$logdir/warmstart.model.pth" ]; then
        ckpt_arg=(SOLVER.ckpt "$logdir/warmstart.model.pth")
        echo "[warmstart] loading $logdir/warmstart.model.pth"
    else
        ckpt_arg=()
        echo "[scratch] training from scratch"
    fi

    python main_fractal.py \
        --config "configs/exp/${EXP}.yaml" \
        "SOLVER.gpu" "(0,)" \
        "${ckpt_arg[@]}" \
        SOLVER.max_epoch "$MAXEP" \
        SOLVER.test_every_epoch "$TESTEVERY" \
        DATA.train.batch_size "$BS" \
        DATA.train.num_workers 8 \
        DATA.test.batch_size 8 \
        2>&1 | tee -a "$logdir/train.log"
    code=${PIPESTATUS[0]}
    echo "[attempt $attempt] python exited code=$code at $(date)"

    # If it exited cleanly AND reached the final epoch, the top-of-loop check
    # will catch it. Otherwise wait a bit and retry (handles external kills).
    [ -f "$final_ckpt" ] && continue
    echo "[retry] sleeping 30s before relaunch"
    sleep 30
done
echo "[finished] $EXP at $(date)"
