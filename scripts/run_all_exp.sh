#!/usr/bin/env bash
# Run all E0-E9 ablation experiments sequentially on a single GPU.
# Usage: bash scripts/run_all_exp.sh [GPU_ID] [BATCH_SIZE]
#   GPU_ID    — which GPU to use (default: 1)
#   BATCH_SIZE — per-GPU batch size (default: 8)

set -uo pipefail

GPU_ID="${1:-1}"
BS="${2:-8}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

EXPS=(
    E0_baseline
    E1_mask_only
    E2_buffer_only
    E3_blocks_only
    E4_sibling_only
    E5_ce_only
    E6_mask_buffer
    E7_mask_buffer_ce
    E8_mask_buffer_blocks
    E9_full_stack
)

echo "============================================"
echo " Ablation sweep: ${#EXPS[@]} experiments"
echo " GPU: $GPU_ID   batch_size: $BS"
echo " Start: $(date)"
echo "============================================"

for exp in "${EXPS[@]}"; do
    cfg="configs/exp/${exp}.yaml"
    logdir="logs/exp/${exp}"

    echo ""
    echo ">>> [$exp] starting at $(date)"

    if [ -f "$logdir/done.flag" ]; then
        echo "    SKIP (done.flag exists)"
        continue
    fi

    if [ -f "$logdir/failed.flag" ]; then
        echo "    RETRY (clearing previous failed.flag)"
        rm "$logdir/failed.flag"
    fi

    mkdir -p "$logdir"

    # Snapshot: config + git sha + gpu info
    cp "$cfg" "$logdir/config_snapshot.yaml"
    git rev-parse HEAD 2>/dev/null > "$logdir/git_sha.txt" || echo "no-git" > "$logdir/git_sha.txt"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader -i "$GPU_ID" > "$logdir/gpu_info.txt" 2>/dev/null

    exp_bs="$BS"

    # The mask+buffer combinations do two mid-transformer passes and carry
    # buffer tokens, so they OOM at bs=16 on 80GB GPUs in the current logs.
    case "$exp" in
        E6_mask_buffer|E7_mask_buffer_ce|E8_mask_buffer_blocks|E9_full_stack)
            if [ "$exp_bs" -gt 8 ]; then
                echo "    NOTE: capping batch_size to 8 for high-memory combo"
                exp_bs=8
            fi
            ;;
    esac

    # Train: single GPU (CUDA_VISIBLE_DEVICES exported at top), override batch_size
    python main_fractal.py \
        --config "$cfg" \
        "SOLVER.gpu" "(0,)" \
        DATA.train.batch_size "$exp_bs" \
        DATA.train.num_workers 8 \
        SOLVER.test_every_epoch 5 \
        2>&1 | tee -a "$logdir/train.log"

    exit_code=${PIPESTATUS[0]}
    if [ $exit_code -ne 0 ]; then
        echo "    FAILED (exit $exit_code). See $logdir/train.log"
        echo "FAILED exit=$exit_code" > "$logdir/failed.flag"
        continue
    fi

    touch "$logdir/done.flag"
    echo ">>> [$exp] done at $(date)"
done

echo ""
echo "============================================"
echo " All experiments finished at $(date)"
echo "============================================"
