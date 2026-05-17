#!/usr/bin/env bash
# Start a v5 improvement experiment. Snapshots env (git SHA, requirements, GPU
# info, config) into experiments/E{ID}_{NAME}/ for reproducibility, then runs
# main_fractal.py in foreground (intended to live inside tmux).
#
# Usage:
#   bash scripts/start_exp.sh <ID> <NAME> <GPUs>
#
# Examples:
#   bash scripts/start_exp.sh 0p baseline_new_metric 4,5,6,7
#   bash scripts/start_exp.sh 1  mask_only           4,5,6,7
#
# Expects config at:  configs/exp/E${ID}_${NAME}.yaml

set -u

if [ $# -lt 2 ]; then
    echo "usage: $0 <ID> <NAME> [GPUs=4,5,6,7]"
    exit 1
fi

EID="$1"
NAME="$2"
GPUS="${3:-4,5,6,7}"

ROOT="/home/dataset-assist-0/usr/lh/ydg/几何计算前沿/3DFractalgen"
cd "$ROOT" || { echo "[FATAL] cannot cd to $ROOT"; exit 1; }

CONFIG="configs/exp/E${EID}_${NAME}.yaml"
if [ ! -f "$CONFIG" ]; then
    echo "[FATAL] config not found: $CONFIG"
    exit 1
fi

DIR="experiments/E${EID}_${NAME}"
mkdir -p "$DIR"

# Use the conda env python directly (avoid `conda run` signal-handling issues)
PY="/home/dataset-assist-0/miniconda/envs/ydg-ocnn/bin/python"

# ---- snapshot reproducibility metadata ----
git rev-parse HEAD          > "$DIR/git_sha.txt"
git status --porcelain      > "$DIR/git_status.txt"
"$PY" -m pip freeze         > "$DIR/requirements.txt" 2>/dev/null
nvidia-smi                  > "$DIR/gpu.txt"
cp "$CONFIG" "$DIR/config.yaml"

LOGDIR="logs/exp/E${EID}_${NAME}"

echo "================================================================"
echo "Experiment E${EID}: ${NAME}"
echo "Config:  $CONFIG"
echo "GPUs:    $GPUS"
echo "Logdir:  $LOGDIR"
echo "Record:  $DIR/"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================"

# Override the logdir from CLI so configs can stay generic
CUDA_VISIBLE_DEVICES="$GPUS" "$PY" main_fractal.py \
    --config "$CONFIG" \
    SOLVER.logdir "$LOGDIR" \
    2>&1 | tee "$DIR/train.log"

RC=${PIPESTATUS[0]}
echo "================================================================"
echo "Experiment E${EID} exited with rc=$RC at $(date '+%Y-%m-%d %H:%M:%S')"
echo "Update experiments/RESULTS.md with final metrics."
echo "================================================================"
exit $RC
