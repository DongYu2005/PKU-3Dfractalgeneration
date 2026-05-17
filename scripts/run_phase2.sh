#!/usr/bin/env bash
# Phase 2 automation: train Fractal im_5 -> train OctGPT im_5 -> run evaluation.
# Designed to run inside a long-lived tmux session.
#
# Usage:
#   tmux new-session -d -s phase2 "bash scripts/run_phase2.sh 4,5,6,7"
#   tmux attach -t phase2
#
# Arg 1: comma-separated GPU list to use (default: 4,5,6,7)

set -u  # error on unset vars, but DO NOT use -e: we want to keep going even
        # if one model crashes (so the other can still produce results).

ROOT="/home/dataset-assist-0/usr/lh/ydg/几何计算前沿/3DFractalgen"
cd "$ROOT" || { echo "[FATAL] cannot cd to $ROOT"; exit 1; }

GPUS="${1:-4,5,6,7}"
N_GPUS=$(echo "$GPUS" | tr ',' '\n' | grep -c .)
LOGDIR_TOP="logs/phase2_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR_TOP"
MASTER_LOG="$LOGDIR_TOP/master.log"

# Use direct python from the conda env to avoid `conda run` swallowing signals.
PY="/home/dataset-assist-0/miniconda/envs/ydg-ocnn/bin/python"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$MASTER_LOG"; }

log "================ Phase 2 starting ================"
log "GPUs: $GPUS (n=$N_GPUS)"
log "Master log dir: $LOGDIR_TOP"

# ---------------- Stage 1: Fractal im_5 training ----------------
FRACTAL_LOGDIR="logs/fractal/im5_gen"
log ""
log "Stage 1/3: Fractal im_5 training -> $FRACTAL_LOGDIR ($N_GPUS-GPU DDP via spawn mode)"

CUDA_VISIBLE_DEVICES="$GPUS" "$PY" main_fractal.py \
    --config configs/shapenet_frac_im5.yaml \
    SOLVER.logdir "$FRACTAL_LOGDIR" \
    2>&1 | tee -a "$LOGDIR_TOP/fractal_train.log"
FRACTAL_RC=${PIPESTATUS[0]}
log "Fractal training exited with rc=$FRACTAL_RC"

# ---------------- Stage 2: OctGPT im_5 training ----------------
OCTGPT_LOGDIR="logs/octgpt/im5_gen"
log ""
log "Stage 2/3: OctGPT im_5 training -> $OCTGPT_LOGDIR ($N_GPUS-GPU DDP, capped at 20 epoch)"

pushd octgpt >/dev/null
CUDA_VISIBLE_DEVICES="$GPUS" "$PY" main_octgpt.py \
    --config configs/ShapeNet/shapenet_uncond_im5.yaml \
    SOLVER.logdir "$OCTGPT_LOGDIR" \
    2>&1 | tee -a "../$LOGDIR_TOP/octgpt_train.log"
OCTGPT_RC=${PIPESTATUS[0]}
popd >/dev/null
log "OctGPT training exited with rc=$OCTGPT_RC"

# ---------------- Stage 3: Generate samples + evaluation ----------------
log ""
log "Stage 3/3: generate samples + run eval (per model)"

run_gen_and_eval() {
    local model="$1"          # fractal|octgpt
    local logdir="$2"         # logdir containing the trained ckpt
    local config="$3"         # config used for training
    local cwd="${4:-$ROOT}"   # cwd to run from
    local results_subdir="$logdir/results"
    local eval_csv="$logdir/eval_im5.csv"

    log "  [$model] generating samples to $results_subdir"
    pushd "$cwd" >/dev/null
    # Single-GPU for generation (no benefit from DDP for inference here)
    CUDA_VISIBLE_DEVICES="$(echo "$GPUS" | cut -d, -f1)" "$PY" \
        "$(if [[ "$model" == "fractal" ]]; then echo main_fractal.py; else echo main_octgpt.py; fi)" \
        --config "$config" \
        SOLVER.run generate \
        SOLVER.logdir "$logdir" \
        SOLVER.gpu '(0,)' \
        2>&1 | tee -a "$ROOT/$LOGDIR_TOP/${model}_generate.log"
    popd >/dev/null

    log "  [$model] running eval_fractal.py -> $eval_csv"
    cd "$ROOT"
    CUDA_VISIBLE_DEVICES="$(echo "$GPUS" | cut -d, -f1)" "$PY" eval/eval_fractal.py \
        --gen_dir "$([[ "$cwd" != "$ROOT" ]] && echo "$cwd/" || echo "")$results_subdir" \
        --ref_dir data/ShapeNet/datasets_256_test \
        --ref_filelist data/ShapeNet/filelist/test_im_5.txt \
        --out_csv "$eval_csv" \
        --n_points 2048 \
        --by_category \
        --max_ref 200 \
        2>&1 | tee -a "$LOGDIR_TOP/${model}_eval.log"
}

run_gen_and_eval fractal "$FRACTAL_LOGDIR" configs/shapenet_frac_im5.yaml "$ROOT"
run_gen_and_eval octgpt  "$OCTGPT_LOGDIR"  configs/ShapeNet/shapenet_uncond_im5.yaml "$ROOT/octgpt"

log ""
log "================ Phase 2 finished ================"
log "All artifacts under: $LOGDIR_TOP/"
log "Per-model results:"
log "  Fractal:  $FRACTAL_LOGDIR/eval_im5.csv  + obj in $FRACTAL_LOGDIR/results/"
log "  OctGPT:   $OCTGPT_LOGDIR/eval_im5.csv  + obj in $OCTGPT_LOGDIR/results/"
log ""
log "Open $MASTER_LOG for the master log."
