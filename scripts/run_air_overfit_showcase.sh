#!/usr/bin/env bash
# Fine-tune the best small ablation variants on the single-airplane overfit case.
# Usage: bash scripts/run_air_overfit_showcase.sh [GPU] [MAX_EPOCH]
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

GPU="${1:-0}"
MAXEP="${2:-30}"
BASE_CKPT="logs/fractal/overfit_thresh050/checkpoints/00300.model.pth"
EXPS=(O1_air_struct_best O2_air_surface_best)

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source /home/dataset-assist-0/miniconda/etc/profile.d/conda.sh
conda activate ydg-ocnn

for EXP in "${EXPS[@]}"; do
  logdir="logs/exp/${EXP}"
  mkdir -p "$logdir"
  echo "===== [$EXP] build warmstart from $BASE_CKPT ====="
  python - "$EXP" "$BASE_CKPT" "$logdir/warmstart.model.pth" <<'PY'
import sys
sys.path.insert(0, "octgpt")
sys.path.insert(0, ".")

import torch
from thsolver.config import parse_args
from fractal_models.fractal_generator import FractalGenerator

exp, base_ckpt, out = sys.argv[1:4]
sys.argv = ["x", "--config", f"configs/exp/{exp}.yaml"]
flags = parse_args()
model = FractalGenerator(**flags.MODEL.FractalGen)
state = torch.load(base_ckpt, map_location="cpu", weights_only=True)
missing, unexpected = model.load_state_dict(state, strict=False)
print("missing:", missing)
print("unexpected:", unexpected)
with torch.no_grad():
    if getattr(model, "use_sibling_attn", False):
        for module in model.feature_expanders:
            module.sibling_attn.out_proj.weight.zero_()
            module.sibling_attn.out_proj.bias.zero_()
    if getattr(model, "leaf_vq_mask", False):
        model.leaf_vq_mask_emb.zero_()
        model.vq_proj.weight.zero_()
        model.vq_proj.bias.zero_()
torch.save(model.state_dict(), out)
print("saved", out)
PY

  echo "===== [$EXP] train on GPU $GPU for $MAXEP epochs ====="
  python main_fractal.py \
    --config "configs/exp/${EXP}.yaml" \
    "SOLVER.gpu" "(0,)" \
    SOLVER.ckpt "$logdir/warmstart.model.pth" \
    SOLVER.max_epoch "$MAXEP" \
    SOLVER.test_every_epoch 10 \
    DATA.train.batch_size 1 \
    DATA.train.num_workers 2 \
    DATA.test.batch_size 1 \
    2>&1 | tee -a "$logdir/train.log"
  code=${PIPESTATUS[0]}
  echo "===== [$EXP] exit code $code ====="
done
