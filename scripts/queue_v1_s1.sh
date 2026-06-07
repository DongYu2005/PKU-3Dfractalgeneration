#!/usr/bin/env bash
# After V1 finishes: build S1 warmstart from V1's final weights (zero the new
# leaf-MaskGIT params for warm-start equivalence), then launch S1. Detached.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
source /home/dataset-assist-0/miniconda/etc/profile.d/conda.sh
conda activate ydg-ocnn

echo "[queue] waiting for V1 done.flag ... $(date)"
until [ -f logs/exp/V1_latent/done.flag ]; do sleep 120; done
echo "[queue] V1 done, building S1 warmstart at $(date)"

mkdir -p logs/exp/S1_leaf_maskgit
python - <<'PY'
import sys; sys.path.insert(0,'octgpt'); sys.path.insert(0,'.')
import glob, torch
from thsolver.config import parse_args
sys.argv=['x','--config','configs/exp/S1_leaf_maskgit.yaml']
F=parse_args()
from fractal_models.fractal_generator import FractalGenerator
m=FractalGenerator(**F.MODEL.FractalGen)
v1=sorted(glob.glob('logs/exp/V1_latent/checkpoints/*.model.pth'))[-1]
sd=torch.load(v1,map_location='cpu',weights_only=True)
miss,unexp=m.load_state_dict(sd,strict=False)
print('from',v1,'| missing',miss,'| unexpected',unexp)
with torch.no_grad():
    m.vq_proj.weight.zero_(); m.vq_proj.bias.zero_()
    m.leaf_vq_mask_emb.zero_()
torch.save(m.state_dict(),'logs/exp/S1_leaf_maskgit/warmstart.model.pth')
print('saved S1 warmstart, keys',len(m.state_dict()))
PY

echo "[queue] launching S1 at $(date)"
bash scripts/run_cond_exp.sh S1_leaf_maskgit 1 16 20 5
echo "[queue] S1 finished at $(date)"
