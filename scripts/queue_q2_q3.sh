#!/usr/bin/env bash
# After Q2 finishes: build Q3 warmstart from Q2's final weights (zero the new
# split-MaskGIT params), then train Q3 (split-structure MaskGIT). Detached.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
source /home/dataset-assist-0/miniconda/etc/profile.d/conda.sh; conda activate ydg-ocnn
echo "[q3] waiting for Q2 done.flag ... $(date)"
until [ -f logs/exp/Q2_scaled/done.flag ]; do sleep 120; done
echo "[q3] Q2 done, building Q3 warmstart at $(date)"
mkdir -p logs/exp/Q3_splitmask
python - <<'PY'
import sys; sys.path.insert(0,'octgpt'); sys.path.insert(0,'.')
import glob, torch
from thsolver.config import parse_args
sys.argv=['x','--config','configs/exp/Q3_splitmask.yaml']; F=parse_args()
from fractal_models.fractal_generator import FractalGenerator
m=FractalGenerator(**F.MODEL.FractalGen)
q2=sorted(glob.glob('logs/exp/Q2_scaled/checkpoints/*.model.pth'))[-1]
sd=torch.load(q2,map_location='cpu',weights_only=True)
miss,unexp=m.load_state_dict(sd,strict=False)
print('from',q2,'| missing(new split params):',miss,'| unexpected:',unexp)
with torch.no_grad():
    m.split_emb.weight.zero_(); m.split_mask_emb.zero_()
torch.save(m.state_dict(),'logs/exp/Q3_splitmask/warmstart.model.pth')
print('saved Q3 warmstart, keys',len(m.state_dict()))
PY
echo "[q3] launching Q3 at $(date)"
bash scripts/run_cond_exp.sh Q3_splitmask 0 8 20 5
echo "[q3] Q3 finished at $(date)"
