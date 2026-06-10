#!/usr/bin/env bash
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
source /home/dataset-assist-0/miniconda/etc/profile.d/conda.sh; conda activate ydg-ocnn
ck=logs/exp/Q3_splitmask/checkpoints/00010.model.pth
echo "[q3gen] waiting for $ck ... $(date)"
until [ -f "$ck" ]; do sleep 60; done
sleep 30  # let checkpoint finish writing
G=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | awk -F, '$2+0<12000 && $1!=0{print $1; exit}')
export CUDA_VISIBLE_DEVICES="${G:-1}"
echo "[q3gen] generating on GPU $CUDA_VISIBLE_DEVICES at $(date)"
out=logs/exp/Q3_splitmask/gen_ep10
python eval/gen_from_ckpt.py --config configs/exp/Q3_splitmask.yaml --ckpt "$ck" \
    --out_dir "$out" --per_class 1 --split_sample 0
for f in "$out"/*.obj; do python render_obj.py "$f" "${f%.obj}.png" 2>/dev/null; done
python - <<'PY'
import trimesh,glob,os
print("=== Q3 ep10 迭代split 组件数 ===")
for f in sorted(glob.glob('logs/exp/Q3_splitmask/gen_ep10/*.obj')):
    m=trimesh.load(f,process=False,force='mesh')
    print(f"  {os.path.basename(f)}: verts={len(m.vertices)} components={len(m.split(only_watertight=False))}")
PY
echo "[q3gen] done at $(date)"
