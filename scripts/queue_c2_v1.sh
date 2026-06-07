#!/usr/bin/env bash
# Wait for C2 to finish, then launch V1 on the same GPU. Detached + resilient.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
source /home/dataset-assist-0/miniconda/etc/profile.d/conda.sh
conda activate ydg-ocnn
echo "[queue] waiting for C2 done.flag ... $(date)"
until [ -f logs/exp/C2_cond_everylevel/done.flag ]; do sleep 120; done
echo "[queue] C2 done, launching V1 at $(date)"
bash scripts/run_cond_exp.sh V1_latent 1 16 20 5
echo "[queue] V1 finished at $(date)"
