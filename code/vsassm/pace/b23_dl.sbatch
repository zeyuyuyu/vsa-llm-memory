#!/bin/bash
#SBATCH -J b23_dl
#SBATCH --account=gts-nimam6-paid
#SBATCH --partition=cpu-small
#SBATCH -N1 --ntasks-per-node=4 --mem=16G
#SBATCH -t 03:00:00
#SBATCH -o /path/to/pace-scratch/logs/b23_dl_%j.out
# B23: the BABILong TRAINING split (RMT-team/babilong-train-5k-samples) for qa1-qa5 at the lengths B18 did not fetch.
cd /path/to/pace-scratch/bench
python3 - <<'PY'
from huggingface_hub import snapshot_download
pats = [f"data/qa{t}/{l}.json" for t in range(1, 6) for l in ("1k", "2k", "8k", "32k", "64k")]
p = snapshot_download("RMT-team/babilong-train-5k-samples", repo_type="dataset",
                      local_dir="RMT-team__babilong-train-5k-samples", allow_patterns=pats)
print("ok", p, flush=True)
PY
ls -la RMT-team__babilong-train-5k-samples/data/qa*/
