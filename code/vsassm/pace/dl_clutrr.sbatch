#!/bin/bash
#SBATCH -J clutrr_dl
#SBATCH --account=gts-nimam6
#SBATCH --partition=cpu-small
#SBATCH -N1 --ntasks-per-node=2 --mem=8G
#SBATCH -t 01:00:00
#SBATCH -o /path/to/pace-scratch/logs/clutrr_dl_%j.out
mkdir -p /path/to/pace-scratch/bench && cd /path/to/pace-scratch/bench
python3 - <<'PY'
from huggingface_hub import snapshot_download, list_repo_files
fs = list_repo_files("CLUTRR/v1", repo_type="dataset")
print(len(fs), [f for f in fs if "gen_train23" in f][:10], flush=True)
p = snapshot_download("CLUTRR/v1", repo_type="dataset", local_dir="CLUTRR__v1")
print("ok", p)
PY
