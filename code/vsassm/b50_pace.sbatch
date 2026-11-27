#!/bin/bash
# B50 on PACE (fallback when the A800 box has no idle GPU): runs b50_run.py commands in order, one per argument
# (each a quoted command line). Same code (md5-checked against git), same model file set as B40 (sha256-checked by
# b50_pace_dl.sbatch). A command that fails stops the job (so a failed smoke never lets a test run start).
# usage: sbatch -J b50_<name> b50_pace.sbatch "eval --task N ..." ["eval ..." ...]
#SBATCH -A gts-nimam6
#SBATCH -p gpu-l40s,gpu-a100,gpu-h100,gpu-h200
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 3:30:00
#SBATCH -o /path/to/pace-scratch/b50/logs/%x_%j.out
cd /path/to/pace-scratch/b50 || exit 1
export HF_HOME=/path/to/pace-scratch/hf HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export B50_ROOT=/path/to/pace-scratch/b50 B40_ROOT=/path/to/pace-scratch/b50
export B40_MODEL=/path/to/pace-scratch/b50/llama31_8b_instruct
echo "B50-PACE start $(date -u +%FT%TZ) host $(hostname) job $SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
md5sum b40_core.py b40_data.py b40_run.py b50_run.py
for c in "$@"; do
  echo "[$(date -u +%FT%TZ)] RUN $c"
  /usr/bin/python3 -u b50_run.py $c || { echo "FAILED rc=$? $c"; exit 1; }
  echo "[$(date -u +%FT%TZ)] DONE $c"
done
echo "B50-PACE end $(date -u +%FT%TZ)"
