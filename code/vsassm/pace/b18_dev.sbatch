#!/bin/bash
#SBATCH -J b18_dev
#SBATCH --account=gts-nimam6
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:h200:1
#SBATCH -N1 --ntasks-per-node=8 --mem=64G
#SBATCH -t 04:00:00
#SBATCH -o /path/to/pace-scratch/logs/b18_dev_%j.out
# B18 development (training split only): 8B teacher, prompt F, grounding check. Args: LENGTHS TASKS N OUTDIR
cd /path/to/pace-scratch/vsassm
export BENCH_ROOT=/path/to/pace-scratch/bench HF_HOME=/path/to/pace-scratch/hf HF_HUB_OFFLINE=1
S8=$(ls -d $HF_HOME/hub/models--Qwen--Qwen3-8B/snapshots/* | head -1)
nvidia-smi --query-gpu=name --format=csv,noheader
python3 -u b18.py --model $S8 --split train --lengths $1 --tasks $2 --n $3 --out $4
