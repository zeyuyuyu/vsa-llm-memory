#!/bin/bash
#SBATCH -J b23_part
#SBATCH --account=gts-nimam6-paid
#SBATCH -p gpu-h200,gpu-h100,gpu-a100,gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --constraint=H200|H100|A100|L40S
#SBATCH -N1 --ntasks-per-node=8 --mem=96G
#SBATCH -t 06:00:00
#SBATCH -o /path/to/pace-scratch/logs/b23_part_%j.out
# B23: part of a run's test in a separate output file, in parallel with the run's own job (same adapter, same eval
# code and window rule). Args: ARM SEED LENGTHS TASKS OUT
cd /path/to/pace-scratch/vsassm
export BENCH_ROOT=/path/to/pace-scratch/bench HF_HOME=/path/to/pace-scratch/hf HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
S17=$(ls -d $HF_HOME/hub/models--Qwen--Qwen3-1.7B/snapshots/* | head -1)
nvidia-smi --query-gpu=name --format=csv,noheader; hostname
python3 -u b23.py eval --arm $1 --model $S17 --adapter b23_models/${1}_s$2 --split test --data b23_data --lengths $3 --tasks $4 --out $5
echo B23 PART DONE
