#!/bin/bash
#SBATCH -J b18_label
#SBATCH --account=gts-nimam6
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:h200:1
#SBATCH -N1 --ntasks-per-node=8 --mem=64G
#SBATCH -t 01:30:00
#SBATCH --array=0-9
#SBATCH -o /path/to/pace-scratch/logs/b18_label_%A_%a.out
# Teacher labels for facts[8192:] in 10 shards (the A800 labels facts[:8192] and the book sentences).
cd /path/to/pace-scratch/vsassm
export BENCH_ROOT=/path/to/pace-scratch/bench HF_HOME=/path/to/pace-scratch/hf HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
S8=$(ls -d $HF_HOME/hub/models--Qwen--Qwen3-8B/snapshots/* | head -1)
mkdir -p b18_data
python3 -u b18_distill.py --model $S8 --start 8192 --nshards 10 --shard $SLURM_ARRAY_TASK_ID \
  --out b18_data/shard_$SLURM_ARRAY_TASK_ID.jsonl
