#!/bin/bash
#SBATCH -J b24
#SBATCH --account=gts-nimam6-paid
#SBATCH -p gpu-h200,gpu-h100,gpu-a100,gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH -N1 --ntasks-per-node=8 --mem=96G
#SBATCH -t 06:00:00
#SBATCH -o /path/to/pace-scratch/logs/b24_%A_%a.out
# B24 (prereg/B24_ruler_store_ablation_prereg.md). Args: PARAMS_FILE, one line per array task:
#   MODE LENGTH TASKS ROOT OUTDIR B21DIR      (MODE armf | armr; ROOT ruler | ruler_dev)
# The GPU constraint is given on the sbatch command line (80 GB cards for the long fact lists).
cd /path/to/pace-scratch/vsassm
export HF_HOME=/path/to/pace-scratch/hf HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
read MODE L T ROOT OUT B21 <<< "$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" $1)"
S8=$(ls -d $HF_HOME/hub/models--Qwen--Qwen3-8B/snapshots/* | head -1)
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; hostname; date
mkdir -p $OUT
python3 -u ruler_b24.py $MODE $OUT/${MODE}_${L}_$(echo $T | cut -d, -f1)_$SLURM_ARRAY_TASK_ID.jsonl --model $S8 \
  --root /path/to/pace-scratch/bench/$ROOT --length $L --tasks $T --b21 $B21 --n 100 || exit 1
date; echo B24 TASK DONE
