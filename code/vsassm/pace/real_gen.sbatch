#!/bin/bash
#SBATCH -J real_gen
#SBATCH --account=gts-nimam6-paid
#SBATCH -N1 --ntasks-per-node=4 --mem=96G
#SBATCH -o /path/to/pace-scratch/logs/real_gen_%j.out
# Realistic application scenarios: Qwen3-32B writes the human lines (shift chat, emails, speech, incident channel)
# and a second greedy pass checks each one against the intended state changes. $1 = split, $2 = batch size.
cd /path/to/pace-scratch/vsassm
export HF_HOME=/path/to/pace-scratch/hf HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SNAP=$(ls -d $HF_HOME/hub/models--Qwen--Qwen3-32B/snapshots/* | head -1)
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python3 -u apps/real/gen.py --split $1 --model $SNAP --bs ${2:-32} ${@:3}
