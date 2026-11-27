#!/bin/bash
#SBATCH -J nsr1_gen
#SBATCH --account=gts-nimam6
#SBATCH -p gpu-h200 --gres=gpu:h200:1 -t 4:00:00
#SBATCH -N1 --ntasks-per-node=4 --mem=96G
#SBATCH -o /path/to/pace-scratch/logs/nsr1_gen_%j.out
# NSR1 held-out logs (prereg/NSR1_prereg.md): Qwen3-32B writes and checks the human lines; seeds 400+, 10 per scenario.
# Free tier first (gts-nimam6); switch to gts-nimam6-paid only if the free balance cannot cover it, and report the spend.
cd /path/to/pace-scratch/vsassm
export HF_HOME=/path/to/pace-scratch/hf HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SNAP=$(ls -d $HF_HOME/hub/models--Qwen--Qwen3-32B/snapshots/* | head -1)
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python3 -u nsr/nsr1_gen.py --split nsr1 --model $SNAP --bs ${1:-32} ${@:2}   # e.g. --scen ops (one shard per scenario)
