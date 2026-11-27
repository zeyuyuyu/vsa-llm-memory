#!/bin/bash
#SBATCH -J nsr1_rdr
#SBATCH --account=gts-nimam6
#SBATCH -p gpu-h200 --gres=gpu:h200:1 -t 2:00:00
#SBATCH -N1 --ntasks-per-node=4 --mem=64G
#SBATCH -o /path/to/pace-scratch/logs/nsr1_rdr_%j.out
# NSR1: the shipped reader (Qwen3-1.7B + b16_models/student_s0, --verify, v4) as committed at 707351b. $1 = split.
cd /path/to/pace-scratch/vsassm
export HF_HOME=/path/to/pace-scratch/hf HF_HUB_OFFLINE=1
S17=$(ls -d $HF_HOME/hub/models--Qwen--Qwen3-1.7B/snapshots/* | head -1)
python3 -u nsr/nsr1_reader.py reader --model $S17 --adapter b16_models/student_s0 --split $1 --verify --tag _v4
