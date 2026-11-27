#!/bin/bash
#SBATCH -J real_rdr
#SBATCH --account=gts-nimam6-paid
#SBATCH -N1 --ntasks-per-node=4 --mem=64G
#SBATCH -o /path/to/pace-scratch/logs/real_rdr_%j.out
# Realistic scenarios: the shipped reader reads the human lines; connectors read the system feeds. $1 = split.
cd /path/to/pace-scratch/vsassm
export HF_HOME=/path/to/pace-scratch/hf HF_HUB_OFFLINE=1
S17=$(ls -d $HF_HOME/hub/models--Qwen--Qwen3-1.7B/snapshots/* | head -1)
python3 -u apps/real/run_real.py reader --model $S17 --adapter b16_models/student_s0 --split $1 ${@:2}
