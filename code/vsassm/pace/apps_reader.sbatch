#!/bin/bash
#SBATCH -J apps_reader
#SBATCH --account=gts-nimam6-paid
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:h200:1
#SBATCH -N1 --ntasks-per-node=8 --mem=64G
#SBATCH -o /path/to/pace-scratch/logs/apps_reader_%j.out
# Application scenarios: the shipped reader (Qwen3-1.7B + distilled LoRA student) reads every log one line at a time
# into the readable memory. $1 = split (dev | test); further arguments are passed on (e.g. --registry).
cd /path/to/pace-scratch/vsassm
export HF_HOME=/path/to/pace-scratch/hf HF_HUB_OFFLINE=1
S17=$(ls -d $HF_HOME/hub/models--Qwen--Qwen3-1.7B/snapshots/* | head -1)
python3 -u apps/run.py reader --model $S17 --adapter b16_models/student_s0 --split $1 ${@:2}
