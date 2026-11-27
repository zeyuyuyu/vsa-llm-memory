#!/bin/bash
#SBATCH -J b18_qa4dev
#SBATCH --account=gts-nimam6
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:h200:1
#SBATCH -N1 --ntasks-per-node=8 --mem=64G
#SBATCH -t 00:40:00
#SBATCH -o /path/to/pace-scratch/logs/b18_qa4dev_%j.out
cd /path/to/pace-scratch/vsassm
export BENCH_ROOT=/path/to/pace-scratch/bench HF_HOME=/path/to/pace-scratch/hf HF_HUB_OFFLINE=1
S17=$(ls -d $HF_HOME/hub/models--Qwen--Qwen3-1.7B/snapshots/* | head -1)
python3 -u b18.py --model $S17 --adapter b18_models/student_s0 --compact --split train --offset 300 --n 50 \
  --lengths 0k,4k,16k --tasks qa4 --out b18_dev/student_s0_qa4fix
