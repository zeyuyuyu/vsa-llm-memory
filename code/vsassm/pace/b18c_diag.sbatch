#!/bin/bash
#SBATCH -J b18c_diag
#SBATCH --account=gts-nimam6-paid
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:h200:1
#SBATCH -N1 --ntasks-per-node=8 --mem=64G
#SBATCH -t 00:40:00
#SBATCH -o /path/to/pace-scratch/logs/b18c_diag_%j.out
# B18c step 0: diagnose the B18b student on qa11 / qa13 bAbI TRAINING stories it was not distilled on (no test data).
cd /path/to/pace-scratch/vsassm
export BENCH_ROOT=/path/to/pace-scratch/bench HF_HOME=/path/to/pace-scratch/hf HF_HUB_OFFLINE=1
S17=$(ls -d $HF_HOME/hub/models--Qwen--Qwen3-1.7B/snapshots/* | head -1)
python3 -u b18c_diag.py --model $S17 --adapter b18_models/student_b18b_s0 --tasks 11,12,13 --start 1000 --n 200 \
  --out b18c_diag
