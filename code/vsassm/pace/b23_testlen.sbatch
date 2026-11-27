#!/bin/bash
#SBATCH -J b23_testlen
#SBATCH --account=gts-nimam6-paid
#SBATCH --partition=cpu-small
#SBATCH -N1 --ntasks-per-node=4 --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o /path/to/pace-scratch/logs/b23_testlen_%j.out
# B23 window rule: prompt token counts of the test inputs (lengths only; no model, no labels).
cd /path/to/pace-scratch/vsassm
export BENCH_ROOT=/path/to/pace-scratch/bench HF_HOME=/path/to/pace-scratch/hf HF_HUB_OFFLINE=1
S17=$(ls -d $HF_HOME/hub/models--Qwen--Qwen3-1.7B/snapshots/* | head -1)
python3 -u b23.py testlen --model $S17 --out b23_data/testlen.json
