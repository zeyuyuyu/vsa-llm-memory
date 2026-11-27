#!/bin/bash
#SBATCH -J b23_prep
#SBATCH --account=gts-nimam6-paid
#SBATCH --partition=cpu-small
#SBATCH -N1 --ntasks-per-node=4 --mem=48G
#SBATCH -t 02:00:00
#SBATCH -o /path/to/pace-scratch/logs/b23_prep_%j.out
# B23: training + dev examples from the BABILong training split (no test data), label sets, token counts.
cd /path/to/pace-scratch/vsassm
export BENCH_ROOT=/path/to/pace-scratch/bench HF_HOME=/path/to/pace-scratch/hf HF_HUB_OFFLINE=1
S17=$(ls -d $HF_HOME/hub/models--Qwen--Qwen3-1.7B/snapshots/* | head -1)
python3 -u b23.py prep --model $S17 --out b23_data
