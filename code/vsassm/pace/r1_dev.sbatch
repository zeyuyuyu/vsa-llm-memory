#!/bin/bash
#SBATCH -J r1_dev
#SBATCH --account=gts-nimam6
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:h200:1
#SBATCH -N1 --ntasks-per-node=8 --mem=64G
#SBATCH -t 05:00:00
#SBATCH -o /path/to/pace-scratch/logs/r1_dev_%j.out
# R1 development on the CLUTRR validation split: reader 8B and 1.7B (no thinking) + VSA rule memory.
cd /path/to/pace-scratch/vsassm
export HF_HOME=/path/to/pace-scratch/hf HF_HUB_OFFLINE=1
D=/path/to/pace-scratch/bench/CLUTRR__v1/gen_train23_test2to10
mkdir -p r1_out
for m in 8B 1.7B; do
  S=$(ls -d $HF_HOME/hub/models--Qwen--Qwen3-$m/snapshots/* | head -1)
  python3 -u r1.py --model $S --data $D --split validation --n 300 --vote --out r1_out/val_v4_$m.jsonl
  python3 -u r1.py --model $S --data $D --split validation --n 300 --vote --think --out r1_out/val_v4think_$m.jsonl
done
