#!/bin/bash
#SBATCH -J r2
#SBATCH --account=gts-nimam6-paid
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:h200:1
#SBATCH -N1 --ntasks-per-node=8 --mem=96G
#SBATCH -t 03:00:00
#SBATCH -o /path/to/pace-scratch/logs/r2_%j.out
# R2 (prereg/R2_prereg.md). $1 = "train" (train both arms, evaluate on validation) or "test" (both arms on test).
cd /path/to/pace-scratch/vsassm
export HF_HOME=/path/to/pace-scratch/hf HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
S17=$(ls -d $HF_HOME/hub/models--Qwen--Qwen3-1.7B/snapshots/* | head -1)
if [ "$1" = "train" ]; then
  python3 -u r2_data.py r1_data r2_data
  python3 -u b16_train.py --model $S17 --data r2_data/reader.jsonl --seed 0 --epochs 2 --holdout 300 --out r2_models/reader_s0
  python3 -u b16_train.py --model $S17 --data r2_data/direct.jsonl --seed 0 --epochs 2 --holdout 300 --out r2_models/direct_s0
  SPLIT=validation; N=2020
else
  SPLIT=test; N=1146
fi
python3 -u r1.py --model $S17 --adapter r2_models/reader_s0 --data r1_data --split $SPLIT --n $N --vote \
  --out r2_out/reader_$SPLIT.jsonl
python3 -u r2_direct.py --model $S17 --adapter r2_models/direct_s0 --data r1_data --split $SPLIT \
  --out r2_out/direct_$SPLIT.jsonl
