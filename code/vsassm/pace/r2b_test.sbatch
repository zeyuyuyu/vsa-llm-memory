#!/bin/bash
#SBATCH -J r2b_test
#SBATCH --account=gts-nimam6-paid
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:h200:1
#SBATCH -N1 --ntasks-per-node=8 --mem=96G
#SBATCH -t 02:00:00
#SBATCH -o /path/to/pace-scratch/logs/r2b_test_%j.out
# R2b (prereg/R2b_prereg.md): train both arms on r2b_data, then evaluate both on validation and test.
cd /path/to/pace-scratch/vsassm
export HF_HOME=/path/to/pace-scratch/hf HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
S17=$(ls -d $HF_HOME/hub/models--Qwen--Qwen3-1.7B/snapshots/* | head -1)
# test only: the R2b models were trained and evaluated on validation in job 14352560 (timed out before test)
for SPLIT in test; do
  N=$([ $SPLIT = test ] && echo 1146 || echo 2020)
  python3 -u r1.py --model $S17 --adapter r2b_models/reader_s0 --data r1_data --split $SPLIT --n $N --vote \
    --out r2b_out/reader_$SPLIT.jsonl
  python3 -u r2_direct.py --model $S17 --adapter r2b_models/direct_s0 --data r1_data --split $SPLIT \
    --out r2b_out/direct_$SPLIT.jsonl
done
