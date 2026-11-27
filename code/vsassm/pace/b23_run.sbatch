#!/bin/bash
#SBATCH -J b23_run
#SBATCH --account=gts-nimam6-paid
#SBATCH -p gpu-h200,gpu-h100,gpu-a100,gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --constraint=H200|H100|A100|L40S
#SBATCH -N1 --ntasks-per-node=8 --mem=96G
#SBATCH -t 30:00:00
#SBATCH -o /path/to/pace-scratch/logs/b23_run_%j.out
# B23 (prereg/B23_babilong_finetune_baseline_prereg.md). Args: ARM (N|Y) SEED. One run: train -> dev -> bar-0 gate ->
# test. Every step resumes (training from its step-100 checkpoints, eval per sample), so a rerun continues.
cd /path/to/pace-scratch/vsassm
export BENCH_ROOT=/path/to/pace-scratch/bench HF_HOME=/path/to/pace-scratch/hf HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=true
ARM=$1; SEED=$2; M=b23_models/${ARM}_s$SEED
S17=$(ls -d $HF_HOME/hub/models--Qwen--Qwen3-1.7B/snapshots/* | head -1)
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; hostname; date
mkdir -p b23_models b23_out
if [ ! -f $M/b23_train.json ]; then
  python3 -u b23.py train --arm $ARM --seed $SEED --model $S17 --data b23_data --out $M || exit 1
fi
python3 -u b23.py eval --arm $ARM --model $S17 --adapter $M --split dev --data b23_data --out b23_out/dev_${ARM}_s$SEED.jsonl || exit 1
python3 -u b23.py gate --dev b23_out/dev_${ARM}_s$SEED.jsonl --labels b23_data/labels.json || { echo "BAR 0 FAILED: stopping before the test"; exit 3; }
python3 -u b23.py eval --arm $ARM --model $S17 --adapter $M --split test --data b23_data --out b23_out/test_${ARM}_s$SEED.jsonl || exit 1
date; echo B23 RUN DONE
