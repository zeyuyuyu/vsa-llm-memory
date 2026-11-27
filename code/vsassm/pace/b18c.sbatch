#!/bin/bash
#SBATCH -J b18c
#SBATCH --account=gts-nimam6-paid
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:h200:1
#SBATCH -N1 --ntasks-per-node=8 --mem=96G
#SBATCH -t 06:00:00
#SBATCH -o /path/to/pace-scratch/logs/b18c_%j.out
# B18c (prereg/B18c_prereg.md): qa11-13 training labels from the coreference rule, retrain the student (seed 0),
# test qa11-13 at 0k and qa1-qa10 at 0k / 4k.
set -e
cd /path/to/pace-scratch/vsassm
export BENCH_ROOT=/path/to/pace-scratch/bench HF_HOME=/path/to/pace-scratch/hf HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
S8=$(ls -d $HF_HOME/hub/models--Qwen--Qwen3-8B/snapshots/* | head -1)
S17=$(ls -d $HF_HOME/hub/models--Qwen--Qwen3-1.7B/snapshots/* | head -1)
python3 -u b18c_data.py b18_data/train_b18b.jsonl b18_data/qa111213_v2.jsonl b18_data/train_b18c.jsonl
[ -f b18_models/student_b18c_s0/adapter_config.json ] || python3 -u b16_train.py --model $S17 \
  --data b18_data/train_b18c.jsonl --seed 0 --epochs 1 --holdout 1000 --out b18_models/student_b18c_s0
python3 -u b18.py --model $S17 --adapter b18_models/student_b18c_s0 --compact --split test100 --n 100 --bs 256 \
  --budget 600000 --lengths 0k --tasks qa11,qa12,qa13,qa1,qa2,qa3,qa4,qa5,qa6,qa7,qa8,qa9,qa10 --out b18c_test/0k
python3 -u b18.py --model $S17 --adapter b18_models/student_b18c_s0 --compact --split test100 --n 100 --bs 256 \
  --budget 600000 --lengths 4k --tasks qa1,qa2,qa3,qa4,qa5,qa6,qa7,qa8,qa9,qa10 --out b18c_test/4k
python3 rescore_b18.py b18c_test && python3 b18_aggregate.py b18c_test
python3 -u b18c_diag.py --model $S17 --adapter b18_models/student_b18c_s0 --tasks 11,12,13 --start 1000 --n 200 --out b18c_diag_after
