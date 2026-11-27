#!/bin/bash
#SBATCH -J b18_student
#SBATCH --account=gts-nimam6
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:h200:1
#SBATCH -N1 --ntasks-per-node=8 --mem=96G
#SBATCH -t 04:00:00
#SBATCH -o /path/to/pace-scratch/logs/b18_student_%j.out
# qa4 teacher labels + book negatives + merge, then the 1.7B student (seed $1), then development on the training split
# (samples 300-349 of each task, never used for distillation).
cd /path/to/pace-scratch/vsassm
export BENCH_ROOT=/path/to/pace-scratch/bench HF_HOME=/path/to/pace-scratch/hf HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
S8=$(ls -d $HF_HOME/hub/models--Qwen--Qwen3-8B/snapshots/* | head -1)
S17=$(ls -d $HF_HOME/hub/models--Qwen--Qwen3-1.7B/snapshots/* | head -1)
if [ ! -f b18_data/train_all.jsonl ]; then
  [ -f b18_data/qa4.jsonl ] || python3 -u b18_distill.py --model $S8 --tasks qa4 --babi 0 --out b18_data/qa4.jsonl
  python3 -u b18_distill.py --model none --negs_only --out b18_data/negs.jsonl
  cat b18_data/a800_head.jsonl b18_data/shard_*.jsonl b18_data/qa4.jsonl b18_data/negs.jsonl > b18_data/train_all.jsonl
fi
wc -l b18_data/train_all.jsonl
python3 -u b16_train.py --model $S17 --data b18_data/train_all.jsonl --seed $1 --epochs 1 --holdout 1000 \
  --out b18_models/student_s$1
python3 -u b18.py --model $S17 --adapter b18_models/student_s$1 --compact --split train --offset 300 --n 50 \
  --lengths 0k,4k,16k --tasks qa1,qa2,qa3,qa4,qa5,qa6,qa7,qa8,qa9,qa10 --out b18_dev/student_s$1
