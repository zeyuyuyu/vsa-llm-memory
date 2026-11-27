#!/bin/bash
#SBATCH -J b18_distill
#SBATCH --account=gts-nimam6
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:h200:1
#SBATCH -N1 --ntasks-per-node=8 --mem=96G
#SBATCH -t 10:00:00
#SBATCH -o /path/to/pace-scratch/logs/b18_distill_%j.out
# bAbI-original download (qa11-13), teacher labels on training stories, then the 1.7B student (seed 0).
cd /path/to/pace-scratch/vsassm
export BENCH_ROOT=/path/to/pace-scratch/bench HF_HOME=/path/to/pace-scratch/hf HF_HUB_OFFLINE=1
mkdir -p $BENCH_ROOT/babi b18_data b18_models
if [ ! -d $BENCH_ROOT/babi/tasks_1-20_v1-2 ]; then
  for u in https://s3.amazonaws.com/text-datasets/babi_tasks_1-20_v1-2.tar.gz https://dl.fbaipublicfiles.com/babi/tasks_1-20_v1-2.tar.gz; do
    curl -sSL -m 300 -o /tmp/babi.tgz $u && tar -xzf /tmp/babi.tgz -C $BENCH_ROOT/babi && break
  done
fi
ls $BENCH_ROOT/babi/tasks_1-20_v1-2/en-10k | head -3
S8=$(ls -d $HF_HOME/hub/models--Qwen--Qwen3-8B/snapshots/* | head -1)
S17=$(ls -d $HF_HOME/hub/models--Qwen--Qwen3-1.7B/snapshots/* | head -1)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python3 -u b18_distill.py --model $S8 --out b18_data/train.jsonl && \
python3 -u b16_train.py --model $S17 --data "b18_data/train.jsonl" --seed 0 --epochs 1 --holdout 1000 --out b18_models/student_s0
