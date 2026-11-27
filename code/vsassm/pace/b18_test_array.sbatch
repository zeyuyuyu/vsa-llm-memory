#!/bin/bash
#SBATCH -J b18_test
#SBATCH --account=gts-nimam6
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:h200:1
#SBATCH -N1 --ntasks-per-node=8 --mem=96G
#SBATCH -t 04:00:00
#SBATCH -o /path/to/pace-scratch/logs/b18_test_%A_%a.out
# B18 test (prereg/B18_prereg.md, amendments 1-4): the 1.7B student (seed $1) on RMT-team/babilong. One line of
# pace/b18_test_params.txt per array task: LENGTH TASKS N.
cd /path/to/pace-scratch/vsassm
export BENCH_ROOT=/path/to/pace-scratch/bench HF_HOME=/path/to/pace-scratch/hf HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
read L T N <<< "$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" pace/b18_test_params.txt)"
S17=$(ls -d $HF_HOME/hub/models--Qwen--Qwen3-1.7B/snapshots/* | head -1)
python3 -u b18.py --model $S17 --adapter b18_models/student_s$1 --compact --split test100 --n $N --bs 256 --budget 600000 \
  --lengths $L --tasks $T --out b18_test/student_s$1/${L}_$(echo $T | cut -d, -f1)_$SLURM_ARRAY_TASK_ID
