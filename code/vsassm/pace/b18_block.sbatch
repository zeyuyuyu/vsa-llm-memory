#!/bin/bash
#SBATCH -J b18_block
#SBATCH --account=gts-nimam6-paid
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:h200:1
#SBATCH -N1 --ntasks-per-node=8 --mem=96G
#SBATCH -o /path/to/pace-scratch/logs/b18_block_%A_%a.out
# B18 amendment 8: the student read one sample at a time in sentence blocks, with per-sample checkpoints.
# One line of $1 per array task: LENGTH TASK N OUTDIR. Time limit is set on the sbatch command line.
cd /path/to/pace-scratch/vsassm
export BENCH_ROOT=/path/to/pace-scratch/bench HF_HOME=/path/to/pace-scratch/hf HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
read L T N O <<< "$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" $1)"
S17=$(ls -d $HF_HOME/hub/models--Qwen--Qwen3-1.7B/snapshots/* | head -1)
python3 -u b18.py --model $S17 --adapter b18_models/student_s0 --compact --split test100 --n $N --bs 256 --budget 600000 \
  --block 2048 --lengths $L --tasks $T --out $O
