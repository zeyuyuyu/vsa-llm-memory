#!/bin/bash
#SBATCH -J b19
#SBATCH --account=gts-nimam6
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:h200:1
#SBATCH -N1 --ntasks-per-node=8 --mem=120G
#SBATCH -t 08:00:00
#SBATCH -o /path/to/pace-scratch/logs/b19_%A_%a.out
# B19 (prereg/B19_ruler_prereg.md). Args: PARAMS_FILE. One line per array task: MODE ROOT LENGTH TASKS N OUTDIR [--yarn]
cd /path/to/pace-scratch/vsassm
export HF_HOME=/path/to/pace-scratch/hf HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
read MODE ROOT L T N OUT EXTRA <<< "$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" $1)"
S8=$(ls -d $HF_HOME/hub/models--Qwen--Qwen3-8B/snapshots/* | head -1)
python3 -u ruler_rm.py $MODE $OUT/${L}_$(echo $T | cut -d, -f1)_$SLURM_ARRAY_TASK_ID.jsonl --model $S8 \
  --root /path/to/pace-scratch/bench/$ROOT --length $L --tasks $T --n $N $EXTRA
