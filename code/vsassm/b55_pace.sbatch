#!/bin/bash
# B55 on PACE (fallback while the A800 box has no idle GPU; paid account, allowed by the brief). Same code (md5 logged),
# same model file set as B40 (sha256-checked for B50, /path/to/pace-scratch/b50/llama31_8b_instruct). Runs the
# smoke on TRAINING rows, then (only if it succeeds and no STOP file exists) the PACE plan in 2 lanes on one GPU.
#SBATCH -A gts-nimam6-paid
#SBATCH -p gpu-h200,gpu-h100,gpu-a100,gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 3:00:00
#SBATCH -o /path/to/pace-scratch/b55/logs/%x_%j.out
B=/path/to/pace-scratch/b55; cd $B || exit 1
export HF_HOME=/path/to/pace-scratch/hf HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export B55_ROOT=$B B50_ROOT=$B B40_ROOT=$B B40_MODEL=/path/to/pace-scratch/b50/llama31_8b_instruct
echo "B55-PACE start $(date -u +%FT%TZ) host $(hostname) job $SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
md5sum b40_core.py b40_data.py b40_run.py b55_run.py
/usr/bin/python3 -u b55_run.py run --jobs jobs_pace.json --limit_split train:16 || { echo "SMOKE FAILED"; exit 1; }
[ -e $B/STOP ] && { echo "STOP file: no test run"; exit 0; }
echo "test start $(date -u +%FT%TZ)"
for i in 0 1; do /usr/bin/python3 -u b55_run.py run --jobs jobs_pace.json --lane $i --lanes 2 > logs/lane$i.log 2>&1 & done
wait
echo "B55-PACE end $(date -u +%FT%TZ)"
