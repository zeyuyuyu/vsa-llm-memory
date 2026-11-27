#!/bin/bash
# B43 job on PACE Phoenix (prereg/B43_natural_text_and_chunked_baseline_prereg.md): runs ONE b43 command (the script's
# arguments) on one GPU with >= 48 GB (H200, H100, A100-80GB or L40S). Same code, same model snapshot
# (Qwen/Qwen3-8B b968826d), same arguments as on the A800 box. For b43_coa.py the batch size follows the card's memory
# (H200 50, 80 GB cards 25, 48 GB cards 10; 100 samples per job): decoding is memory-bound, so larger batches finish sooner; a batch that
# runs out of memory is split; batching changes greedy outputs only through numerics (B29 measured this code path).
# usage: sbatch -J b43_<name> b43_pace.sbatch b43_coa.py ruler out/<file>.jsonl --model $Q8 ...
#SBATCH -A gts-nimam6
#SBATCH -p gpu-h200,gpu-h100,gpu-a100,gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH -C H200|H100|A100-80GB|L40S
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 4:00:00
#SBATCH -o /path/to/pace-scratch/b43/logs/%x_%j.out
cd /path/to/pace-scratch/b43 || exit 1
export HF_HOME=/path/to/pace-scratch/hf HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BENCH_ROOT=/path/to/pace-scratch/bench
export PROPARA_DIR=/path/to/pace-scratch/b43/data/propara_dl PROPARA_LB=/path/to/pace-scratch/b43/data/propara_lb/propara
echo "B43-PACE start $(date -u +%FT%TZ) host $(hostname) job $SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
EXTRA=""
if [ "$1" = "b43_coa.py" ]; then
  if [ "$MEM" -gt 100000 ]; then EXTRA="--bs 50"; elif [ "$MEM" -gt 60000 ]; then EXTRA="--bs 25"; else EXTRA="--bs 10"; fi
fi
echo "extra args: $EXTRA"
/usr/bin/python3 -u "$@" $EXTRA
rc=$?
echo "B43-PACE end $(date -u +%FT%TZ) rc=$rc"
