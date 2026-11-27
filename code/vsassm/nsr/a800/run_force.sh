#!/bin/bash
# NSR1 budget forcing on the A800 box (apps/real/force_answer.py as committed at 707351b).
# Usage: bash nsr/a800/run_force.sh <gpus> <qwen3-32b|qwen3-8b> <problems.jsonl> <out_dir>
cd /path/to/work/vsassm
source ~/miniconda3/etc/profile.d/conda.sh && conda activate dino_wm
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4
case $2 in
  qwen3-32b) SNAP=$(ls -d /data0/path/to/HOME/hf32/hub/models--Qwen--Qwen3-32B/snapshots/* | head -1) ;;
  qwen3-8b) SNAP=$(ls -d /path/to/work/hf/hub/models--Qwen--Qwen3-8B/snapshots/* | head -1) ;;
esac
CUDA_VISIBLE_DEVICES=$1 python3 -u nsr/vendor_707351b/force_answer.py --model $SNAP --name $2 --problems $3 \
  --runs "$4/$2_NSR1_n*.jsonl" --out $4/forced_$2.jsonl --bs 4
