#!/bin/bash
# NSR1 LLM arms on the A800 box, apps/real protocol (b6_qwen.py: thinking on, greedy, 8,192 new tokens).
# Usage: bash nsr/a800/run_llm.sh <gpus> <qwen3-32b|qwen3-8b> <problems.jsonl> <out_dir> <ns, comma-separated> [bs]
cd /path/to/work/vsassm
source ~/miniconda3/etc/profile.d/conda.sh && conda activate dino_wm
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4
case $2 in
  qwen3-32b) SNAP=$(ls -d /data0/path/to/HOME/hf32/hub/models--Qwen--Qwen3-32B/snapshots/* | head -1) ;;
  qwen3-8b) SNAP=$(ls -d /path/to/work/hf/hub/models--Qwen--Qwen3-8B/snapshots/* | head -1) ;;
esac
mkdir -p $4
CUDA_VISIBLE_DEVICES=$1 python3 -u nsr/nsr1_qwen.py --model $SNAP --name $2 --task NSR1 --problems $3 --out $4 --bs ${6:-8} \
  --max_new 8192 --ns $5
