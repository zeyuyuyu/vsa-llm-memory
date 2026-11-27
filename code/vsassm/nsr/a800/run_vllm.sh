#!/bin/bash
# NSR1 LLM arms with vLLM 0.11 on the A800 box (conda env rsi-vllm11). See nsr/nsr1_vllm.py for why.
# Usage: bash nsr/a800/run_vllm.sh <gpus> <qwen3-32b|qwen3-8b> <problems.jsonl> <out_dir> <ns> [tp]
cd /path/to/work/vsassm
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 VLLM_WORKER_MULTIPROC_METHOD=spawn
case $2 in
  qwen3-32b) SNAP=$(ls -d /data0/path/to/HOME/hf32/hub/models--Qwen--Qwen3-32B/snapshots/* | head -1) ;;
  qwen3-8b) SNAP=$(ls -d /path/to/work/hf/hub/models--Qwen--Qwen3-8B/snapshots/* | head -1) ;;
esac
CUDA_VISIBLE_DEVICES=$1 ~/miniconda3/envs/rsi-vllm11/bin/python -u nsr/nsr1_vllm.py --model $SNAP --name $2 \
  --problems $3 --out $4 --ns $5 --tp ${6:-2}
