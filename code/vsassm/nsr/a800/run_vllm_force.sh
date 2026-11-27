#!/bin/bash
# NSR1 budget forcing with vLLM on the A800 box. Usage: bash nsr/a800/run_vllm_force.sh <gpus> <model> <problems> <runs_dir> [tp]
cd /path/to/work/vsassm
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 VLLM_WORKER_MULTIPROC_METHOD=spawn
case $2 in
  qwen3-32b) SNAP=$(ls -d /data0/path/to/HOME/hf32/hub/models--Qwen--Qwen3-32B/snapshots/* | head -1) ;;
  qwen3-8b) SNAP=$(ls -d /path/to/work/hf/hub/models--Qwen--Qwen3-8B/snapshots/* | head -1) ;;
esac
CUDA_VISIBLE_DEVICES=$1 ~/miniconda3/envs/rsi-vllm11/bin/python -u nsr/nsr1_vllm_force.py --model $SNAP --name $2 \
  --problems $3 --runs "$4/$2_NSR1_n*.jsonl" --out $4/forced${6:-}_$2.jsonl --tp ${5:-2} --max_tokens ${7:-64}
