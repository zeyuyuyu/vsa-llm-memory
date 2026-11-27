#!/bin/bash
# NSR1 held-out generation on the A800 box (PACE gpu-h200 estimated a week of queueing on 2026-11-07; user rule:
# A800 free GPUs when PACE queues, never GPU3). Qwen3-32B bf16 spread over two A800 80GB per process.
# Usage: bash nsr/a800/run_gen.sh "1,4" warehouse,biobank   ;   bash nsr/a800/run_gen.sh "6,7" home,ops
cd /path/to/work/vsassm
source ~/miniconda3/etc/profile.d/conda.sh && conda activate dino_wm
export HF_HOME=/data0/path/to/HOME/hf32 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4
SNAP=$(ls -d $HF_HOME/hub/models--Qwen--Qwen3-32B/snapshots/* | head -1)
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader
CUDA_VISIBLE_DEVICES=$1 python3 -u nsr/nsr1_gen.py --split ${3:-nsr1} --model $SNAP --bs 32 --scen $2
