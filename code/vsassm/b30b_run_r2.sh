#!/bin/bash
# B30b S32 R2 (seed 1), the registered secondary replicate; started while R1 runs because the ledger projection is far
# below the 55 GPU-hour condition (compute note 3). Two jobs, each claiming a free GPU pair when one appears.
cd /path/to/work/b30b
B32=/data0/path/to/HOME/hf32/hub/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137
./b30b_run.sh S32_r2_iw 2 vllm b30b_gen.py --model $B32 --name qwen3-32b --mode think32k --prompts out/logs/prompts_test.jsonl \
  --only impact,window --out out/logs/s32_r2_iw.jsonl --tp 2 --seed 1 --every 20 > logs/S32_r2_iw.log 2>&1 &
sleep 120
./b30b_run.sh S32_r2_where 2 vllm b30b_gen.py --model $B32 --name qwen3-32b --mode think32k --prompts out/logs/prompts_test.jsonl \
  --only where --out out/logs/s32_r2_where.jsonl --tp 2 --seed 1 --every 20 > logs/S32_r2_where.log 2>&1 &
wait
