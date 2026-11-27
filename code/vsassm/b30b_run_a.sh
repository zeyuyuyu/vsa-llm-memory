#!/bin/bash
# B30b part (a) on the A800 box, in the registered order: G-E (impact, window; two shards each, then forcing), S32 R1
# (two jobs: location; impact + window), G-E location. Every GPU job goes through b30b_run.sh (claim, lock, ledger).
cd /path/to/work/b30b
B32=/data0/path/to/HOME/hf32/hub/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137
NS1=0,1,2,3,4,5,6,7,8,9,100,101,102,103,104,105,106,107,108,109
NS2=200,201,202,203,204,205,206,207,208,209,300,301,302,303,304,305,306,307,308,309
mkdir -p out/logs/ge/impact out/logs/ge/window out/logs/ge/where
ge() {  # <set> <task> <forcing tokens> <forced file>
  ./b30b_run.sh GE_$1_a 2 vllm b30b_ge.py b30b_frozen/nsr/nsr1_vllm.py --model $B32 --name qwen3-32b --task $2 \
    --problems out/logs/ge_problems_$1.jsonl --out out/logs/ge/$1 --ns $NS1 --tp 2 > logs/GE_$1_a.log 2>&1 &
  sleep 90
  ./b30b_run.sh GE_$1_b 2 vllm b30b_ge.py b30b_frozen/nsr/nsr1_vllm.py --model $B32 --name qwen3-32b --task $2 \
    --problems out/logs/ge_problems_$1.jsonl --out out/logs/ge/$1 --ns $NS2 --tp 2 > logs/GE_$1_b.log 2>&1 &
  wait
  ./b30b_run.sh GE_$1_f 2 vllm b30b_ge.py b30b_frozen/nsr/nsr1_vllm_force.py --model $B32 --name qwen3-32b \
    --problems out/logs/ge_problems_$1.jsonl --runs "out/logs/ge/$1/qwen3-32b_$2_n*.jsonl" \
    --out out/logs/ge/$1/$4 --tp 2 --max_tokens $3 > logs/GE_$1_f.log 2>&1
}
ge impact NSR1 512 forced512_qwen3-32b.jsonl &
sleep 200
ge window NSR3 512 forced512_qwen3-32b.jsonl &
wait
./b30b_run.sh S32_r1_iw 2 vllm b30b_gen.py --model $B32 --name qwen3-32b --mode think32k --prompts out/logs/prompts_test.jsonl \
  --only impact,window --out out/logs/s32_r1_iw.jsonl --tp 2 --seed 0 --every 20 > logs/S32_r1_iw.log 2>&1 &
sleep 120
./b30b_run.sh S32_r1_where 2 vllm b30b_gen.py --model $B32 --name qwen3-32b --mode think32k --prompts out/logs/prompts_test.jsonl \
  --only where --out out/logs/s32_r1_where.jsonl --tp 2 --seed 0 --every 20 > logs/S32_r1_where.log 2>&1 &
wait
ge where REAL 64 forced_qwen3-32b.jsonl
echo "part a done $(date -u +%FT%TZ)" >> logs/gpu_ledger_notes.txt
