#!/bin/bash
# B30b: G-E location (vLLM greedy 8,192, 64-token forcing as published) on GPUs left free while S32 R1 runs
# (compute note 2). b30b_run_a2.sh reaches the same step only after S32 R1 and then finds every problem done.
cd /path/to/work/b30b
B32=/data0/path/to/HOME/hf32/hub/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137
NS1=0,1,2,3,4,5,6,7,8,9,100,101,102,103,104,105,106,107,108,109
NS2=200,201,202,203,204,205,206,207,208,209,300,301,302,303,304,305,306,307,308,309
mkdir -p out/logs/ge/where
for x in "a $NS1" "b $NS2"; do set -- $x
  ./b30b_run.sh GE_where_$1 2 vllm b30b_ge.py b30b_frozen/nsr/nsr1_vllm.py --model $B32 --name qwen3-32b --task REAL \
    --problems out/logs/ge_problems_where.jsonl --out out/logs/ge/where --ns $2 --tp 2 > logs/GE_where_$1.log 2>&1 &
  sleep 90
done
wait
./b30b_run.sh GE_where_f 2 vllm b30b_ge.py b30b_frozen/nsr/nsr1_vllm_force.py --model $B32 --name qwen3-32b \
  --problems out/logs/ge_problems_where.jsonl --runs "out/logs/ge/where/qwen3-32b_REAL_n*.jsonl" \
  --out out/logs/ge/where/forced_qwen3-32b.jsonl --tp 2 --max_tokens 64 > logs/GE_where_f.log 2>&1
