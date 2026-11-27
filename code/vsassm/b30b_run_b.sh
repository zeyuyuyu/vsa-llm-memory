#!/bin/bash
# B30b part (b) on the A800 box: reader prompts (CPU), one reader pass over the distinct prompts of all three lengths,
# replay + gate + notepads per length (CPU), then the six answerer arms per length (128k first).
cd /path/to/work/b30b
PY=/path/to/HOME/miniconda3/envs/dino_wm/bin/python
export BENCH_ROOT=/path/to/work/bench HF_HUB_OFFLINE=1
B8=/path/to/work/hf/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
B17=/path/to/work/hf/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e
RD=models/reader_s0_merged
mkdir -p out/bab
[ -f out/bab/reader_prompts.jsonl ] || $PY -I -B b30b_bab.py prompts 128k,512k,1M out/bab/reader_prompts.jsonl > logs/BR_prompts.log 2>&1
./b30b_run.sh BR_read 1 vllm b30b_gen.py --model $RD --name reader --mode reader --prompts out/bab/reader_prompts.jsonl \
  --out out/bab/reads.jsonl --max_model_len 40960 --gpu_util 0.85 --inflight 4096 --every 20000 > logs/BR_read.log 2>&1
for L in 128k 512k 1M; do
  $PY -I -B b30b_bab.py assemble $L out/bab/reads.jsonl out/bab/assembled_$L.jsonl > logs/BR_assemble_$L.log 2>&1
  $PY -I -B b30b_bab.py gate $L out/bab/assembled_$L.jsonl ref/b18_s0 > out/bab/gate_$L.txt 2>&1
  $PY -I -B b30b_bab.py notepad out/bab/assembled_$L.jsonl out/bab/notepad_$L.jsonl > logs/BR_notepad_$L.log 2>&1
done
for L in 128k 512k 1M; do
  for arm in q8T q17T rdT q8N q17N rdN; do
    case $arm in q8*) M=$B8; N=qwen3-8b;; q17*) M=$B17; N=qwen3-1.7b;; rd*) M=$RD; N=reader;; esac
    case $arm in *T) MODE="--mode think32k --max_model_len 0";; *N) MODE="--mode nothink --max_new 64 --max_model_len 0";; esac
    ./b30b_run.sh NP_${L}_$arm 1 vllm b30b_gen.py --model $M --name $N $MODE --prompts out/bab/notepad_$L.jsonl \
      --out out/bab/ans_${L}_$arm.jsonl --gpu_util 0.90 --every 100 > logs/NP_${L}_$arm.log 2>&1
  done
done
echo "part b done $(date -u +%FT%TZ)" >> logs/gpu_ledger_notes.txt
