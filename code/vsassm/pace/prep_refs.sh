#!/bin/bash
# Build the reference problem files (login node: light CPU work only) and submit the reference jobs.
cd /path/to/pace-scratch/vsassm
export BENCH_ROOT=/path/to/pace-scratch/bench
D=$BENCH_ROOT/CLUTRR__v1/gen_train23_test2to10
T=qa1,qa2,qa3,qa5,qa6,qa7,qa8,qa9,qa10
mkdir -p refs
python3 b18_llm.py prep test100 4k $T 50 refs/b18_4k.jsonl
python3 b18_llm.py prep test100 16k $T 25 refs/b18_16k.jsonl
wc -l refs/*.jsonl
sbatch pace/llm_ref.sbatch 8B refs/b18_4k.jsonl qwen3-8b B18_4k 8 refs/out_b18
sbatch pace/llm_ref.sbatch 8B refs/b18_16k.jsonl qwen3-8b B18_16k 2 refs/out_b18
