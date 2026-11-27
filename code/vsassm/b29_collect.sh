#!/bin/bash
# B29: fetch finished outputs from the A800 box and turn them into per-sample / per-question files (local CPU).
# Idempotent; each step runs only when its input exists and its output is older than the input.
cd /path/to/vsassm
R=b29_out/ruler; L=b29_out/logs
timeout 900 rsync -a --exclude 'test_*prompts*' --exclude 'dev_*' cavin:/path/to/work/b29/b29_out/ruler/ $R/
timeout 900 rsync -a --exclude 'calls_dev*' --exclude 'reader_dev*' cavin:/path/to/work/b29/b29_out/logs/ $L/
timeout 120 rsync -a cavin:/path/to/work/b29/logs/gpu_ledger.txt b29_out/gpu_ledger.txt
done_job() { grep -q "^$1 .* end .*rc=0" b29_out/gpu_ledger.txt; }
newer() { [ ! -f "$2" ] || [ "$1" -nt "$2" ]; }
for m in q8 q32 q17; do
  case $m in q8) J=G1_r8;; q32) J=R32_rel;; q17) J=R17_rel;; esac
  if done_job $J && newer $R/test_rel_${m}_reads.jsonl $R/test_rel_$m.jsonl; then
    python3 -B b29_ruler.py assemble rel /path/to/HOME/bench_dl/ruler $R/test_rel_${m}_reads.jsonl $R/test_rel_$m.jsonl
  fi
done
# part1 was stopped after 80,000 prompts and its remaining 63,713 prompts were split over four GPUs (p1r00-03)
if done_job R32_full0 && done_job R32_full1r00 && done_job R32_full1r01 && done_job R32_full1r02 && done_job R32_full1r03 \
   && [ ! -f $R/test_full_q32.jsonl ]; then
  python3 -B b29_ruler.py assemble full /path/to/HOME/bench_dl/ruler \
    $R/test_full_q32_reads.part0.jsonl,$R/test_full_q32_reads.part1.jsonl,$R/test_full_q32_reads.p1r00.jsonl,$R/test_full_q32_reads.p1r01.jsonl,$R/test_full_q32_reads.p1r02.jsonl,$R/test_full_q32_reads.p1r03.jsonl \
    $R/test_full_q32.jsonl --lengths 131072
fi
for s in test2 nsr1 nsr2; do
  for t in ft17 q8 q32 gpt61low opus_low; do
    [ -f $L/reader_${s}_$t.jsonl.meta.json ] || continue
    if newer $L/reader_${s}_$t.jsonl $L/ans_${s}_$t.jsonl; then
      python3 -I -B b29_logs.py answer --split $s --tag $t
    fi
    if newer $L/calls_${s}_$t.jsonl $L/reader_${s}_${t}_dict.jsonl; then
      python3 -I -B b29_logs.py read --split $s --reader replay --calls $L/calls_${s}_$t.jsonl --memory dict --tag ${t}_dict \
        > $L/replay_${s}_$t.log 2>&1 && python3 -I -B b29_logs.py answer --split $s --tag ${t}_dict
    fi
  done
done
