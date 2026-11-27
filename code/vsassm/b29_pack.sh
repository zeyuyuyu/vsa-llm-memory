#!/bin/bash
# B29: gzip the per-sample / per-question outputs and reader logs for the commit (originals stay on disk, uncommitted).
cd /path/to/vsassm
R=b29_out/ruler; L=b29_out/logs
for f in $R/test_rel_q8.jsonl $R/test_rel_q32.jsonl $R/test_rel_q17.jsonl $R/test_full_q32.jsonl \
         $R/test_rel_q8_reads.jsonl $R/test_rel_q32_reads.jsonl $R/test_rel_q17_reads.jsonl \
         $R/test_full_q32_reads.part0.jsonl $R/test_full_q32_reads.part1.jsonl \
         $R/test_full_q32_reads.p1r00.jsonl $R/test_full_q32_reads.p1r01.jsonl $R/test_full_q32_reads.p1r02.jsonl \
         $R/test_full_q32_reads.p1r03.jsonl $R/dev_rel_q32.jsonl $R/dev_full_v32.jsonl $R/dev_asm_rel_q32.jsonl; do
  [ -f "$f" ] && { [ -f "$f.gz" ] && [ "$f.gz" -nt "$f" ] || gzip -kf9 "$f"; }
done
for f in $L/reader_*.jsonl $L/calls_*.jsonl $L/run1_*.jsonl; do
  case "$f" in *banked*) continue;; esac           # copies of banked files already in the repository
  [ -f "$f" ] && { [ -f "$f.gz" ] && [ "$f.gz" -nt "$f" ] || gzip -kf9 "$f"; }
done
ls $R/*.gz $L/*.gz | wc -l
du -ch $R/*.gz $L/*.gz | tail -1
