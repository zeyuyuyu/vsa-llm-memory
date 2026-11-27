#!/bin/bash
# B30b: fetch finished outputs from the A800 box (idempotent; read-only on the box). Reader reads stay on the box
# (only a gzipped copy is fetched); everything the verdict needs lands under b30b_out/.
cd /path/to/vsassm
CM="-o ControlMaster=auto -o ControlPath=/path/to/HOME/.b30b_cm/%C -o ControlPersist=3600 -o BatchMode=yes -o ConnectTimeout=30"
R=cavin:/path/to/work/b30b
timeout 900 rsync -a -e "ssh $CM" --exclude 'reads.jsonl' --exclude 'reader_prompts.jsonl' $R/out/bab/ b30b_out/bab/
timeout 900 rsync -a -e "ssh $CM" --exclude 'prompts_test.jsonl' --exclude 'ge_problems_*' $R/out/logs/ b30b_out/logs/
timeout 300 rsync -a -e "ssh $CM" $R/logs/gpu_ledger.txt b30b_out/gpu_ledger.txt
cat b30b_out/logs/s32_r1_iw.jsonl b30b_out/logs/s32_r1_where.jsonl > b30b_out/logs/s32_r1.jsonl 2>/dev/null
[ -f b30b_out/logs/s32_r2_iw.jsonl ] && cat b30b_out/logs/s32_r2_iw.jsonl b30b_out/logs/s32_r2_where.jsonl > b30b_out/logs/s32_r2.jsonl 2>/dev/null
wc -l b30b_out/logs/s32_r1.jsonl b30b_out/bab/ans_*.jsonl 2>/dev/null | tail -20
