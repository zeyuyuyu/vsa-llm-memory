#!/bin/bash
# Opus 5.5 queue (NSR5 amendment 6); each step resumes and stops at the shared USD 180 ledger.
cd /path/to/vsassm
while pgrep -f "nsr5_K8_claude-opus-5-5_medium[.]jsonl" >/dev/null; do sleep 30; done
R="python3 nsr/nsr_claude.py --model claude-opus-5-5 --sites 4 --inner 6"
$R --effort medium --problems nsr/out/held/nsr1_llm_problems_nsr1_arm3_interleaved.jsonl --out nsr/out/api/nsr1_claude-opus-5-5_medium.jsonl >> nsr/out/api/opus_queue.log 2>&1
$R --effort medium --problems nsr/out/nsr5_problems_K4.jsonl --out nsr/out/api/nsr5_K4_claude-opus-5-5_medium.jsonl >> nsr/out/api/opus_queue.log 2>&1
$R --effort medium --problems nsr/out/held/nsr3_llm_problems_nsr2.jsonl --out nsr/out/api/nsr3_claude-opus-5-5_medium.jsonl >> nsr/out/api/opus_queue.log 2>&1
# re-run any errored / cut calls once per file before the sensitivity runs
for P in "nsr/out/nsr5_problems_K8.jsonl nsr/out/api/nsr5_K8_claude-opus-5-5_medium.jsonl" "nsr/out/held/nsr1_llm_problems_nsr1_arm3_interleaved.jsonl nsr/out/api/nsr1_claude-opus-5-5_medium.jsonl" "nsr/out/nsr5_problems_K4.jsonl nsr/out/api/nsr5_K4_claude-opus-5-5_medium.jsonl" "nsr/out/held/nsr3_llm_problems_nsr2.jsonl nsr/out/api/nsr3_claude-opus-5-5_medium.jsonl"; do
  set -- $P; $R --effort medium --problems $1 --out $2 >> nsr/out/api/opus_queue.log 2>&1
done
$R --effort xhigh --problems nsr/out/nsr5_problems_K8.jsonl --out nsr/out/api/nsr5_K8_claude-opus-5-5_xhigh.jsonl >> nsr/out/api/opus_queue.log 2>&1
$R --effort xhigh --problems nsr/out/held/nsr1_llm_problems_nsr1_arm3_interleaved.jsonl --out nsr/out/api/nsr1_claude-opus-5-5_xhigh.jsonl >> nsr/out/api/opus_queue.log 2>&1
echo "QUEUE DONE" >> nsr/out/api/opus_queue.log
