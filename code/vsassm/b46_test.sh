#!/bin/bash
# B46 test phase (prereg/B46_second_llm_family_prereg.md, 'Test'): run once, after the dev choices are committed.
# B32's test (VSA / index at 4-64 bits per fact), B36's test of the new families, B32's arms at 1-2 bits per fact,
# merges; then the verdict and the checks. Every fit is read from the caches written in the dev phase.
cd /path/to/vsassm || exit 1
export PYTHONPATH=/path/to/HOME/venvs/b32_site PYTHONDONTWRITEBYTECODE=1
L=b46_out/logs
ts() { date '+%F %T'; }
git diff --quiet HEAD -- b46_out/results/b32_choice_b46.json b46_out/results/b36_choice_b46.json || { echo "choices not committed"; exit 1; }
echo "$(ts) test start (HEAD $(git rev-parse --short HEAD))" >> $L/steps.log
OMP_NUM_THREADS=4 python3 b46_run.py b32test vsa > $L/b32test_vsa.log 2>&1 &
A=$!
OMP_NUM_THREADS=4 python3 b46_run.py b32test idx > $L/b32test_idx.log 2>&1 &
B=$!
OMP_NUM_THREADS=2 python3 b46_run.py test --workers 4 > $L/test.log 2>&1 &
C=$!
wait $A; echo "$(ts) b32test vsa rc=$?" >> $L/steps.log
wait $B; echo "$(ts) b32test idx rc=$?" >> $L/steps.log
python3 b46_run.py b32merge > $L/b32merge.log 2>&1; echo "$(ts) b32merge rc=$?" >> $L/steps.log
OMP_NUM_THREADS=4 python3 b46_run.py test32 > $L/test32.log 2>&1; echo "$(ts) test32 rc=$?" >> $L/steps.log
wait $C; echo "$(ts) test (new families) rc=$?" >> $L/steps.log
python3 b46_run.py merge > $L/merge.log 2>&1; echo "$(ts) merge rc=$?" >> $L/steps.log
OMP_NUM_THREADS=8 python3 b46_verdict.py > $L/verdict.log 2>&1; echo "$(ts) verdict rc=$?" >> $L/steps.log
echo "$(ts) TEST_DONE" >> $L/steps.log
