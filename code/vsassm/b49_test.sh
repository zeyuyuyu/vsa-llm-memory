#!/bin/bash
# B49 panel, test phase for one new encoder (prereg/B49_panel_mechanism_competitors_prereg.md, 'Test'): run once, after
# the encoder's dev choices are committed. B32's test (VSA / index at 4-64 bits per fact), B36's test of the key-free
# families, B32's arms at 1-2 bits per fact, merges (b46_test.sh's steps through b49_run.py).
# usage: b49_test.sh ENC
ENC=$1
cd /path/to/vsassm || exit 1
export PYTHONPATH=/path/to/HOME/venvs/b32_site PYTHONDONTWRITEBYTECODE=1
W=b49_out/enc_$ENC; L=$W/logs
ts() { date '+%F %T'; }
R="python3 b49_run.py $ENC"
git diff --quiet HEAD -- $W/results/b32_choice_$ENC.json $W/results/b36_choice_$ENC.json || { echo "choices not committed"; exit 1; }
git ls-files --error-unmatch $W/results/b36_choice_$ENC.json > /dev/null || { echo "choices not committed"; exit 1; }
echo "$(ts) test start $ENC (HEAD $(git rev-parse --short HEAD))" >> $L/steps.log
OMP_NUM_THREADS=3 $R b32test vsa > $L/b32test_vsa.log 2>&1 &
A=$!
OMP_NUM_THREADS=3 $R b32test idx > $L/b32test_idx.log 2>&1 &
B=$!
OMP_NUM_THREADS=2 $R test --workers 3 > $L/test.log 2>&1 &
C=$!
wait $A; echo "$(ts) b32test vsa rc=$?" >> $L/steps.log
wait $B; echo "$(ts) b32test idx rc=$?" >> $L/steps.log
$R b32merge > $L/b32merge.log 2>&1; echo "$(ts) b32merge rc=$?" >> $L/steps.log
OMP_NUM_THREADS=3 $R test32 > $L/test32.log 2>&1; echo "$(ts) test32 rc=$?" >> $L/steps.log
wait $C; echo "$(ts) test (key-free families) rc=$?" >> $L/steps.log
$R merge > $L/merge.log 2>&1; echo "$(ts) merge rc=$?" >> $L/steps.log
echo "$(ts) TEST_DONE" >> $L/steps.log
