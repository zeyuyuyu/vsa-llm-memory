#!/bin/bash
# B46 dev phase, continued by hand after b46_dev.sh was stopped for memory (b46_out/logs/steps.log): same steps, same
# code; the whiten OPQ fits run after the centre fits; the B36 families' dev grid starts once the VSA grid and the B36 PQ
# codebooks are done; the index grids start after every fit. Dev split only.
cd /path/to/vsassm || exit 1
export PYTHONPATH=/path/to/HOME/venvs/b32_site PYTHONDONTWRITEBYTECODE=1
L=b46_out/logs
ts() { date '+%F %T'; }
running() { pgrep -u zeyu -f "$1" > /dev/null; }
while running "b46_run.py fit center opq8"; do sleep 20; done
echo "$(ts) centre OPQ fits done" >> $L/steps.log
for p in 1 2 4 8 16 32 64; do
  B32_FAISS_THREADS=1 OMP_NUM_THREADS=1 python3 b46_run.py fit whiten opq8 $p > $L/fit_whiten_opq8_$p.log 2>&1 &
done
while running "b46_run.py b32dev vsa" || running "b46_run.py pqfit"; do sleep 20; done
echo "$(ts) b32dev vsa and pqfit done (vsa lines: $(grep -c . b46_out/b32_out/dev_b46_vsa.jsonl))" >> $L/steps.log
OMP_NUM_THREADS=2 python3 b46_run.py dev --workers 4 > $L/dev.log 2>&1 &
P3=$!
while running "b46_run.py fit whiten"; do sleep 20; done
echo "$(ts) whiten fits done ($(ls b46_out/b32_out/fitcache | wc -l) cached fits)" >> $L/steps.log
OMP_NUM_THREADS=4 python3 b46_run.py b32dev idx > $L/b32dev_idx.log 2>&1
echo "$(ts) b32dev idx rc=$?" >> $L/steps.log
OMP_NUM_THREADS=4 python3 b46_run.py dev32 > $L/dev32.log 2>&1
echo "$(ts) dev32 rc=$?" >> $L/steps.log
wait $P3; echo "$(ts) dev (new families) rc=$?" >> $L/steps.log
python3 b46_run.py b32choose > $L/b32choose.log 2>&1; echo "$(ts) b32choose rc=$?" >> $L/steps.log
python3 b46_run.py choose > $L/choose.log 2>&1; echo "$(ts) choose rc=$?" >> $L/steps.log
echo "$(ts) DEV_DONE" >> $L/steps.log
