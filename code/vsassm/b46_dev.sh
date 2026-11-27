#!/bin/bash
# B46 dev phase on the local CPU (prereg/B46_second_llm_family_prereg.md): every step of B32's and B36's dev protocol on
# the B46 codes, through b46_run.py (B32 / B36 code unchanged). Logs to b46_out/logs/. Dev split only; no test store is
# built here.
# usage: b46_dev.sh            (after: python3 b46_run.py setup b46_out/b32_data/emb_b46_*.npy)
cd /path/to/vsassm || exit 1
export PYTHONPATH=/path/to/HOME/venvs/b32_site PYTHONDONTWRITEBYTECODE=1
L=b46_out/logs; mkdir -p $L
ts() { date '+%F %T'; }
echo "$(ts) dev start" >> $L/steps.log
# 1. fits that depend only on the training split: OPQ (B32 compute note 2: one process per fit), B36 PQ codebooks
for prep in center whiten; do
  for p in 1 2 4 8 16 32 64; do
    B32_FAISS_THREADS=1 OMP_NUM_THREADS=1 python3 b46_run.py fit $prep opq8 $p > $L/fit_${prep}_opq8_$p.log 2>&1 &
  done
done
for prep in center whiten; do  # PQ fits too, so that no two later processes write the same cache file at once
  ( B32_FAISS_THREADS=2 python3 b46_run.py fit $prep pq4 1,2,4,8,16,32,64,128
    B32_FAISS_THREADS=2 python3 b46_run.py fit $prep pq8 1,2,4,8,16,32,64 ) > $L/fit_${prep}_pq.log 2>&1 &
done
OMP_NUM_THREADS=4 python3 b46_run.py pqfit > $L/pqfit.log 2>&1 &
# 2. the B32 VSA dev grid needs no fit
OMP_NUM_THREADS=4 python3 b46_run.py b32dev vsa > $L/b32dev_vsa.log 2>&1
echo "$(ts) b32dev vsa rc=$?" >> $L/steps.log
wait
echo "$(ts) fits done" >> $L/steps.log
# 3. index dev grids (4-64 and 1-2 bits per fact) and the B36 families, in parallel
OMP_NUM_THREADS=4 python3 b46_run.py b32dev idx > $L/b32dev_idx.log 2>&1 &
P1=$!
OMP_NUM_THREADS=2 python3 b46_run.py dev32 > $L/dev32.log 2>&1 &
P2=$!
OMP_NUM_THREADS=2 python3 b46_run.py dev --workers 4 > $L/dev.log 2>&1 &
P3=$!
wait $P1; echo "$(ts) b32dev idx rc=$?" >> $L/steps.log
wait $P2; echo "$(ts) dev32 rc=$?" >> $L/steps.log
wait $P3; echo "$(ts) dev (new families) rc=$?" >> $L/steps.log
# 4. choices (dev only)
python3 b46_run.py b32choose > $L/b32choose.log 2>&1; echo "$(ts) b32choose rc=$?" >> $L/steps.log
python3 b46_run.py choose > $L/choose.log 2>&1; echo "$(ts) choose rc=$?" >> $L/steps.log
echo "$(ts) DEV_DONE" >> $L/steps.log
