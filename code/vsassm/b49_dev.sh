#!/bin/bash
# B49 panel, dev phase for one new encoder (prereg/B49_panel_mechanism_competitors_prereg.md, 'Panel'): every step of
# B32's and B36's dev protocol through b49_run.py (B32 / B36 code unchanged). Order (no grid, rule or seed depends on
# it): all OPQ / PQ fits, the B36 PQ codebooks and the VSA grid at once; the B36 families' grid when the codebooks are
# done; the index grids (4-64 and 1-2 bits per fact) when every fit is cached. Dev split only; no test store is built.
# usage: b49_dev.sh ENC [WORKERS]    (after: python b49_run.py ENC setup CODES.npy)
# environment: B49_ROOT (repository copy; default /path/to/vsassm), B49_PY (python; default python3 with B32's
# site directory on PYTHONPATH), B49_NICE (default 10).
ENC=$1; WK=${2:-3}
ROOT=${B49_ROOT:-/path/to/vsassm}
cd $ROOT || exit 1
PY=${B49_PY:-python3}
[ -z "$B49_PY" ] && export PYTHONPATH=/path/to/HOME/venvs/b32_site
export PYTHONDONTWRITEBYTECODE=1
W=b49_out/enc_$ENC; L=$W/logs; mkdir -p $L
ts() { date '+%F %T'; }
R="nice -n ${B49_NICE:-10} $PY b49_run.py $ENC"
running() { pgrep -u $(whoami) -f "$1" > /dev/null; }
echo "$(ts) dev start $ENC ($(hostname))" >> $L/steps.log
# every PQ / OPQ size a budget can reach is fitted here, so that no two later processes write the same cache file
OPQL=$($PY b49_run.py $ENC fitlist opq8); PQ4L=$($PY b49_run.py $ENC fitlist pq4); PQ8L=$($PY b49_run.py $ENC fitlist pq8)
echo "$(ts) fit lists opq8 $OPQL pq4 $PQ4L pq8 $PQ8L" >> $L/steps.log
for prep in center whiten; do
  for p in ${OPQL//,/ }; do
    [ -f $W/b32_out/fitcache/${ENC}_${prep}_opq8_$p.faiss ] && continue
    B32_FAISS_THREADS=1 OMP_NUM_THREADS=1 $R fit $prep opq8 $p > $L/fit_${prep}_opq8_$p.log 2>&1 &
  done
  ( B32_FAISS_THREADS=2 $R fit $prep pq4 $PQ4L
    B32_FAISS_THREADS=2 $R fit $prep pq8 $PQ8L ) > $L/fit_${prep}_pq.log 2>&1 &
done
OMP_NUM_THREADS=4 $R pqfit > $L/pqfit.log 2>&1
echo "$(ts) pqfit rc=$?" >> $L/steps.log
OMP_NUM_THREADS=2 $R dev --workers $WK > $L/dev.log 2>&1 &
P3=$!
OMP_NUM_THREADS=4 $R b32dev vsa > $L/b32dev_vsa.log 2>&1 &
P0=$!
while running "b49_run.py $ENC fit "; do sleep 20; done
echo "$(ts) fits done ($(ls $W/b32_out/fitcache | wc -l) cached fits)" >> $L/steps.log
OMP_NUM_THREADS=4 $R b32dev idx > $L/b32dev_idx.log 2>&1 &
P1=$!
OMP_NUM_THREADS=2 $R dev32 > $L/dev32.log 2>&1
echo "$(ts) dev32 rc=$?" >> $L/steps.log
wait $P1; echo "$(ts) b32dev idx rc=$?" >> $L/steps.log
wait $P0; echo "$(ts) b32dev vsa rc=$?" >> $L/steps.log
wait $P3; echo "$(ts) dev (key-free families) rc=$?" >> $L/steps.log
$R b32choose > $L/b32choose.log 2>&1; echo "$(ts) b32choose rc=$?" >> $L/steps.log
$R choose > $L/choose.log 2>&1; echo "$(ts) choose rc=$?" >> $L/steps.log
echo "$(ts) DEV_DONE" >> $L/steps.log
