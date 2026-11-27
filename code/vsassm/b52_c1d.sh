#!/bin/bash
# B52 part (c) pipeline on the A800 box CPUs (prereg/B52_robustness_layers_maps_1day_prereg.md); B38 environment.
# usage (in /path/to/work/b52): nohup nice -n 10 bash b52_c1d.sh 24 > logs/c1d.log 2>&1 < /dev/null &
cd /path/to/work/b52 || exit 1
PY=/path/to/HOME/miniconda3/envs/dino_wm/bin/python
W=${1:-24}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export B38_OUT=/path/to/work/b38/b38_out B52C_OUT=out_c
echo "c1d start $(date -u +%FT%TZ) workers $W" >> logs/c1d_ledger.txt
for step in dev choose plan test extend closed verdict; do
  echo "== $step $(date -u +%FT%TZ)"
  $PY -u b52_b38_1d.py $step --workers $W || { echo "c1d FAILED at $step $(date -u +%FT%TZ)" >> logs/c1d_ledger.txt; exit 1; }
done
echo "c1d end $(date -u +%FT%TZ) workers $W" >> logs/c1d_ledger.txt
