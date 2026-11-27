#!/bin/bash
# B30a test run: one process per length, single-threaded (torch / BLAS oversubscription stalled the first attempt,
# which was stopped before any length finished; its partial files are overwritten here).
cd /path/to/vsassm
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for L in 0k 4k 16k 64k 128k 512k 1M 10M; do
  ( python3 -B b30a_floor.py test --train /path/to/HOME/bench_dl/babilong_hf/train --test /path/to/HOME/bench_dl/babilong_hf/test \
      --out b30a_out/test_$L --lengths $L --arms F1,F1m,F2 --all > b30a_out/test_$L.log 2>&1; echo "EXIT $L $?" >> b30a_out/test_$L.log ) &
done
wait
echo ALL_DONE
