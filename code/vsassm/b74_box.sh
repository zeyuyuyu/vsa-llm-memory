#!/bin/bash
# B74 on the A800 box: runplan PLAN with W workers (CPU only, nice 5, one BLAS thread per process), log to LOG.
# ~/b74_run = ~/b72_run's code (md5-checked) + b57_run.py, b60_vsapc.py, b74_run.py; data via symlinks (b49_out, dat/).
# usage: bash b74_box.sh PLAN W LOG
W=$HOME/b74_run
PY=/path/to/HOME/miniconda3/envs/dino_wm/bin/python
cd $W || exit 1
export B53_DATA=$W/dat B60_DATA=$W/b49_out B64_DATA=$W/b49_out OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=
nohup nice -n 5 $PY b74_run.py runplan "$1" --workers "$2" > "$3" 2>&1 < /dev/null &
echo "PID $!"
