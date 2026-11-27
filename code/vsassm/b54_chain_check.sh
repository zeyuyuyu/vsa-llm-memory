#!/bin/bash
# B54 checks on the box after the test chain (b54_check.py)
cd /path/to/work/b54
PY=/path/to/HOME/miniconda3/envs/dino_wm/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 B54_CAL=b54_out/b54_channel.npz
until grep -q chain-test-done logs/test_files.txt 2>/dev/null; do sleep 30; done
F=$(head -1 logs/test_files.txt | sed 's/result files: //')
S=b54_out/test_stores.json.gz
nice -n 10 $PY b54_check.py device $S $F b54_out/b54_check_device.json --workers 60 > logs/check_device.log 2>&1 &
nice -n 10 $PY b54_check.py closed $S $F b54_out/b54_check_closed.json --workers 30 > logs/check_closed.log 2>&1 &
wait
nice -n 10 $PY b54_check.py second $S $F b54_out/b54_check_second.txt --workers 60 > logs/check_second.log 2>&1
nice -n 10 $PY b54_check.py handread $S $F b54_out/b54_handread.txt > logs/handread.log 2>&1
echo chain-check-done > logs/check_done.txt
