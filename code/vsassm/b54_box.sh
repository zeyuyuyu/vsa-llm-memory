#!/bin/bash
# B54 stages on the A800 box's CPUs (no GPU), nice 10. On the box, in /path/to/work/b54:
#   nohup bash b54_box.sh STAGE [WORKERS] > logs/STAGE.log 2>&1 < /dev/null &
cd /path/to/work/b54
PY=/path/to/HOME/miniconda3/envs/dino_wm/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export B54_CAL=b54_out/b54_channel.npz
W=${2:-96}
O=b54_out
case "$1" in
  dev1)    nice -n 10 $PY b54_mlc.py dev1 $O/dev_stores.json.gz $O/b54_dev1.json --workers $W ;;
  dev2)    nice -n 10 $PY b54_mlc.py dev2 $O/dev_stores.json.gz $O/b54_dev1.json $O/b54_dev2.json --workers $W ;;
  test)    nice -n 10 $PY b54_mlc.py run $O/test_stores.json.gz $O/b54_plan_test.json $O/b54_test.json --workers $W ;;
  test-x)  # $3 = pass number, $4 = comma list of finished test files
           nice -n 10 $PY b54_mlc.py extend $O/test_stores.json.gz $O/b54_choice.json $4 $O/b54_plan_test_x$3.json && \
           nice -n 10 $PY b54_mlc.py run $O/test_stores.json.gz $O/b54_plan_test_x$3.json $O/b54_test_x$3.json --workers $W ;;
  *)       nice -n 10 $PY b54_mlc.py "${@:3}" ;;
esac
echo "exit $? $(date)"
