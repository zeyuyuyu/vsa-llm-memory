#!/bin/bash
# B63 on the A800 box's CPUs (prereg/B63_online_directory_prereg.md, 'Compute'): nice 10, no GPU, one BLAS thread per
# worker, at most P workers (P <= 64). In ~/b63_run (code and stores MD5-checked against the local commit):
#   nohup bash b63_box.sh dev1 > logs/dev1.out 2>&1 < /dev/null &
#   nohup bash b63_box.sh dev2 > logs/dev2.out 2>&1 < /dev/null &
#   nohup bash b63_box.sh test PLAN.json OUT.json > logs/test_X.out 2>&1 < /dev/null &   # only after the choices are committed
cd ~/b63_run || exit 1
PY=/path/to/HOME/miniconda3/envs/dino_wm/bin/python
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1
mkdir -p logs results
ts() { date '+%F %T'; }
P=${B63_P:-60}
case $1 in
md5)
  md5sum b63_online.py b38_eq.py b34_inmem.py b25_pcm.py ruler_vsa.py ruler_rm.py b22_robust.py b26b_pcm_table.py \
    b34_out/dev_stores.json.gz b34_out/test_stores.json.gz ;;
selftest)
  nice -n 10 $PY -u b63_online.py selftest ;;
dev1)
  echo "$(ts) dev1 start" >> logs/steps.log
  nice -n 10 $PY -u b63_online.py dev b34_out/dev_stores.json.gz results/b63_dev1.json --workers $P
  echo "$(ts) DEV1_DONE rc=$?" >> logs/steps.log ;;
dev2)
  echo "$(ts) dev2 start" >> logs/steps.log
  nice -n 10 $PY -u b63_online.py dev2 b34_out/dev_stores.json.gz results/b63_dev1.json results/b63_dev2.json --workers $P
  echo "$(ts) DEV2_DONE rc=$?" >> logs/steps.log ;;
test)
  [ -f "$2" ] || { echo "missing plan $2"; exit 1; }
  echo "$(ts) test start $2 -> $3" >> logs/steps.log
  nice -n 10 $PY -u b63_online.py run b34_out/test_stores.json.gz "$2" "$3" --workers $P
  echo "$(ts) TEST_DONE $3 rc=$?" >> logs/steps.log ;;
esac
