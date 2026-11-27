#!/bin/bash
# B75 on the A800 box's CPUs (prereg/B75_strongest_options_prereg.md, 'Compute'): nice 10, no GPU, one BLAS / numba
# thread per worker, at most P <= 48 workers. In ~/b75_run (code and stores MD5-checked against the local commit):
#   nohup bash b75_box.sh dev  PLAN.json OUT.json [ONLY] > logs/dev_X.out 2>&1 < /dev/null &
#   nohup bash b75_box.sh test PLAN.json OUT.json [ONLY] > logs/test_X.out 2>&1 < /dev/null &   # only after the choices are committed
#   nohup bash b75_box.sh fresh PLAN.json OUT.json [ONLY] > logs/fresh_X.out 2>&1 < /dev/null &
cd ~/b75_run || exit 1
PY=/path/to/HOME/miniconda3/envs/dino_wm/bin/python
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1
mkdir -p logs results
ts() { date '+%F %T'; }
P=${B75_P:-48}
[ "$P" -le 48 ] || P=48
ONLY=${4:+--only $4}
case $1 in
md5)
  md5sum b75_strong.py b38_eq.py b61_hd.py b63_online.py b69_vocab.py b25_pcm.py b34_inmem.py b22_robust.py \
    b26b_pcm_table.py ruler_vsa.py ruler_rm.py b73_cells.py b34_out/dev_stores.json.gz b34_out/test_stores.json.gz \
    b73_out/test_stores.json.gz 2>/dev/null ;;
selftest)
  nice -n 10 $PY -u b75_strong.py selftest --quick ;;
dev)
  [ -f "$2" ] || { echo "missing plan $2"; exit 1; }
  echo "$(ts) dev start $2 -> $3 $ONLY" >> logs/steps.log
  nice -n 10 $PY -u b75_strong.py run b34_out/dev_stores.json.gz "$2" "$3" --workers $P $ONLY
  echo "$(ts) DEV_DONE $3 rc=$?" >> logs/steps.log ;;
test)
  [ -f "$2" ] || { echo "missing plan $2"; exit 1; }
  echo "$(ts) test start $2 -> $3 $ONLY" >> logs/steps.log
  nice -n 10 $PY -u b75_strong.py run b34_out/test_stores.json.gz "$2" "$3" --workers $P $ONLY
  echo "$(ts) TEST_DONE $3 rc=$?" >> logs/steps.log ;;
fresh)
  [ -f "$2" ] || { echo "missing plan $2"; exit 1; }
  echo "$(ts) fresh start $2 -> $3 $ONLY" >> logs/steps.log
  nice -n 10 $PY -u b75_strong.py run b73_out/test_stores.json.gz "$2" "$3" --workers $P $ONLY --conds k1,k8,k16
  echo "$(ts) FRESH_DONE $3 rc=$?" >> logs/steps.log ;;
esac
