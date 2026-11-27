#!/bin/bash
# B69 on the A800 box's CPUs (prereg/B69_vocab_sweep_prereg.md, 'Compute'): nice 10, no GPU, one BLAS thread per
# worker, at most P workers (P <= 64). In ~/b69_run (code and stores MD5-checked against the local commit):
#   nohup bash b69_box.sh run STORES PLAN OUT > logs/X.out 2>&1 < /dev/null &
cd ~/b69_run || exit 1
PY=/path/to/HOME/miniconda3/envs/dino_wm/bin/python
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1
mkdir -p logs results
ts() { date '+%F %T'; }
P=${B69_P:-60}
case $1 in
md5)
  md5sum b69_vocab.py b38_eq.py b34_inmem.py b25_pcm.py ruler_vsa.py ruler_rm.py b22_robust.py b26b_pcm_table.py \
    b61_hd.py b63_online.py b69_out/sweep_dev.json.gz b69_out/sweep_test.json.gz results/b69_theory.json ;;
selftest)
  nice -n 10 $PY -u b69_vocab.py selftest ;;
run)
  [ -f "$3" ] || { echo "missing plan $3"; exit 1; }
  echo "$(ts) run start $2 $3 -> $4" >> logs/steps.log
  nice -n 10 $PY -u b69_vocab.py run "$2" "$3" "$4" --workers $P
  echo "$(ts) RUN_DONE $4 rc=$?" >> logs/steps.log ;;
c0)
  echo "$(ts) c0 start" >> logs/steps.log
  nice -n 10 $PY -u b69_check.py c0 results/b69_check_c0.json
  echo "$(ts) C0_DONE rc=$?" >> logs/steps.log ;;
esac
