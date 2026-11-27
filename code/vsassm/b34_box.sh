#!/bin/bash
# B34: run one stage on the A800 box's CPUs (no GPU). Usage (on the box, in /path/to/work/b34):
#   nohup bash b34_box.sh dev-grid|dev-full|test WORKERS > logs/STAGE.log 2>&1 &
cd /path/to/work/b34
PY=/path/to/HOME/miniconda3/envs/dino_wm/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
W=${2:-48}
case "$1" in
  dev-grid) $PY b34_inmem.py dev b34_out/dev_stores.json.gz b34_out/ber.json b34_out/dev_grid.json grid --workers $W ;;
  dev-full) $PY b34_inmem.py dev b34_out/dev_stores.json.gz b34_out/ber.json b34_out/dev_full.json full b34_out/b34_choice.json --workers $W ;;
  test)     $PY b34_inmem.py test b34_out/test_stores.json.gz b34_out/ber.json b34_out/test_full.json full b34_out/b34_choice.json --workers $W ;;
esac
echo "exit $? $(date)"
# extra stage (added after T-vhdd): the T-vhdd share grid with the chosen MPHF lambda
if [ "$1" = "dev-grid-vhdd" ]; then
  $PY b34_inmem.py dev b34_out/dev_stores.json.gz b34_out/ber.json b34_out/dev_grid_vhdd.json grid T-vhdd 5 --workers $W
  echo "exit $? $(date)"
fi
# closed-form checks (b34_check.py closed)
if [ "$1" = "dev-check" ]; then
  $PY b34_check.py closed b34_out/dev_stores.json.gz b34_out/ber.json b34_out/b34_choice.json b34_out/dev_full.json b34_out/dev_check.txt --workers $W
  echo "exit $? $(date)"
fi
if [ "$1" = "test-check" ]; then
  $PY b34_check.py closed b34_out/test_stores.json.gz b34_out/ber.json b34_out/b34_choice.json b34_out/test_full.json b34_out/test_check.txt --workers $W
  echo "exit $? $(date)"
fi
