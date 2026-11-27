#!/bin/bash
# B59 part 1 on the A800 box's CPUs (prereg/B59_maps_all_encoders_protected_full_prereg.md, 'Compute'): nice 10, no GPU,
# at most P jobs x 3 threads (P <= 14 -> <= 42 cores). In ~/b59_run:
#   nohup bash b59_box.sh dev1 > logs/dev1.out 2>&1 < /dev/null &      # MF grids (72 parts) + LAM rp dev (12)
#   nohup bash b59_box.sh dev2 > logs/dev2.out 2>&1 < /dev/null &      # MF choices (12) + engine dev (48)
#   nohup bash b59_box.sh test > logs/test.out 2>&1 < /dev/null &      # only after the choices are committed locally
cd ~/b59_run || exit 1
PY=/path/to/HOME/miniconda3/envs/dino_wm/bin/python
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3
L=logs; mkdir -p $L
ts() { date '+%F %T'; }
P=${B59_P:-14}
job() { OMP_NUM_THREADS=3 nice -n 10 $PY b59_run.py "$@" > $L/$(echo "$@" | tr ' :' '__').log 2>&1; echo "$(ts) $* rc=$?" >> $L/steps.log; }
export -f job ts; export PY L
NEW="b46 llama mistral bge"
case $1 in
dev1)
  echo "$(ts) dev1 start" >> $L/steps.log
  { for e in b46 llama mistral; do for m in 1 2 3; do for p in center:lin center:sgn center:sgn3 whiten:lin whiten:sgn whiten:sgn3; do
      echo "$e devm $m $p"; done; done; done
    for e in b46 llama mistral; do for m in 1 2 3; do echo "$e lamm $m dev"; done; done
    for m in 1 2 3; do for p in center:lin center:sgn center:sgn3 whiten:lin whiten:sgn whiten:sgn3; do echo "bge devm $m $p"; done; done
    for m in 1 2 3; do echo "bge lamm $m dev"; done; } > $L/q1.txt
  xargs -P $P -L 1 bash -c 'job "$@"' _ < $L/q1.txt
  echo "$(ts) DEV1_DONE" >> $L/steps.log ;;
dev2)
  for e in $NEW; do for m in 1 2 3; do $PY b59_run.py $e choosem $m >> $L/choose.log 2>&1; done; done
  echo "$(ts) MF choices written" >> $L/steps.log
  { for e in b46 llama mistral bge; do for m in 1 2 3; do for M in 1024 256; do for V in 2 16; do echo "$e decm $m dev $M $V"; done; done; done; done
  } > $L/q2.txt
  xargs -P $P -L 1 bash -c 'job "$@"' _ < $L/q2.txt
  echo "$(ts) DEV2_DONE" >> $L/steps.log ;;
test)
  for e in $NEW; do [ -f results/b59_choice_$e.json ] || { echo "missing choice $e"; exit 1; }; done
  echo "$(ts) test start" >> $L/steps.log
  { for e in b46 llama mistral; do for m in 1 2 3; do for M in 1024 256; do for V in 2 16; do echo "$e decm $m test $M $V"; done; done
      echo "$e lamm $m test"; done; done
    for e in qwen8 b46 llama mistral; do for m in 1 2 3; do echo "$e mft $m"; done; done
    for m in 1 2 3; do for M in 1024 256; do for V in 2 16; do echo "bge decm $m test $M $V"; done; done; echo "bge lamm $m test"; done
    for e in e5 bge minilm; do for m in 1 2 3; do echo "$e mft $m"; done; done; } > $L/qt.txt
  xargs -P $P -L 1 bash -c 'job "$@"' _ < $L/qt.txt
  echo "$(ts) TEST_DONE" >> $L/steps.log ;;
esac
