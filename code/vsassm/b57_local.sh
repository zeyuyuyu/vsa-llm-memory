#!/bin/bash
# B57 on the local CPU (prereg/B57_pca_store_maps_online_prereg.md, 'Compute'): nice 10, no GPU.
# usage: b57_local.sh dev1    # MF grids (VSAP 28 parts, maps 54 parts), LAM rp on new maps (9), NLMS dev (7)
#        b57_local.sh dev2    # MF choices; LM / RT on VSAP (28 jobs), LM / RT / LR on the maps (36 jobs)
#        b57_local.sh choose
#        b57_local.sh test    # only after the choices are committed
cd /path/to/vsassm || exit 1
export PYTHONPATH=/path/to/HOME/venvs/b32_site PYTHONDONTWRITEBYTECODE=1
L=b57_out/logs; mkdir -p $L
ts() { date '+%F %T'; }
R="nice -n 10 python3 b57_run.py"
ENCS="qwen8 b46 llama mistral e5 bge minilm"
MENCS="qwen8 e5 minilm"
P=${B57_P:-4}
job() { OMP_NUM_THREADS=3 $R "$@" > $L/$(echo "$@" | tr ' :' '__').log 2>&1; echo "$(ts) $* rc=$?" >> $L/steps.log; }
export -f job ts; export R L
case $1 in
dev1)
  echo "$(ts) dev1 start (HEAD $(git rev-parse --short HEAD))" >> $L/steps.log
  { for m in 1 2 3; do for p in center:lin center:sgn center:sgn3 whiten:lin whiten:sgn whiten:sgn3; do echo "qwen8 devm $m $p"; done; done
    for m in 1 2 3; do echo "qwen8 lamm $m dev"; done
    for e in qwen8 b46 llama mistral; do echo "$e onl dev"; done
    for e in $ENCS; do for p in center:plin center:psgn whiten:plin whiten:psgn; do echo "$e devp $p"; done; done
    for e in e5 minilm; do for m in 1 2 3; do for p in center:lin center:sgn center:sgn3 whiten:lin whiten:sgn whiten:sgn3; do
      echo "$e devm $m $p"; done; echo "$e lamm $m dev"; done; done
    for e in e5 bge minilm; do echo "$e onl dev"; done; } > $L/q1.txt
  xargs -P $P -L 1 bash -c 'job "$@"' _ < $L/q1.txt
  echo "$(ts) DEV1_DONE" >> $L/steps.log ;;
dev2)
  for e in $ENCS; do python3 b57_run.py $e choosep >> $L/choose.log 2>&1; done
  for e in $MENCS; do for m in 1 2 3; do python3 b57_run.py $e choosem $m >> $L/choose.log 2>&1; done; done
  echo "$(ts) MF choices written" >> $L/steps.log
  { for e in qwen8; do for m in 1 2 3; do for M in 1024 256; do for V in 2 16; do echo "$e decm $m dev $M $V"; done; done; done; done
    for e in $ENCS; do for M in 1024 256; do for V in 2 16; do echo "$e decp dev $M $V"; done; done; done
    for e in e5 minilm; do for m in 1 2 3; do for M in 1024 256; do for V in 2 16; do echo "$e decm $m dev $M $V"; done; done; done; done
  } > $L/q2.txt
  xargs -P $P -L 1 bash -c 'job "$@"' _ < $L/q2.txt
  echo "$(ts) DEV2_DONE" >> $L/steps.log ;;
choose)
  for e in $ENCS; do python3 b57_run.py $e choose >> $L/choose.log 2>&1; echo "$(ts) choose $e rc=$?" >> $L/steps.log; done ;;
test)
  for e in $ENCS; do
    for f in results/b57_choice_$e.json results/b57_choicep_$e.json; do
      git ls-files --error-unmatch $f > /dev/null && git diff --quiet HEAD -- $f || { echo "choices not committed: $f"; exit 1; }
    done
  done
  echo "$(ts) test start (HEAD $(git rev-parse --short HEAD))" >> $L/steps.log
  { for m in 1 2 3; do for M in 1024 256; do for V in 2 16; do echo "qwen8 decm $m test $M $V"; done; done; echo "qwen8 lamm $m test"; done
    for e in $ENCS; do for M in 1024 256; do for V in 2 16; do echo "$e decp test $M $V"; done; done; echo "$e onl test"; echo "$e rlstest"; done
    for e in e5 minilm; do for m in 1 2 3; do for M in 1024 256; do for V in 2 16; do echo "$e decm $m test $M $V"; done; done
      echo "$e lamm $m test"; done; done; } > $L/qt.txt
  xargs -P $P -L 1 bash -c 'job "$@"' _ < $L/qt.txt
  for e in $ENCS; do python3 b57_run.py $e merge >> $L/merge.log 2>&1; echo "$(ts) merge $e rc=$?" >> $L/steps.log; done
  echo "$(ts) TEST_DONE" >> $L/steps.log ;;
esac
