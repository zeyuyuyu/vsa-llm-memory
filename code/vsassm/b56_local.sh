#!/bin/bash
# B56 on the local CPU (prereg/B56_decoders_retraining_prereg.md, 'Compute'): nice 10, no GPU.
# usage: b56_local.sh dev     # 42 sparse parts of the widened grid (3 at a time) + the LM/RT/LR engine on the dev stores
#        b56_local.sh choose  # -> results/b56_choice_ENC.json
#        b56_local.sh test    # only after the choices are committed
cd /path/to/vsassm || exit 1
export PYTHONPATH=/path/to/HOME/venvs/b32_site PYTHONDONTWRITEBYTECODE=1
L=b56_out/logs; mkdir -p $L
ts() { date '+%F %T'; }
R="nice -n 10 python3 b56_run.py"
ENCS="qwen8 b46 llama mistral e5 bge minilm"
case $1 in
dev)
  echo "$(ts) dev start (HEAD $(git rev-parse --short HEAD))" >> $L/steps.log
  for e in $ENCS; do for p in center:tern4 center:tern16 center:tern64 whiten:tern4 whiten:tern16 whiten:tern64; do
    echo "$e $p"; done; done > $L/qW.txt
  { for e in qwen8 b46 llama mistral; do echo "$e 1024 2"; echo "$e 1024 16"; done
    for e in e5 bge minilm; do echo "$e 1024 2"; echo "$e 1024 16"; done
    for e in $ENCS; do echo "$e 256 2"; echo "$e 256 16"; done; } > $L/qD.txt
  runw() { OMP_NUM_THREADS=2 $R $1 devw $2 > $L/devw_$1_${2/:/_}.log 2>&1; echo "$(ts) devw $1 $2 rc=$?" >> $L/steps.log; }
  rund() { OMP_NUM_THREADS=3 $R $1 dec dev $2 $3 > $L/decdev_$1_$2_$3.log 2>&1; echo "$(ts) decdev $1 $2 $3 rc=$?" >> $L/steps.log; }
  export -f runw rund ts; export R L
  xargs -P 3 -L 1 bash -c 'runw $0 $1' < $L/qW.txt &
  A=$!
  xargs -P 3 -L 1 bash -c 'rund $0 $1 $2' < $L/qD.txt &
  B=$!
  wait $A $B
  echo "$(ts) DEV_DONE" >> $L/steps.log ;;
choose)
  for e in $ENCS; do
    python3 b56_run.py $e choose > $L/choose_$e.log 2>&1; echo "$(ts) choose $e rc=$?" >> $L/steps.log
  done ;;
test)
  for e in $ENCS; do
    git ls-files --error-unmatch results/b56_choice_$e.json > /dev/null && git diff --quiet HEAD -- results/b56_choice_$e.json \
      || { echo "choices not committed: $e"; exit 1; }
  done
  echo "$(ts) test start (HEAD $(git rev-parse --short HEAD))" >> $L/steps.log
  for e in $ENCS; do for M in 1024 256; do for V in 2 16; do echo "$e $M $V"; done; done; done > $L/qT.txt
  runt() { OMP_NUM_THREADS=3 $R $1 dec test $2 $3 > $L/dectest_$1_$2_$3.log 2>&1; echo "$(ts) dectest $1 $2 $3 rc=$?" >> $L/steps.log; }
  runtw() { OMP_NUM_THREADS=2 $R $1 testw > $L/testw_$1.log 2>&1; echo "$(ts) testw $1 rc=$?" >> $L/steps.log; }
  export -f runt runtw ts; export R L
  xargs -P 3 -L 1 bash -c 'runt $0 $1 $2' < $L/qT.txt &
  A=$!
  echo $ENCS | tr ' ' '\n' | xargs -P 2 -L 1 bash -c 'runtw $0' &
  B=$!
  wait $A $B
  for e in $ENCS; do python3 b56_run.py $e merge >> $L/merge.log 2>&1; echo "$(ts) merge $e rc=$?" >> $L/steps.log; done
  echo "$(ts) TEST_DONE" >> $L/steps.log ;;
esac
