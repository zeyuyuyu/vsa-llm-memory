#!/bin/bash
# B60 VSAPc on the local CPU (prereg/B60_corrected_lam_prereg.md): B57's code path through b60_vsapc.py; nice 10.
# usage: b60_local.sh devp | choosep | decdev | choose | dectest
cd /path/to/vsassm || exit 1
export PYTHONPATH=/path/to/HOME/venvs/b32_site PYTHONDONTWRITEBYTECODE=1
L=b60_out/logs; mkdir -p $L
ts() { date '+%F %T'; }
ENCS="qwen8 b46 llama mistral e5 bge minilm"
P=${B60_P:-3}
job() { OMP_NUM_THREADS=3 nice -n 10 python3 b60_vsapc.py "$@" > $L/vsapc_$(echo "$@" | tr ' :' '__').log 2>&1; echo "$(ts) vsapc $* rc=$?" >> $L/steps.log; }
export -f job ts; export L
case $1 in
devp)
  for e in $ENCS; do for p in whiten:plin whiten:psgn; do echo "$e devp $p"; done; done > $L/qv1.txt
  xargs -P $P -L 1 bash -c 'job "$@"' _ < $L/qv1.txt ;;
choosep)
  for e in $ENCS; do python3 b60_vsapc.py $e choosep >> $L/vsapc_choose.log 2>&1; done ;;
decdev)
  for e in $ENCS; do for M in 1024 256; do for V in 2 16; do echo "$e decp dev $M $V"; done; done; done > $L/qv2.txt
  xargs -P $P -L 1 bash -c 'job "$@"' _ < $L/qv2.txt ;;
choose)
  for e in $ENCS; do python3 b60_vsapc.py $e choose >> $L/vsapc_choose.log 2>&1; done ;;
dectest)
  for e in $ENCS; do f=results/b60_choicepc_$e.json
    git ls-files --error-unmatch $f > /dev/null && git diff --quiet HEAD -- $f || { echo "choices not committed: $f"; exit 1; }; done
  for e in $ENCS; do for M in 1024 256; do for V in 2 16; do echo "$e decp test $M $V"; done; done; done > $L/qv3.txt
  xargs -P $P -L 1 bash -c 'job "$@"' _ < $L/qv3.txt ;;
esac
