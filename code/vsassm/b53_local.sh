#!/bin/bash
# B53 on the local CPU (prereg/B53_value_codes_prereg.md, 'Compute'): nice 10, no GPU.
# usage: b53_local.sh dev     # every (encoder, prep, fmap) part of the VSA dev grid; two queues bound the memory
#                             # (the four LLMs' sgn3 parts, 1 at a time x 6 threads; every other part, 3 at a time x 3)
#        b53_local.sh choose  # B32's rule per encoder -> b53_out/ENC/choice53_ENC.json, copied to results/b53_choice_ENC.json
#        b53_local.sh test    # only after the choices are committed; LLMs one at a time (memory), sentence encoders together
cd /path/to/vsassm || exit 1
export PYTHONPATH=/path/to/HOME/venvs/b32_site PYTHONDONTWRITEBYTECODE=1
L=b53_out/logs; mkdir -p $L
ts() { date '+%F %T'; }
R="nice -n 10 python3 b53_run.py"
case $1 in
dev)
  echo "$(ts) dev start" >> $L/steps.log
  for e in qwen8 b46 llama mistral; do for p in center whiten; do echo "$e $p:sgn3"; done; done > $L/qA.txt
  { for e in qwen8 b46 llama mistral; do for p in center:lin center:sgn whiten:lin whiten:sgn; do echo "$e $p"; done; done
    for e in e5 bge minilm; do for p in center:lin center:sgn center:sgn3 whiten:lin whiten:sgn whiten:sgn3; do
      echo "$e $p"; done; done; } > $L/qB.txt
  run() { OMP_NUM_THREADS=$1 $R $2 dev $3 > $L/dev_$2_${3/:/_}.log 2>&1; echo "$(ts) dev $2 $3 rc=$?" >> $L/steps.log; }
  export -f run ts; export R L
  xargs -P 1 -L 1 bash -c 'run 6 $0 $1' < $L/qA.txt &
  A=$!
  xargs -P 3 -L 1 bash -c 'run 3 $0 $1' < $L/qB.txt &
  B=$!
  wait $A $B
  echo "$(ts) DEV_DONE" >> $L/steps.log ;;
choose)
  for e in qwen8 b46 llama mistral e5 bge minilm; do
    python3 b53_run.py $e choose > $L/choose_$e.log 2>&1 && cp b53_out/$e/choice53_$e.json results/b53_choice_$e.json
    echo "$(ts) choose $e rc=$?" >> $L/steps.log
  done ;;
test)
  for e in qwen8 b46 llama mistral e5 bge minilm; do
    git ls-files --error-unmatch results/b53_choice_$e.json > /dev/null && git diff --quiet HEAD -- results/b53_choice_$e.json \
      || { echo "choices not committed: $e"; exit 1; }
  done
  echo "$(ts) test start (HEAD $(git rev-parse --short HEAD))" >> $L/steps.log
  ( for e in e5 bge minilm; do OMP_NUM_THREADS=2 $R $e test results/b53_choice_$e.json > $L/test_$e.log 2>&1 &
    done; wait ) &
  S=$!
  for e in qwen8 b46 llama mistral; do
    OMP_NUM_THREADS=10 $R $e test results/b53_choice_$e.json > $L/test_$e.log 2>&1; echo "$(ts) test $e rc=$?" >> $L/steps.log
  done
  wait $S; echo "$(ts) test sentence encoders done" >> $L/steps.log
  echo "$(ts) TEST_DONE" >> $L/steps.log ;;
esac
