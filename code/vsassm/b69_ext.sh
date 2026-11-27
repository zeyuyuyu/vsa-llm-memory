#!/bin/bash
# B69: B38's extension rule on the test, mechanically, at most 3 rounds (prereg/B69_vocab_sweep_prereg.md, 'Test').
cd ~/b69_run || exit 1
PY=/path/to/HOME/miniconda3/envs/dino_wm/bin/python
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1
ts() { date '+%F %T'; }
PLANS=results/b69_plan_test1.json
TESTS=results/b69_test1.json
for r in 1 2 3; do
  $PY b69_vocab.py extend results/b69_choice.json $PLANS $TESTS results/b69_plan_x$r.json > logs/extend_x$r.out 2>&1
  n=$(grep "extension evaluations" logs/extend_x$r.out | awk '{print $3}')
  echo "$(ts) extend round $r: $n evaluations" >> logs/steps.log
  [ "$n" = "0" ] && break
  nice -n 10 $PY -u b69_vocab.py run b69_out/sweep_test.json.gz results/b69_plan_x$r.json results/b69_test_x$r.json --workers ${B69_P:-60} > logs/test_x$r.out 2>&1
  echo "$(ts) RUN_DONE results/b69_test_x$r.json rc=$?" >> logs/steps.log
  PLANS=$PLANS,results/b69_plan_x$r.json
  TESTS=$TESTS,results/b69_test_x$r.json
done
echo "$(ts) EXT_DONE plans=$PLANS tests=$TESTS" >> logs/steps.log
for f in ${TESTS//,/ }; do gzip -k -f $f; done
