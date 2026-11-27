#!/bin/bash
# B65 orchestration on the box: runs queue phases in order (each phase = one or more b65_worker.sh workers on the same
# queue; a phase starts only after the previous one has ended). Every choice is b65_plan.py's mechanical dev rule. No
# retries.
# usage: b65_orch.sh selftest | dev1 | dev2 <pos> | test
B=/path/to/work/b65; cd $B || exit 1
PY=/path/to/HOME/miniconda3/envs/dino_wm/bin/python
stamp() { echo "$1 $(date -u +%FT%TZ)" >> logs/gpu_ledger.txt; }
case $1 in
selftest)
  while ! grep -q "^S1 gpu. end" logs/gpu_ledger.txt; do sleep 10; done
  bash b65_worker.sh q/S2 4 S2
  grep '^b65_run.py train' q/S3 > q/S3a; grep '^b65_run.py eval' q/S3 > q/S3b
  bash b65_worker.sh q/S3a 3 S3a
  bash b65_worker.sh q/S3b 3 S3b
  stamp "orch selftest end" ;;
dev1)  # dev caches, ops layer scan, readers; then the first write-back grids of both positions
  stamp "orch dev1 start"
  $PY b65_plan.py D1 > logs/plan_D1.out 2>&1
  bash b65_worker.sh q/D1_rdlast 4 D1r &
  bash b65_worker.sh q/D1_cache 1 D1c && bash b65_worker.sh q/D1_scan 4 D1s
  wait
  $PY b65_plan.py L > logs/plan_L.out 2>&1
  $PY b65_plan.py R last > logs/plan_R_last.out 2>&1
  for i in 1 2 3; do bash b65_worker.sh q/R_wb_last 4 RWl$i & sleep 25; done
  bash b65_worker.sh q/L_rdops 4 Lr
  $PY b65_plan.py R ops > logs/plan_R_ops.out 2>&1
  for i in 1 2 3; do bash b65_worker.sh q/R_wb_ops 4 RWo$i & sleep 25; done
  wait
  stamp "orch dev1 end" ;;
dev2)  # edge extensions until every choice is interior (or 3 extensions per side), then seeds 1 and 2
  P=$2; k=0
  stamp "orch dev2 $P start"
  while true; do
    k=$((k+1)); flock out/choices.lock $PY b65_plan.py W $P > logs/plan_W_${P}_$k.out 2>&1
    [ -s q/W_ext_$P ] || break
    bash b65_worker.sh q/W_ext_$P 3 We${P}$k
  done
  flock out/choices.lock $PY b65_plan.py F $P > logs/plan_F_$P.out 2>&1
  bash b65_worker.sh q/F_rd_$P 4 Fr$P
  for i in 1 2; do bash b65_worker.sh q/F_wb_$P 3 Fw$P$i & sleep 25; done
  wait
  stamp "orch dev2 $P end" ;;
test)
  stamp "orch test start"
  $PY b65_plan.py T > logs/plan_T.out 2>&1
  for i in 1 2 3; do bash b65_worker.sh q/T_test 3 T$i & sleep 25; done
  wait
  bash b65_worker.sh q/T_rerun 3 Tre
  stamp "orch test end" ;;
esac
