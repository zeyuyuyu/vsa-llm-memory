#!/bin/bash
# B75 test on the box (prereg/B75_strongest_options_prereg.md; compute note 3): the registered test plan
# (results/b75_plan_test.json, from the committed dev choices) run once, split into three launches by memory and
# condition only (the rows are those of one launch of the whole plan; at most 48 workers in all at any time):
#   R1  U-LS, O2-BCH, O2-LDPC, U-J (every condition)            24 workers
#   R2a H-SBC-J at k1 and k8                                     24 workers, concurrently with R1
#   R2b H-SBC-J at k16                                           24 workers, after R2a
#   F   (D) fresh stores, U-LS / O2-BCH / O2-LDPC at k1, k8, k16 24 workers, after R1 ((D) for the joint decoders cut)
#   nohup bash b75_test_chain.sh > logs/chain.out 2>&1 < /dev/null &
cd ~/b75_run || exit 1
PY=/path/to/HOME/miniconda3/envs/dino_wm/bin/python
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1
ts() { date '+%F %T'; }
P=results/b75_plan_test.json
run() {  # name stores out workers args...
  local name=$1 st=$2 out=$3 w=$4; shift 4
  echo "$(ts) $name start" >> logs/steps.log
  nice -n 10 $PY -u b75_strong.py run $st $P $out --workers $w "$@" > logs/test_$name.out 2>&1
  echo "$(ts) ${name}_DONE rc=$?" >> logs/steps.log
}
( run R1 b34_out/test_stores.json.gz results/b75_test_R1.json 24 --only U-LS,O2-BCH,O2-LDPC,U-J
  run F b73_out/test_stores.json.gz results/b75_fresh_F.json 24 --only U-LS,O2-BCH,O2-LDPC --conds k1,k8,k16 ) &
( run R2a b34_out/test_stores.json.gz results/b75_test_R2a.json 24 --only H-SBC-J --conds k1,k8
  run R2b b34_out/test_stores.json.gz results/b75_test_R2b.json 24 --only H-SBC-J --conds k16 ) &
wait
echo "$(ts) CHAIN_DONE" >> logs/steps.log
