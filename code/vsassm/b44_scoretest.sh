#!/bin/bash
# B44 (PACE job body): score the test split once from the saved dev-best state of each finished run given as
# "<task> <variant> <n> <lr> <seed>" quintuples on the command line; a run that has not finished is skipped.
while [ $# -ge 5 ]; do
  nm=$1_$2_n$3_r8_lr$4_s$5
  if [ -f out/lora/$nm.json ]; then
    /usr/bin/python3 -u b44_lora.py --task $1 --variant $2 --n $3 --rank 8 --lr $4 --seed $5 --score_test --out out/lora/$nm.json || echo "FAILED $nm"
  else echo "NOT FINISHED $nm"; fi
  shift 5
done
