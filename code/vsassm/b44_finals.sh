#!/bin/bash
# B44 finals for one LoRA cell after its selection (b44_select.py): the seed-42 test scoring of the selected tuning run
# and the seed-123 / 456 runs (each scores the test split once from its own dev-best state, in a chained job).
# usage: b44_finals.sh <task> <variant> <n> <lr> <nseg> [seeds...]
TASK=$1; V=$2; N=$3; LR=$4; NS=$5; shift 5; SEEDS=${@:-"123 456"}
cd /path/to/vsassm
for s in $SEEDS; do
  nm=${TASK}_${V}_n${N}_r8_lr${LR}_s$s
  TL=1:30:00 ./b44_pace_submit.sh $nm $NS b44_lora.py --task $TASK --variant $V --n $N --rank 8 --lr $LR --seed $s --max_hours 1.3 --final --out out/lora/$nm.json
done
