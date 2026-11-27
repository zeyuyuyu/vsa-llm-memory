#!/bin/bash
# B42 (prereg/B42_label_free_discrete_prereg.md): when every arena-A/B tuning output exists (240 runs), select on dev,
# write the A/B final queue and start GPU workers for it (they claim idle GPUs only, as b42_worker.sh always does).
B=/path/to/work/b42; cd $B || exit 1
PY=/path/to/HOME/miniconda3/envs/dino_wm/bin/python
until [ $(ls out/tune/*_[AB]_*.json 2>/dev/null | wc -l) -ge 240 ]; do sleep 30; done
echo "tuning complete $(date -u +%FT%TZ)"
$PY b42_select.py --root out --out out/select.json --arenas ABC > logs/select_ABC.log 2>&1
$PY b42_queues.py final --arenas AB
for n in fAB1 fAB2 fAB3 fAB4 fAB5 fAB6; do nohup bash b42_worker.sh q/final_AB 4 $n > logs/$n.worker.log 2>&1 & sleep 1; done
echo "final workers started $(date -u +%FT%TZ)"
