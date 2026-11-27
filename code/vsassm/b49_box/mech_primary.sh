#!/bin/bash
# B49: the primary mechanism runs first (qwen8 T1, qwen8 T3, e5 T2; prereg section 6); replaces phase1.sh's mechanism
# loop, which would have started the replications (b46 T1 / T3) before e5 T2. Replications are started later if time allows.
export B49_ROOT=/path/to/work/b49 B49_PY=/path/to/HOME/envs/b49/bin/python
cd $B49_ROOT || exit 1
PY=$B49_PY; N="nice -n 10"; L=b49_out/logs
ts() { date '+%F %T'; }
while pgrep -f "b49_mech.py qwen8 T1 mak[e]" > /dev/null; do sleep 15; done
echo "$(ts) mech make qwen8 T1 finished (phase1 loop stopped; codes $(ls -la b49_out/b32_data/emb_qwen8_T1.npy | awk '{print $5}') bytes)" >> $L/phase1.log
nohup bash b49_box/mech_dev.sh qwen8 T1 > $L/mech_dev_qwen8_T1.log 2>&1 &
for bt in qwen8:T3 e5:T2 "$@"; do
  b=${bt%%:*}; t=${bt##*:}
  OMP_NUM_THREADS=8 $N $PY b49_mech.py $b $t make > $L/mech_make_${b}_$t.log 2>&1
  echo "$(ts) mech make $b $t rc=$?" >> $L/phase1.log
  nohup bash b49_box/mech_dev.sh $b $t > $L/mech_dev_${b}_$t.log 2>&1 &
done
wait
