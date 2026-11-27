#!/bin/bash
# B49 phase 2 on the A800 box's CPU (prereg/B49_compute_note1.md): test once, after every choice file was committed in
# the repository and copied here (their md5 are checked against b49_box/choices_md5.txt, written from the commit).
# usage: phase2.sh panel ENC... | fam ENC... | mech BASE:T...
export B49_ROOT=/path/to/work/b49 B49_PY=/path/to/HOME/envs/b49/bin/python
export PYTHONDONTWRITEBYTECODE=1
cd $B49_ROOT || exit 1
PY=$B49_PY; N="nice -n 10"
L=b49_out/logs; mkdir -p $L
ts() { date '+%F %T'; }
md5sum -c --quiet b49_box/choices_md5.txt || { echo "$(ts) choice md5 mismatch: no test" >> $L/phase2.log; exit 1; }
what=$1; shift
for x in "$@"; do
  if [ $what = panel ]; then
    ( W=b49_out/enc_$x; R="$N $PY b49_run.py $x"
      echo "$(ts) test start $x" >> $W/logs/steps.log
      OMP_NUM_THREADS=4 $R b32test vsa > $W/logs/b32test_vsa.log 2>&1 &
      A=$!
      OMP_NUM_THREADS=4 $R b32test idx > $W/logs/b32test_idx.log 2>&1 &
      B=$!
      OMP_NUM_THREADS=2 $R test --workers 6 > $W/logs/test.log 2>&1 &
      C=$!
      wait $A; echo "$(ts) b32test vsa rc=$?" >> $W/logs/steps.log
      wait $B; echo "$(ts) b32test idx rc=$?" >> $W/logs/steps.log
      $R b32merge > $W/logs/b32merge.log 2>&1; echo "$(ts) b32merge rc=$?" >> $W/logs/steps.log
      OMP_NUM_THREADS=4 $R test32 > $W/logs/test32.log 2>&1; echo "$(ts) test32 rc=$?" >> $W/logs/steps.log
      wait $C; echo "$(ts) test (key-free families) rc=$?" >> $W/logs/steps.log
      $R merge > $W/logs/merge.log 2>&1; echo "$(ts) merge rc=$?" >> $W/logs/steps.log
      echo "$(ts) TEST_DONE" >> $W/logs/steps.log ) &
  elif [ $what = fam ]; then
    ( OMP_NUM_THREADS=2 $N $PY b49_fam.py $x test --workers 6 > $L/fam_test_$x.log 2>&1
      echo "$(ts) fam test $x rc=$?" >> $L/phase2.log ) &
  elif [ $what = mech ]; then
    ( b=${x%%:*}; t=${x##*:}; W=b49_out/mech_${b}_$t
      echo "$(ts) mech test start" >> $W/logs/steps.log
      OMP_NUM_THREADS=4 $N $PY b49_mech.py $b $t test32vsa > $W/logs/test32vsa.log 2>&1 &
      A=$!
      OMP_NUM_THREADS=4 $N $PY b49_mech.py $b $t test32idx > $W/logs/test32idx.log 2>&1 &
      B=$!
      OMP_NUM_THREADS=2 $N $PY b49_mech.py $b $t test36 --workers 4 > $W/logs/test36.log 2>&1 &
      C=$!
      B49_GRID_V=2,16 B49_GRID_BPF=4,8,16 OMP_NUM_THREADS=2 $N $PY b49_fam.py ${b}_$t test --workers 4 > $W/logs/fam_test.log 2>&1 &
      D=$!
      wait $A $B $C; $PY b49_mech.py $b $t merge > $W/logs/merge.log 2>&1; echo "$(ts) mech merge rc=$?" >> $W/logs/steps.log
      wait $D; echo "$(ts) fam test rc=$?" >> $W/logs/steps.log
      echo "$(ts) MECH_TEST_DONE" >> $W/logs/steps.log ) &
  fi
done
wait
echo "$(ts) phase2 $what $* done" >> $L/phase2.log
