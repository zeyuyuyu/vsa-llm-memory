#!/bin/bash
# B49 phase 3 on the box (after every test of the given panel encoders and mechanism runs): merges, verdict (one process
# per encoder, then the combination), mechanism verdict, checks, hand reading, summary tables, predictions.
# usage: phase3.sh "ENC ..." "BASE:T ..."
export B49_ROOT=/path/to/work/b49
cd $B49_ROOT || exit 1
PY=/path/to/HOME/envs/b49/bin/python; N="nice -n 10"; L=b49_out/logs
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4
ts() { date '+%F %T'; }
ENCS="$1"; MECH="$2"
echo "$(ts) phase3 start encs [$ENCS] mech [$MECH]" >> $L/phase3.log
for e in $ENCS; do $PY b49_fam.py $e merge >> $L/phase3.log 2>&1 || echo "$(ts) merge $e FAILED" >> $L/phase3.log; done
for bt in $MECH; do b=${bt%%:*}; t=${bt##*:}
  B49_GRID_V=2,16 B49_GRID_BPF=4,8,16 $PY b49_fam.py ${b}_$t merge >> $L/phase3.log 2>&1 || echo "$(ts) merge $bt FAILED" >> $L/phase3.log
done
for e in $ENCS; do $N $PY b49_verdict.py panel1 $e > $L/verdict_part_$e.log 2>&1 & done
wait
$PY b49_verdict.py panel --encs $(echo $ENCS | tr ' ' ',') > $L/verdict.log 2>&1; echo "$(ts) verdict rc=$?" >> $L/phase3.log
[ -n "$MECH" ] && { $PY b49_verdict.py mech > $L/mech_verdict.log 2>&1; echo "$(ts) mech verdict rc=$?" >> $L/phase3.log; }
# checks
$PY b49_check.py score --encs $(echo $ENCS | tr ' ' ',') > $L/check_score.log 2>&1 &
$PY b49_check.py bits --encs $(echo $ENCS | tr ' ' ',') > $L/check_bits.log 2>&1 &
for e in $ENCS; do
  $N $PY b49_check.py lam $e --frac 0.1 > $L/check_lam_$e.log 2>&1 &
  $N $PY b49_check.py lh $e > $L/check_lh_$e.log 2>&1 &
  case $e in qwen8|e5|b46) ;; *)
    $N $PY b49_check36.py $e bits > $L/check36_bits_$e.log 2>&1 &
    $N $PY b49_check36.py $e rederive --frac 0.1 > $L/check36_rederive_$e.log 2>&1 &
    $N $PY b49_check36.py $e rederive32 --frac 0.1 > $L/check32_rederive_$e.log 2>&1 & ;;
  esac
done
$PY b49_handread.py $ENCS > $L/handread.log 2>&1
$PY b49_summary.py > $L/summary.log 2>&1
$PY -c "
import json, glob
d = json.load(open('b49_out/lh_choice_part1.json'))
for f in glob.glob('b49_out/lh_choice_*.json'):
    if 'part1' in f or 'all' in f: continue
    d.update(json.load(open(f)))
json.dump(d, open('b49_out/lh_choice_all.json', 'w'), indent=1)"
$PY b49_predcheck.py > $L/predcheck.log 2>&1
wait
echo "$(ts) PHASE3_DONE" >> $L/phase3.log
