#!/bin/bash
# B49 (local): after the box's dev phase of the given encoders, fetch, make the new families' choices, copy every choice
# file into results/, commit them, push them to the box with their md5 list. No test is run here.
# usage: choose_local.sh panel ENC...   |   choose_local.sh mech BASE:T...
cd /path/to/vsassm || exit 1
export PYTHONPATH=/path/to/HOME/venvs/b32_site PYTHONDONTWRITEBYTECODE=1
what=$1; shift
bash b49_box/fetch.sh || exit 1
FILES=()
for x in "$@"; do
  if [ $what = panel ]; then
    e=$x
    case $e in qwen8|e5|b46) ;; *)
      grep -q CHOICES_DONE b49_out/enc_$e/logs/steps.log || { echo "$e: panel choices not done"; exit 1; }
      cp b49_out/enc_$e/results/b32_choice_$e.json results/b49_choice32_$e.json
      cp b49_out/enc_$e/results/b36_choice_$e.json results/b49_choice36_$e.json
      FILES+=(results/b49_choice32_$e.json results/b49_choice36_$e.json) ;;
    esac
    python3 b49_fam.py $e choose || exit 1
    FILES+=(results/b49_choice_$e.json)
  else
    b=${x%%:*}; t=${x##*:}; W=b49_out/mech_${b}_$t
    grep -q MECH_DEV_DONE $W/logs/steps.log || { echo "$x: mech dev not done"; exit 1; }
    cp $W/results/b32_choice_$b.json results/b49_mech_choice32_${b}_$t.json
    cp $W/results/b36_choice_$b.json results/b49_mech_choice36_${b}_$t.json
    B49_GRID_V=2,16 B49_GRID_BPF=4,8,16 python3 b49_fam.py ${b}_$t choose || exit 1
    FILES+=(results/b49_mech_choice32_${b}_$t.json results/b49_mech_choice36_${b}_$t.json results/b49_choice_${b}_$t.json)
  fi
done
git add "${FILES[@]}" && git commit -q -m "B49 dev choices ($what $*), committed before the corresponding test run (dev split only; box dev outputs fetched into b49_out/)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git log --oneline -1
# push to the box: the new families' choices into its results/, and the md5 list of every choice file the tests read
for f in "${FILES[@]}"; do case $f in results/b49_choice_*) scp -q -o ConnectTimeout=20 $f cavin:/path/to/work/b49/results/ ;; esac; done
python3 - "$what" "$@" <<'PY'
import hashlib, sys, os
what, xs = sys.argv[1], sys.argv[2:]
lines = []
def md5(f): return hashlib.md5(open(f, 'rb').read()).hexdigest()
for x in xs:
    if what == 'panel':
        lines.append(f'{md5(f"results/b49_choice_{x}.json")}  results/b49_choice_{x}.json')
        if x not in ('qwen8', 'e5', 'b46'):
            lines.append(f'{md5(f"results/b49_choice32_{x}.json")}  b49_out/enc_{x}/results/b32_choice_{x}.json')
            lines.append(f'{md5(f"results/b49_choice36_{x}.json")}  b49_out/enc_{x}/results/b36_choice_{x}.json')
    else:
        b, t = x.split(':')
        lines.append(f'{md5(f"results/b49_choice_{b}_{t}.json")}  results/b49_choice_{b}_{t}.json')
        lines.append(f'{md5(f"results/b49_mech_choice32_{b}_{t}.json")}  b49_out/mech_{b}_{t}/results/b32_choice_{b}.json')
        lines.append(f'{md5(f"results/b49_mech_choice36_{b}_{t}.json")}  b49_out/mech_{b}_{t}/results/b36_choice_{b}.json')
open('b49_box/choices_md5.txt', 'w').write('\n'.join(lines) + '\n')
PY
scp -q -o ConnectTimeout=20 b49_box/choices_md5.txt b49_box/phase2.sh cavin:/path/to/work/b49/b49_box/
echo "pushed; md5 list:"; cat b49_box/choices_md5.txt
