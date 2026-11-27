#!/bin/bash
# B31 PACE bridge (compute note 4), run on the local machine: every 3 minutes
#  1. takes the Qwen3-8B phase-3 jobs out of the A800 queue q/p3_heavy (under the workers' flock; lines are moved,
#     never copied) and submits each once to PACE with b31_pace.sbatch (no retry: a failed sbatch is logged, and the
#     line is written to pace_failed for a manual squeue check before any resubmission);
#  2. copies finished PACE outputs (out/lora_final, out/lora_tune) back to the A800 box.
# Stops when the file STOP exists in the scratch directory.
S=/path/to/scratch
mkdir -p $S/got
PK="-i $HOME/.ssh/id_ed25519_pace -o BatchMode=yes -o ConnectTimeout=25"
PH=<user>@login-phoenix.pace.gatech.edu
PR=/path/to/pace-scratch/b31
while [ ! -f $S/STOP ]; do
  timeout 60 ssh -o BatchMode=yes cavin 'cd /path/to/work/b31 && flock q/p3_heavy.lock -c "cat q/p3_heavy; : > q/p3_heavy"' < /dev/null > $S/new.txt 2>/dev/null
  while read -r line; do
    [ -z "$line" ] && continue
    out=$(echo "$line" | sed 's/.*--out \([^ ]*\).*/\1/')
    name=b31_$(basename $out .json)
    r=$(timeout 90 ssh $PK $PH "cd $PR && sbatch -J $name -p gpu-h200,gpu-h100,gpu-a100 -C 'H200|H100|A100-80GB' b31_pace.sbatch $line" < /dev/null 2>&1 | tail -1)
    echo "$(date -u +%FT%TZ) $name: $r" >> $S/bridge.log
    case "$r" in Submitted*) ;; *) echo "$line" >> $S/pace_failed ;; esac
  done < $S/new.txt
  for sub in lora_final lora_tune; do
    for f in $(timeout 60 ssh $PK $PH "ls $PR/out/$sub/ 2>/dev/null" < /dev/null); do
      [ -f $S/got/$f ] && continue
      case $f in *.json) ;; *) continue ;; esac
      timeout 60 scp -O -q $PK $PH:$PR/out/$sub/$f $S/got/$f && \
        timeout 60 rsync -a $S/got/$f cavin:/path/to/work/b31/out/$sub/$f && \
        echo "$(date -u +%FT%TZ) copied $sub/$f" >> $S/bridge.log
    done
  done
  sleep 180
done
