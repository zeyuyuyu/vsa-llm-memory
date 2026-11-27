#!/bin/bash
# B44 monitor (PACE only): job ends and run milestones of the remaining lora 80,000 chains
S=/path/to/vsassm/b44_out/mon; st=$S/states.txt; sl=$S/lines.txt; touch $st $sl
while true; do
  timeout 60 ssh -i $HOME/.ssh/id_ed25519_pace -o BatchMode=yes -o ConnectTimeout=25 <user>@login-phoenix.pace.gatech.edu 'sacct -u <user> -S now-3hours -n -X -o JobID,JobName%40,State%14,Elapsed | grep b44_ | grep -v -E "PENDING|CANCELLED"; cd /path/to/pace-scratch/b44/logs && grep -h -E "step [0-9]+/10000 |TEST|Traceback|Error|time budget" b44_cls_*n80000*_158754*.out 2>/dev/null | cut -c1-200' < /dev/null 2>/dev/null | awk '{$1=$1};1' | sort > $st.new
  comm -13 $st $st.new | grep -v RUNNING
  cat $st.new >> $st; sort -u $st -o $st
  sleep 120
done
