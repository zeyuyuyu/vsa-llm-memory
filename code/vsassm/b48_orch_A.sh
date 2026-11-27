#!/bin/bash
# B48 part A on the A800 box: wait until all 134 dev tuning outputs exist (final .json files; a run still writing has
# only .part), then selection + final queues (b48_after_tune_box.sh), then the finals on two CPU workers (same queue,
# same flock). Stops and reports if the tuning queue is empty, no tuning run is alive and outputs are missing.
cd /path/to/work/b48 || exit 1
n() { ls out/tune/*.json out/g0/*.json out/g1/*.json out/bindtune/*.json out/probe_tune/*.json out/nametune/*.json 2>/dev/null | wc -l; }
while [ "$(n)" -lt 134 ]; do
  if [ ! -s q/tuneA ] && ! pgrep -u <user> -f "mode tune" > /dev/null; then
    echo "$(date -u +%FT%TZ) STOP: queue empty, no tuning run alive, $(n) of 134 outputs" >> logs/orch_A.log; exit 1
  fi
  sleep 60
done
echo "$(date -u +%FT%TZ) all 134 tuning outputs present" >> logs/orch_A.log
bash b48_after_tune_box.sh >> logs/orch_A.log 2>&1 || { echo "after_tune failed" >> logs/orch_A.log; exit 1; }
setsid nohup bash b48_cpuworker.sh q/finalA 9 6 c_finalA > logs/c_finalA.out 2>&1 < /dev/null &
setsid nohup bash b48_cpuworker.sh q/finalA 8 6 c_finalA2 > logs/c_finalA2.out 2>&1 < /dev/null &
echo "$(date -u +%FT%TZ) finals launched: $(wc -l < q/finalA) lines" >> logs/orch_A.log
