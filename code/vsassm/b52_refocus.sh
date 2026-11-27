#!/bin/bash
# B52 compute note 2 (coordinator deadline, 2026-11-20 ~03:50 UTC): part (a) is restricted to the primary cell
# qwen8L27 x C. Keeps only that cell's lines in the queues (under the queues' flock), stops this experiment's own runs on
# the other cells (python children of the B52 workers, matched on their argv read from /proc), restarts the orchestrator
# restricted to the cell. Never touches another user's or experiment's process.
cd /path/to/work/b52 || exit 1
for Q in q/tune q/final; do
  flock "$Q.lock" -c "grep -- '--enc_model qwen8L27 --arena C ' '$Q' > '$Q.keep'; mv '$Q.keep' '$Q'"
done
wc -l q/tune q/final
for pid in $(pgrep -u <user> -f "b52_run[.]py"); do
  args=$(tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null)
  case "$args" in
    *"--enc_model qwen8L27 --arena C "*) echo "keep $pid $(echo $args | cut -c1-160)";;
    *) echo "stop $pid $(echo $args | cut -c1-160)"; kill $pid;;
  esac
done
kill $(cat logs/orch.pid) 2>/dev/null
B52_CELLS=qwen8L27_C nohup /path/to/HOME/miniconda3/envs/dino_wm/bin/python -u b52_queues.py orch >> logs/orch.log 2>&1 < /dev/null &
echo $! > logs/orch.pid
echo "refocus $(date -u +%FT%TZ)" >> logs/cpu_ledger.txt
