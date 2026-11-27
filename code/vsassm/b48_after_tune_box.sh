#!/bin/bash
# B48 on the A800 box (compute notes 2-3): after the dev tuning runs, part A selection (b48_select.py), part B
# selection (b48_alias.py select) and the final queues; the finals are then served by b48_cpuworker.sh / b48_worker.sh
# from q/finalA and q/finalB (same flock, no run twice).
cd /path/to/work/b48 || exit 1
PY=/path/to/HOME/miniconda3/envs/dino_wm/bin/python
$PY b48_select.py --root out --out out/select.json || exit 1
$PY b48_alias.py select --root out/b --out out/b/select.json || exit 1
$PY b48_queues.py final || exit 1
