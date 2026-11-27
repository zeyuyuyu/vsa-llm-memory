#!/bin/bash
# B42 CPU worker on the A800 box (prereg/B42_label_free_discrete_prereg.md, compute note 1). Uses no GPU: pops command
# lines from a shared queue file (the same flock as b42_worker.sh, so a job never runs twice) and runs each with
# --device cpu, P lanes of T threads each. Lane start / end go to logs/cpu_ledger.txt (core-hours = lanes x threads x h).
# usage: b42_cpuworker.sh <queue_file> <P> <T> <name>
Q=$1; P=${2:-4}; T=${3:-6}; NAME=${4:-c}
B=/path/to/work/b42; cd $B || exit 1; mkdir -p logs
export PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=$T MKL_NUM_THREADS=$T CUDA_VISIBLE_DEVICES=""
PY=/path/to/HOME/miniconda3/envs/dino_wm/bin/python
pop() { flock "$Q.lock" -c "head -n1 '$Q'; sed -i '1d' '$Q'"; }
lane() {
  echo "$NAME lane$1 start $(date -u +%FT%TZ) threads $T" >> logs/cpu_ledger.txt
  while true; do
    c=$(pop); [ -z "$c" ] && break
    echo "[$(date -u +%FT%TZ)] cpu lane$1 $c" >> logs/$NAME.cmds
    $PY -u $c --device cpu >> logs/$NAME.lane$1.log 2>&1
    rc=$?
    echo "[$(date -u +%FT%TZ)] cpu lane$1 rc=$rc $c" >> logs/$NAME.cmds
  done
  echo "$NAME lane$1 end $(date -u +%FT%TZ) threads $T" >> logs/cpu_ledger.txt
}
for i in $(seq 1 $P); do lane $i & done
wait
