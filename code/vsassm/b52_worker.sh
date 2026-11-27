#!/bin/bash
# B52 part (a) worker on the A800 box (prereg/B52_robustness_layers_maps_1day_prereg.md). Copy of B42's two workers
# (b42_cpuworker.sh / b42_worker.sh) with the B52 directory and PRIORITY queues: each lane pops the first line of the
# first non-empty queue file in the order given (q/final before q/tune), under the queue's flock, so no job runs twice.
#   cpu mode: no GPU; P lanes of T threads each (--device cpu); lane start / end -> logs/cpu_ledger.txt
#   gpu mode: claims ONE idle GPU (never GPU 3; < 500 MiB used and no compute process, re-checked after 20 s; lock =
#             mkdir in the claim directory shared with the other experiments; only our own lock is removed); P lanes on
#             it; occupancy -> logs/gpu_ledger.txt. Never touches another process.
# usage: b52_worker.sh cpu <P> <T> <name> <queue>...    |    b52_worker.sh gpu <P> <name> <queue>...
MODE=$1; shift
if [ "$MODE" = cpu ]; then P=$1; T=$2; NAME=$3; shift 3; else P=$1; T=4; NAME=$2; shift 2; fi
QS=("$@")
B=/path/to/work/b52; cd $B || exit 1; mkdir -p logs
export HF_HOME=/path/to/work/hf_home HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=$T MKL_NUM_THREADS=$T PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/path/to/HOME/miniconda3/envs/dino_wm/bin/python
LOCK=/path/to/work/b29/claims; mkdir -p $LOCK
pop1() { flock "$1.lock" -c "head -n1 '$1'; sed -i '1d' '$1'"; }
pop() { for q in "${QS[@]}"; do [ -s "$q" ] || continue; c=$(pop1 "$q"); [ -n "$c" ] && { echo "$c"; return; }; done; }
idle() {
  local used apps
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $1 2>/dev/null)
  apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i $1 2>/dev/null | grep -c .)
  [ -n "$used" ] && [ "$used" -lt 500 ] && [ "$apps" = 0 ]
}
claim() {
  while true; do
    for g in 0 1 2 4 5 6 7; do
      idle $g || continue
      mkdir $LOCK/gpu$g 2>/dev/null || continue
      echo "b52 $NAME $$" > $LOCK/gpu$g/owner
      sleep 20
      if idle $g; then echo $g; return; fi
      rm -f $LOCK/gpu$g/owner; rmdir $LOCK/gpu$g
    done
    sleep 60
  done
}
lane() {
  [ "$MODE" = cpu ] && echo "$NAME lane$1 start $(date -u +%FT%TZ) threads $T" >> logs/cpu_ledger.txt
  idle_rounds=0
  while true; do
    c=$(pop)
    if [ -z "$c" ]; then          # queues may still grow (the orchestrator appends cw grids and finals)
      idle_rounds=$((idle_rounds+1)); [ $idle_rounds -gt ${WAIT_ROUNDS:-30} ] && break; sleep 60; continue
    fi
    idle_rounds=0
    echo "[$(date -u +%FT%TZ)] $MODE$G lane$1 $c" >> logs/$NAME.cmds
    if [ "$MODE" = cpu ]; then CUDA_VISIBLE_DEVICES="" $PY -u $c --device cpu >> logs/$NAME.lane$1.log 2>&1
    else CUDA_VISIBLE_DEVICES=$G $PY -u $c >> logs/$NAME.lane$1.log 2>&1; fi
    rc=$?
    echo "[$(date -u +%FT%TZ)] $MODE$G lane$1 rc=$rc $c" >> logs/$NAME.cmds
  done
  [ "$MODE" = cpu ] && echo "$NAME lane$1 end $(date -u +%FT%TZ) threads $T" >> logs/cpu_ledger.txt
}
G=""
if [ "$MODE" = gpu ]; then G=$(claim); echo "$NAME gpu$G start $(date -u +%FT%TZ)" >> logs/gpu_ledger.txt; fi
for i in $(seq 1 $P); do lane $i & done
wait
if [ "$MODE" = gpu ]; then
  echo "$NAME gpu$G end $(date -u +%FT%TZ)" >> logs/gpu_ledger.txt
  rm -f $LOCK/gpu$G/owner; rmdir $LOCK/gpu$G
fi
