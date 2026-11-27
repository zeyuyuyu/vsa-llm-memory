#!/bin/bash
# B68 worker on the A800 box (prereg/B68_early_layer_labelfree_prereg.md); a copy of b52_worker.sh's GPU mode with the
# B68 directory, the GPUs this experiment may use (1, 2, 4, 5, 6, 7: NEVER 0 or 3) and the idle rule of the brief
# (memory.used < 1,000 MiB and no compute process, re-checked after 20 s; lock = mkdir in the claim directory shared
# with the other experiments; only our own lock is removed). P lanes on the claimed GPU; each lane pops the first line
# of the first non-empty queue file in the order given, under the queue's flock, so no job runs twice. Never touches
# another process. Occupancy -> logs/gpu_ledger.txt.
# usage: b68_worker.sh <P> <T> <name> <queue>...
P=$1; T=$2; NAME=$3; shift 3
QS=("$@")
B=/data0/path/to/HOME/b68; cd $B || exit 1; mkdir -p logs
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
  [ -n "$used" ] && [ "$used" -lt 1000 ] && [ "$apps" = 0 ]
}
claim() {
  local tries=0
  while true; do
    for g in 1 2 4 5 6 7; do
      idle $g || continue
      mkdir $LOCK/gpu$g 2>/dev/null || continue
      echo "b68 $NAME $$" > $LOCK/gpu$g/owner
      sleep 20
      if idle $g; then echo $g; return; fi
      rm -f $LOCK/gpu$g/owner; rmdir $LOCK/gpu$g
    done
    tries=$((tries+1)); [ $tries -gt ${CLAIM_ROUNDS:-120} ] && { echo none; return; }
    sleep 30
  done
}
lane() {
  idle_rounds=0
  while true; do
    c=$(pop)
    if [ -z "$c" ]; then          # queues may still grow (the orchestrator appends cw grids; finals are released later)
      idle_rounds=$((idle_rounds+1)); [ $idle_rounds -gt ${WAIT_ROUNDS:-30} ] && break; sleep 30; continue
    fi
    idle_rounds=0
    echo "[$(date -u +%FT%TZ)] gpu$G lane$1 $c" >> logs/$NAME.cmds
    CUDA_VISIBLE_DEVICES=$G $PY -u $c >> logs/$NAME.lane$1.log 2>&1
    rc=$?
    echo "[$(date -u +%FT%TZ)] gpu$G lane$1 rc=$rc $c" >> logs/$NAME.cmds
  done
}
G=$(claim)
[ "$G" = none ] && { echo "$NAME no idle GPU $(date -u +%FT%TZ)" >> logs/gpu_ledger.txt; exit 0; }
echo "$NAME gpu$G start $(date -u +%FT%TZ) lanes $P" >> logs/gpu_ledger.txt
for i in $(seq 1 $P); do lane $i & done
wait
echo "$NAME gpu$G end $(date -u +%FT%TZ)" >> logs/gpu_ledger.txt
rm -f $LOCK/gpu$G/owner; rmdir $LOCK/gpu$G
