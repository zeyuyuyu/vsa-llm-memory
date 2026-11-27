#!/bin/bash
# B65 worker on the A800 box (from b40_worker.sh). Claims ONE idle GPU from 1 2 4 5 6 7 (never 0 = another user's job,
# never 3; idle = memory.used < 1024 MiB and no compute process, re-checked 20 s later; lock = mkdir in the claim
# directory shared with the B29-B55 wrappers, so no two wrappers take the same GPU; only our own lock is removed), then
# runs python commands popped from a shared queue file, P at a time on that GPU. Several workers may share one queue.
# GPU occupancy -> logs/gpu_ledger.txt. Never touches another process. No retries: a failed command is logged (rc) only.
# usage: b65_worker.sh <queue_file> <P> <name>
Q=$1; P=${2:-1}; NAME=${3:-w}
B=/path/to/work/b65; cd $B || exit 1; mkdir -p logs
LOCK=/path/to/work/b29/claims; mkdir -p $LOCK
export HF_HOME=/path/to/work/hf_home HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export B65_ROOT=$B B40_ROOT=/path/to/work/b40 B40_MODEL=/path/to/work/b40/llama31_8b_instruct
PY=/path/to/HOME/miniconda3/envs/dino_wm/bin/python
pop() { flock "$Q.lock" -c "head -n1 '$Q'; sed -i '1d' '$Q'"; }
idle() {
  local used apps
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $1 2>/dev/null)
  apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i $1 2>/dev/null | grep -c .)
  [ -n "$used" ] && [ "$used" -lt 1024 ] && [ "$apps" = 0 ]
}
claim() {
  local n=0
  while [ $n -lt 240 ]; do
    for g in 1 2 4 5 6 7; do
      idle $g || continue
      mkdir $LOCK/gpu$g 2>/dev/null || continue
      echo "b65 $NAME $$" > $LOCK/gpu$g/owner
      sleep 20
      if idle $g; then echo $g; return 0; fi
      rm -f $LOCK/gpu$g/owner; rmdir $LOCK/gpu$g
    done
    n=$((n+1)); sleep 30
  done
  return 1
}
[ -s "$Q" ] || { echo "empty queue $Q"; exit 0; }
G=$(claim) || { echo "$NAME no idle GPU $(date -u +%FT%TZ)" >> logs/gpu_ledger.txt; exit 0; }
echo "$NAME gpu$G start $(date -u +%FT%TZ)" >> logs/gpu_ledger.txt
lane() {
  while true; do
    c=$(pop); [ -z "$c" ] && break
    echo "[$(date -u +%FT%TZ)] gpu$G lane$1 $c" >> logs/$NAME.cmds
    CUDA_VISIBLE_DEVICES=$G $PY -u $c >> logs/$NAME.lane$1.log 2>&1
    rc=$?
    echo "[$(date -u +%FT%TZ)] gpu$G lane$1 rc=$rc $c" >> logs/$NAME.cmds
  done
}
for i in $(seq 1 $P); do lane $i & done
wait
echo "$NAME gpu$G end $(date -u +%FT%TZ)" >> logs/gpu_ledger.txt
rm -f $LOCK/gpu$G/owner; rmdir $LOCK/gpu$G
