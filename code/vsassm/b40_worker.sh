#!/bin/bash
# B40 worker on the A800 box (copied from b35_worker.sh). Claims ONE idle GPU (never GPU 3; < 500 MiB used and no
# compute process, re-checked 20 s later; lock = mkdir in the claim directory shared with the B29-B35 wrappers, so no
# two wrappers take the same GPU; only our own lock is removed), then runs python commands popped from a shared queue
# file, P at a time on that GPU. GPU occupancy goes to logs/gpu_ledger.txt. Never touches another process.
# usage: b40_worker.sh <queue_file> <P> <name>
Q=$1; P=${2:-1}; NAME=${3:-w}
B=/path/to/work/b40; cd $B || exit 1; mkdir -p logs
LOCK=/path/to/work/b29/claims; mkdir -p $LOCK
export HF_HOME=/path/to/work/hf_home HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True B40_ROOT=$B B40_MODEL=$B/llama31_8b_instruct
PY=/path/to/HOME/miniconda3/envs/dino_wm/bin/python
pop() { flock "$Q.lock" -c "head -n1 '$Q'; sed -i '1d' '$Q'"; }
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
      echo "b40 $NAME $$" > $LOCK/gpu$g/owner
      sleep 20
      if idle $g; then echo $g; return; fi
      rm -f $LOCK/gpu$g/owner; rmdir $LOCK/gpu$g
    done
    sleep 60
  done
}
[ -s "$Q" ] || { echo "empty queue $Q"; exit 0; }
G=$(claim)
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
