#!/bin/bash
# B42 worker on the A800 box (prereg/B42_label_free_discrete_prereg.md; copy of b31_worker.sh with the B42 directory). Claims ONE idle GPU (never GPU 3;
# < 500 MiB used and no compute process, re-checked 20 s later; lock = mkdir in the claim directory shared with the
# B29/B30b wrappers, so no two wrappers take the same GPU; only our own lock is removed), then runs python commands
# popped from a shared queue file, P at a time on that GPU. GPU occupancy (worker start / end) goes to
# logs/gpu_ledger.txt, the GPU-hour ledger. Never touches another process.
# usage: b31_worker.sh <queue_file> <P> <name> [<queue_file2> <P2> ...]   (later queues run on the same GPU after the
# earlier ones are empty, e.g. light jobs with 4 lanes, then Phi-2 LoRA with 2)
Q=$1; P=${2:-1}; NAME=${3:-w}; shift 3; MORE=("$@")
B=/path/to/work/b42; cd $B || exit 1; mkdir -p logs
LOCK=/path/to/work/b29/claims; mkdir -p $LOCK
export HF_HOME=/path/to/work/hf_home HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
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
      echo "b42 $NAME $$" > $LOCK/gpu$g/owner
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
while [ ${#MORE[@]} -ge 2 ]; do
  Q=${MORE[0]}; P=${MORE[1]}; MORE=("${MORE[@]:2}")
  for i in $(seq 1 $P); do lane $i & done
  wait
done
echo "$NAME gpu$G end $(date -u +%FT%TZ)" >> logs/gpu_ledger.txt
rm -f $LOCK/gpu$G/owner; rmdir $LOCK/gpu$G
