#!/bin/bash
# B33 worker (copy of b28_worker.sh; directory /path/to/work/b33, PYTHONHASHSEED=0, HF datasets offline) on the A800 box: claim ONE idle GPU (never GPU 3; < 500 MiB used, re-checked 20 s later; a mkdir lock
# keeps two B33 workers off the same GPU), then run python commands popped from a shared queue file, P at a time on
# that GPU. GPU occupancy (start/end per worker) goes to logs/gpu_ledger.txt for the GPU-hour count.
# usage: b27_worker.sh <queue_file> <P> <name>
Q=$1; P=${2:-1}; NAME=${3:-w}
B=/path/to/work/b33; cd $B || exit 1; mkdir -p claims logs
export HF_HOME=/path/to/work/hf_home HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
PY=/path/to/HOME/miniconda3/envs/dino_wm/bin/python
pop() { flock "$Q.lock" -c "head -n1 '$Q'; sed -i '1d' '$Q'"; }
claim() {
  while true; do
    for g in 0 2 4 5 6 7 1; do
      used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g 2>/dev/null)
      { [ -n "$used" ] && [ "$used" -lt 500 ]; } || continue
      mkdir claims/gpu$g 2>/dev/null || continue
      sleep 20
      used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g 2>/dev/null)
      if [ -n "$used" ] && [ "$used" -lt 500 ]; then echo $g; return; fi
      rmdir claims/gpu$g
    done
    sleep 60
  done
}
[ -s "$Q" ] || { echo "empty queue $Q"; exit 0; }
G=$(claim)
echo $$ > claims/gpu$G/pid
echo "$NAME gpu$G start $(date -u +%FT%TZ)" >> logs/gpu_ledger.txt
lane() {
  while true; do
    c=$(pop); [ -z "$c" ] && break
    echo "[$(date -u +%FT%TZ)] gpu$G lane$1 $c" >> logs/$NAME.cmds
    CUDA_VISIBLE_DEVICES=$G $PY -u $c >> logs/$NAME.lane$1.log 2>&1
    echo "[$(date -u +%FT%TZ)] gpu$G lane$1 rc=$? $c" >> logs/$NAME.cmds
  done
}
for i in $(seq 1 $P); do lane $i & done
wait
echo "$NAME gpu$G end $(date -u +%FT%TZ)" >> logs/gpu_ledger.txt
rm -f claims/gpu$G/pid; rmdir claims/gpu$G
