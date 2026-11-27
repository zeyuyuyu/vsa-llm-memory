#!/bin/bash
# B30b job wrapper on the A800 box (prereg/B30b_referee_baselines_prereg.md). Claims N idle GPUs (never GPU 3; < 500 MiB
# used and no compute process, re-checked 20 s later), runs one command on them, and appends start / end and the GPU
# list to logs/gpu_ledger.txt (the GPU-hour ledger). Locks: mkdir in B29's claim directory, so that B29's wrapper and
# this one never take the same GPU (each removes only its own locks). Never touches another process.
# usage: b30b_run.sh <name> <n_gpus> <env: hf|vllm> <script> [args...]
NAME=$1; N=$2; ENV=$3; shift 3
B=/path/to/work/b30b; cd $B || exit 1; mkdir -p logs
LOCK=/path/to/work/b29/claims; mkdir -p $LOCK
export HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True BENCH_ROOT=/path/to/work/bench
case $ENV in
  hf) PY=/path/to/HOME/miniconda3/envs/dino_wm/bin/python ;;
  vllm) PY=/path/to/HOME/miniconda3/envs/rsi-vllm11/bin/python; export VLLM_WORKER_MULTIPROC_METHOD=spawn
        export NCCL_P2P_DISABLE=1 ;;  # compute note 1: NCCL P2P over these PCIe pairs crashed (GPUs 0-1)
esac
idle() {  # GPU $1 has < 500 MiB in use and no compute process on it
  local used apps
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $1 2>/dev/null)
  apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i $1 2>/dev/null | grep -c .)
  [ -n "$used" ] && [ "$used" -lt 500 ] && [ "$apps" = 0 ]
}
claim() {
  local got=()
  while true; do
    got=()
    for g in 0 1 2 4 5 6 7; do
      idle $g || continue
      mkdir $LOCK/gpu$g 2>/dev/null || continue
      echo "b30b $NAME $$" > $LOCK/gpu$g/owner
      got+=($g)
      [ ${#got[@]} -ge $N ] && break
    done
    if [ ${#got[@]} -ge $N ]; then
      sleep 20
      ok=1
      for g in "${got[@]}"; do idle $g || ok=0; done
      if [ $ok = 1 ]; then echo "${got[@]}"; return; fi
    fi
    for g in "${got[@]}"; do rm -f $LOCK/gpu$g/owner; rmdir $LOCK/gpu$g; done
    sleep 60
  done
}
G=$(claim)
release() { for g in $G; do rm -f $LOCK/gpu$g/owner; rmdir $LOCK/gpu$g 2>/dev/null; done; }
trap release EXIT
GL=$(echo $G | tr ' ' ',')
echo "$NAME gpus=$GL start $(date -u +%FT%TZ)" >> logs/gpu_ledger.txt
CUDA_VISIBLE_DEVICES=$GL $PY -I -B -u "$@"
RC=$?
echo "$NAME gpus=$GL end $(date -u +%FT%TZ) rc=$RC" >> logs/gpu_ledger.txt
exit $RC
