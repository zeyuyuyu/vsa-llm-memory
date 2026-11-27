#!/bin/bash
# B43 job wrapper (copy of b29_run.sh) on the A800 box: claim N idle GPUs (never GPU 3; < 500 MiB used, re-checked 20 s later; a mkdir
# lock keeps two B43 jobs off the same GPU), run one command on them, and record GPU occupancy (start / end, GPUs) in
# logs/gpu_ledger.txt for the GPU-hour ledger. Never touches another process.
# usage: b29_run.sh <name> <n_gpus> <env: hf|vllm> <script> [args...]
NAME=$1; N=$2; ENV=$3; shift 3
B=/path/to/work/b43; cd $B || exit 1; mkdir -p claims logs
export HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
case $ENV in
  hf) PY=/path/to/HOME/miniconda3/envs/dino_wm/bin/python ;;
  vllm) PY=/path/to/HOME/miniconda3/envs/rsi-vllm11/bin/python; export VLLM_WORKER_MULTIPROC_METHOD=spawn ;;
esac
claim() {
  local got=()
  while true; do
    got=()
    for g in 0 1 2 4 5 6 7; do
      used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g 2>/dev/null)
      { [ -n "$used" ] && [ "$used" -lt 500 ]; } || continue
      mkdir claims/gpu$g 2>/dev/null || continue
      got+=($g)
      [ ${#got[@]} -ge $N ] && break
    done
    if [ ${#got[@]} -ge $N ]; then
      sleep 20
      ok=1
      for g in "${got[@]}"; do
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g 2>/dev/null)
        { [ -n "$used" ] && [ "$used" -lt 500 ]; } || ok=0
      done
      if [ $ok = 1 ]; then echo "${got[@]}"; return; fi
    fi
    for g in "${got[@]}"; do rmdir claims/gpu$g; done
    sleep 60
  done
}
G=$(claim)
GL=$(echo $G | tr ' ' ',')
echo "$NAME gpus=$GL start $(date -u +%FT%TZ)" >> logs/gpu_ledger.txt
CUDA_VISIBLE_DEVICES=$GL $PY -I -B -u "$@"
RC=$?
echo "$NAME gpus=$GL end $(date -u +%FT%TZ) rc=$RC" >> logs/gpu_ledger.txt
for g in $G; do rmdir claims/gpu$g; done
