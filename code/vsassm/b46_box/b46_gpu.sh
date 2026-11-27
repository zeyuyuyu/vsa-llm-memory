#!/bin/bash
# B46 GPU job on the A800 box (prereg/B46_second_llm_family_prereg.md): claims ONE idle GPU (never GPU 3; < 500 MiB used
# and no compute process, re-checked 20 s later; lock = mkdir in the claim directory shared with the B29-B44 wrappers, so
# no two wrappers take the same GPU; only our own lock is removed), runs b46_embed.py for the given models one after the
# other, then releases the GPU. GPU occupancy goes to logs/gpu_ledger.txt. Never touches another process.
# usage: b46_gpu.sh NAME MODEL... with MODEL in llama31i mistral7i granite33i qwen8chk
NAME=$1; shift
B=/path/to/work/b46; cd $B || exit 1; mkdir -p logs b32_data
LOCK=/path/to/work/b29/claims; mkdir -p $LOCK
export HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/path/to/HOME/miniconda3/envs/dino_wm/bin/python
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
      echo "b46 $NAME $$" > $LOCK/gpu$g/owner
      sleep 20
      if idle $g; then echo $g; return; fi
      rm -f $LOCK/gpu$g/owner; rmdir $LOCK/gpu$g
    done
    sleep 60
  done
}
spec() {  # model -> "path layers"
  case $1 in
    llama31i) echo "/path/to/work/b40/llama31_8b_instruct 8,16,24,32" ;;
    mistral7i) echo "$B/mistral7b_instruct_v03 8,16,24,32" ;;
    granite33i) echo "$(ls -d /path/to/HOME/.cache/huggingface/hub/models--ibm-granite--granite-3.3-8b-instruct/snapshots/*/ | head -1) 10,20,30,40" ;;
    qwen8chk) echo "$(ls -d /path/to/HOME/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/*/ | head -1) 9" ;;
  esac
}
for m in "$@"; do
  if [ $m = mistral7i ]; then  # the download must have finished with every sha256 verified
    until grep -q ALLDONE logs/dl_mistral.log 2>/dev/null; do sleep 60; done
    grep -q SHA_OK logs/dl_mistral.log || { echo "Mistral download not verified: skipped"; continue; }
  fi
done
G=$(claim)
echo "$NAME gpu$G start $(date -u +%FT%TZ)" >> logs/gpu_ledger.txt
for m in "$@"; do
  set -- $(spec $m)
  [ $m = mistral7i ] && ! grep -q SHA_OK logs/dl_mistral.log && continue
  echo "[$(date -u +%FT%TZ)] $m $1 layers $2"
  CUDA_VISIBLE_DEVICES=$G $PY -u b46_embed.py $1 $m --layers $2
  echo "[$(date -u +%FT%TZ)] $m rc=$?"
done
echo "$NAME gpu$G end $(date -u +%FT%TZ)" >> logs/gpu_ledger.txt
rm -f $LOCK/gpu$G/owner; rmdir $LOCK/gpu$G
echo EMBED_DONE $NAME
