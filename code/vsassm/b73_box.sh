#!/bin/bash
# B73 on the A800 box (prereg/B73_confirmatory_prereg.md, 'Compute'): fresh directory ~/b73_run, code copied from the
# repository (md5 in ~/b73_run/code_md5.txt). GPU encoding: one model per GPU, a GPU is used only if idle (memory.used
# < 1024 MiB and no compute app, re-checked 20 s later), only GPUs 1 2 4 5 6 7 (never 0 or 3), lock = mkdir in the
# claim directory shared with the earlier wrappers; GPU occupancy to logs/gpu_ledger.txt. CPU runs: one process per
# unit, OMP / BLAS threads 1, nice 5, at most 48 workers. Never touches another process.
# usage (on the box): bash b73_box.sh setup
#                     bash b73_box.sh embed MODEL      (qwen8 | b46 | llama | mistral)
#                     bash b73_box.sh run PLAN.txt WORKERS
set -u
W=$HOME/b73_run
PY=/path/to/HOME/miniconda3/envs/dino_wm/bin/python
export B73_ROOT=$W/fresh B73_OUT=$W/run
LOCK=/path/to/work/b29/claims
idle() {
  local used apps
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $1 2>/dev/null)
  apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i $1 2>/dev/null | grep -c .)
  [ -n "$used" ] && [ "$used" -lt 1024 ] && [ "$apps" = 0 ]
}
claim() {  # one pass over the allowed GPUs; prints the claimed GPU or nothing
  for g in 1 2 4 5 6 7; do
    idle $g || continue
    mkdir $LOCK/gpu$g 2>/dev/null || continue
    echo "b73 $1 $$" > $LOCK/gpu$g/owner
    sleep 20
    if idle $g; then echo $g; return; fi
    rm -f $LOCK/gpu$g/owner; rmdir $LOCK/gpu$g
  done
}
case $1 in
setup)
  mkdir -p $W/fresh/b32_data $W/run/logs $W/logs $LOCK
  (cd $W && md5sum *.py > code_md5.txt; cat code_md5.txt) ;;
embed)
  M=$2
  cd $W/fresh || exit 1
  export HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
  G=$(claim $M)
  [ -z "$G" ] && { echo "NO_IDLE_GPU $M $(date +%T)"; exit 2; }
  echo "b73 $M gpu$G start $(date -u +%FT%TZ)" >> $W/logs/gpu_ledger.txt
  case $M in
    qwen8) P=$(ls -d /path/to/HOME/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218/)
           CUDA_VISIBLE_DEVICES=$G $PY -u $W/b32_embed.py lm $P qwen8 --layers 9; rc=$?
           SRC=b32_data/emb_qwen8_mean_L9.npy; DST=b32_data/emb_qwen8_mean_L9_f16.npy ;;
    b46)   P=/path/to/HOME/.cache/huggingface/hub/models--ibm-granite--granite-3.3-8b-instruct/snapshots/51dd4bc2ade4059a6bd87649d68aa11e4fb2529b/
           CUDA_VISIBLE_DEVICES=$G $PY -u $W/b46_embed.py $P granite33i --layers 30 --out b32_data; rc=$?
           SRC=b32_data/emb_granite33i_mean_L30.npy; DST=b32_data/emb_b46_granite33i_mean_L30_f16.npy ;;
    llama) CUDA_VISIBLE_DEVICES=$G $PY -u $W/b46_embed.py /path/to/work/b40/llama31_8b_instruct llama31i --layers 24 --out b32_data; rc=$?
           SRC=b32_data/emb_llama31i_mean_L24.npy; DST=b32_data/emb_llama31i_mean_L24_f16.npy ;;
    mistral) CUDA_VISIBLE_DEVICES=$G $PY -u $W/b46_embed.py /path/to/work/b46/mistral7b_instruct_v03 mistral7i --layers 24 --out b32_data; rc=$?
           SRC=b32_data/emb_mistral7i_eol_L24.npy; DST=b32_data/emb_mistral7i_eol_L24_f16.npy ;;
  esac
  echo "b73 $M gpu$G end $(date -u +%FT%TZ) rc=$rc" >> $W/logs/gpu_ledger.txt
  rm -f $LOCK/gpu$G/owner; rmdir $LOCK/gpu$G
  $PY -c "import numpy as np; X = np.load('$SRC'); np.save('$DST', X.astype(np.float16)); print('$DST', X.shape)"
  md5sum $DST
  echo EMBED_DONE $M rc=$rc ;;
run)
  cd $W || exit 1
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
  nohup nice -n 5 $PY b73_run.py runplan $W/$2 --workers $3 > $W/logs/runplan_${2%.txt}.log 2>&1 &
  echo "started $2 pid $!" ;;
*) echo "unknown $1" ;;
esac
