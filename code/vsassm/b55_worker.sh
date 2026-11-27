#!/bin/bash
# B55 worker on the A800 box (from b50_worker.sh): claims ONE idle GPU (never GPU 3; < 500 MiB used and no compute
# process, re-checked 20 s later; lock = mkdir in the claim directory shared with the B29-B54 wrappers; only our own
# lock is removed), runs the smoke on TRAINING rows (16 per task), and only if it succeeds runs the registered test
# jobs in P lanes (one process per lane, model loaded once per lane). Never touches another process.
# usage: b55_worker.sh <P>
P=${1:-2}
B=/path/to/work/b55; cd $B || exit 1; mkdir -p logs
LOCK=/path/to/work/b29/claims; mkdir -p $LOCK
export HF_HOME=/path/to/work/hf_home HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export B55_ROOT=$B B50_ROOT=/path/to/work/b50 B40_ROOT=/path/to/work/b40 B40_MODEL=/path/to/work/b40/llama31_8b_instruct
PY=/path/to/HOME/miniconda3/envs/dino_wm/bin/python
idle() {
  local used apps
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $1 2>/dev/null)
  apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i $1 2>/dev/null | grep -c .)
  [ -n "$used" ] && [ "$used" -lt 500 ] && [ "$apps" = 0 ]
}
claim() {
  while true; do
    [ -e $B/STOP ] && return 1
    for g in 0 1 2 4 5 6 7; do
      idle $g || continue
      mkdir $LOCK/gpu$g 2>/dev/null || continue
      echo "b55 $$" > $LOCK/gpu$g/owner
      sleep 20
      if idle $g; then echo $g; return 0; fi
      rm -f $LOCK/gpu$g/owner; rmdir $LOCK/gpu$g
    done
    sleep 60
  done
}
G=$(claim) || { echo "stopped before claim $(date -u +%FT%TZ)" >> logs/gpu_ledger.txt; exit 0; }
echo "b55 gpu$G start $(date -u +%FT%TZ)" >> logs/gpu_ledger.txt
md5sum b40_core.py b40_data.py b40_run.py b55_run.py >> logs/gpu_ledger.txt
CUDA_VISIBLE_DEVICES=$G $PY -u b55_run.py run --jobs jobs_a800.json --limit_split train:16 > logs/smoke.log 2>&1
rc=$?; echo "smoke rc=$rc $(date -u +%FT%TZ)" >> logs/gpu_ledger.txt
if [ $rc = 0 ] && [ ! -e $B/STOP ]; then
  echo "test start $(date -u +%FT%TZ)" >> logs/gpu_ledger.txt
  for i in $(seq 0 $((P-1))); do
    CUDA_VISIBLE_DEVICES=$G $PY -u b55_run.py run --jobs jobs_a800.json --lane $i --lanes $P > logs/lane$i.log 2>&1 &
  done
  wait
fi
echo "b55 gpu$G end $(date -u +%FT%TZ)" >> logs/gpu_ledger.txt
rm -f $LOCK/gpu$G/owner; rmdir $LOCK/gpu$G
