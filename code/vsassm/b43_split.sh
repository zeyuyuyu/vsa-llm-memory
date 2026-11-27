#!/bin/bash
# B43 compute note 1: cancel a 128k job right after a completed batch and run its remaining samples as slices.
# usage (PACE login node): bash b43_split.sh JOBID TASK START1 STOP1 [START2 STOP2 ...]
cd /path/to/pace-scratch/b43 || exit 1
JOB=$1; T=$2; shift 2
Q8=/path/to/pace-scratch/hf/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
R=/path/to/pace-scratch/bench/ruler
scancel $JOB
sleep 5
echo "done in main file: $(wc -l < out/coa_ruler_131072_$T.jsonl)"
while [ $# -ge 2 ]; do
  sbatch -J b43_r128s_${T}_$1 -t 4:00:00 b43_pace.sbatch b43_coa.py ruler out/coa_ruler_131072_${T}_s$1.jsonl \
    --model $Q8 --root $R --length 131072 --tasks $T --n 100 --bs 16 --start $1 --stop $2
  shift 2
done
