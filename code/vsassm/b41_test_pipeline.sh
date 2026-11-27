#!/bin/sh
# B41 test pipeline (box). Started only after the dev record is committed. Runs once:
#   test bank features -> test logits -> 45 engine runs (seeds 1-3) on GPUs 4-7 -> exploratory D = 262,144 VSA-only
#   runs (N >= 320; not in the bar) -> second implementation (G0) on CPU -> test verdict.
set -e
PY=/path/to/HOME/miniconda3/envs/dino_wm/bin/python
R=/path/to/work/b41
RHOS="0.03 0.06 0.2"
cd $R/code
echo "start $(date -u)"
CUDA_VISIBLE_DEVICES=7 $PY b41_feats.py --split test --rhos $RHOS > $R/logs/feats_test.log 2>&1
CUDA_VISIBLE_DEVICES=7 $PY b41_head.py logits --split test > $R/logs/logits_test.log 2>&1
echo "logits done $(date -u)"
for w in 0 1 2 3; do
  $PY b41_jobs.py --split test --rhos $RHOS --seeds 1 2 3 --nq 1000 --worker $w --nworkers 4 > $R/logs/testjobs_$w.sh
  CUDA_VISIBLE_DEVICES=$((4 + w)) sh $R/logs/testjobs_$w.sh > $R/logs/test_w$w.log 2>&1 &
done
wait
echo "engine done $(date -u)"
mkdir -p $R/runs_x
i=0
for N in 100000 32000 3200 320; do for s in 1 2 3; do for r in $RHOS; do
  out=$(printf "%s/runs_x/test_s%d_N%d_r%.3f.npz" $R $s $N $r)
  echo "[ -f $out ] || $PY $R/code/b41_engine.py --split test --seed $s --N $N --rho $r --nq 1000 --logits $R/head/test_logits.npy --out $out --Ds 262144 --no_tables" >> $R/logs/xjobs_$((i % 4)).sh
  i=$((i + 1))
done; done; done
for w in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$((4 + w)) sh $R/logs/xjobs_$w.sh > $R/logs/x_w$w.log 2>&1 &
done
mkdir -p $R/indep
$PY b41_indep_jobs.py --split test --rhos $RHOS --seeds 1 2 3 > $R/logs/indepjobs.sh
xargs -P 16 -I{} sh -c "{}" < $R/logs/indepjobs.sh || true
wait
echo "exploratory + indep done $(date -u)"
$PY b41_verdict.py --split test --root $R/runs --seeds 1 2 3 --rhos $RHOS --choice $R/results/b41_dev_choice.json \
  --out $R/results/b41_test_verdict.txt > $R/logs/test_verdict.log 2>&1
echo "PIPELINE DONE $(date -u)"
