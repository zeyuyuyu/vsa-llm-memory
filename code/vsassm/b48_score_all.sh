#!/bin/bash
# B48 on the A800 box: after the finals and the LM test runs (CPU only): part B scoring of every arm on the test split,
# the second scorer (A and B), the verdict and the hand-reading files.
cd /path/to/work/b48 || exit 1
PY=/path/to/HOME/miniconda3/envs/dino_wm/bin/python
export OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=""
for k in W P; do
  $PY b48_alias.py score --kind $k --split test --head out/b/final/${k}_head.json --lm out/lm/${k}_test.json \
      --out out/b/scores_${k}_test.json > logs/score_${k}.log 2>&1 &
done
wait
$PY b48_indep.py A --root out --out out/indepA.json > logs/indepA.log 2>&1 &
$PY b48_indep.py B --root out --out out/indepB.json > logs/indepB.log 2>&1 &
wait
