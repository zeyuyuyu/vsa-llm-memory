#!/bin/bash
# B52 part (a) after the qwen8L27 x C finals (prereg/B52_robustness_layers_maps_1day_prereg.md): second scorer
# (b42_indep.py), verdict (b52_verdict_a.py), hand-reading (b42_handread.py). On the box, in /path/to/work/b52.
cd /path/to/work/b52 || exit 1
PY=/path/to/HOME/miniconda3/envs/dino_wm/bin/python
export OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=""
mkdir -p results
mkdir -p out/final; $PY -u b52_merge_seeds.py --root out --cell qwen8L27_C > logs/merge.log 2>&1
$PY -u b42_indep.py --root out --arenas C --glob 'qwen8L27_%s_*.json' --out out/indep52.json > logs/indep52.log 2>&1
$PY -u b52_verdict_a.py --root out --b31 /path/to/work/b31/out --out results/b52a_verdict.txt --scores results/b52a_scores.json --cells qwen8L27_C > logs/verdict_a.log 2>&1
$PY -u b42_handread.py --enc qwen8L27 --root out --n 12 > results/b52a_handread_qwen8L27_C.txt 2> logs/handread.err
echo "a_post done $(date -u +%FT%TZ)"
