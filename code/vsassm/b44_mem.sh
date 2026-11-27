#!/bin/bash
# B44 frozen-feature arms for one variant (std | ho): exact rows at every n, naming head (3 seeds) and equalised probe
# at 800 and 8,000. The naming head at 80,000 runs as one job per seed (b44_name.py --seeds <s>).
V=$1; PY=${2:-/usr/bin/python3}
for n in 800 8000 80000; do $PY -u b44_name.py --cell $V:$n --arm exact --out out/name/qwen8_exact_${V}_n${n}.json || echo "FAILED exact $n"; done
for n in 800 8000; do $PY -u b44_name.py --cell $V:$n --arm name --out out/name/qwen8_name_${V}_n${n}.json || echo "FAILED name $n"; done
for n in 800 8000; do $PY -u b44_name.py --cell $V:$n --arm probe --out out/name/qwen8_probe_${V}_n${n}.json || echo "FAILED probe $n"; done
