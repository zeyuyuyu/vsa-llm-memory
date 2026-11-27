#!/bin/bash
# B68 after the test runs (prereg/B68_early_layer_labelfree_prereg.md): merge the per-seed finals, run the second
# scorer (b42_indep.py, unchanged) per cell in parallel on the box CPU, merge its shards into out/indep68.json, then the
# verdict. usage: b68_post.sh "<cut cells, comma-separated>"
B=/data0/path/to/HOME/b68; cd $B || exit 1
PY=/path/to/HOME/miniconda3/envs/dino_wm/bin/python
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
CUT="$1"
$PY b68_queues.py merge > logs/post_merge.log 2>&1
mkdir -p out/indep
for enc in qwen8L0 qwen8L2 qwen8L4 phi2L0 phi2L2 phi2L4; do for ar in A B C; do
  case ",$CUT," in *",${enc}_${ar},"*) continue;; esac
  ls out/final/${enc}_${ar}_*.json >/dev/null 2>&1 || continue
  nice $PY b42_indep.py --root out --arenas $ar --glob "${enc}_%s_*.json" --out out/indep/${enc}_${ar}.json > logs/indep_${enc}_${ar}.log 2>&1 &
done; done
wait
$PY -c "
import json,glob
d={}
for p in sorted(glob.glob('out/indep/*.json')): d.update(json.load(open(p)))
json.dump(d, open('out/indep68.json','w'), indent=1)
print('indep entries', len(d))
"
$PY b68_verdict.py --root out --b31 /path/to/work/b31/out --feats feats --out b68_verdict.txt --scores b68_scores.json --cut "$CUT" > logs/verdict.log 2>&1
echo "post done rc=$? $(date -u +%FT%TZ)"
