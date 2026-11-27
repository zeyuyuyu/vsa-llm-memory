#!/bin/sh
R=/path/to/work/b41; P=/path/to/HOME/miniconda3/envs/dino_wm/bin/python
until grep -q "PIPELINE DONE" $R/logs/test_pipeline.log; do sleep 10; done
cd $R/code
$P b41_report.py --split test --seeds 1 2 3 --scores $R/results/b41_test_verdict.json --runs $R/runs --runs_x $R/runs_x --dev_head $R/head/choice.json --out $R/results/b41_report.txt > $R/logs/report.log 2>&1
$P b41_ledger.py $R > $R/results/b41_gpu_ledger.txt 2>&1
$P - > $R/results/b41_indep_summary.txt 2>&1 <<'PY'
import json, glob
fs = sorted(glob.glob("/path/to/work/b41/indep/*_indep.json"))
n_cfg = n_q = 0; low = []; cost_bad = []
agree_tab = []
for f in fs:
    d = json.load(open(f))
    for k, v in d.items():
        n_cfg += 1; n_q += v["n"]
        if v["cost_mismatch"]:
            cost_bad.append((f.split("/")[-1], k, v["cost_mismatch"]))
        if v["answer_agree"] is not None:
            if not k.startswith("VSA"):
                agree_tab.append(v["answer_agree"])
            if v["answer_agree"] < 0.99:
                low.append((f.split("/")[-1], k, round(v["answer_agree"], 4)))
print("G0 second implementation: %d run files, %d (run, config) pairs, %d (query, h) comparisons" % (len(fs), n_cfg, n_q))
print("table/EXACT/ENUM configs: min answer agreement %.4f over %d pairs" % (min(agree_tab), len(agree_tab)))
print("cost-count mismatches (ops/traffic/state, 1e-6 rel): %d %s" % (len(cost_bad), cost_bad[:5]))
print("configs below 99%% answer agreement: %d" % len(low))
for x in low:
    print("  ", x)
PY
echo POST DONE > $R/logs/post.log
