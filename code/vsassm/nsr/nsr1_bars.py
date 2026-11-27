"""NSR1 bars (prereg/NSR1_prereg.md + amendment 1) from the independent scorer's JSON (nsr/nsr1_score.py) and the
per-question answer files. Run after every arm is parsed; prints one block per registered bar.

python3 nsr/nsr1_bars.py nsr/out/nsr1_score_nsr1.json
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
S = json.load(open(sys.argv[1]))


def preds(arm):
    return {(r["log"], r["qtype"], r["target"]): (sorted(r["pred"]) if r["pred"] is not None else None)
            for r in map(json.loads, open(os.path.join(HERE, "out", f"nsr1_nsr1_{arm}.jsonl")))}


a1 = S["arm1"]["ALL"]
a3 = S["arm3_qwen3-32b_forced"]["ALL"]
a4 = S["arm4"]["ALL"]
none = S["always_none"]
p1, p2a, p2b = preds("arm1"), preds("arm2a"), preds("arm2b")
same_2b = sum(p1[k] == p2b[k] for k in p1) / len(p1)
same_2a = sum(p1[k] == p2a[k] for k in p1) / len(p1)
out = {
    "n_questions": S["n_questions"], "always_none": none,
    "P1 arm1 >= 0.80": (a1, a1 >= 0.80),
    "P2 arm1 - arm3(32B, forced) >= 0.20, arm1 >= 0.60": (round(a1 - a3, 4), a1 - a3 >= 0.20 and a1 >= 0.60),
    "P3 arm1 - arm4(SoR) >= 0.30": (round(a1 - a4, 4), a1 - a4 >= 0.30),
    "P3 vs always-none (reported)": round(a1 - none, 4),
    "P4 arm1 == arm2b on >= 99%": (round(same_2b, 4), same_2b >= 0.99),
    "P4 arm1 == arm2a on >= 99%": (round(same_2a, 4), same_2a >= 0.99),
    "P5 trace validity == 1.00": (S["trace"]["validity"], S["trace"]["validity"] == 1.0),
    "trace correctness (reported)": S["trace"]["correctness"],
    "sanity == 1.000": (S["sanity"]["ALL"], S["sanity"]["ALL"] == 1.0),
    "negctl <= always-none + 0.05": (S["negctl"]["ALL"], S["negctl"]["ALL"] <= none + 0.05),
    "D1 arm3b(32B over RM state) vs arm2b (reported)": (S["arm3b_qwen3-32b_forced"]["ALL"], S["arm2b"]["ALL"]),
    "arm3 strict / forced (32B)": (S["arm3_qwen3-32b_strict"]["ALL"], a3),
    "arm3 8B strict / forced": (S["arm3_qwen3-8b_strict"]["ALL"], S["arm3_qwen3-8b_forced"]["ALL"]),
}
print(json.dumps(out, indent=1))
