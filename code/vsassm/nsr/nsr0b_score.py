"""NSR0b independent scorer (prereg/NSR0b_prereg.md). Recomputes accuracies from nsr/out/nsr0b_preds.jsonl.gz and
r1_data/test.csv; shares no code with nsr0.py / nsr0b.py. Also reads NSR0's T records to evaluate B2 on NSR0's A2 cells.

python3 nsr/nsr0b_score.py
"""
import collections
import csv
import gzip
import json

gold = {r["id"]: r["target_text"] for r in csv.DictReader(open("r1_data/test.csv", encoding="utf-8-sig"))}


def acc(rec):
    assert set(rec["ids"]) == set(gold) and len(rec["ids"]) == len(gold)
    return sum(p == gold[i] for i, p in zip(rec["ids"], rec["pred"])) / len(gold)


def table(path, arms=None):
    A = collections.defaultdict(dict)
    for line in gzip.open(path, "rt"):
        r = json.loads(line)
        if arms is None or r["arm"] in arms:
            A[(r["cell"], r["arm"])][r["seed"]] = acc(r)
    return A


B = table("nsr/out/nsr0b_preds.jsonl.gz")
Z = table("nsr/out/nsr0_preds.jsonl.gz", arms={"T", "V-delta"})
cells = sorted({c for c, _ in B})
mean = lambda d: sum(d.values()) / len(d)
b1, b2 = [], []
print("cell          T(NSR0)  T-prune  V-delta  diff(VD-TP)  seeds VD>TP")
for c in cells:
    tp, vd = B[(c, "T-prune")], B[(c, "V-delta")]
    d = [vd[s] - tp[s] for s in sorted(tp)]
    win = sum(d) / len(d) >= 0.02 and sum(x > 0 for x in d) >= 9 and mean(vd) >= 0.50
    b1.append((c, win))
    t0, vd0 = Z[(c, "T")], Z[(c, "V-delta")]
    d0 = [vd0[s] - t0[s] for s in sorted(t0)]
    a2 = sum(d0) / len(d0) >= 0.02 and sum(x > 0 for x in d0) >= 9 and mean(vd0) >= 0.50 and c[:2] in ("L_", "F_")
    if a2:
        b2.append((c, mean(tp) >= mean(vd) - 0.02))
    assert abs(mean(vd) - mean(vd0)) < 1e-12, f"V-delta in NSR0b differs from NSR0 in {c}"
    print(f"{c:13s} {mean(t0):.4f}   {mean(tp):.4f}   {mean(vd):.4f}   {sum(d) / len(d):+.4f}      {sum(x > 0 for x in d)}/10"
          + ("   <- NSR0 A2 cell" if a2 else ""))
print(json.dumps({"B1_pass": any(w for _, w in b1), "B1_cells": [c for c, w in b1 if w],
                  "B2_cells": b2, "B2_hold": all(h for _, h in b2)}, indent=1))
