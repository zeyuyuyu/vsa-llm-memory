"""NSR0d independent scorer: accuracies from nsr/out/nsr0d_preds.jsonl.gz and r1_data/test.csv; D1 and D2."""
import collections
import csv
import gzip
import json

gold = {r["id"]: r["target_text"] for r in csv.DictReader(open("r1_data/test.csv", encoding="utf-8-sig"))}
best = json.load(open("nsr/out/nsr0d_tuning.json"))["choice"]["best_symbolic"]
A = collections.defaultdict(dict)
for line in gzip.open("nsr/out/nsr0d_preds.jsonl.gz", "rt"):
    r = json.loads(line)
    assert set(r["ids"]) == set(gold) and len(r["ids"]) == len(gold)
    A[(r["cell"], r["arm"])][r["seed"]] = sum(p == gold[i] for i, p in zip(r["ids"], r["pred"])) / len(gold)
mean = lambda d: sum(d.values()) / len(d)
d1, d2 = [], {}
print(f"best symbolic arm (validation): {best}")
print("cell          T-stage  T-st+vote  V-delta  diff(VD-best)  seeds VD>best")
for c in sorted({c for c, _ in A}):
    b, vd = A[(c, best)], A[(c, "V-delta")]
    assert sorted(b) == list(range(20, 30))
    d = [vd[s] - b[s] for s in sorted(b)]
    md, w = sum(d) / len(d), sum(x > 0 for x in d)
    if md >= 0.02 and w >= 9 and mean(vd) >= 0.50:
        d1.append(c)
    if c in ("L_p0.1_m1", "L_p0.2_m1", "L_p0.3_m1"):
        d2[c] = mean(b) >= mean(vd) - 0.02
    print(f"{c:13s} {mean(A[(c, 'T-stage')]):.4f}   {mean(A[(c, 'T-stage+vote')]):.4f}     {mean(vd):.4f}   {md:+.4f}        {w}/10")
print(json.dumps({"D1_pass": bool(d1), "D1_cells": d1, "D2": d2, "D2_hold_all": all(d2.values())}, indent=1))
