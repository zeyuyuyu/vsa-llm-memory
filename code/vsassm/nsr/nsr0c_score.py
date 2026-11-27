"""NSR0c independent scorer: recomputes accuracies from nsr/out/nsr0c_preds.jsonl.gz and r1_data/test.csv."""
import collections
import csv
import gzip
import json

gold = {r["id"]: r["target_text"] for r in csv.DictReader(open("r1_data/test.csv", encoding="utf-8-sig"))}
A = collections.defaultdict(dict)
for line in gzip.open("nsr/out/nsr0c_preds.jsonl.gz", "rt"):
    r = json.loads(line)
    assert set(r["ids"]) == set(gold) and len(r["ids"]) == len(gold)
    A[(r["cell"], r["arm"])][r["seed"]] = sum(p == gold[i] for i, p in zip(r["ids"], r["pred"])) / len(gold)
mean = lambda d: sum(d.values()) / len(d)
c1, c2 = [], []
print("cell          T        T-prune  V-delta  diff(VD-TP)  seeds VD>TP")
for c in sorted({c for c, _ in A}):
    tp, vd, t = A[(c, "T-prune")], A[(c, "V-delta")], A[(c, "T")]
    assert sorted(tp) == list(range(10, 20))
    d = [vd[s] - tp[s] for s in sorted(tp)]
    md, wins = sum(d) / len(d), sum(x > 0 for x in d)
    if md >= 0.02 and wins >= 9 and mean(vd) >= 0.50:
        c1.append(c)
    if mean(vd) >= 0.50:
        c2.append((c, mean(tp) >= mean(vd) - 0.02))
    print(f"{c:13s} {mean(t):.4f}   {mean(tp):.4f}   {mean(vd):.4f}   {md:+.4f}      {wins}/10")
print(json.dumps({"C1_pass": bool(c1), "C1_cells": c1, "C2_hold": all(h for _, h in c2),
                  "C2_fail_cells": [c for c, h in c2 if not h]}, indent=1))
