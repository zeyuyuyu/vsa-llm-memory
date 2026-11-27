"""NSR0e independent scorer: accuracies from nsr/out/nsr0e_preds.jsonl.gz and r1_data/test.csv; E1 and E2."""
import collections
import csv
import gzip
import json

gold = {r["id"]: r["target_text"] for r in csv.DictReader(open("r1_data/test.csv", encoding="utf-8-sig"))}
A = collections.defaultdict(dict)
for line in gzip.open("nsr/out/nsr0e_preds.jsonl.gz", "rt"):
    r = json.loads(line)
    assert set(r["ids"]) == set(gold) and len(r["ids"]) == len(gold)
    A[(r["cell"], r["arm"])][r["seed"]] = sum(p == gold[i] for i, p in zip(r["ids"], r["pred"])) / len(gold)
mean = lambda d: sum(d.values()) / len(d)
e1, e2_fail = [], []
print("cell          T-st+vote  V-delta-d  V-delta-0")
for c in sorted({c for c, _ in A}):
    t = A[(c, "T-stage+vote")]
    assert sorted(t) == list(range(30, 40))
    for arm in ("V-delta-d", "V-delta-0"):
        v = A[(c, arm)]
        d = [v[s] - t[s] for s in sorted(t)]
        if sum(d) / len(d) >= 0.02 and sum(x > 0 for x in d) >= 9 and mean(v) >= 0.50:
            e1.append((c, arm))
        if mean(t) < mean(v) - 0.02:
            e2_fail.append((c, arm))
    print(f"{c:13s} {mean(t):.4f}     {mean(A[(c, 'V-delta-d')]):.4f}     {mean(A[(c, 'V-delta-0')]):.4f}")
print(json.dumps({"E1_pass": bool(e1), "E1_cells": e1, "E2_hold": not e2_fail, "E2_fail_cells": e2_fail}, indent=1))
