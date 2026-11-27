"""NSR0b exploratory upper bound: per cell, the best T-prune (s, q) and the best V-delta (tau, alpha), each chosen on
test by mean accuracy over 10 seeds. Exploratory only (prereg/NSR0b_prereg.md); never quoted as a result."""
import collections
import csv
import gzip
import json

gold = {r["id"]: r["target_text"] for r in csv.DictReader(open("r1_data/test.csv", encoding="utf-8-sig"))}
A = collections.defaultdict(list)
for line in gzip.open("nsr/out/nsr0b_upper.jsonl.gz", "rt"):
    r = json.loads(line)
    A[(r["cell"], r["arm"])].append(sum(p == gold[i] for i, p in zip(r["ids"], r["pred"])) / len(gold))
cells = sorted({c for c, _ in A})
out, vd_better = [], []
print("cell          best T-prune (s|q)      best V-delta (tau|alpha)   diff")
for c in cells:
    tp = max(((a, sum(v) / len(v)) for (cc, a), v in A.items() if cc == c and a.startswith("T-prune")), key=lambda x: x[1])
    vd = max(((a, sum(v) / len(v)) for (cc, a), v in A.items() if cc == c and a.startswith("V-delta")), key=lambda x: x[1])
    d = vd[1] - tp[1]
    if d > 0.005:
        vd_better.append((c, round(d, 4)))
    print(f"{c:13s} {tp[1]:.4f} {tp[0][8:]:10s}   {vd[1]:.4f} {vd[0][8:]:10s}   {d:+.4f}")
print(json.dumps({"cells_where_best_VD_beats_best_TP_by_>0.005": vd_better}))
