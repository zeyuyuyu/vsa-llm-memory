"""NSR0 independent scorer (written before the run; prereg/NSR0_prereg.md).

Reads the per-story predictions (nsr/out/nsr0_preds.jsonl.gz, one record per cell x arm x seed with ids, preds, confs)
and recomputes every number from r1_data/test.csv on its own. It shares no code with nsr0.py. If the run's own summary
(nsr/out/nsr0_summary.json) exists, every accuracy in it must match to 1e-9.

python3 nsr/nsr0_score.py [r1_data] [nsr/out]
"""
import collections
import csv
import gzip
import json
import math
import os
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else "r1_data"
OUT = sys.argv[2] if len(sys.argv) > 2 else "nsr/out"

gold = {}
for row in csv.DictReader(open(os.path.join(ROOT, "test.csv"), encoding="utf-8-sig")):
    gold[row["id"]] = (row["target_text"], len(row["f_comb"].split("-")))


def auroc(conf, ok):
    """Mann-Whitney AUROC with average ranks for ties; None if one class is empty."""
    pos = sum(ok)
    neg = len(ok) - pos
    if pos == 0 or neg == 0:
        return None
    order = sorted(range(len(conf)), key=lambda i: conf[i])
    ranks = [0.0] * len(conf)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and conf[order[j + 1]] == conf[order[i]]:
            j += 1
        for t in range(i, j + 1):
            ranks[order[t]] = (i + j) / 2.0 + 1.0
        i = j + 1
    rank_pos = sum(r for r, o in zip(ranks, ok) if o)
    return (rank_pos - pos * (pos + 1) / 2.0) / (pos * neg)


def score(rec):
    ids, preds = rec["ids"], rec["pred"]
    assert len(ids) == len(gold) and set(ids) == set(gold), "prediction ids do not cover the test split exactly"
    ok = [preds[i] == gold[ids[i]][0] for i in range(len(ids))]
    ks = [gold[i][1] for i in ids]
    byk = collections.defaultdict(list)
    for o, k in zip(ok, ks):
        byk[k].append(o)
    long_ = [o for o, k in zip(ok, ks) if k >= 7]
    return {
        "acc": sum(ok) / len(ok),
        "acc_k7": sum(long_) / len(long_),
        "by_k": {k: sum(v) / len(v) for k, v in sorted(byk.items())},
        "unknown": sum(p == "UNKNOWN" for p in preds) / len(preds),
        "ambig": sum(p == "AMBIG" for p in preds) / len(preds),
        "auroc": auroc(rec["conf"], ok) if rec.get("conf") else None,
    }


recs = [json.loads(l) for l in gzip.open(os.path.join(OUT, "nsr0_preds.jsonl.gz"), "rt")]
S = {}  # (cell, arm) -> {seed: scores}
meta = {}
for r in recs:
    S.setdefault((r["cell"], r["arm"]), {})[r["seed"]] = score(r)
    meta[(r["cell"], r["arm"], r["seed"])] = r.get("meta", {})


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


cells = sorted({c for c, _ in S}, key=lambda c: (c[0], c))
arms = ["T", "V-dict", "V-sum", "V-delta", "V-dict-q4", "V-sum-q4"]
print("cell            arm        seeds  acc     k>=7    unk    ambig  auroc")
for c in cells:
    for a in arms:
        if (c, a) not in S:
            continue
        v = S[(c, a)]
        print(f"{c:15s} {a:10s} {len(v):3d}   {mean(x['acc'] for x in v.values()):.4f}  "
              f"{mean(x['acc_k7'] for x in v.values()):.4f}  {mean(x['unknown'] for x in v.values()):.3f}  "
              f"{mean(x['ambig'] for x in v.values()):.3f}  {mean(x['auroc'] for x in v.values()):.3f}")


def wins(prefixes):
    """A1/A2 bar: pure VSA arm beats T by >= 0.02 mean overall, diff > 0 in >= 9/10 seeds, arm acc >= 0.50."""
    out = []
    for c in cells:
        if not any(c.startswith(p) for p in prefixes) or (c, "T") not in S:
            continue
        t = S[(c, "T")]
        for a in ("V-sum", "V-delta"):
            if (c, a) not in S:
                continue
            v = S[(c, a)]
            seeds = sorted(set(t) & set(v))
            d = [v[s]["acc"] - t[s]["acc"] for s in seeds]
            acc_a = mean(v[s]["acc"] for s in seeds)
            out.append((c, a, mean(d), sum(x > 0 for x in d), len(seeds), acc_a,
                        mean(d) >= 0.02 and sum(x > 0 for x in d) >= 9 and acc_a >= 0.50))
    return out


verdict = {}
for name, pref in (("A1", ("S_", "C_")), ("A2", ("L_", "F_"))):
    rows = wins(pref)
    best = max(rows, key=lambda r: r[2]) if rows else None
    verdict[name] = {"pass": any(r[-1] for r in rows), "passing_cells": [r[:2] for r in rows if r[-1]],
                     "largest_mean_diff": best[:6] if best else None}

# A3 size
t_all = mean(x["acc"] for x in S[("S_all", "T")].values())
t_meta = meta[("S_all", "T", min(S[("S_all", "T")]))]
n_rel = t_meta["n_relations"]
table_bytes = t_meta["n_rules"] * 3 * math.ceil(math.log2(n_rel)) / 8
a3 = {}
for a, bpd in (("V-dict", 8), ("V-sum", 8), ("V-dict-q4", 0.5), ("V-sum-q4", 0.5)):
    ok_d = sorted(int(c.split("_D")[1]) for c in cells if c.startswith("Z_D") and (c, a) in S
                  and mean(x["acc"] for x in S[(c, a)].values()) >= t_all - 0.01)
    a3[a] = {"min_D": ok_d[0] if ok_d else None, "bytes": ok_d[0] * bpd if ok_d else None}
verdict["A3"] = {"table_bytes": table_bytes, "T_acc_all": t_all, "vsa": a3,
                 "pass": any(v["bytes"] is not None and v["bytes"] < table_bytes for v in a3.values())}

# A4 confidence
a4 = {}
for c in ("L_p0.3_m3", "F_p0.3_m3"):
    if (c, "T") not in S:
        continue
    at = mean(x["auroc"] for x in S[(c, "T")].values())
    for a in ("V-dict", "V-sum", "V-delta"):
        if (c, a) in S:
            av = mean(x["auroc"] for x in S[(c, a)].values())
            a4[f"{c}/{a}"] = (at, av, av - at >= 0.05)
verdict["A4"] = {"cells": a4, "pass": any(v[2] for v in a4.values())}

nc1 = mean(x["acc"] for x in S[("NC1", "T")].values())
nc2 = mean(x["acc"] for x in S[("NC2", "T")].values())
verdict["NC"] = {"NC1": nc1, "NC2": nc2, "hold": nc1 <= 0.20 and nc2 <= 0.20}
vd_all = mean(x["acc"] for x in S[("S_all", "V-dict")].values())
verdict["repro"] = {"T": t_all, "V-dict": vd_all, "R1": 1117 / 1146,
                    "hold": abs(t_all - 1117 / 1146) < 1e-9 and abs(vd_all - 1117 / 1146) < 1e-9}

summ = os.path.join(OUT, "nsr0_summary.json")
if os.path.exists(summ):
    own = json.load(open(summ))
    bad = []
    for key, v in own["acc"].items():
        c, a, s = key.split("|")
        mine = S[(c, a)][int(s)]["acc"]
        if abs(mine - v) > 1e-9:
            bad.append((key, v, mine))
    verdict["summary_match"] = {"checked": len(own["acc"]), "mismatches": bad[:10], "n_mismatch": len(bad)}

print(json.dumps(verdict, indent=1, default=str))
json.dump(verdict, open(os.path.join(OUT, "nsr0_score.json"), "w"), indent=1, default=str)
