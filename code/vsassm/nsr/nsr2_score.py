"""NSR2 independent scorer (prereg/NSR2_prereg.md). Recomputes gold answers from the logs (nsr1_score.gold, which uses
only the catalog's constant tables), fact labels from the logs' truth, and every measure with its own AUROC (Mann-Whitney
with average ranks). Shares no code with nsr2.py / nsr2_run.py.

python3 nsr/nsr2_score.py --split nsr2 --logs nsr/data
"""
import argparse
import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from nsr1_score import gold  # noqa: E402


def auroc(score, label):
    pos = sum(label)
    neg = len(label) - pos
    if not pos or not neg:
        return float("nan")
    order = sorted(range(len(score)), key=lambda i: score[i])
    ranks = [0.0] * len(score)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and score[order[j + 1]] == score[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return (sum(r for r, l in zip(ranks, label) if l) - pos * (pos + 1) / 2) / (pos * neg)


def selective(score, ok, cov):
    k = max(1, int(round(cov * len(ok))))
    order = sorted(range(len(ok)), key=lambda i: (-score[i], i))[:k]
    return sum(ok[i] for i in order) / k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="nsr2")
    ap.add_argument("--logs", default=os.path.join(HERE, "data"))
    a = ap.parse_args()
    logs = {}
    for sc in ("warehouse", "biobank", "home", "ops"):
        for line in open(os.path.join(a.logs, f"{sc}_{a.split}.jsonl")):
            d = json.loads(line)
            logs[d["id"]] = d
    Q = [json.loads(l) for l in open(os.path.join(HERE, "out", f"nsr2_{a.split}_records.jsonl"))]
    F = [json.loads(l) for l in open(os.path.join(HERE, "out", f"nsr2_{a.split}_facts.jsonl"))]
    ok_det = [sorted(r["pred_det"]) == gold(logs[r["log"]], r["qtype"], r["target"]) for r in Q]
    ok_ns = [sorted(r["pred_ns"]) == gold(logs[r["log"]], r["qtype"], r["target"]) for r in Q]
    f_ok = [logs[f["log"]]["truth"].get(f["x"]) == f["p"] for f in F]
    f_rel = [f["rel"] for f in F]
    bins = collections.defaultdict(lambda: [0, 0.0, 0])
    for p, o in zip(f_rel, f_ok):
        b = min(int(p * 10), 9)
        bins[b][0] += 1
        bins[b][1] += p
        bins[b][2] += o
    ece = sum(abs(s / n - c / n) * n for n, s, c in bins.values()) / len(F)
    arms = list(Q[0]["conf"])
    res = {"n_questions": len(Q), "n_facts": len(F), "wrong_facts": len(F) - sum(f_ok),
           "fact_auroc": auroc(f_rel, f_ok), "fact_ece": ece,
           "acc_det": sum(ok_det) / len(Q), "acc_ns_map": sum(ok_ns) / len(Q), "arms": {}}
    for arm in arms:
        c = [r["conf"][arm] for r in Q]
        res["arms"][arm] = {"auroc": auroc(c, ok_det),
                            **{f"sel@{int(v * 100)}": selective(c, ok_det, v) for v in (0.7, 0.8, 0.9)}}
    if "marg_problog" in Q[0]:
        res["problog_max_abs_diff"] = max((abs(r["marg"][x] - r["marg_problog"].get(x, 0.0)) for r in Q for x in r["marg"]),
                                          default=0.0)
    per = collections.defaultdict(lambda: [0, 0])
    for r, o in zip(Q, ok_det):
        per[r["scenario"]][0] += 1
        per[r["scenario"]][1] += o
    res["acc_det_per_scenario"] = {k: v[1] / v[0] for k, v in per.items()}
    ns = res["arms"]["ns"]["auroc"]
    best_base = max(res["arms"][k]["auroc"] for k in ("min_rel", "human", "size", "log_rel"))
    vsa2048 = sum(res["arms"][f"vsa2048_s{k}"]["auroc"] for k in range(3)) / 3
    res["vsa_mean_auroc"] = {D: sum(res["arms"][f"vsa{D}_s{k}"]["auroc"] for k in range(3)) / 3 for D in (256, 1024, 2048)}
    res["bars"] = {
        "floor fact AUROC >= 0.75 and >= 8 wrong facts": res["fact_auroc"] >= 0.75 and res["wrong_facts"] >= 8,
        "B1 NS AUROC >= 0.72": (round(ns, 4), ns >= 0.72),
        "B2 NS - best of (min_rel, human, size, log_rel) >= 0.03": (round(ns - best_base, 4), ns - best_base >= 0.03),
        "B3 NS sel@80 - det acc >= 0.025": (round(res["arms"]["ns"]["sel@80"] - res["acc_det"], 4),
                                            res["arms"]["ns"]["sel@80"] - res["acc_det"] >= 0.025),
        "B4 ProbLog max |diff| <= 1e-6": res.get("problog_max_abs_diff", float("nan")) <= 1e-6,
        "B5 NS - structure-only (ns_const) >= 0.03": (round(ns - res["arms"]["ns_const"]["auroc"], 4),
                                                      ns - res["arms"]["ns_const"]["auroc"] >= 0.03),
        "B5b fact-level targeting: NS - permuted (reported; predicted <= 0.01)": round(ns - res["arms"]["negctl"]["auroc"], 4),
        "B6 VSA D=2048 (mean of 3 codebooks) within 0.02 of exact": (round(vsa2048 - ns, 4), abs(vsa2048 - ns) <= 0.02),
    }
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
