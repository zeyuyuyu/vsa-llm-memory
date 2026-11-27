"""NSR5 scorer: gold = replay of the concatenated true events (own replay here) + nsr1_score.gold on the merged site;
scores answer files nsr/out/nsr5_K<K>_<arm>.jsonl (set-exact and micro-F1). python3 nsr/nsr5_score.py --K 8 --arms ..."""
import argparse
import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from nsr1_score import gold  # noqa: E402
import nsr5_sites as S  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--K", type=int, required=True)
ap.add_argument("--arms", nargs="+", required=True)
a = ap.parse_args()
sites = {}
for site, _ in S.sites(a.K):
    st = {}
    for line in site["lines"]:
        for ev in line.get("events") or []:
            if ev[0] == "move":
                st[ev[1]] = ev[2]
            elif ev[0] == "swap":
                moved = {o: (ev[2] if p == ev[1] else ev[1]) for o, p in st.items() if p in (ev[1], ev[2])}
                st.update(moved)
    assert st == site["truth"]
    sites[site["id"]] = site
res = {}
for arm in a.arms:
    per, f1 = collections.defaultdict(list), [0, 0, 0]
    for r in map(json.loads, open(os.path.join(HERE, "out", f"nsr5_K{a.K}_{arm}.jsonl"))):
        sid, qt, tg = r["id"].split("|")
        g = gold(sites[sid], qt, tg)
        ok = r["pred"] is not None and sorted(r["pred"]) == g
        per[r["scenario"]].append(ok)
        per["ALL"].append(ok)
        ps, gs = set(r["pred"] or []), set(g)
        f1[0] += len(ps & gs)
        f1[1] += len(ps)
        f1[2] += len(gs)
    p_, r_ = f1[0] / max(f1[1], 1), f1[0] / max(f1[2], 1)
    res[arm] = {**{k: round(sum(v) / len(v), 4) for k, v in per.items()}, "micro_f1": round(2 * p_ * r_ / max(p_ + r_, 1e-9), 4),
                "n": len(per["ALL"])}
print(json.dumps(res, indent=1))
