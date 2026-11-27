"""Second, independently written implementation of NSR0d's T-stage+vote (staged support thresholds, derivation-majority
answers). Shares no learning or inference code with nsr0.py / nsr0d.py; only the training sample is regenerated with
NSR0's sampling function (the object under test is the learner and the inference). Compares per-story answers with
nsr/out/nsr0d_preds.jsonl.gz.

python3 nsr/nsr0d_check2.py CELL SEED
"""
import collections
import csv
import functools
import gzip
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nsr0 import train_set  # sampling only  # noqa: E402


def fit(stories, s2, sc, q):
    votes, born = {}, {}

    def ok(pair, frozen_votes):
        c = frozen_votes.get(pair)
        if not c:
            return False
        top = max(c.values())
        need = s2 if born[pair] == 1 else sc
        return top >= need and top / sum(c.values()) >= q

    def best(c):
        top = max(c.values())
        return sorted(v for v, n in c.items() if n == top)[0]

    def add(pair, val, stage):
        born.setdefault(pair, stage)
        votes.setdefault(pair, {})
        votes[pair][val] = votes[pair].get(val, 0) + 1

    for st in stories:
        if len(st["rels"]) == 2:
            add(tuple(st["rels"]), st["target"], 1)
    for _ in range(3):
        frozen = {p: dict(c) for p, c in votes.items()}
        for st in stories:
            if len(st["rels"]) != 3:
                continue
            a, b, c = st["rels"]
            if ok((a, b), frozen):
                x = best(frozen[(a, b)])
                if (x, c) not in frozen or not ok((x, c), frozen):
                    if (x, c) not in frozen:
                        add((x, c), st["target"], 2)
                    else:
                        add((x, c), st["target"], born[(x, c)])
            if ok((b, c), frozen):
                y = best(frozen[(b, c)])
                if (a, y) not in frozen or not ok((a, y), frozen):
                    if (a, y) not in frozen:
                        add((a, y), st["target"], 2)
                    else:
                        add((a, y), st["target"], born[(a, y)])
    return {p: best(c) for p, c in votes.items() if ok(p, votes)}


def answer(chain, rules):
    chain = tuple(chain)

    @functools.lru_cache(maxsize=None)
    def ways(lo, hi):  # value -> number of derivations of chain[lo:hi]
        if hi - lo == 1:
            return {chain[lo]: 1}
        out = collections.Counter()
        for cut in range(lo + 1, hi):
            for x, nx in ways(lo, cut).items():
                for y, ny in ways(cut, hi).items():
                    if (x, y) in rules:
                        out[rules[(x, y)]] += nx * ny
        return dict(out)

    w = ways(0, len(chain))
    if not w:
        return "UNKNOWN"
    top = max(w.values())
    win = [v for v, n in w.items() if n == top]
    return win[0] if len(win) == 1 else "AMBIG"


if __name__ == "__main__":
    cell, seed = sys.argv[1], int(sys.argv[2])
    tr = train_set(cell, np.random.default_rng([ord(c) for c in cell] + [seed]))
    rules = fit(tr, 1, 3, 0.5)
    test = {r["id"]: r["f_comb"].split("-") for r in csv.DictReader(open("r1_data/test.csv", encoding="utf-8-sig"))}
    for line in gzip.open("nsr/out/nsr0d_preds.jsonl.gz", "rt"):
        r = json.loads(line)
        if r["cell"] == cell and r["seed"] == seed and r["arm"] == "T-stage+vote":
            mine = [answer(test[i], rules) for i in r["ids"]]
            print(f"{cell} seed {seed}: {len(rules)} rules here vs {r['meta']['n_rules']}; "
                  f"{sum(a != b for a, b in zip(mine, r['pred']))} of {len(mine)} answers differ")
