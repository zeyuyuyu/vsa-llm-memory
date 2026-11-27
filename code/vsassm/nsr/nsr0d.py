"""NSR0d (prereg/NSR0d_prereg.md): T-stage, T-stage+vote and V-delta, all tuned on validation with m = 1 cells in the
criterion, evaluated on fresh seeds 20-29. NSR0's learning, sampling and noise code is imported unchanged.

python3 nsr/nsr0d.py tune ; python3 nsr/nsr0d.py run
"""
import collections
import gzip
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nsr0 as N  # noqa: E402

OUT = N.OUT


class TableStage(N.Table):
    """Separate support thresholds for rules first written from k = 2 stories and for rules created in closure."""

    def __init__(self, s2=1, sc=1, q=0.0, **_):
        super().__init__()
        self.s2, self.sc, self.q = s2, sc, q
        self.stage = {}
        self.closure = False

    def write(self, a, b, v):
        self.stage.setdefault((a, b), "c" if self.closure else "k2")
        super().write(a, b, v)

    def known(self, a, b):
        c = self.votes.get((a, b))
        if not c:
            return False
        top, tot = max(c.values()), sum(c.values())
        s = self.s2 if self.stage[(a, b)] == "k2" else self.sc
        return top >= s and top / tot >= self.q

    def snapshot(self):
        t = TableStage(self.s2, self.sc, self.q)
        t.votes = collections.defaultdict(collections.Counter, {k: collections.Counter(c) for k, c in self.votes.items()})
        t.stage = dict(self.stage)
        self.closure = True  # snapshots are taken only at the start of each closure round
        return t


def compose_vote(rels, VAL_, KNOWN):
    """CYK counting derivations per value; answer = value with the most derivations (ties: AMBIG)."""
    n, k = len(N.RELS), len(rels)
    T = {}
    for i, r in enumerate(rels):
        c = np.zeros(n)
        c[N.IDX[r]] = 1.0
        T[(i, i)] = c
    for span in range(2, k + 1):
        for i in range(k - span + 1):
            j = i + span - 1
            out = np.zeros(n)
            for m in range(i, j):
                comb = np.outer(T[(i, m)], T[(m + 1, j)])
                ok = KNOWN & (comb > 0)
                if ok.any():
                    np.add.at(out, VAL_[ok], comb[ok])
            T[(i, j)] = out
    res = T[(0, k - 1)]
    if res.max() <= 0:
        return "UNKNOWN", 0.0
    top = np.flatnonzero(res == res.max())
    return (N.RELS[top[0]], float(res.max() / res.sum())) if len(top) == 1 else ("AMBIG", 0.0)


def predict(R, test, vote):
    if not vote:
        return N.predict(R, test)
    n = len(N.RELS)
    VAL_, KNOWN = np.zeros((n, n), int), np.zeros((n, n), bool)
    for (a, b), (v, _c) in R.items():
        VAL_[N.IDX[a], N.IDX[b]], KNOWN[N.IDX[a], N.IDX[b]] = N.IDX[v], True
    out = [compose_vote(s["rels"], VAL_, KNOWN) for s in test]
    return [p for p, _ in out], [c for _, c in out]


SYM_GRID = [(s2, sc, q) for s2 in (1, 2) for sc in (1, 2, 3) for q in (0.0, 0.5, 0.7)]
VD_GRID = [(t, a) for t in (0.3, 0.4, 0.5, 0.6, 0.7) for a in (0.3, 0.5, 1.0)]
VAL_CELLS = [(1, None), (1, "L"), (3, None), (3, "L"), (3, "F")]


def build(kind, params, seed):
    if kind == "VD":
        return N.VDelta(2048, np.random.default_rng([seed, 7]), *params)
    return TableStage(*params)


def tune():
    res = collections.defaultdict(list)
    for seed in range(3):
        for m, noise in VAL_CELLS:
            rng = np.random.default_rng(1000 + seed)
            tr = N.per_key(m, rng)
            if noise:
                tr = N.noisy(tr, noise, 0.2, rng)
            for p in SYM_GRID:
                R = N.rule_tables(N.learn(TableStage(*p), tr, np.random.default_rng(seed)))
                for vote in (False, True):
                    res[("T-stage+vote" if vote else "T-stage", p)].append(N.acc(predict(R, N.VAL, vote)[0], N.VAL))
            for p in VD_GRID:
                R = N.rule_tables(N.learn(N.VDelta(2048, np.random.default_rng(seed), *p), tr, np.random.default_rng(seed)))
                res[("V-delta", p)].append(N.acc(N.predict(R, N.VAL)[0], N.VAL))
    choice = {}
    for arm in ("T-stage", "T-stage+vote", "V-delta"):
        best = max((k for k in res if k[0] == arm), key=lambda k: np.mean(res[k]))  # first in grid order on ties
        choice[arm] = {"params": list(best[1]), "val": float(np.mean(res[best]))}
    choice["best_symbolic"] = max(("T-stage", "T-stage+vote"), key=lambda a: choice[a]["val"])
    json.dump({"choice": choice, "grid": {f"{k[0]}|{k[1]}": float(np.mean(v)) for k, v in res.items()}},
              open(f"{OUT}/nsr0d_tuning.json", "w"), indent=1)
    print(json.dumps(choice, indent=1))


def job(cell):
    ch = json.load(open(f"{OUT}/nsr0d_tuning.json"))["choice"]
    recs = []
    for seed in range(20, 30):
        rng = np.random.default_rng([ord(c) for c in cell] + [seed])
        tr = N.train_set(cell, rng)
        for arm in ("T-stage", "T-stage+vote", "V-delta"):
            p = tuple(ch[arm]["params"])
            st = N.learn(N.VDelta(2048, np.random.default_rng([seed, 7]), *p) if arm == "V-delta" else TableStage(*p),
                         tr, np.random.default_rng([seed, 11]))
            R = N.rule_tables(st)
            pr, cf = predict(R, N.TEST, arm == "T-stage+vote")
            recs.append({"cell": cell, "arm": arm, "seed": seed, "ids": N.IDS, "pred": pr, "conf": cf,
                         "meta": {"n_rules": len(R), "params": list(p)}})
    return recs


def run():
    import multiprocessing as mp
    from nsr0b import CELLS
    with mp.Pool(14) as pool, gzip.open(f"{OUT}/nsr0d_preds.jsonl.gz", "wt") as f:
        for recs in pool.imap(job, CELLS):
            by = collections.defaultdict(list)
            for r in recs:
                f.write(json.dumps(r) + "\n")
                by[r["arm"]].append(N.acc(r["pred"], N.TEST))
            print(recs[0]["cell"], {a: round(float(np.mean(v)), 4) for a, v in by.items()}, flush=True)


if __name__ == "__main__":
    {"tune": tune, "run": run}[sys.argv[1]]()
