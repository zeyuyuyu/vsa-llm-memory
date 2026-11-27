"""NSR0b (prereg/NSR0b_prereg.md): is the V-delta advantage under label noise a VSA property, or rule pruning?

Adds T-prune, a count table that treats a rule as known only when its winning value has >= s votes and a vote share
>= q, and re-runs every NSR0 S/C/L/F cell for T-prune and V-delta with NSR0's seeds (same training sets, same
codebooks). Imports NSR0's stores, learning and inference unchanged.

python3 nsr/nsr0b.py tune     # choose (s, q) on validation with NSR0's criterion -> nsr/out/nsr0b_tuning.json
python3 nsr/nsr0b.py run      # registered cells -> nsr/out/nsr0b_preds.jsonl.gz
python3 nsr/nsr0b.py upper    # exploratory: best parameters per cell chosen on test, for both arms
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


class TablePrune(N.Table):
    def __init__(self, s=1, q=0.0, **_):
        super().__init__()
        self.s, self.q = s, q

    def known(self, a, b):
        c = self.votes.get((a, b))
        if not c:
            return False
        top, tot = max(c.values()), sum(c.values())
        return top >= self.s and top / tot >= self.q

    def snapshot(self):
        t = TablePrune(self.s, self.q)
        t.votes = collections.defaultdict(collections.Counter, {k: collections.Counter(c) for k, c in self.votes.items()})
        return t


GRID_S, GRID_Q = [1, 2, 3], [0.0, 0.5, 0.6, 0.7]


def tune():
    res = {}
    for seed in range(3):
        for cell in ("clean", "L0.2"):
            rng = np.random.default_rng(1000 + seed)  # NSR0's validation draws
            tr = N.per_key(3, rng)
            if cell == "L0.2":
                tr = N.noisy(tr, "L", 0.2, rng)
            for s in GRID_S:
                for q in GRID_Q:
                    st = N.learn(TablePrune(s, q), tr, np.random.default_rng(seed))
                    res.setdefault(f"{s}|{q}", []).append(N.acc(N.predict(N.rule_tables(st), N.VAL)[0], N.VAL))
    best = max(res.items(), key=lambda kv: np.mean(kv[1]))  # first in grid order on ties
    s, q = best[0].split("|")
    out = {"grid": {k: float(np.mean(v)) for k, v in res.items()},
           "choice": {"s": int(s), "q": float(q), "val_acc": float(np.mean(best[1]))}}
    json.dump(out, open(f"{OUT}/nsr0b_tuning.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


CELLS = ["S_m1", "S_m3", "S_m10", "S_all"] + [f"C_N{n}" for n in (50, 100, 200, 400, 800)]
CELLS += [f"{k}_p{p}_m{m}" for k in "LF" for p in (0.1, 0.2, 0.3) for m in (1, 3, 10)]


def job(args):
    cell, arms = args  # arms: list of (name, factory-args)
    recs = []
    for seed in N.SEEDS:
        rng = np.random.default_rng([ord(c) for c in cell] + [seed])  # NSR0's draw for this cell and seed
        tr = N.train_set(cell, rng)
        for name, kind, kw in arms:
            if kind == "TP":
                st = TablePrune(**kw)
            else:
                st = N.VDelta(2048, np.random.default_rng([seed, 7]), kw["tau"], kw["alpha"])
            st = N.learn(st, tr, np.random.default_rng([seed, 11]))
            R = N.rule_tables(st)
            p, c = N.predict(R, N.TEST)
            recs.append({"cell": cell, "arm": name, "seed": seed, "ids": N.IDS, "pred": p, "conf": c,
                         "meta": {"n_rules": len(R), "params": kw}})
    return recs


def run_jobs(specs, path):
    import multiprocessing as mp
    with mp.Pool(14) as pool, gzip.open(path, "wt") as f:
        for recs in pool.imap(job, specs):
            for r in recs:
                f.write(json.dumps(r) + "\n")
            by = collections.defaultdict(list)
            for r in recs:
                by[r["arm"]].append(N.acc(r["pred"], N.TEST))
            print(recs[0]["cell"], {a: round(float(np.mean(v)), 4) for a, v in by.items()}, flush=True)


def run():
    tp = json.load(open(f"{OUT}/nsr0b_tuning.json"))["choice"]
    vd = json.load(open(f"{OUT}/nsr0_tuning.json"))["choice"]["V-delta"]
    arms = [("T-prune", "TP", {"s": tp["s"], "q": tp["q"]}), ("V-delta", "VD", {"tau": vd["tau"], "alpha": vd["alpha"]})]
    run_jobs([(c, arms) for c in CELLS], f"{OUT}/nsr0b_preds.jsonl.gz")


def upper():
    arms = [(f"T-prune|{s}|{q}", "TP", {"s": s, "q": q}) for s in GRID_S for q in GRID_Q]
    arms += [(f"V-delta|{t}|{a}", "VD", {"tau": t, "alpha": a}) for t in (0.3, 0.4, 0.5, 0.6, 0.7) for a in (0.3, 0.5, 1.0)]
    run_jobs([(c, arms) for c in CELLS], f"{OUT}/nsr0b_upper.jsonl.gz")


if __name__ == "__main__":
    {"tune": tune, "run": run, "upper": upper}[sys.argv[1]]()
