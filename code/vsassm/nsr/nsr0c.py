"""NSR0c (prereg/NSR0c_prereg.md): T-prune vs V-delta, both re-tuned on validation with fact noise in the criterion,
evaluated on fresh seeds 10-19. Reuses NSR0/NSR0b code paths unchanged.

python3 nsr/nsr0c.py tune ; python3 nsr/nsr0c.py run
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nsr0 as N  # noqa: E402
import nsr0b as B  # noqa: E402

OUT = N.OUT


def tune():
    res = {"T-prune": {}, "V-delta": {}}
    for seed in range(3):
        for cell in ("clean", "L0.2", "F0.2"):
            rng = np.random.default_rng(1000 + seed)
            tr = N.per_key(3, rng)
            if cell != "clean":
                tr = N.noisy(tr, cell[0], 0.2, rng)
            for s in B.GRID_S:
                for q in B.GRID_Q:
                    st = N.learn(B.TablePrune(s, q), tr, np.random.default_rng(seed))
                    res["T-prune"].setdefault(f"{s}|{q}", []).append(N.acc(N.predict(N.rule_tables(st), N.VAL)[0], N.VAL))
            for tau in (0.3, 0.4, 0.5, 0.6, 0.7):
                for alpha in (0.3, 0.5, 1.0):
                    st = N.learn(N.VDelta(2048, np.random.default_rng(seed), tau, alpha), tr, np.random.default_rng(seed))
                    res["V-delta"].setdefault(f"{tau}|{alpha}", []).append(N.acc(N.predict(N.rule_tables(st), N.VAL)[0], N.VAL))
    out = {"grid": {a: {k: float(np.mean(v)) for k, v in d.items()} for a, d in res.items()}, "choice": {}}
    for a, d in res.items():
        k = max(d, key=lambda key: np.mean(d[key]))
        x, y = k.split("|")
        out["choice"][a] = ({"s": int(x), "q": float(y)} if a == "T-prune" else {"tau": float(x), "alpha": float(y)})
        out["choice"][a]["val_acc"] = float(np.mean(d[k]))
    json.dump(out, open(f"{OUT}/nsr0c_tuning.json", "w"), indent=1)
    print(json.dumps(out["choice"], indent=1))


def run():
    ch = json.load(open(f"{OUT}/nsr0c_tuning.json"))["choice"]
    N.SEEDS = range(10, 20)  # fresh data draws; inherited by the forked workers
    arms = [("T", "TP", {"s": 1, "q": 0.0}),
            ("T-prune", "TP", {"s": ch["T-prune"]["s"], "q": ch["T-prune"]["q"]}),
            ("V-delta", "VD", {"tau": ch["V-delta"]["tau"], "alpha": ch["V-delta"]["alpha"]})]
    B.run_jobs([(c, arms) for c in B.CELLS], f"{OUT}/nsr0c_preds.jsonl.gz")


if __name__ == "__main__":
    {"tune": tune, "run": run}[sys.argv[1]]()
