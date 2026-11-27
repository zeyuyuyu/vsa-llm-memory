"""NSR0e (prereg/NSR0e_prereg.md): T-stage+vote vs two fixed V-delta settings, fresh seeds 30-39, no tuning."""
import collections
import gzip
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nsr0 as N  # noqa: E402
import nsr0d as D  # noqa: E402
from nsr0b import CELLS  # noqa: E402

ARMS = [("T-stage+vote", (1, 3, 0.5)), ("V-delta-d", (0.3, 0.3)), ("V-delta-0", (0.4, 0.3))]


def job(cell):
    recs = []
    for seed in range(30, 40):
        rng = np.random.default_rng([ord(c) for c in cell] + [seed])
        tr = N.train_set(cell, rng)
        for arm, p in ARMS:
            vote = arm == "T-stage+vote"
            st = D.TableStage(*p) if vote else N.VDelta(2048, np.random.default_rng([seed, 7]), *p)
            R = N.rule_tables(N.learn(st, tr, np.random.default_rng([seed, 11])))
            pr, cf = D.predict(R, N.TEST, vote)
            recs.append({"cell": cell, "arm": arm, "seed": seed, "ids": N.IDS, "pred": pr, "conf": cf,
                         "meta": {"n_rules": len(R), "params": list(p)}})
    return recs


if __name__ == "__main__":
    import multiprocessing as mp
    with mp.Pool(14) as pool, gzip.open(f"{N.OUT}/nsr0e_preds.jsonl.gz", "wt") as f:
        for recs in pool.imap(job, CELLS):
            by = collections.defaultdict(list)
            for r in recs:
                f.write(json.dumps(r) + "\n")
                by[r["arm"]].append(N.acc(r["pred"], N.TEST))
            print(recs[0]["cell"], {a: round(float(np.mean(v)), 4) for a, v in by.items()}, flush=True)
