"""Exploratory, after the VSA1 test (not registered): do the level-3 failures yield to 5x the time or 2x the dimension?"""
import json, sys, os
import numpy as np
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import resonator as R

def job(a):
    it, D = a
    rng = np.random.default_rng(it["seed"] * 10 + it["level"] + D)
    p = R.vans_problem(it["data"], it["optimum"], D, rng)
    r = R.resonate(p, rng, rule="attention", beta=10.0, noise=0.05, sweeps=50, restarts=100000, time_limit=600)
    return dict(id=it["id"], level=it["level"], D=D, **{k: v for k, v in r.items() if k != "choice"})

if __name__ == "__main__":
    its = [json.loads(l) for l in open("data/vsa1_items.jsonl")]
    its = [i for i in its if i["level"] == 3]
    with open("out/vsa/explore_budget.jsonl", "w") as out, ProcessPoolExecutor(14) as ex:
        for r in ex.map(job, [(i, D) for i in its for D in (10000, 20000)]):
            out.write(json.dumps(r) + "\n"); out.flush()
