"""Development runs (burned data only): subset-sum replication and vans on NSR8 levels 1-3."""
import json, sys, os, time
import numpy as np
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import resonator as R

def ss_job(a):
    n, D, seed, rule = a
    rng = np.random.default_rng(seed)
    S = [int(x) for x in rng.integers(0, 5001, n)]
    pick = rng.random(n) < 0.5
    T = int(sum(s for s, b in zip(S, pick) if b))
    p = R.subset_sum_problem(S, T, D, rng)
    r = R.resonate(p, rng, rule=rule, sweeps=100, restarts=10, time_limit=60)
    return dict(task="ss", n=n, D=D, seed=seed, rule=rule, **{k: v for k, v in r.items() if k != "choice"})

def vans_job(a):
    iid, D, rule, beta, seed = a
    it = [json.loads(l) for l in open("data/nsr8_items.jsonl") if iid in l][0]
    rng = np.random.default_rng(seed)
    p = R.vans_problem(it["data"], it["optimum"], D, rng)
    r = R.resonate(p, rng, rule=rule, beta=beta, sweeps=200, restarts=10, time_limit=90)
    return dict(task="vans", id=iid, level=it["level"], D=D, rule=rule, beta=beta, factors=p.n_factors,
                **{k: v for k, v in r.items() if k != "choice"})

if __name__ == "__main__":
    jobs_ss = [(n, 10000, s, "standard") for n in (10, 20, 30, 40) for s in range(5)]
    ids = [json.loads(l)["id"] for l in open("data/nsr8_items.jsonl") if '"family": "vans"' in l]
    ids = [i for i in ids if "-L1-" in i or "-L2-" in i or "-L3-" in i]
    jobs_v = [(i, D, rule, beta, 1) for i in ids for D, rule, beta in
              [(10000, "standard", 0), (10000, "attention", 20.0), (10000, "attention", 50.0)]]
    out = open("out/vsa/dev1.jsonl", "w")
    with ProcessPoolExecutor(14) as ex:
        for r in ex.map(ss_job, jobs_ss):
            out.write(json.dumps(r) + "\n"); out.flush()
        for r in ex.map(vans_job, jobs_v):
            out.write(json.dumps(r) + "\n"); out.flush()
    print("done")
