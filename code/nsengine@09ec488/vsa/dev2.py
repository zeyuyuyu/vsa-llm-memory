"""Development round 2 (burned NSR8 vans, levels 1-3): resonator variants from the literature."""
import json, sys, os
import numpy as np
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import resonator as R

CONFIGS = {
    "att20": dict(rule="attention", beta=20.0, noise=0.0),
    "att20n": dict(rule="attention", beta=20.0, noise=0.05),
    "att10n": dict(rule="attention", beta=10.0, noise=0.05),
    "stdn": dict(rule="standard", beta=0.0, noise=0.05),
}

def job(a):
    iid, cname = a
    it = [json.loads(l) for l in open("data/nsr8_items.jsonl") if iid in l][0]
    rng = np.random.default_rng(2)
    p = R.vans_problem(it["data"], it["optimum"], 10000, rng)
    r = R.resonate(p, rng, sweeps=50, restarts=1000, time_limit=120, **CONFIGS[cname])
    return dict(id=iid, level=it["level"], config=cname, factors=p.n_factors,
                **{k: v for k, v in r.items() if k != "choice"})

if __name__ == "__main__":
    ids = [json.loads(l)["id"] for l in open("data/nsr8_items.jsonl") if '"family": "vans"' in l]
    ids = [i for i in ids if "-L1-" in i or "-L2-" in i or "-L3-" in i]
    jobs = [(i, c) for i in ids for c in CONFIGS]
    with open("out/vsa/dev2.jsonl", "w") as out, ProcessPoolExecutor(14) as ex:
        for r in ex.map(job, jobs):
            out.write(json.dumps(r) + "\n"); out.flush()
    print("done")
