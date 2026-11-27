"""VSA1 test run (prereg/VSA1_prereg.md). Steps, in order:
    python3 vsa/test.py gen     -> data/vsa1_items.jsonl (fresh vans seeds, after the prereg commit)
    python3 vsa/test.py gate    -> out/vsa/test_gate.jsonl
    python3 vsa/test.py run     -> out/vsa/test_V.jsonl, out/vsa/test_C.jsonl
    python3 vsa/test.py score   -> out/vsa/test_score.json
"""
import json, os, sys
import numpy as np
from concurrent.futures import ProcessPoolExecutor
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path[:0] = [HERE, os.path.join(ROOT, "bench", "heldout")]
import resonator as R  # noqa: E402

ITEMS = os.path.join(ROOT, "data", "vsa1_items.jsonl")
OUT = os.path.join(ROOT, "out", "vsa")
V_CONFIG = dict(rule="attention", beta=10.0, noise=0.05, sweeps=50, restarts=100000)  # prereg: att10n
TIME = 120


def gen():
    import vans
    assert not os.path.exists(ITEMS), "items already exist"
    jobs = [(s, L) for L in range(1, 6) for s in range(9300, 9310)] + [(s, L) for L in (6, 7, 8) for s in range(9300, 9305)]
    with open(ITEMS, "w") as f:
        for s, L in jobs:
            it = vans.generate(s, L)
            f.write(json.dumps({k: it[k] for k in ("id", "family", "level", "seed", "data", "gold", "optimum")}) + "\n")
            print(it["id"], len(it["data"]["parcels"]), it["optimum"], flush=True)


def gate_job(a):
    n, i = a
    rng = np.random.default_rng(10_000 + 100 * n + i)
    S = [int(x) for x in rng.integers(0, 5001, n)]
    T = int(sum(s for s, b in zip(S, rng.random(n) < 0.5) if b))
    p = R.subset_sum_problem(S, T, 10000, rng)
    r = R.resonate(p, rng, rule="standard", sweeps=100, restarts=10, time_limit=TIME)
    return dict(n=n, i=i, **{k: v for k, v in r.items() if k != "choice"})


def v_job(it):
    rng = np.random.default_rng(it["seed"] * 10 + it["level"])
    p = R.vans_problem(it["data"], it["optimum"], 10000, rng)
    r = R.resonate(p, rng, time_limit=TIME, **V_CONFIG)
    return dict(id=it["id"], level=it["level"], factors=p.n_factors, **{k: v for k, v in r.items() if k != "choice"})


def c_job(it):
    import cpsat_arm as C
    r = C.solve_vans(it["data"], it["optimum"], time_limit=TIME, fix_heaviest=True)
    ok = bool(r["assign"]) and R.vans_valid(it["data"], r["assign"], it["optimum"])
    return dict(id=it["id"], level=it["level"], solved=ok, seconds=r["seconds"], status=r["status"])


def run_pool(fn, args, path, workers):
    with open(path, "w") as out, ProcessPoolExecutor(workers) as ex:
        for r in ex.map(fn, args):
            out.write(json.dumps(r) + "\n")
            out.flush()


def main(step):
    os.makedirs(OUT, exist_ok=True)
    if step == "gen":
        gen()
    elif step == "gate":
        run_pool(gate_job, [(n, i) for n in (10, 20, 30, 40) for i in range(10)], os.path.join(OUT, "test_gate.jsonl"), 14)
    elif step == "run":
        items = [json.loads(l) for l in open(ITEMS)]
        run_pool(v_job, items, os.path.join(OUT, "test_V.jsonl"), 14)
        run_pool(c_job, items, os.path.join(OUT, "test_C.jsonl"), 14)
    elif step == "score":
        g = [json.loads(l) for l in open(os.path.join(OUT, "test_gate.jsonl"))]
        res = {"gate": {"solved": sum(r["solved"] for r in g), "of": len(g),
                        "by_n": {n: sum(r["solved"] for r in g if r["n"] == n) for n in (10, 20, 30, 40)}}}
        for arm in ("V", "C"):
            rs = [json.loads(l) for l in open(os.path.join(OUT, "test_%s.jsonl" % arm))]
            res[arm] = {L: {"solved": sum(r["solved"] for r in rs if r["level"] == L),
                            "of": sum(1 for r in rs if r["level"] == L),
                            "median_s": float(np.median([r["seconds"] for r in rs if r["level"] == L and r["solved"]] or [float("nan")]))}
                        for L in range(1, 9)}
        V, C = res["V"], res["C"]
        res["B1"] = V[1]["solved"] >= C[1]["solved"] - 1
        res["B2"] = sum(C[L]["solved"] - V[L]["solved"] for L in range(2, 6)) >= 20
        res["B3_against"] = any(V[L]["solved"] >= 5 for L in range(3, 9)) or \
            sum(V[L]["solved"] for L in (6, 7, 8)) > sum(C[L]["solved"] for L in (6, 7, 8))
        res["G"] = res["gate"]["solved"] >= 24
        json.dump(res, open(os.path.join(OUT, "test_score.json"), "w"), indent=1)
        print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
