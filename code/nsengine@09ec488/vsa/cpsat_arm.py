"""The solver arm: the same vans feasibility problem (k vans) with CPMpy + OR-Tools CP-SAT, one worker."""
import time
import cpmpy as cp


def solve_vans(data, k, time_limit=60, fix_heaviest=False):
    P = data["parcels"]
    idx = {p["id"]: i for i, p in enumerate(P)}
    x = cp.intvar(1, k, shape=len(P), name="van")
    m = cp.Model()
    for v in range(1, k + 1):
        m += cp.sum([(x[i] == v) * p["w"] for i, p in enumerate(P)]) <= data["W"]
        m += cp.sum([(x[i] == v) * p["s"] for i, p in enumerate(P)]) <= data["S"]
    for a, b in data["apart"]:
        m += x[idx[a]] != x[idx[b]]
    for a, b in data["together"]:
        m += x[idx[a]] == x[idx[b]]
    if fix_heaviest:
        # the resonator's symmetry fix: the heaviest group of must-travel-together parcels goes in van 1
        import resonator as R
        groups, _, _ = R.vans_groups(data)
        g = max(groups, key=lambda g: (g["w"], g["s"]))
        m += x[idx[g["members"][0]]] == 1
    t0 = time.time()
    ok = m.solve("ortools", time_limit=time_limit, num_workers=1)
    dt = time.time() - t0
    assign = {p["id"]: int(x[i].value()) for i, p in enumerate(P)} if ok else None
    return dict(solved=bool(ok), seconds=dt, status=str(m.status().exitstatus), assign=assign)
