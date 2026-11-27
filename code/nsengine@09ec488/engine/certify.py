"""Impossibility certificates for maintenance plans: a minimal set of services that alone cannot all stay available
under any split of the hosts into the allowed number of waves (a minimal unsatisfiable subset, MUS).

Why a core and not just "unsat": a reviewer can check a 10-30-service core by hand or with any solver, and it says
what to change (add a replica, allow another wave, lower a quorum). Kept apart from sre_engine.maintenance_waves so the
answers of the registered runs are unchanged.

    core = impossibility_core(dsl)        # dsl["query"] is a maintenance_waves query; None if a plan exists
"""
import z3

from sre_engine import DSLError, Topology, _norm


def _encode(topo, H, W):
    wave = {h: z3.Int("w_" + h) for h in H}
    base = [z3.And(wave[h] >= 0, wave[h] < W) for h in H]
    lits, cons = {}, {}
    for s, v in topo.services.items():
        on = [h for h in v["replicas"] if h in wave]
        cap = len(v["replicas"]) - v["quorum"]           # replicas the service can lose and stay available
        if len(on) <= cap:
            continue                                     # cannot lose quorum whatever the plan
        lits[s] = z3.Bool("keep_" + s)
        cons[s] = z3.And([z3.Sum([z3.If(wave[h] == w, 1, 0) for h in on]) <= cap for w in range(W)])
    return wave, base, lits, cons


def impossibility_core(dsl, minimise=True, timeout_ms=60000):
    """Services whose own quorum rules cannot all hold in any plan. When every service keeps its own quorum in every
    wave, no dependency can fail, so the dependency graph does not enter the core."""
    q = dsl.get("query") or {}
    if q.get("type") != "maintenance_waves":
        raise DSLError("impossibility_core needs a maintenance_waves query")
    topo = Topology(dsl)
    H = sorted({_norm(h) for h in q["hosts"]})
    W = int(q.get("max_waves", 2))
    wave, base, lits, cons = _encode(topo, H, W)
    sol = z3.Solver()
    sol.set("timeout", timeout_ms)
    sol.add(base)
    for s in lits:
        sol.add(z3.Implies(lits[s], cons[s]))
    r = sol.check(list(lits.values()))
    if r == z3.sat:
        return None
    if r != z3.unsat:
        raise DSLError("solver gave %s" % r)
    names = {str(l): s for s, l in lits.items()}
    core = sorted(names[str(l)] for l in sol.unsat_core())
    if minimise:                                         # deletion-based: drop any service the core does not need
        i = 0
        while i < len(core):
            trial = core[:i] + core[i + 1:]
            if sol.check([lits[s] for s in trial]) == z3.unsat:
                core = sorted(names[str(l)] for l in sol.unsat_core())
                core = [s for s in core if s in trial]
            else:
                i += 1
    return core


def explain(dsl, core):
    """Plain-language statement of a core: each service, its replicas on the drain set and how many it can lose."""
    topo = Topology(dsl)
    q = dsl["query"]
    H = {_norm(h) for h in q["hosts"]}
    lines = ["These %d services cannot all stay available if the hosts are patched in %d waves:"
             % (len(core), int(q.get("max_waves", 2)))]
    for s in core:
        v = topo.services[s]
        on = sorted(h for h in v["replicas"] if h in H)
        cap = len(v["replicas"]) - v["quorum"]
        lines.append("- %s needs %d of %d replicas healthy, so at most %d of %s may be down together."
                     % (s, v["quorum"], len(v["replicas"]), cap, ", ".join(on)))
    lines.append("Any plan breaks at least one of these; removing any one service from the list makes a plan possible.")
    return "\n".join(lines)
