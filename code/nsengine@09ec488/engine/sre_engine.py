"""Exact solver for infrastructure change-risk questions, over the translated DSL (v0).

DSL (what a translator produces from the runbook text and the question):
    {"racks":    {"R1": {"zone": "use1-a", "hosts": ["node01", ...]}, ...},
     "services": {"payments-api": {"replicas": ["node07", ...], "quorum": 2, "needs": ["users-db"]}, ...},
                 (instead of "quorum", a service may give "can_lose": t, the number of replicas it can lose)
     "query":    {"type": "blast_radius", "failed_hosts": [...], "failed_racks": [...]}
               | {"type": "min_failure_sets", "service": "payments-api"}
               | {"type": "maintenance_waves", "hosts": [...], "max_waves": 2}}

Every answer comes with a certificate: the reason each service is down (blast radius); the minimum, a proof that no
smaller set works and the exhaustive list (minimal failure sets); a wave assignment with each wave re-checked, or
an UNSAT result (maintenance). DSLError means the translation is not a well-formed problem (the verifier abstains).
"""
import z3


class DSLError(ValueError):
    pass


def _norm(x):
    return str(x).strip().lower()


class Topology:
    def __init__(self, dsl):
        try:
            racks = {_norm(r): {"zone": _norm(v.get("zone", "")), "hosts": sorted({_norm(h) for h in v["hosts"]})}
                     for r, v in dsl["racks"].items()}
            svcs = {}
            for s, v in dsl["services"].items():
                reps = sorted({_norm(h) for h in v["replicas"]})
                # the translator may copy "can lose up to t of them" literally; the engine does the arithmetic
                q = len(reps) - int(v["can_lose"]) if v.get("quorum") is None and v.get("can_lose") is not None else int(v["quorum"])
                svcs[_norm(s)] = {"replicas": reps, "quorum": q, "needs": sorted({_norm(n) for n in v.get("needs", [])})}
        except (KeyError, TypeError, ValueError, AttributeError) as e:
            raise DSLError("malformed DSL: %r" % (e,))
        self.racks, self.services = racks, svcs
        self.rack_of = {}
        for r, v in racks.items():
            for h in v["hosts"]:
                if h in self.rack_of:
                    raise DSLError("host %s in two racks" % h)
                self.rack_of[h] = r
        for s, v in svcs.items():
            if not 1 <= v["quorum"] <= len(v["replicas"]):
                raise DSLError("%s: quorum %d with %d replicas" % (s, v["quorum"], len(v["replicas"])))
            for n in v["needs"]:
                if n not in svcs:
                    raise DSLError("%s needs unknown service %s" % (s, n))
            for h in v["replicas"]:
                if h not in self.rack_of:
                    raise DSLError("%s: replica host %s is in no rack" % (s, h))
        self.order = self._topo_order()
        self.hosts = sorted(self.rack_of)

    def _topo_order(self):
        order, state = [], {}

        def visit(s):
            if state.get(s) == 1:
                raise DSLError("dependency cycle through %s" % s)
            if state.get(s) == 2:
                return
            state[s] = 1
            for n in self.services[s]["needs"]:
                visit(n)
            state[s] = 2
            order.append(s)
        for s in sorted(self.services):
            visit(s)
        return order

    # ------------------------------------------------------------------------------------------ propagation
    def propagate(self, down_hosts):
        down_hosts, status = set(down_hosts), {}
        for s in self.order:
            v = self.services[s]
            healthy = [h for h in v["replicas"] if h not in down_hosts]
            dead_deps = [n for n in v["needs"] if status[n][0]]
            if len(healthy) < v["quorum"]:
                status[s] = (True, "only %d of %d replicas healthy, needs %d" % (len(healthy), len(v["replicas"]), v["quorum"]))
            elif dead_deps:
                status[s] = (True, "depends on unavailable %s" % ", ".join(dead_deps))
            else:
                status[s] = (False, "")
        return status

    def _hosts_down(self, failed_hosts, failed_racks):
        out = {_norm(h) for h in failed_hosts}
        for r in failed_racks:
            r = _norm(r)
            if r not in self.racks:
                raise DSLError("unknown rack %s" % r)
            out |= set(self.racks[r]["hosts"])
        unknown = out - set(self.hosts)
        if unknown:
            raise DSLError("unknown hosts %s" % sorted(unknown))
        return out

    # ------------------------------------------------------------------------------------------ queries
    def blast_radius(self, failed_hosts, failed_racks):
        st = self.propagate(self._hosts_down(failed_hosts, failed_racks))
        down = sorted(s for s, (d, _) in st.items() if d)
        return {"unavailable": down}, {"reasons": {s: st[s][1] for s in down}}

    def _z3_down(self, fail):
        down = {}
        for s in self.order:
            v = self.services[s]
            healthy = z3.Sum([z3.If(fail[h], 0, 1) for h in v["replicas"]])
            down[s] = z3.Or([healthy < v["quorum"]] + [down[n] for n in v["needs"]])
        return down

    def min_failure_sets(self, service, max_sets=1000):
        service = _norm(service)
        if service not in self.services:
            raise DSLError("unknown service %s" % service)
        fail = {h: z3.Bool("f_" + h) for h in self.hosts}
        down = self._z3_down(fail)
        size = z3.Sum([z3.If(fail[h], 1, 0) for h in self.hosts])
        opt = z3.Optimize()
        opt.add(down[service])
        opt.minimize(size)
        if opt.check() != z3.sat:
            raise DSLError("%s cannot be made unavailable" % service)
        m = opt.model().eval(size).as_long()
        below = z3.Solver()
        below.add(down[service], size <= m - 1)
        proof_min = str(below.check())                      # 'unsat': no set of m-1 hosts works
        sol, sets = z3.Solver(), []
        sol.add(down[service], size == m)
        while sol.check() == z3.sat and len(sets) < max_sets:
            md = sol.model()
            chosen = sorted(h for h in self.hosts if z3.is_true(md.eval(fail[h], model_completion=True)))
            sets.append(chosen)
            sol.add(z3.Not(z3.And([fail[h] for h in chosen])))
        for c in sets:                                       # re-check each set by plain propagation
            assert self.propagate(c)[service][0], c
        return {"size": m, "sets": sorted(sets)}, {"no_smaller_set": proof_min, "enumeration_complete": len(sets) < max_sets}

    def maintenance_waves(self, hosts, max_waves):
        H = sorted({_norm(h) for h in hosts})
        unknown = set(H) - set(self.hosts)
        if unknown:
            raise DSLError("unknown hosts %s" % sorted(unknown))
        W = int(max_waves)
        wave = {h: z3.Int("w_" + h) for h in H}
        sol = z3.Solver()
        for h in H:
            sol.add(wave[h] >= 0, wave[h] < W)
        for w in range(W):                                   # while wave w is down, no service may be unavailable
            fail = {h: (wave[h] == w) if h in wave else z3.BoolVal(False) for h in self.hosts}
            down = self._z3_down(fail)
            sol.add(z3.Not(z3.Or(list(down.values()))))
        if sol.check() == z3.sat:
            md = sol.model()
            waves = [sorted(h for h in H if md.eval(wave[h]).as_long() == w) for w in range(W)]
            waves = [w for w in waves if w]
            for w in waves:                                  # re-check each wave by plain propagation
                assert not any(d for d, _ in self.propagate(set(w)).values()), w
            return {"waves": waves}, {"waves_checked": True}
        return {"impossible": True}, {"assignment": "unsat"}


def solve(dsl):
    topo = Topology(dsl)
    q = dsl.get("query") or {}
    t = q.get("type")
    if t == "blast_radius":
        return topo.blast_radius(q.get("failed_hosts", []), q.get("failed_racks", []))
    if t == "min_failure_sets":
        return topo.min_failure_sets(q["service"])
    if t == "maintenance_waves":
        return topo.maintenance_waves(q["hosts"], q.get("max_waves", 2))
    raise DSLError("unknown query type %r" % (t,))
