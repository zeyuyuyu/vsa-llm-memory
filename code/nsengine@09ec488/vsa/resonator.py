"""A resonator network that searches in superposition (Frady et al. 2020; Kymn et al. 2025), used to ask whether a
vector symbolic search can take the constraint solver's place.

Codes are unit phasors in C^D (FHRR). An integer x in a channel is coded by a phase code z(x)[d] = exp(i x th[d]);
th[d] = 2 pi * sum_j r_jd / m_j for coprime moduli m_j (Kymn et al.'s residue code), so that binding (elementwise
product) adds the integers modulo M = prod m_j. Codes of different channels are independent, so a product of codes
from several channels binds them all.

A problem is a set of factors, each with a codebook (D x K, unit phasors), and a target t such that a choice of one
codebook entry per factor is a solution iff the product of the chosen entries equals t. The resonator keeps one
estimate per factor and updates each in turn:
    u   = t * conj(product of the other estimates)
    a   = C^H u / D                      (similarity to each entry)
    x_f = phasor(C w),  w = a            ("standard", Frady et al. 2020)
                        w = softmax(beta * Re a)   ("attention", Yeung et al. 2024, arXiv 2403.13218)
After every sweep, each factor is decoded to its most similar entry and the decoded choice is checked by the
problem's own exact verifier; the search stops at the first valid choice.
"""
import numpy as np

TWO_PI = 2.0 * np.pi


def phasor(v, eps=1e-12):
    return v / np.maximum(np.abs(v), eps)


class Channel:
    """Phase code for integers modulo prod(moduli)."""

    def __init__(self, rng, D, moduli):
        self.moduli = tuple(int(m) for m in moduli)
        self.M = int(np.prod(self.moduli))
        th = np.zeros(D)
        for m in self.moduli:
            th += TWO_PI * rng.integers(0, m, size=D) / m
        self.th = th

    def code(self, x):
        return np.exp(1j * np.outer(np.atleast_1d(x), self.th)).T  # D x len(x)

    def one(self, x):
        return np.exp(1j * x * self.th)


class Problem:
    """factors: list of codebooks (D x K complex). target: D complex. verify(choice) -> bool."""

    def __init__(self, factors, target, verify, fixed=None):
        self.C = factors
        self.t = target
        self.verify = verify
        self.fixed = fixed or {}   # factor index -> entry index (symmetry breaking)


def resonate(prob, rng, rule="standard", beta=20.0, sweeps=200, restarts=10, time_limit=None, noise=0.0):
    """noise: std of Gaussian noise added to the real similarities at every update (a stochastic resonator, as
    device noise does in Langenegger et al. 2023)."""
    import time
    t0 = time.time()
    D = prob.t.shape[0]
    F = len(prob.C)
    total = 0
    for r in range(restarts):
        x = []
        for f, C in enumerate(prob.C):
            if f in prob.fixed:
                x.append(C[:, prob.fixed[f]].copy())
            else:
                x.append(phasor(C @ rng.uniform(0.0, 1.0, C.shape[1])))
        P = np.ones(D, dtype=complex)
        for v in x:
            P *= v
        for s in range(sweeps):
            total += 1
            for f in rng.permutation(F):
                if f in prob.fixed:
                    continue
                C = prob.C[f]
                u = prob.t * np.conj(P) * x[f]
                a = (C.conj().T @ u) / D
                if noise:
                    a = a + noise * rng.standard_normal(a.shape[0])
                if rule == "standard":
                    w = a
                else:
                    z = beta * a.real
                    z -= z.max()
                    w = np.exp(z)
                    w /= w.sum()
                new = phasor(C @ w)
                P *= np.conj(x[f]) * new
                x[f] = new
            # renormalise the running product against drift
            P = phasor(P)
            choice = [prob.fixed[f] if f in prob.fixed else int(np.argmax((prob.C[f].conj().T @ x[f]).real))
                      for f in range(F)]
            if prob.verify(choice):
                return dict(solved=True, restarts=r + 1, sweeps=total, seconds=time.time() - t0, choice=choice)
            if time_limit and time.time() - t0 > time_limit:
                return dict(solved=False, restarts=r + 1, sweeps=total, seconds=time.time() - t0, timeout=True)
    return dict(solved=False, restarts=restarts, sweeps=total, seconds=time.time() - t0)


# ------------------------------------------------------------------ subset sum (Kymn et al. 2025, sec. 2.6.2)
def subset_sum_problem(S, T, D, rng, m=1000):
    ch = Channel(rng, D, (m - 1, m, m + 1))
    assert ch.M > sum(S)
    factors = [ch.code([0, s]) for s in S]
    target = ch.one(T)

    def verify(choice):
        return sum(s for s, c in zip(S, choice) if c == 1) == T
    return Problem(factors, target, verify)


# ------------------------------------------------------------------ vans (bench/heldout/vans.py) at k vans
def vans_groups(data):
    """Must-travel-together parcels merged into groups (union-find), as the family's own solver does."""
    ids = [p["id"] for p in data["parcels"]]
    par = {i: i for i in ids}

    def f(a):
        while par[a] != a:
            par[a] = par[par[a]]
            a = par[a]
        return a
    for a, b in data["together"]:
        par[f(a)] = f(b)
    groups = {}
    for i in ids:
        groups.setdefault(f(i), []).append(i)
    by = {p["id"]: p for p in data["parcels"]}
    out = [dict(members=m, w=sum(by[i]["w"] for i in m), s=sum(by[i]["s"] for i in m)) for m in groups.values()]
    gid = {i: g for g, grp in enumerate(out) for i in grp["members"]}
    apart = sorted({tuple(sorted((gid[a], gid[b]))) for a, b in data["apart"]})
    return out, apart, gid


def vans_valid(data, assign, k):
    """assign: parcel id -> van in 1..k. The family's own rules."""
    W, S = data["W"], data["S"]
    load = {}
    for p in data["parcels"]:
        v = assign.get(p["id"])
        if v is None or not 1 <= v <= k:
            return False
        lw, ls = load.get(v, (0, 0))
        load[v] = (lw + p["w"], ls + p["s"])
    if any(lw > W or ls > S for lw, ls in load.values()):
        return False
    if any(assign[a] == assign[b] for a, b in data["apart"]):
        return False
    if any(assign[a] != assign[b] for a, b in data["together"]):
        return False
    return True


def vans_problem(data, k, D, rng, unit=10, symmetry=True):
    """Feasibility of loading into k vans, as one product of codes.
    Channels: weight of van v, spaces of van v (integers), and for each apart pair p and van v a counter mod 3.
    Factor per group g: entry v = Wv(w_g) * Sv(s_g) * prod_{p containing g} Qpv(1).
    Slack factors: van v weight slack Wv(0..W), spaces slack Sv(0..S); apart slack (p, v): Qpv(0) or Qpv(1).
    Target: prod_v Wv(W) * Sv(S) * prod_{p,v} Qpv(1). A choice reaches the target iff every van's load plus its
    slack equals the limit (so the load is within the limit) and every apart pair has at most one member per van."""
    groups, apart, gid = vans_groups(data)
    assert all(g["w"] % unit == 0 for g in groups) and data["W"] % unit == 0
    Wu = data["W"] // unit
    Su = data["S"]
    tot_w = sum(g["w"] for g in groups) // unit
    tot_s = sum(g["s"] for g in groups)
    # moduli large enough that no partial sum wraps: m-1, m, m+1 with (m-1)^3 > tot + limit
    def mods(maxval):
        m = 8
        while (m - 1) ** 3 <= maxval:
            m += 1
        return (m - 1, m, m + 1)
    Wc = [Channel(rng, D, mods(tot_w + Wu)) for _ in range(k)]
    Sc = [Channel(rng, D, mods(tot_s + Su)) for _ in range(k)]
    Qc = {(p, v): Channel(rng, D, (3,)) for p in range(len(apart)) for v in range(k)}
    in_pairs = {g: [p for p, (a, b) in enumerate(apart) if g in (a, b)] for g in range(len(groups))}
    factors = []
    for g, grp in enumerate(groups):
        cols = []
        for v in range(k):
            c = Wc[v].one(grp["w"] // unit) * Sc[v].one(grp["s"])
            for p in in_pairs[g]:
                c = c * Qc[(p, v)].one(1)
            cols.append(c)
        factors.append(np.stack(cols, axis=1))
    nG = len(groups)
    for v in range(k):
        factors.append(Wc[v].code(np.arange(Wu + 1)))
        factors.append(Sc[v].code(np.arange(Su + 1)))
    for p in range(len(apart)):
        for v in range(k):
            factors.append(Qc[(p, v)].code([0, 1]))
    target = np.ones(D, dtype=complex)
    for v in range(k):
        target *= Wc[v].one(Wu) * Sc[v].one(Su)
        for p in range(len(apart)):
            target *= Qc[(p, v)].one(1)
    fixed = {}
    if symmetry:
        heaviest = max(range(nG), key=lambda g: (groups[g]["w"], groups[g]["s"]))
        fixed[heaviest] = 0

    def verify(choice):
        assign = {}
        for g, grp in enumerate(groups):
            for i in grp["members"]:
                assign[i] = choice[g] + 1
        return vans_valid(data, assign, k)
    prob = Problem(factors, target, verify, fixed)
    prob.n_groups = nG
    prob.n_factors = len(factors)
    prob.space_log10 = float(sum(np.log10(C.shape[1]) for C in factors))
    return prob
