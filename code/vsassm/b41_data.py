"""B41 contexts and queries (prereg/B41_superposed_uncertain_multihop_prereg.md). Pure numpy, seeded; no machine files.

A context is a set of N facts (s, r, o) over the 4,096 entities and 32 relations, functional in (s, r). Q query chains
of HMAX = 6 hops are planted first (start e0, relations r1..r6, intermediate entities e1..e6 distinct within the chain);
for each chain, with probability 1/2, a DECOY chain is planted that starts at another member of e0's confusable cluster
and follows the same relations through other members of the clusters of e1..e6 (so a misreading at any hop can jump
to a parallel chain with a different end). The rest of the N facts are random distractors. Queries of h in {2, 4, 6}
hops read the first h relations of a chain; the answer is e_h.

Every mention (the query's start and both arguments of every fact) is an independent draw of one variant of the
true entity's name from the split's bank at the cell's typo rate rho (variant index j, uniform). The same graph and the
same j are used at every rho of a (split, seed, N), so rho levels are paired.
"""
import numpy as np
import b41_names as NM

HMAX = 6
QCAP = 1000


def q_per_context(N):
    return int(max(1, min(QCAP, N // 18)))


def make(seed, N, n_queries, split):
    """Returns dict of numpy arrays: facts (ctx, rel, s, o, js, jo), queries (ctx, e0, j0, rels[6], ents[7])."""
    nvar = {"dev": 3, "test": 8}[split]
    sid = {"dev": 1, "test": 2}[split]
    rng = np.random.default_rng([41, 3, sid, seed, N])
    E, M = NM.E, NM.M_REL
    Q = q_per_context(N)
    n_ctx = int(np.ceil(n_queries / Q))
    F_ctx, F_rel, F_s, F_o = [], [], [], []
    Q_ctx, Q_rels, Q_ents, Q_decoy = [], [], [], []
    for c in range(n_ctx):
        used = set()                                        # (s, r)
        facts = []
        nq = min(Q, n_queries - c * Q)
        planted = 0
        for _ in range(nq):
            for _try in range(1000):
                ents = rng.choice(E, HMAX + 1, replace=False)
                rels = rng.integers(0, M, HMAX)
                if all((int(ents[t]), int(rels[t])) not in used for t in range(HMAX)):
                    break
            else:
                raise RuntimeError("no chain")
            for t in range(HMAX):
                used.add((int(ents[t]), int(rels[t])))
                facts.append((int(rels[t]), int(ents[t]), int(ents[t + 1])))
            decoy = rng.random() < 0.5
            if decoy:
                dents = []
                for t in range(HMAX + 1):
                    cl = NM.cluster_of(int(ents[t]))
                    others = [cl * 4 + k for k in range(4) if cl * 4 + k != int(ents[t])]
                    dents.append(int(others[rng.integers(3)]))
                for t in range(HMAX):
                    if (dents[t], int(rels[t])) in used:
                        break                               # decoy truncated at the first conflict
                    used.add((dents[t], int(rels[t])))
                    facts.append((int(rels[t]), dents[t], dents[t + 1]))
            Q_ctx.append(c); Q_rels.append(rels); Q_ents.append(ents); Q_decoy.append(decoy)
            planted += 1
        if len(facts) > N:
            raise RuntimeError("planted facts exceed N: %d > %d" % (len(facts), N))
        while len(facts) < N:
            s, r = int(rng.integers(E)), int(rng.integers(M))
            if (s, r) in used:
                continue
            used.add((s, r))
            facts.append((r, s, int(rng.integers(E))))
        order = rng.permutation(len(facts))
        for i in order:
            r, s, o = facts[i]
            F_ctx.append(c); F_rel.append(r); F_s.append(s); F_o.append(o)
    vr = np.random.default_rng([41, 4, sid, seed, N])
    nf, nq = len(F_s), len(Q_ctx)
    return {"N": N, "Q_per_ctx": Q, "n_ctx": n_ctx, "seed": seed, "split": split,
            "f_ctx": np.array(F_ctx, np.int32), "f_rel": np.array(F_rel, np.int32),
            "f_s": np.array(F_s, np.int32), "f_o": np.array(F_o, np.int32),
            "f_js": vr.integers(0, nvar, nf).astype(np.int32), "f_jo": vr.integers(0, nvar, nf).astype(np.int32),
            "q_ctx": np.array(Q_ctx, np.int32), "q_rels": np.array(Q_rels, np.int32),
            "q_ents": np.array(Q_ents, np.int32), "q_decoy": np.array(Q_decoy, bool),
            "q_j0": vr.integers(0, nvar, nq).astype(np.int32)}


def row_index(split, rhos):
    """Map (entity, rho, j) -> bank row, for the bank built with these rhos."""
    nvar = {"dev": 3, "test": 8}[split]
    allr = NM.DEV_RHOS if split == "dev" else rhos
    per_e = len(allr) * nvar

    def idx(e, rho, j):
        k = [round(x, 3) for x in allr].index(round(rho, 3))
        return e * per_e + k * nvar + j
    return idx


if __name__ == "__main__":
    import time
    for N in (32, 320, 3200, 32000, 100000):
        t0 = time.time()
        d = make(1, N, 1000, "dev")
        nf = len(d["f_s"])
        print(N, "Q/ctx", d["Q_per_ctx"], "ctx", d["n_ctx"], "facts", nf, "decoy", d["q_decoy"].mean().round(3),
              "%.1fs" % (time.time() - t0))
