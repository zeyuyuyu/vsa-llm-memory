"""B41 second implementation (CPU, numpy float64, explicit loops and counters) of the scoring and of the cost counts,
written from the cost model in the prereg, not from b41_engine.py. Shared with the engine: the instance generator
(b41_data.make), the bank row map, the logits file and the codebook/permutation specification (seeded generator).

For the first --nsub queries of an engine run it recomputes, per config: answer, ops, traffic, state at h = 2, 4, 6,
and reports agreement with the engine's per-query outputs.

  python b41_indep.py --run <engine .npz> --logits <split>_logits.npy --nsub 150
"""
import argparse, json, os, sys
from collections import defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import b41_names as NM                                                      # noqa: E402
import b41_data as DA                                                       # noqa: E402

E = NM.E
HOUT = (2, 4, 6)


def softmax64(x):
    x = x.astype(np.float64)
    x = x - x.max(-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(-1, keepdims=True)


def topk(p, k):
    """{entity: prob} of the k largest entries (ties: lower index first), zero entries dropped."""
    if k >= E:
        idx = np.nonzero(p)[0]
    else:
        idx = np.argsort(-p, kind="stable")[:k]
    return {int(i): float(p[i]) for i in idx if p[i] > 0}


def first_argmax(d):
    best, bv = -1, -1.0
    for e in sorted(d):
        if d[e] > bv:
            best, bv = e, d[e]
    return best


class Ctx:
    def __init__(self, run, logits):
        meta = json.load(open(run.replace(".npz", ".json")))
        self.meta = meta
        d = DA.make(meta["seed"], meta["N"], meta["nq"], meta["split"])
        self.d = d
        lmeta = json.load(open(logits.replace("_logits.npy", "_logits_meta.json")))
        idx = DA.row_index(meta["split"], lmeta["rhos"])
        lg = np.load(logits, mmap_mode="r")
        rho = meta["rho"]
        self.ps = lambda f: softmax64(lg[idx(int(d["f_s"][f]), rho, int(d["f_js"][f]))])
        self.po = lambda f: softmax64(lg[idx(int(d["f_o"][f]), rho, int(d["f_jo"][f]))])
        self.p0 = lambda q: softmax64(lg[idx(int(d["q_ents"][q, 0]), rho, int(d["q_j0"][q]))])
        self.slot = defaultdict(list)                          # (ctx, rel) -> facts in file order
        for f in range(len(d["f_s"])):
            self.slot[(int(d["f_ctx"][f]), int(d["f_rel"][f]))].append(f)
        self._ps, self._po = {}, {}
        self._tk = {}

    def TK(self, side, f, k):
        key = (side, f, k)
        if key not in self._tk:
            self._tk[key] = topk(self.PS(f) if side == "s" else self.PO(f), k)
        return self._tk[key]

    def facts(self, q, t):
        return self.slot.get((int(self.d["q_ctx"][q]), int(self.d["q_rels"][q, t])), [])

    def PS(self, f):
        if f not in self._ps:
            self._ps[f] = self.ps(f)
        return self._ps[f]

    def PO(self, f):
        if f not in self._po:
            self._po[f] = self.po(f)
        return self._po[f]


def beam(C, q, k, B):
    """Explicit posting-list beam. Returns per h: (answer, ops, traffic, state)."""
    out = {}
    st = topk(C.p0(q), k)
    ops, tr = float(E), float(E)
    if B and len(st) > B:
        st = dict(sorted(st.items(), key=lambda kv: (-kv[1], kv[0]))[:B])
    state_max = 2 * len(st)
    for t in range(6):
        fl = C.facts(q, t)
        post = {}                                              # posting lists of this slot: e -> [(f, w)]
        for f in fl:
            for e, w in C.TK('s', f, k).items():
                post.setdefault(e, []).append((f, w))
        acc = defaultdict(float)
        n_probe = len(st)
        n_post = 0
        for e, m in st.items():
            for (f, w) in post.get(e, []):
                acc[f] += m * w
                n_post += 1
        new = defaultdict(float)
        n_scat = 0
        for f, wf in acc.items():
            for o, po in C.TK('o', f, k).items():
                new[o] += wf * po
                n_scat += 1
        new = {e: v for e, v in new.items() if v > 0}
        tot = sum(new.values())
        if tot > 0:
            new = {e: v / tot for e, v in new.items()}
        ops += n_probe + n_post + n_scat + len(new)
        tr += 2 * n_probe + 2 * n_post + 2 * n_scat
        if B and len(new) > B:
            ops += len(new)
            new = dict(sorted(new.items(), key=lambda kv: (-kv[1], kv[0]))[:B])
        st = new
        state_max = max(state_max, 2 * len(st))
        if t + 1 in HOUT:
            ans = first_argmax(st) if st else -1
            out[t + 1] = (ans, ops + len(st), tr + len(st), float(state_max))
    return out


def exact(C, q):
    out = {}
    p = C.p0(q)
    ops = tr = float(E)
    for t in range(6):
        fl = C.facts(q, t)
        new = np.zeros(E)
        for f in fl:
            w = float(np.dot(C.PS(f), p))
            new += w * C.PO(f)
        ops += len(fl) * E * 2 + E
        tr += len(fl) * E * 2 + E
        s = new.sum()
        p = new / s if s > 0 else new
        if t + 1 in HOUT:
            out[t + 1] = (int(np.argmax(p)) if s > 0 else -1, ops + E, tr + E, float(E))
    return out


def enum(C, q, k, cap=2_000_000):
    out = {}
    paths = list(topk(C.p0(q), k).items())
    ops = tr = float(E)
    smax = 2 * len(paths)
    for t in range(6):
        fl = C.facts(q, t)
        post = {}
        for f in fl:
            for e, w in C.TK('s', f, k).items():
                post.setdefault(e, []).append((f, w))
        new = []
        mac = 0
        for e, m in paths:
            for (f, w) in post.get(e, []):
                ob = C.TK('o', f, k)
                mac += 1 + len(ob)
                for o, po in ob.items():
                    new.append((o, m * w * po))
            if len(new) > cap:
                return None
        ops += len(paths) + mac
        tr += 2 * len(paths) + 2 * mac
        paths = new
        smax = max(smax, 2 * len(paths))
        if t + 1 in HOUT:
            out[t + 1] = (None, ops + len(paths), tr + len(paths), float(smax))
    return out


def codebook(seed, D):
    import torch
    g = torch.Generator(device="cpu").manual_seed(41_000_000 + 1000 * seed + D % 997)
    PHI = (torch.randint(0, 2, (E, D), generator=g, dtype=torch.int8) * 2 - 1).numpy().astype(np.float64)
    P = torch.randperm(D, generator=g).numpy()
    return PHI, P


def vsa(C, q, D, kb, c, PHI, P, mem, trace=None):
    out = {}
    w0 = topk(C.p0(q), kb)
    x = np.zeros(D)
    for e, w in w0.items():
        x += w * PHI[e]
    ops = E + len(w0) * D
    tr = E + len(w0) * D + 2 * len(w0)
    n = np.linalg.norm(x); x = x / n if n > 0 else x; ops += 2 * D
    dead = n == 0
    Pinv = np.argsort(P)
    for t in range(6):
        key = (int(C.d["q_ctx"][q]), int(C.d["q_rels"][q, t]))
        Mr = mem(key)
        z = x * Mr if Mr is not None else np.zeros(D)
        x = z[Pinv]
        ops += D; tr += 2 * D
        n = np.linalg.norm(x); ops += 2 * D
        dead = dead or n == 0
        x = x / n if n > 0 else x
        if c:
            sc = PHI @ x
            ops += E * D + E; tr += E * D
            order = np.argsort(-sc, kind="stable")[:c]
            ans = int(order[0])
            if trace is not None:
                trace[t + 1] = sc.copy()
            if t + 1 in HOUT:
                out[t + 1] = (-1 if dead else ans, ops, tr, float(D))
            if t + 1 < 6:
                x = np.zeros(D)
                for e in order:
                    x += max(sc[e], 0.0) * PHI[e]
                ops += c * D; tr += c * D
                n = np.linalg.norm(x); ops += 2 * D
                dead = dead or n == 0
                x = x / n if n > 0 else x
        elif t + 1 in HOUT:
            sc = PHI @ x
            if trace is not None:
                trace[t + 1] = sc.copy()
            out[t + 1] = (-1 if dead else int(np.argmax(sc)), ops + E * D + E, tr + E * D, float(D))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--logits", required=True)
    ap.add_argument("--nsub", type=int, default=150)
    ap.add_argument("--Ds", type=int, nargs="*", default=[256, 1024, 4096])
    ap.add_argument("--out", required=True)
    ap.add_argument("--no_vsa", action="store_true")
    a = ap.parse_args()
    C = Ctx(a.run, a.logits)
    Z = np.load(a.run)
    qs = list(range(min(a.nsub, len(C.d["q_ctx"]))))
    rep = {}

    def cmp(name, fn):
        agree = n = na = 0
        cost_bad = defaultdict(int)
        for q in qs:
            r = fn(q)
            if r is None:
                continue
            for hi, h in enumerate(HOUT):
                ans, ops, tr, st = r[h]
                if ans is not None:
                    agree += int(ans == Z["%s|answer" % name][q, hi])
                    na += 1
                n += 1
                for key, v in (("ops", ops), ("traffic", tr), ("state", st)):
                    ev = Z["%s|%s" % (name, key)][q, hi]
                    if abs(ev - v) > 1e-6 * max(1.0, abs(v)):
                        cost_bad[key] += 1
        rep[name] = {"n": n, "answer_agree": agree / na if na else None, "cost_mismatch": dict(cost_bad)}
        print(name, rep[name], flush=True)

    for (k, B) in [(1, 1), (2, 4), (4, 16), (8, 64), (16, 0), (4, 0)]:
        nm = "ARGMAX" if (k, B) == (1, 1) else "BEAM_k%d_B%s" % (k, "inf" if B == 0 else B)
        cmp(nm, lambda q, k=k, B=B: beam(C, q, k, B))
    cmp("EXACT", lambda q: exact(C, q))
    for k in (2, 4):
        cmp("ENUM_k%d" % k, lambda q, k=k: enum(C, q, k))
    seed = C.meta["seed"]
    for D in ([] if a.no_vsa else a.Ds):
        PHI, P = codebook(seed, D)
        for kb in (1, 4):
            cache = {}

            def mem(key, kb=kb):
                if key not in cache:
                    fl = C.slot.get(key, [])
                    if not fl:
                        cache[key] = None
                    else:
                        m = np.zeros(D)
                        for f in fl:
                            cs = np.zeros(D); co = np.zeros(D)
                            for e, w in C.TK('s', f, kb).items():
                                cs += w * PHI[e]
                            for e, w in C.TK('o', f, kb).items():
                                co += w * PHI[e]
                            m += cs * co[P]
                        cache[key] = m
                return cache[key]
            for mode, c in (("END", 0), ("HOP1", 1), ("HOP4", 4)):
                cmp("VSA_%s_D%d_kb%d" % (mode, D, kb), lambda q, c=c: vsa(C, q, D, kb, c, PHI, P, mem))
    json.dump(rep, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
