"""B22 (prereg/B22_hardware_robustness_prereg.md): the B21 RULER vector stores under hardware-like precision,
read noise and failed components, and the trade back through load or vector length.

python3 b22_robust.py B21_DIR OUT.json [--workers 4]
"""
import collections, glob, json, math, os, sys
import numpy as np
import ruler_vsa as V

THR = 0.5
CELLS = [(131072, "niah_multikey_2"), (32768, "niah_multikey_2"), (131072, "niah_multikey_3"),
         (131072, "niah_multivalue"), (131072, "vt")]
NKEYS = 150


def load_stores(d):
    out = []
    for L, task in CELLS:
        rows = [json.loads(l) for f in sorted(glob.glob(os.path.join(d, "%d_*.jsonl" % L))) for l in open(f)]
        rows = [r for r in rows if r["task"] == task][:10]
        assert len(rows) == 10, (L, task, len(rows))
        out += [(L, task, r["index"], [tuple(p) for p in r["pairs"]]) for r in rows]
    return out


def phases(strings, slot, d):
    return np.stack([np.random.default_rng(V._seed("code:%s:%s" % (slot, s))).uniform(-np.pi, np.pi, d)
                     .astype(np.float32) for s in strings])


def to_code(ph, bits):
    if bits:
        step = 2 * np.pi / 2 ** bits
        ph = np.round(ph / step) * step
    return np.exp(1j * ph).astype(np.complex64)


def quant_part(x, bits):
    if bits == 1:
        return np.sign(x) + (x == 0)
    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True)) + 1e-12
    clip = 3 * rms
    step = 2 * clip / 2 ** bits
    q = (np.floor(x / step) + 0.5) * step
    return np.clip(q, -clip + step / 2, clip - step / 2)


def quant_mem(m, bits):
    rms0 = np.sqrt(np.mean(np.abs(m) ** 2, axis=-1, keepdims=True))
    q = quant_part(m.real, bits) + 1j * quant_part(m.imag, bits)
    rms1 = np.sqrt(np.mean(np.abs(q) ** 2, axis=-1, keepdims=True)) + 1e-12
    return (q * (rms0 / rms1)).astype(np.complex64)


def run_store(pairs, d=8192, per_shard=16, bits=None, r=0.0, p=0.0, bipolar=False):
    rng = np.random.default_rng(0)
    table = V.table_of(pairs)
    keys = list(table)
    if len(keys) > NKEYS:
        keys = keys[:: max(1, len(keys) // NKEYS)][:NKEYS]
    S = max(1, math.ceil(len(pairs) / per_shard))
    shard = lambda k: V._seed("shard:" + k) % S
    vocab = list(dict.fromkeys(v for _, v, _ in pairs))
    vi = {v: i for i, v in enumerate(vocab)}
    allk = list(dict.fromkeys(k for k, _, _ in pairs))
    ki = {k: i for i, k in enumerate(allk)}
    maxj = max(j for _, _, j in pairs) + 2
    Kc = to_code(phases(allk, "key", d), bits)
    Vc = to_code(phases(vocab, "value", d), bits)
    Pc = to_code(phases([str(j) for j in range(maxj)], "pos", d), bits)
    mem = np.zeros((S, d), np.complex64)
    for k, v, j in pairs:
        mem[shard(k)] += Kc[ki[k]] * Pc[j] * Vc[vi[v]]
    if bits and bipolar:                      # amendment 2: real part only, sign, random ties (seed 0 per store)
        assert bits == 1 and np.abs(mem.imag).max() < 1e-3
        rms0 = np.sqrt(np.mean(mem.real ** 2, axis=-1, keepdims=True))
        q = np.sign(mem.real)
        ties = q == 0
        q[ties] = np.random.default_rng(0).choice([-1.0, 1.0], size=int(ties.sum()))
        mem = (q * rms0).astype(np.complex64)
    elif bits:
        mem = quant_mem(mem, bits)
    if p:
        mem = mem * (rng.random(mem.shape) >= p)
    rms = np.sqrt(np.mean(np.abs(mem) ** 2, axis=-1))
    sh = np.array([shard(k) for k in keys])
    X = mem[sh] * np.conj(Kc[[ki[k] for k in keys]])
    got = [[] for _ in keys]
    active = np.arange(len(keys))
    j = 0
    VH = Vc.conj().T
    while len(active) and j < maxj:
        Y = X[active] * np.conj(Pc[j])
        if r:
            sd = (r * rms[sh[active]] / np.sqrt(2))[:, None]
            Y = Y + (sd * (rng.standard_normal(Y.shape) + 1j * rng.standard_normal(Y.shape))).astype(np.complex64) \
                * np.conj(Kc[[ki[keys[a]] for a in active]] * Pc[j])
        sims = (Y @ VH).real / d
        b = sims.argmax(1)
        ok = sims[np.arange(len(active)), b] >= THR
        for a, bb, o in zip(active, b, ok):
            if o:
                got[a].append(vocab[bb])
        active = active[ok]
        j += 1
    exact = [g == table[k] for g, k in zip(got, keys)]
    return sum(exact), len(exact), int(S), int(np.bincount([shard(k) for k, _, _ in pairs]).max())


def conditions():
    c = [("float", {})]
    c += [("Q b=%d" % b, {"bits": b}) for b in (1, 2, 3, 4, 6, 8)]
    c += [("N r=%g" % r, {"r": r}) for r in (0.25, 0.5, 1, 2)]
    c += [("F p=%g" % p, {"p": p}) for p in (0.05, 0.1, 0.2, 0.3)]
    for name, kw in (("Q b=1", {"bits": 1}), ("Q b=2", {"bits": 2}), ("N r=2", {"r": 2}), ("F p=0.3", {"p": 0.3})):
        c += [(name + " | 8 per vector", dict(kw, per_shard=8)), (name + " | 4 per vector", dict(kw, per_shard=4)),
              (name + " | d=16384", dict(kw, d=16384))]
    return c


def one(args):
    name, kw, stores = args
    tot_e = tot_n = all_ok = 0
    per = collections.defaultdict(lambda: [0, 0])
    for L, task, idx, pairs in stores:
        e, n, S, ml = run_store(pairs, **kw)
        tot_e += e; tot_n += n; all_ok += e == n
        per["%d/%s" % (L, task)][0] += e; per["%d/%s" % (L, task)][1] += n
    res = {"cond": name, "kw": kw, "keys_exact": tot_e, "keys": tot_n, "frac": tot_e / tot_n,
           "stores_all_exact": all_ok, "stores": len(stores), "per_cell": dict(per)}
    print("%-28s %6d/%-6d %.4f  stores %d/%d" % (name, tot_e, tot_n, tot_e / tot_n, all_ok, len(stores)), flush=True)
    return res


def conditions_ext():
    """Amendment 1: added levels, descriptive (no bar); amendment 2: bipolar 1-bit store (post hoc)."""
    bip = [("Q b=1 bipolar", {"bits": 1, "bipolar": True})] + \
          [("Q b=1 bipolar | " + n, dict({"bits": 1, "bipolar": True}, **kw)) for n, kw in
           (("8 per vector", {"per_shard": 8}), ("4 per vector", {"per_shard": 4}), ("d=16384", {"d": 16384}))]
    return bip + [("N r=%g" % r, {"r": r}) for r in (3, 4, 6, 8)] + [("F p=%g" % p, {"p": p}) for p in (0.4, 0.5, 0.6, 0.7)]


if __name__ == "__main__":
    src, out = sys.argv[1], sys.argv[2]
    workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 4
    stores = load_stores(src)
    print("stores", len(stores), "pairs max", max(len(s[3]) for s in stores), flush=True)
    # float baseline must reproduce B21's own store on probed keys
    for L, task, idx, pairs in stores[:3] + stores[20:22]:
        st, t = V.Store(pairs), V.table_of(pairs)
        ks = list(t)[:: max(1, len(t) // 20)][:20]
        assert all(st.read(k) == t[k] for k in ks)
    if "--selftest" in sys.argv:
        print(one(("float", {}, stores[:2]))); sys.exit()
    from multiprocessing import Pool
    with Pool(workers) as pool:
        conds = conditions_ext() if "--ext" in sys.argv else conditions()
        if "--trades-ext" in sys.argv:       # amendment 1: trades at the first added level below 99.5% (r = 3, p = 0.5)
            conds = [(nm + " | " + t, dict(kw, **tk)) for nm, kw in (("N r=3", {"r": 3}), ("F p=0.5", {"p": 0.5}))
                     for t, tk in (("8 per vector", {"per_shard": 8}), ("4 per vector", {"per_shard": 4}), ("d=16384", {"d": 16384}))]
        if "--bip" in sys.argv:
            conds = [c for c in conditions_ext() if "bipolar" in c[0]]
        res = pool.map(one, [(n, kw, stores) for n, kw in conds], chunksize=1)
    json.dump(res, open(out, "w"), indent=1)
