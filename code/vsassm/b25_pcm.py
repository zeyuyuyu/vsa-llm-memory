"""B25 (prereg/B25_pcm_device_model_prereg.md): the B21 RULER stores, bipolar (+-1) variant, on a phase-change-memory
(PCM) device model whose parameters are the published measurements of Joshi et al. 2020 (Nat. Commun. 11, 2473,
Supplementary Note 2, Eqs. 8-17): programming noise at write, conductance drift with a per-device drift exponent,
1/f read noise, an 8-bit ADC, and global drift compensation (GDC).

python3 b25_pcm.py B21_DIR OUT.json [--workers 4]     # the registered grid + descriptive extras
python3 b25_pcm.py --selftest                         # synthetic pairs only (never the real stores)

Hardware mapping (prereg):
  store crossbar  : one row per bundled vector. B1 = the B22 amendment-2 bipolar store (sign of the integer sum, random
                    ties), each component a differential pair (+1 -> (G_ON, RESET), -1 -> (RESET, G_ON)), read device by
                    device with the ADC and taken as a sign. ML = the integer sum itself, |m| mapped linearly onto
                    [0, G_ON] per vector (scale max|m| kept digitally), m > 0 on the + device, m < 0 on the - device.
  unbind          : digital (+-1 key and position codes: sign flips).
  clean-up crossbar: the store's value codes (+-1) as d x N differential pairs; the unbound vector is applied as inputs
                    (B1: 1-bit, ML: 8-bit), column currents are digitized per 256-row tile with the ADC and summed.
"""
import collections
import json
import math
import os
import sys

import numpy as np

import b22_robust as B
import ruler_vsa as V

D, THR, NKEYS = 8192, 0.5, 150

# ---- PCM device model: Joshi et al. 2020, Supplementary Note 2 (conductance in uS, time in s) ----
G_ON = 25.0                              # programmed device target (main text: 0-25 uS range; Gmax = 25 uS)
SIG_PROG = {True: 1.2, False: 0.2}       # programming-noise sd per event (T0 and T1): G_T > 0 / RESET (G_T = 0)
T_REF, T0, T1 = 20.0, 27.36, 55.39       # targets defined at 20 s; programming-noise events at T0, T1 (SI Eqs. 13-14)
T_READ, GAMMA, T_Q = 250e-9, 1.21, 1.089e5
QS_CAP = 0.2                             # Eq. 10 diverges at G_T = 0; cap (as in IBM aihwkit's PCM model)
TILE, ADC_BITS = 256, 8
TIMES = [("60 s", 60.0), ("1 h", 3600.0), ("1 d", 86400.0), ("30 d", 30 * 86400.0), ("1 y", 365 * 86400.0)]
LOADS = (16, 32, 64)
CH = 512                                 # codebook columns per generation chunk (memory bound)


def mu_nu(g):                            # SI Eq. 8
    return np.maximum(-0.0045 * g + 0.08, np.minimum(0.06, np.maximum(-0.00169 * g + 0.0825, 0.055)))


def sd_nu(g):                            # SI Eq. 9
    return 1.0 / (8.0 * g + 30.0320)


def nu_q(g):                             # SI Eq. 11; the second term applies for G_T > 13.21 uS (where it is >= 0)
    return np.minimum(0.081, 0.04 * g + 0.02) - np.where(g > 13.21, np.minimum(0.035, -0.2048 + 0.0155 * g), 0.0)


def q_s(g, t):                           # SI Eq. 10
    return np.minimum(0.0388 / np.maximum(g, 1e-9) ** 0.5729, QS_CAP) * (t / T_Q) ** (-nu_q(g))


def f_int(t):                            # SI Eq. 17 frequency integral, f_max = 1/(2 T_read), f_min = 1/(t + T_read)
    fmax, fmin = 1.0 / (2 * T_READ), 1.0 / (t + T_READ)
    return math.sqrt((fmax ** (1 - GAMMA) - fmin ** (1 - GAMMA)) / (1 - GAMMA))


def program(gt, shape, rng, k=1.0):
    """SI Eqs. 12-14: per-device drift exponent (Gaussian, truncated at 0: drift is relaxation, conductance does not
    grow), then the two programming-noise events. Conductance clamped at >= 0."""
    gt = np.asarray(gt, np.float32)
    nu = np.maximum(mu_nu(gt) + k * sd_nu(gt) * rng.standard_normal(shape, np.float32), 0).astype(np.float32)
    sp = (k * np.where(gt > 0, SIG_PROG[True], SIG_PROG[False])).astype(np.float32)
    g = np.maximum(gt * np.exp(-nu * math.log(T0 / T_REF)) + sp * rng.standard_normal(shape, np.float32), 0)
    g = np.maximum(g * np.exp(-nu * math.log(T1 / T0)) + sp * rng.standard_normal(shape, np.float32), 0)
    return nu, g.astype(np.float32)


def at_time(gt, nu, g1, t, rng, k=1.0):
    """SI Eqs. 16-17: drift from T1 to t, plus the 1/f read noise accumulated since programming (one draw per device
    per time point: the array state read at time t). Returns (G(t), G_drift(t), read-noise sd)."""
    gt = np.asarray(gt, np.float32)
    gd = (g1 * np.exp(-nu * math.log(t / T1))).astype(np.float32)
    sig = (k * q_s(gt, t) * f_int(t) * gd).astype(np.float32)
    g = np.maximum(gd + sig * rng.standard_normal(gd.shape, np.float32), 0) if rng is not None else gd
    return g.astype(np.float32), gd, sig


def adc(x, fs, bits):
    """Symmetric mid-tread quantizer, 2^bits - 1 levels over [-fs, fs], clipped."""
    n = 2 ** (bits - 1) - 1
    step = fs / n
    return np.clip(np.round(x / step), -n, n) * step


def bip(strings, slot, d):
    """The B22 1-bit codes (phase rounded to 0 or pi), as real +-1."""
    return np.where(np.cos(np.round(B.phases(strings, slot, d) / np.pi) * np.pi) > 0, 1.0, -1.0).astype(np.float32)


def codebook_array(Vc, seed, t, k=1.0, redraw=False):
    """Clean-up crossbar at time t: W (d x N, uS) = G+ - G- per value component, and the GDC scale alpha =
    (summed conductance of all devices, one calibration read) / (summed targets). With redraw=True, W holds the drifted
    conductances without the per-time read-noise draw, and S2 the per-weight read-noise variance (fresh noise is then
    drawn on every MVM at the output)."""
    N, d = Vc.shape
    W = np.empty((d, N), np.float32)
    S2 = np.empty((d, N), np.float32) if redraw else None
    gsum = 0.0
    for c0 in range(0, N, CH):
        c1 = min(N, c0 + CH)
        shp = (d, c1 - c0)
        r = np.random.default_rng([seed, 1, c0])
        nu_h, g_h = program(G_ON, shp, r, k)
        nu_l, g_l = program(0.0, shp, r, k)
        rr = np.random.default_rng([seed, 2, c0, int(t)])
        Gh, gdh, sh = at_time(G_ON, nu_h, g_h, t, rr, k)
        Gl, gdl, sl = at_time(0.0, nu_l, g_l, t, rr, k)
        s = Vc[c0:c1].T
        gsum += float(Gh.sum(dtype=np.float64) + Gl.sum(dtype=np.float64))
        if redraw:
            W[:, c0:c1] = s * (gdh - gdl)
            S2[:, c0:c1] = sh ** 2 + sl ** 2
        else:
            W[:, c0:c1] = s * (Gh - Gl)
    return W, S2, gsum / (N * d * G_ON)


def store_array(kind, mem, b1, seed, t, k=1.0, bits=ADC_BITS):
    """Store crossbar read at time t. Returns {comp: rows as read (S x d, before the per-vector digital scale)} and
    diagnostics. kind B1: rows = +-1 signs read through the ADC (ADC zero -> random tie); kind ML: rows = m-hat."""
    S, d = mem.shape
    r = np.random.default_rng([seed, 3, S, 0 if kind == "B1" else 1])
    if kind == "B1":
        gp = np.where(b1 > 0, G_ON, 0.0).astype(np.float32)
        gm = np.where(b1 < 0, G_ON, 0.0).astype(np.float32)
    else:
        M = np.maximum(np.abs(mem).max(1, keepdims=True), 1.0)
        gp = (np.maximum(mem, 0) * (G_ON / M)).astype(np.float32)
        gm = (np.maximum(-mem, 0) * (G_ON / M)).astype(np.float32)
    nu_p, g1p = program(gp, gp.shape, r, k)
    nu_m, g1m = program(gm, gm.shape, r, k)
    rr = np.random.default_rng([seed, 4, S, 0 if kind == "B1" else 1, int(t)])
    Gp, _, _ = at_time(gp, nu_p, g1p, t, rr, k)
    Gm, _, _ = at_time(gm, nu_m, g1m, t, rr, k)
    q = adc(Gp - Gm, G_ON, bits)
    alpha = float((Gp.sum(dtype=np.float64) + Gm.sum(dtype=np.float64)) / max(gp.sum(dtype=np.float64) + gm.sum(dtype=np.float64), 1e-9))
    if kind == "B1":
        sgn = np.sign(q)
        ties = sgn == 0
        sgn[ties] = np.random.default_rng([seed, 5, S, int(t)]).choice([-1.0, 1.0], size=int(ties.sum()))
        out = {"gdc": sgn.astype(np.float32), "none": sgn.astype(np.float32)}
        diag = {"alpha_store": alpha, "bit_err": float((sgn != b1).mean()), "ties": float(ties.mean())}
    else:
        base = q * (M / G_ON)
        out = {"gdc": (base / alpha).astype(np.float32), "none": base.astype(np.float32)}
        rms = np.sqrt(np.mean(mem ** 2)) + 1e-12
        diag = {"alpha_store": alpha, "rel_err_gdc": float(np.sqrt(np.mean((out["gdc"] - mem) ** 2)) / rms),
                "rel_err_none": float(np.sqrt(np.mean((out["none"] - mem) ** 2)) / rms)}
    return out, diag


def pcm_cleanup(W, alpha, n_in, bits, S2=None, rng=None):
    """In-memory clean-up: inputs quantized to n_in levels per sign (B1: 1 -> exact +-1; ML: 127 = 8-bit DAC), per-tile
    column currents digitized with a full-range ADC (fs = TILE * n_in * G_ON, never clips), summed digitally, divided by
    alpha * G_ON * d and rescaled by the input scale."""
    d = W.shape[0]

    def f(Y):
        sx = np.abs(Y).max(1, keepdims=True) / n_in + 1e-30
        xq = np.clip(np.round(Y / sx), -n_in, n_in).astype(np.float32)
        out = np.zeros((Y.shape[0], W.shape[1]), np.float32)
        for a in range(0, d, TILE):
            p = xq[:, a:a + TILE] @ W[a:a + TILE]
            if S2 is not None:
                p += np.sqrt((xq[:, a:a + TILE] ** 2) @ S2[a:a + TILE]) * rng.standard_normal(p.shape, np.float32)
            out += adc(p, TILE * n_in * G_ON, bits)
        return out * (sx / (alpha * G_ON * d))
    return f


def read_keys(rows, scale, sh, kidx, Kc, Pc, cleanup, maxj):
    """B21/B22 read: unbind key and position j, clean up, accept if the best similarity >= 0.5, stop at the first j below."""
    X = rows[sh] * scale[sh][:, None] * Kc[kidx]
    got = [[] for _ in sh]
    active = np.arange(len(sh))
    j = 0
    while len(active) and j < maxj:
        sims = cleanup(X[active] * Pc[j])
        b = sims.argmax(1)
        ok = sims[np.arange(len(active)), b] >= THR
        for a, bb, o in zip(active, b, ok):
            if o:
                got[a].append(int(bb))
        active = active[ok]
        j += 1
    return got


def codes(pairs, d=D):
    vocab = list(dict.fromkeys(v for _, v, _ in pairs))
    allk = list(dict.fromkeys(k for k, _, _ in pairs))
    maxj = max(j for _, _, j in pairs) + 2
    return dict(vocab=vocab, allk=allk, maxj=maxj, Kc=bip(allk, "key", d), Vc=bip(vocab, "value", d),
                Pc=bip([str(j) for j in range(maxj)], "pos", d))


def build(pairs, per_shard, cd, d=D):
    table = V.table_of(pairs)
    keys = list(table)
    if len(keys) > NKEYS:
        keys = keys[:: max(1, len(keys) // NKEYS)][:NKEYS]
    S = max(1, math.ceil(len(pairs) / per_shard))
    shard = lambda k: V._seed("shard:" + k) % S
    vi = {v: i for i, v in enumerate(cd["vocab"])}
    ki = {k: i for i, k in enumerate(cd["allk"])}
    Kc, Vc, Pc, maxj = cd["Kc"], cd["Vc"], cd["Pc"], cd["maxj"]
    mem = np.zeros((S, d), np.float32)
    for k, v, j in pairs:
        mem[shard(k)] += Kc[ki[k]] * Pc[j] * Vc[vi[v]]
    b1 = np.sign(mem)                    # B22 amendment 2: sign, random ties (seed 0 per store)
    ties = b1 == 0
    b1[ties] = np.random.default_rng(0).choice([-1.0, 1.0], size=int(ties.sum()))
    rms0 = np.sqrt(np.mean(mem ** 2, axis=1))
    want = [[vi[v] for v in table[k]] for k in keys]
    sh = np.array([shard(k) for k in keys])
    kidx = np.array([ki[k] for k in keys])
    return dict(keys=keys, Kc=Kc, Vc=Vc, Pc=Pc, mem=mem, b1=b1.astype(np.float32), rms0=rms0.astype(np.float32),
                want=want, sh=sh, kidx=kidx, maxj=maxj, S=S)


def conditions():
    """(name, variant, stage, comp, time label, load, k, bits, redraw). Registered grid first, then descriptive extras."""
    c = []
    for load in LOADS:
        for var in ("B1", "ML"):
            c.append(("%s ideal | %d/vec" % (var, load), var, "ideal", "gdc", "-", load, 1.0, ADC_BITS, False))
    for tl, _ in TIMES:
        for load in LOADS:
            for var in ("B1", "ML"):
                for stage in ("S", "C", "SC"):
                    for comp in ("gdc", "none"):
                        c.append(("%s-%s %s | %s | %d/vec" % (var, stage, comp, tl, load), var, stage, comp, tl, load,
                                  1.0, ADC_BITS, False))
    for var in ("B1", "ML"):                                  # descriptive extras at 30 d, 16 per vector, SC, GDC
        for k in (2.0, 4.0, 8.0, 16.0, 32.0):
            for stage in ("S", "C", "SC"):
                c.append(("X %s-%s gdc | 30 d | 16/vec | noise x%g" % (var, stage, k), var, stage, "gdc", "30 d", 16, k,
                          ADC_BITS, False))
        for bits in (6, 4):
            c.append(("X %s-SC gdc | 30 d | 16/vec | ADC %d-bit" % (var, bits), var, "SC", "gdc", "30 d", 16, 1.0, bits, False))
        c.append(("X %s-SC gdc | 30 d | 16/vec | read noise redrawn per MVM" % var, var, "SC", "gdc", "30 d", 16, 1.0,
                  ADC_BITS, True))
    return c


def run_store(args):
    L, task, idx, pairs, conds = args
    seed = V._seed("b25:%d:%s:%d" % (L, task, idx)) % (2 ** 62)
    tmap = dict(TIMES)
    res = collections.defaultdict(lambda: [0, 0])
    diag = {}
    built = {}
    cd = codes(pairs)

    def get(load):
        if load not in built:
            built[load] = build(pairs, load, cd)
        return built[load]

    def score(st, got):
        return sum(g == w for g, w in zip(got, st["want"])), len(got)

    for name, var, stage, comp, tl, load, k, bits, redraw in conds:
        if stage != "ideal":
            continue
        st = get(load)
        rows, scale = (st["b1"], st["rms0"]) if var == "B1" else (st["mem"], np.ones(st["S"], np.float32))
        dig = lambda Y, Vc=st["Vc"]: Y @ Vc.T / D
        res[name] = list(score(st, read_keys(rows, scale, st["sh"], st["kidx"], st["Kc"], st["Pc"], dig, st["maxj"])))
    groups = collections.OrderedDict()
    for cnd in conds:
        if cnd[2] != "ideal":
            groups.setdefault((cnd[4], cnd[6], cnd[8]), []).append(cnd)
    for (tl, k, redraw), cl in groups.items():
        t = tmap[tl]
        Wcache = {}
        for load in sorted({c[5] for c in cl}):
            st = get(load)
            if any(c[2] in ("C", "SC") for c in cl if c[5] == load):
                if "W" not in Wcache:          # the clean-up crossbar does not depend on the load
                    Wcache["W"] = codebook_array(st["Vc"], seed, t, k, redraw)
                    diag["alpha_cb | %s | x%g%s" % (tl, k, " redraw" if redraw else "")] = Wcache["W"][2]
            for var in ("B1", "ML"):
                cv = [c for c in cl if c[5] == load and c[1] == var]
                if not cv:
                    continue
                stored = {}
                for bits in sorted({c[7] for c in cv}):
                    if any(c[2] in ("S", "SC") for c in cv if c[7] == bits):
                        stored[bits], dg = store_array(var, st["mem"], st["b1"], seed, t, k, bits)
                        diag["%s store | %s | %d/vec | x%g | %d-bit" % (var, tl, load, k, bits)] = dg
                for name, _, stage, comp, _, _, _, bits, _ in cv:
                    if stage in ("S", "SC"):
                        rows = stored[bits][comp]
                    else:
                        rows = st["b1"] if var == "B1" else st["mem"]
                    scale = st["rms0"] if var == "B1" else np.ones(st["S"], np.float32)
                    if stage == "S":
                        clean = lambda Y, Vc=st["Vc"]: Y @ Vc.T / D
                    else:
                        W, S2, a = Wcache["W"]
                        clean = pcm_cleanup(W, a if comp == "gdc" else 1.0, 1 if var == "B1" else 2 ** (bits - 1) - 1,
                                            bits, S2, np.random.default_rng([seed, 6, int(t), len(name)]))
                    res[name] = list(score(st, read_keys(rows, scale, st["sh"], st["kidx"], st["Kc"], st["Pc"], clean,
                                                         st["maxj"])))
        Wcache.clear()
    return {"store": "%d/%s/%d" % (L, task, idx), "cell": "%d/%s" % (L, task), "res": dict(res), "diag": diag}


def selftest():
    """Synthetic pairs only: code paths, the B22 equivalence of the ideal 1-bit path, and timing."""
    import time
    rng = np.random.default_rng(1)
    pairs, seen = [], collections.defaultdict(int)
    for i in range(1500):
        k = "key%d" % rng.integers(0, 1200)
        pairs.append((k, "val%d" % i, seen[k]))
        seen[k] += 1
    e, n, _, _ = B.run_store(pairs, bits=1, bipolar=True)
    conds = conditions()
    t0 = time.time()
    out = run_store((0, "synthetic", 0, pairs, [c for c in conds if c[2] == "ideal"]))
    print("ideal", out["res"], "B22 bipolar 16/vec:", e, n)
    assert out["res"]["B1 ideal | 16/vec"] == [e, n], "ideal 1-bit path must reproduce B22 amendment 2"
    sub = [c for c in conds if (c[4] in ("60 s", "1 d", "1 y") and c[5] == 16) or c[0].startswith("X B1")]
    out = run_store((0, "synthetic", 0, pairs, sub))
    for kk, v in out["res"].items():
        print("%-60s %d/%d" % (kk, v[0], v[1]))
    for kk, v in out["diag"].items():
        print(kk, v)
    print("time %.1fs" % (time.time() - t0))


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
        sys.exit()
    src, out = sys.argv[1], sys.argv[2]
    workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 4
    stores = B.load_stores(src)
    conds = conditions()
    # registered check: the ideal 1-bit path reproduces B22 amendment 2 on every store (key counts at 16 per vector)
    jobs = sorted([(L, t, i, p, conds) for L, t, i, p in stores], key=lambda j: -len(set(v for _, v, _ in j[3])))
    from multiprocessing import Pool
    with Pool(workers) as pool:
        per = []
        for r in pool.imap_unordered(run_store, jobs):
            per.append(r)
            print("done %-28s %d/50  B1-SC gdc 1 d 16/vec %s" % (r["store"], len(per),
                  r["res"].get("B1-SC gdc | 1 d | 16/vec")), flush=True)
    for L, t, i, p in stores:
        e, n, _, _ = B.run_store(p, bits=1, bipolar=True)
        r = [x for x in per if x["store"] == "%d/%s/%d" % (L, t, i)][0]
        assert r["res"]["B1 ideal | 16/vec"] == [e, n], (L, t, i, r["res"]["B1 ideal | 16/vec"], e, n)
    tot = []
    for name, *_ in conds:
        ke = sum(r["res"][name][0] for r in per)
        kn = sum(r["res"][name][1] for r in per)
        cells = collections.defaultdict(lambda: [0, 0])
        for r in per:
            cells[r["cell"]][0] += r["res"][name][0]
            cells[r["cell"]][1] += r["res"][name][1]
        tot.append({"cond": name, "keys_exact": ke, "keys": kn, "frac": ke / kn,
                    "stores_all_exact": sum(r["res"][name][0] == r["res"][name][1] for r in per),
                    "per_cell": dict(cells)})
        print("%-62s %5d/%-5d %.4f" % (name, ke, kn, ke / kn), flush=True)
    json.dump({"totals": tot, "per_store": per}, open(out, "w"), indent=1)
