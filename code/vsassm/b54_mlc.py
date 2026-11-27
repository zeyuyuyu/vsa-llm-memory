"""B54 (prereg/B54_mlc_cells_prereg.md): multi-level cells (2 and 3 bits per cell) for the VSA store and for the
error-corrected tables of B38; cells per stored fact for >= 99.5% exact reads at each noise multiplier, 30 days and 1
day, and the overhead of every memory relative to its own noise-free requirement.

python3 b54_mlc.py calib OUT.npz                               # per-level read distributions from B25's device model
python3 b54_mlc.py selftest                                    # synthetic pairs only
python3 b54_mlc.py dev1  STORES PASS1.json [--workers N]        # dev pass 1 (octave budgets, every design)
python3 b54_mlc.py dev2  STORES PASS1.json PASS2.json [--workers N]   # dev pass 2 (quarter steps around the octave N*)
python3 b54_mlc.py choose DEV1.json,DEV2.json CHOICE.json
python3 b54_mlc.py plan  STORES CHOICE.json PLAN.json            # test windows: dev N* +- 4 grid steps
python3 b54_mlc.py extend STORES CHOICE.json TEST1.json[,..] PLAN.json
python3 b54_mlc.py run   STORES PLAN.json OUT.json [--workers N]

Cell model. A cell is B38's differential pair of PCM devices (B25's device model: Joshi et al. 2020 programming
noise, per-device drift, 1/f read noise, 8-bit ADC over +-G_ON). With b bits per cell the differential target takes
L = 2^b equally spaced levels in [-G_ON, +G_ON] (a positive level on the + device with the - device RESET, and
vice versa; b = 1 is B38's cell). Every noise source is multiplied by k. For each (b, k, time) the read value
(G+ - G-, after the ADC) of 2^21 cells per level is simulated once (calib); cells are then read i.i.d. by inverse-CDF
sampling from these per-level distributions (one uniform per cell, shared across k and time: monotone coupling). The
reader is calibrated on reference cells (a standard MLC technique): decision thresholds between adjacent levels at the
point minimizing the pairwise error (from the per-level distributions), and per-level medians for the analog read.

Packed mode (every table, and the VSA store's sign vector): the arm's bits (bit budget b x cells) are spread over the
cells by a seeded random interleaver, b bits per cell, Gray-labelled levels; the read decides a level and returns its
Gray bits (hard), and, for E-LDPC and the packed VSA read, the a-posteriori log-likelihood ratio of every bit given the
decided level (from the calibrated confusion matrix). Analog mode (VSA only): one component of the integer bundle per
cell, uniformly quantized to L levels with the outermost level at c x the shard's rms (c chosen on dev), read as an
analog value mapped through the calibrated per-level mean read values (piecewise linear; soft, as an in-memory
crossbar uses it).
"""
import collections
import gzip
import json
import math
import os
import sys
import time

import numpy as np

import b25_pcm as P
import b38_eq as E
import ruler_vsa as V

GRID = [float(2 ** (i / 4)) for i in range(4, 49)]          # 2 ... 4096 cells per stored pair, quarter octaves
OCT = list(range(0, len(GRID), 4))                            # octave budgets (dev pass 1)
TARGET = 0.995
TIMES = collections.OrderedDict([("30d", 30 * 86400.0), ("1d", 86400.0)])
KS = (1, 2, 4, 8, 16, 32)
BITS = (2, 3)
P1D = {1: 0.0, 2: 8.2e-06, 4: 0.0097645, 8: 0.1017177, 16: 0.25640955, 32: 0.36504285}   # b34_ber.json, 1 d
NREF = 2 ** 21
ZG = np.linspace(-14.5, 14.5, 4097)                          # logit grid of the stored quantiles
VL_CLIPS = (1.0, 1.5, 2.0, 2.5, 3.0)
ARMS = ("V-last", "T-rep", "T-vhdd", "E-BCH", "E-LDPC")
CAL_PATH = os.environ.get("B54_CAL", "b54_out/b54_channel.npz")


def conditions(b):
    if b == 1:                                                # binary T-vhdd at one day (missing from B52c)
        return ["m1-k%d-1d" % k for k in KS]
    return ["m%d-ideal" % b] + ["m%d-k%d-%s" % (b, k, t) for t in TIMES for k in KS]


def parse(cond):
    p = cond.split("-")
    b = int(p[0][1:])
    if p[1] == "ideal":
        return dict(b=b, k=None, t=None, ideal=True)
    return dict(b=b, k=int(p[1][1:]), t=p[2], ideal=False)


def next_harder(cond):
    c = parse(cond)
    if c["ideal"]:
        return "m%d-k1-30d" % c["b"]
    i = KS.index(c["k"])
    return None if i == len(KS) - 1 else "m%d-k%d-%s" % (c["b"], KS[i + 1], c["t"])


def gray(x):
    return x ^ (x >> 1)


def levels(b):
    L = 2 ** b
    return (2 * np.arange(L) - (L - 1)) / (L - 1)


# =============================================================================================== calibration
def calib_one(b, k, tl):
    t = TIMES[tl]
    lv = levels(b)
    L = len(lv)
    srt = []
    for l, s in enumerate(lv):
        rng = np.random.default_rng([54, b, k, int(t), l])
        gp = np.full(NREF, max(s, 0.0) * P.G_ON, np.float32)
        gm = np.full(NREF, max(-s, 0.0) * P.G_ON, np.float32)
        nu_p, g1p = P.program(gp, gp.shape, rng, float(k))
        nu_m, g1m = P.program(gm, gm.shape, rng, float(k))
        Gp, _, _ = P.at_time(gp, nu_p, g1p, t, rng, float(k))
        Gm, _, _ = P.at_time(gm, nu_m, g1m, t, rng, float(k))
        srt.append(np.sort(P.adc(Gp - Gm, P.G_ON, P.ADC_BITS).astype(np.float64)))
    med = np.array([x[NREF // 2] for x in srt])
    step = P.G_ON / (2 ** (P.ADC_BITS - 1) - 1)
    thr = []
    for l in range(L - 1):
        lo, hi = med[l], med[l + 1]
        if hi <= lo:                                          # fully overlapped (not expected): midpoint
            thr.append(0.5 * (lo + hi))
            continue
        cand = np.arange(math.floor(lo / step), math.ceil(hi / step) + 1) * step + step / 2
        cand = cand[(cand > lo - step) & (cand < hi + step)]
        err = (1 - np.searchsorted(srt[l], cand, side="right") / NREF) + np.searchsorted(srt[l + 1], cand,
                                                                                        side="right") / NREF
        m = np.flatnonzero(err <= err.min() + 1e-12)
        thr.append(float(cand[m[len(m) // 2]]))
    thr = np.array(thr)
    C = np.zeros((L, L))
    for l in range(L):
        cdf = np.r_[0.0, np.searchsorted(srt[l], thr, side="right") / NREF, 1.0]
        C[l] = np.diff(cdf)
    u = 1 / (1 + np.exp(-ZG))
    Q = np.stack([np.quantile(x, u) for x in srt])
    g = gray(np.arange(L))
    bitsof = ((g[:, None] >> (b - 1 - np.arange(b))[None, :]) & 1)       # Gray label bits of each level (MSB first)
    berpos = np.array([(C * (bitsof[:, i][:, None] != bitsof[:, i][None, :])).sum() / L for i in range(b)])
    llr = np.zeros((L, b))
    for j in range(L):
        for i in range(b):
            p1 = (C[:, j] * (bitsof[:, i] == 1)).sum() + 1e-300
            p0 = (C[:, j] * (bitsof[:, i] == 0)).sum() + 1e-300
            llr[j, i] = np.clip(math.log(p1 / p0), -30, 30)
    return dict(Q=Q, med=med, thr=thr, C=C, ser=float(1 - np.trace(C) / L), berpos=berpos, ber=float(berpos.mean()),
                llr=llr, mean=np.array([x.mean() for x in srt]), sd=np.array([x.std() for x in srt]))


def _calib_job(a):
    b, k, tl = a
    return "m%d-k%d-%s" % (b, k, tl), calib_one(b, k, tl)


def calib(out, workers=4):
    from multiprocessing import Pool
    res = {}
    jobs = [(b, k, tl) for b in (1,) + BITS for tl in TIMES for k in KS]
    with Pool(workers) as pool:
        for key, r in pool.imap(_calib_job, jobs):
            for f, v in r.items():
                res["%s|%s" % (key, f)] = np.asarray(v)
            print("%-12s SER %.3g  BER %.3g  per position %s  medians %s" % (
                key, r["ser"], r["ber"], np.array2string(r["berpos"], precision=4),
                np.array2string(r["med"], precision=2)), flush=True)
    np.savez_compressed(out, **res)


_CAL = None


def cal(cond):
    global _CAL
    if _CAL is None:
        z = np.load(CAL_PATH)
        _CAL = {kk: z[kk] for kk in z.files}
    return {f: _CAL["%s|%s" % (cond, f)] for f in ("Q", "med", "mean", "thr", "C", "ber", "berpos", "llr")}


def ber(cond):
    c = parse(cond)
    if c["ideal"]:
        return 0.0
    if c["b"] == 1:
        return (P1D if c["t"] == "1d" else E.P30)[c["k"]]
    return float(cal(cond)["ber"])


def sample_read(level, u, cc):
    """Analog read value of cells at `level` (int array) for uniforms u (inverse CDF of the calibrated quantiles)."""
    z = np.clip(np.log(u) - np.log1p(-u), ZG[0], ZG[-1])
    pos = (z - ZG[0]) / (ZG[1] - ZG[0])
    i0 = np.minimum(pos.astype(np.int64), len(ZG) - 2)
    w = (pos - i0).astype(np.float64)
    Q = cc["Q"]
    return Q[level, i0] * (1 - w) + Q[level, i0 + 1] * w


def device_read(level, b, k, t, seed):
    """B25's device model on the actual cells (fresh devices): the read value (G+ - G-, after the ADC) per cell."""
    s = levels(b)[level]
    rng = np.random.default_rng([seed, 54, b, k, int(t)])
    gp = (np.maximum(s, 0.0) * P.G_ON).astype(np.float32)
    gm = (np.maximum(-s, 0.0) * P.G_ON).astype(np.float32)
    nu_p, g1p = P.program(gp, gp.shape, rng, float(k))
    nu_m, g1m = P.program(gm, gm.shape, rng, float(k))
    Gp, _, _ = P.at_time(gp, nu_p, g1p, t, rng, float(k))
    Gm, _, _ = P.at_time(gm, nu_m, g1m, t, rng, float(k))
    return P.adc(Gp - Gm, P.G_ON, P.ADC_BITS).astype(np.float64)


# =============================================================================================== MLC channel
class MLCChannel:
    """Per (store, arm, design, budget, b): the interleaver (bits -> cell slots) and one uniform per cell, shared by
    every k and time (monotone coupling)."""

    def __init__(self, sid, tag, B, b, nunits):
        self.sid, self.tag, self.B, self.b, self.n = sid, tag, B, b, nunits
        self._perm = self._u = None

    def u(self, ncell):
        if self._u is None:
            self._u = np.random.default_rng(E.seed_of("b54", self.sid, self.tag, "%.4f" % self.B, self.b, "u")).random(
                ncell)
            self._u = np.clip(self._u, 1e-12, 1 - 1e-12)
        return self._u

    device_seed = None                                         # b54_check device: B25's device model itself

    def _read(self, level, cond, cc, ncell):
        if self.device_seed is None:
            return sample_read(level, self.u(ncell), cc)
        c = parse(cond)
        return device_read(level, c["b"], c["k"], TIMES[c["t"]], self.device_seed)

    def packed(self, bits_pm, cond):
        """bits_pm: int8 +-1 (natural order, + = bit 1). Returns (hard int8 +-1, llr float32 for bit = 1)."""
        b = self.b
        nb = len(bits_pm)
        ncell = -(-nb // b)
        if self._perm is None:
            self._perm = np.random.default_rng(E.seed_of("b54", self.sid, self.tag, "%.4f" % self.B, b,
                                                         "perm")).permutation(ncell * b)
        slot = self._perm[:nb]
        lab = np.zeros(ncell * b, np.int64)
        lab[slot] = bits_pm > 0
        lab = lab.reshape(ncell, b)
        g = (lab << (b - 1 - np.arange(b))[None, :]).sum(1)
        inv = np.argsort(gray(np.arange(2 ** b)))               # Gray label -> level
        level = inv[g]
        c = parse(cond)
        if c["ideal"]:
            dec = level
            llrt = np.where(((gray(np.arange(2 ** b))[:, None] >> (b - 1 - np.arange(b))[None, :]) & 1) > 0, 30.0,
                            -30.0)
        else:
            cc = cal(cond)
            r = self._read(level, cond, cc, ncell)
            dec = np.searchsorted(cc["thr"], r, side="left")
            llrt = cc["llr"]
        gl = gray(dec)
        rb = ((gl[:, None] >> (b - 1 - np.arange(b))[None, :]) & 1).reshape(-1)
        hard = (rb[slot] * 2 - 1).astype(np.int8)
        llr = llrt[dec].reshape(-1)[slot].astype(np.float32)
        return hard, llr

    def analog(self, level, cond, L):
        """level: int array (natural order), one component per cell. Returns the read in level units, centered."""
        c = parse(cond)
        if c["ideal"]:
            return (level - (L - 1) / 2).astype(np.float32)
        cc = cal(cond)
        r = self._read(level, cond, cc, len(level))
        med = cc["mean"].astype(np.float64)                  # calibrated per-level mean read values
        med = med + 1e-6 * np.arange(L)                      # strictly increasing (levels that drift together)
        x = np.interp(r, med, np.arange(L, dtype=np.float64))
        x = np.where(r < med[0], (r - med[0]) / (med[1] - med[0]), x)
        x = np.where(r > med[-1], L - 1 + (r - med[-1]) / (med[-1] - med[-2]), x)
        return (x - (L - 1) / 2).astype(np.float32)


# =============================================================================================== arms
def bits_budget(S38, B, b):
    """Bit budget B' (per pair, for b38_eq's builders) with int(B' n) = b int(B n) exactly."""
    return (b * int(B * S38.n) + 0.5) / S38.n


def vsa_analog_build(S38, B, b, clip):
    n = S38.n
    total = int(B * n)
    S = max(1, math.ceil(total / E.D))
    d = min(E.D, total // S)
    if d < 16:
        return None
    shard_k = np.array([V._seed("shard:" + k) % S for k in S38.s.cd["allk"]])
    ps = shard_k[S38.pk]
    prod = (S38.Kc[S38.pk, :d] * S38.Pc[S38.pj, :d] * S38.Vc[S38.pv, :d]).astype(np.int32)
    prod = np.where(S38.plast[:, None] > 0, prod * S38.L[None, :d], prod)
    mem = np.zeros((S, d), np.int32)
    order = np.argsort(ps, kind="stable")
    sp_ = ps[order]
    starts = np.flatnonzero(np.r_[True, sp_[1:] != sp_[:-1]])
    mem[sp_[starts]] = np.add.reduceat(prod[order], starts, axis=0)
    rms = np.sqrt(np.mean(mem.astype(np.float64) ** 2, axis=1))
    L = 2 ** b
    delta = np.maximum(2 * clip * rms / (L - 1), 1e-9)[:, None]
    m = mem.astype(np.float64)
    zero = mem == 0                                           # exact zeros: a random half-step (B38's random ties)
    m = m + np.where(zero, np.random.default_rng(0).choice([-0.5, 0.5], size=mem.shape), 0.0) * delta
    level = np.clip(np.floor(m / delta) + L // 2, 0, L - 1).astype(np.int64)
    return dict(level=level.ravel(), S=S, d=d, shard_k=shard_k, rms0=rms.astype(np.float32), kind="last", L=L,
                cells=S * d)


def ldpc_decode_llr(llr_sum, lay, keep, rng):
    from ldpc import BpDecoder
    code = E.ira_code(lay["K"], lay["R"])
    full = np.zeros(lay["ncw"] * lay["N"], np.float64)
    full[keep] = llr_sum
    full[~keep] = -60.0
    hard = (full > 0).astype(np.uint8)
    ties = full == 0
    hard[ties] = rng.integers(0, 2, int(ties.sum()))
    pr = np.clip(1.0 / (1.0 + np.exp(np.minimum(np.abs(full), 60.0))), 1e-15, 0.5)
    if "bp" not in code["dec"]:
        code["dec"]["bp"] = BpDecoder(code["H"], error_rate=0.1, max_iter=E.LDPC_ITERS, bp_method="product_sum",
                                      input_vector_type="received_vector")
    bpd = code["dec"]["bp"]
    out, nfail = [], 0
    for w in range(lay["ncw"]):
        sl = slice(w * lay["N"], (w + 1) * lay["N"])
        bpd.update_channel_probs(pr[sl])
        d = bpd.decode(hard[sl])
        nfail += int(not bpd.converge)
        out.append(np.asarray(d[:lay["K"]], np.uint8))
    return np.concatenate(out), nfail


def build(S38, arm, B, b, design):
    """Returns (A, nbits_or_cells, mode) or None. design: V-last 'packed' or clip c; T-vhdd share; E-LDPC rate;
    E-BCH (n, k, t)."""
    if arm == "V-last" and design != "packed":
        A = vsa_analog_build(S38, B, b, float(design))
        return None if A is None else (A, A["cells"], "analog")
    Bb = bits_budget(S38, B, b) if b > 1 else B
    A = E.build_arm(S38, arm, Bb, None if arm in ("V-last", "T-rep") else design)
    if A is None:
        return None
    return A, len(A["cells"]), "packed"


def read(S38, arm, A, mode, ch, cond, rng):
    b = parse(cond)["b"]
    if mode == "analog":
        x = ch.analog(A["level"], cond, A["L"])
        rd = E.vsa_read(S38, A, x)
        return E.vsa_rule(rd, "last"), {}
    if b == 1:                                                # binary (T-vhdd at one day): B38's i.i.d. flips
        c = parse(cond)
        eps = ber(cond)
        hard = np.where(ch.u(len(A["cells"])) < eps, -A["cells"], A["cells"]).astype(np.int8)
        llr = hard * (math.log((1 - eps) / eps) if eps > 0 else 30.0)
    else:
        hard, llr = ch.packed(A["cells"], cond)
    if arm == "V-last":
        rd = E.vsa_read(S38, A["bd"], np.tanh(llr / 2))
        return E.vsa_rule(rd, "last"), {}
    if arm == "E-LDPC":
        lay = A["lay"]
        starts = np.r_[0, np.cumsum(lay["counts"])[:-1]]
        bits, nf = ldpc_decode_llr(np.add.reduceat(llr.astype(np.float64), starts), lay, A["keep"], rng)
        return E.mphf_read(S38, A["cont"], bits[:len(A["cont"]["bits"])]), {"cw_nonconv": nf}
    res, diag = E.read_arm(S38, arm, A, hard, "k1", rng)      # T-rep, T-vhdd, E-BCH: hard bits; cond unused
    return res[""], diag


def designs_of(arm):
    return {"V-last": ["packed"] + list(VL_CLIPS), "T-rep": ["rep"], "T-vhdd": list(E.VHDD_SHARES),
            "E-LDPC": list(E.RATES), "E-BCH": ["auto"]}[arm]


# =============================================================================================== jobs
def run_job(job):
    """job: dict(sid, pairs, arm, b, evals=[(B, design, [conds])]); E-BCH 'auto' resolved per (B, cond)."""
    t0 = time.time()
    S38 = E.Store38(job["sid"], job["pairs"])
    nq = len(S38.want)
    arm, b = job["arm"], job["b"]
    evals = []
    if arm == "E-BCH":
        cont = E.mphf_content(S38)
        needs = E.mphf_needs(S38, cont, list(S38.s.count))
        grp = collections.OrderedDict()
        for B, d, conds in job["evals"]:
            for c in conds:
                nh = next_harder(c)
                bits = int(bits_budget(S38, B, b) * S38.n) if b > 1 else int(B * S38.n)
                des = E.bch_choose(len(cont["bits"]), needs, bits, ber(c), ber(nh) if nh else None)
                grp.setdefault((B, json.dumps(None if des is None else list(des))), []).append(c)
        evals = [(B, json.loads(d), cs) for (B, d), cs in grp.items()]
    else:
        evals = job["evals"]
    out = []
    for B, design, conds in evals:
        A = None if design is None else build(S38, arm, B, b, design)
        if A is None:
            for c in conds:
                out.append(dict(B=B, design=design, cond=c, exact=0, n=nq, failed=None, infeasible=True))
            continue
        A, nunits, mode = A
        tag = "%s|%s|%s" % (arm, json.dumps(design), mode)
        ch = MLCChannel(job["sid"], tag, B, b, nunits)
        cells = nunits if (mode == "analog" or b == 1) else -(-nunits // b)
        for c in conds:
            rng = np.random.default_rng(E.seed_of("b54", job["sid"], tag, "%.4f" % B, c, "dec"))
            got, diag = read(S38, arm, A, mode, ch, c, rng)
            f = E.exact(got, S38.want)
            out.append(dict(B=B, design=design, cond=c, exact=nq - len(f), n=nq, failed=f, cells=int(cells),
                            budget=int(B * S38.n), diag=diag))
    return dict(sid=job["sid"], arm=arm, b=b, n_pairs=S38.n, nq=nq, res=out, sec=time.time() - t0)


def load_stores(path):
    return E.load_stores(path)


def pool_run(jobs, workers, out_path):
    from multiprocessing import Pool
    done = []
    last = time.time()
    with Pool(workers) as pool:
        for r in pool.imap_unordered(run_job, jobs):
            done.append(r)
            print("done %-28s %-7s m%d %5.0fs  %d/%d" % (r["sid"], r["arm"], r["b"], r["sec"], len(done), len(jobs)),
                  flush=True)
            if time.time() - last > 600:
                json.dump(done, open(out_path + ".partial", "w"))
                last = time.time()
    json.dump(done, open(out_path, "w"))
    return done


def mkjobs(stores, need, chunk=4):
    """need: {(arm, b): {(gi, design_json): set(conds)}} -> jobs (chunked evaluations, biggest stores first)."""
    jobs = []
    for sid, pairs in stores:
        for (arm, b), ev in need.items():
            items = [(GRID[gi], json.loads(d), sorted(cs)) for (gi, d), cs in sorted(ev.items())]
            ck = 1 if arm == "E-BCH" else chunk                  # decoding of failing BCH words dominates
            for i in range(0, len(items), ck):
                jobs.append(dict(sid=sid, pairs=pairs, arm=arm, b=b, evals=items[i:i + ck]))
    jobs.sort(key=lambda j: (-len(j["pairs"]), j["arm"] != "E-BCH"))
    return jobs


# =============================================================================================== aggregation / choice
def gidx(B):
    return min(range(len(GRID)), key=lambda i: abs(GRID[i] - B))


def aggregate(results):
    """acc[(arm, b)][design_key][cond][gi] = [exact, n] summed over stores (E-BCH: design key 'auto')."""
    acc = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(
        lambda: collections.defaultdict(lambda: [0, 0]))))
    for r in results:
        for x in r["res"]:
            dk = "auto" if r["arm"] == "E-BCH" else json.dumps(x["design"])
            a = acc[(r["arm"], r["b"])][dk][x["cond"]][gidx(x["B"])]
            a[0] += x["exact"]
            a[1] += x["n"]
    return acc


def nstar(curve):
    """Smallest evaluated gi with share >= TARGET there and at every larger evaluated gi; None if none."""
    ok = None
    for gi in sorted(curve, reverse=True):
        if curve[gi] >= TARGET:
            ok = gi
        else:
            break
    return ok


def best_curve(acc_v, cond):
    """Per gi, the best design (highest share; ties: share at the next harder condition, then design order)."""
    curve, ch = {}, {}
    nh = next_harder(cond)
    gis = set()
    for dk in acc_v:
        gis |= set(acc_v[dk][cond])
    for gi in gis:
        cands = []
        for o, dk in enumerate(acc_v):
            a = acc_v[dk][cond].get(gi)
            if a is None or a[1] == 0:
                continue
            a2 = acc_v[dk][nh].get(gi) if nh else None
            cands.append((-a[0] / a[1], -(a2[0] / a2[1] if a2 and a2[1] else 0.0), o, dk, a[0] / a[1]))
        if cands:
            cands.sort()
            ch[gi], curve[gi] = cands[0][3], cands[0][4]
    return curve, ch


def all_variants():
    v = [(a, b) for b in BITS for a in ARMS] + [("T-vhdd", 1)]
    only = os.environ.get("B54_ARMS")
    return [x for x in v if x[0] in only.split(",")] if only else v


def dev1_need():
    need = {}
    for arm, b in all_variants():
        need[(arm, b)] = {(gi, json.dumps(d)): set(conditions(b)) for gi in OCT for d in designs_of(arm)}
    return need


def dev2_need(dev1):
    acc = aggregate(json.load(open(dev1)))
    need = {}
    for arm, b in all_variants():
        ev = collections.defaultdict(set)
        for c in conditions(b):
            curve, _ = best_curve(acc[(arm, b)], c)
            ns = nstar(curve)
            gs = range(len(GRID) - 4, len(GRID)) if ns is None else range(max(0, ns - 3), min(len(GRID), ns + 4))
            for gi in gs:
                if gi in OCT:
                    continue
                for d in designs_of(arm):
                    ev[(gi, json.dumps(d))].add(c)
        need[(arm, b)] = ev
    return need


def choose(dev_files, out):
    res = []
    for f in dev_files.split(","):
        res += json.load(open(f))
    acc = aggregate(res)
    ch = {"designs": {}, "dev_nstar": {}, "dev_curve": {}}
    for arm, b in all_variants():
        v = "%s|m%d" % (arm, b)
        ch["designs"][v], ch["dev_nstar"][v], ch["dev_curve"][v] = {}, {}, {}
        for c in conditions(b):
            curve, des = best_curve(acc[(arm, b)], c)
            ns = nstar(curve)
            ch["designs"][v][c] = {str(gi): json.loads(d) if d != "auto" else "auto" for gi, d in des.items()}
            ch["dev_curve"][v][c] = {str(gi): round(s, 6) for gi, s in sorted(curve.items())}
            ch["dev_nstar"][v][c] = None if ns is None else GRID[ns]
    json.dump(ch, open(out, "w"), indent=1)
    for b in BITS + (1,):
        cs = conditions(b)
        print("dev N* (m%d):" % b, " ".join("%10s" % c[3:] for c in cs))
        for arm in ARMS:
            v = "%s|m%d" % (arm, b)
            if v in ch["dev_nstar"]:
                print("  %-7s" % arm, " ".join("%10s" % ("%.0f" % x if x else "-") for x in
                                              (ch["dev_nstar"][v][c] for c in cs)))


def design_at(ch, v, c, gi):
    d = ch["designs"][v][c]
    if not d:
        return designs_of(v.split("|")[0])[0]
    near = min(d, key=lambda g: (abs(int(g) - gi), -int(g)))
    return d[near]


def plan(stores_f, choice_f, out):
    ch = json.load(open(choice_f))
    need = {}
    for arm, b in all_variants():
        v = "%s|m%d" % (arm, b)
        ev = collections.defaultdict(set)
        for c in conditions(b):
            ns = ch["dev_nstar"][v][c]
            gs = range(len(GRID) - 5, len(GRID)) if ns is None else range(max(0, gidx(ns) - 4),
                                                                         min(len(GRID), gidx(ns) + 5))
            for gi in gs:
                ev[(gi, json.dumps(design_at(ch, v, c, gi)))].add(c)
        need[(arm, b)] = ev
    json.dump({"need": [[a, b, [[gi, d, sorted(cs)] for (gi, d), cs in ev.items()]] for (a, b), ev in need.items()]},
              open(out, "w"))
    print("plan:", sum(len(cs) for ev in need.values() for cs in ev.values()), "(budget, condition) cells per store")


def extend(stores_f, choice_f, test_fs, out):
    ch = json.load(open(choice_f))
    res = []
    for f in test_fs.split(","):
        res += json.load(open(f))
    acc = aggregate(res)
    need, nadd = {}, 0
    for arm, b in all_variants():
        v = "%s|m%d" % (arm, b)
        ev = collections.defaultdict(set)
        for c in conditions(b):
            curve = {}
            for dk in acc[(arm, b)]:
                for gi, a in acc[(arm, b)][dk][c].items():
                    curve[gi] = a[0] / a[1]
            if not curve:
                continue
            top, bot = max(curve), min(curve)
            ns = nstar(curve)
            add = []
            if curve[top] < TARGET and top < len(GRID) - 1:
                add = range(top + 1, min(len(GRID), top + 5))
            elif ns is not None and ns == bot and bot > 0:
                add = range(max(0, bot - 4), bot)
            for gi in add:
                ev[(gi, json.dumps(design_at(ch, v, c, gi)))].add(c)
                nadd += 1
        need[(arm, b)] = ev
    json.dump({"need": [[a, b, [[gi, d, sorted(cs)] for (gi, d), cs in ev.items()]] for (a, b), ev in need.items()]},
              open(out, "w"))
    print("extension:", nadd, "(budget, condition) cells per store")
    return nadd


def need_of_plan(pf):
    p = json.load(open(pf))
    return {(a, b): {(gi, d): set(cs) for gi, d, cs in ev} for a, b, ev in p["need"]}


# =============================================================================================== self-test
def selftest():
    rng = np.random.default_rng(1)
    pairs, seen = [], collections.defaultdict(int)
    for i in range(1500):
        k = "key%d" % rng.integers(0, 1200)
        pairs.append((k, "val%d" % i, seen[k]))
        seen[k] += 1
    S38 = E.Store38("selftest/0", pairs)
    nq = len(S38.want)
    # Gray / interleaver round trip on ideal cells, every arm, b = 2, 3
    for b in BITS:
        for arm in ARMS:
            for design in designs_of(arm)[:2]:
                if arm == "E-BCH":
                    design = (255, 215, 5)
                B = 128.0 if arm == "V-last" else 64.0
                Ab = build(S38, arm, B, b, design)
                A, nu, mode = Ab
                ch = MLCChannel("selftest/0", arm, B, b, nu)
                got, _ = read(S38, arm, A, mode, ch, "m%d-ideal" % b, np.random.default_rng(0))
                ok = nq - len(E.exact(got, S38.want))
                cells = nu if mode == "analog" else -(-nu // b)
                print("m%d %-7s %-8s ideal: %d/%d exact, cells %d <= budget %d" % (b, arm, design, ok, nq, cells,
                                                                                  int(B * S38.n)))
                assert cells <= int(B * S38.n)
                if mode == "packed":
                    hard, llr = ch.packed(A["cells"], "m%d-ideal" % b)
                    assert (hard == A["cells"]).all() and ((llr > 0) == (A["cells"] > 0)).all()
    if os.path.exists(CAL_PATH):
        for b in BITS:
            A, nu, mode = build(S38, "T-rep", 64.0, b, None)
            for c in conditions(b)[1:]:
                ch = MLCChannel("selftest/0", "T-rep", 64.0, b, nu)
                hard, _ = ch.packed(A["cells"], c)
                print("%-12s simulated bit-error rate %.5f  calibrated %.5f" % (c, (hard != A["cells"]).mean(), ber(c)))


if __name__ == "__main__":
    mode = sys.argv[1]
    W = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 4
    if mode == "calib":
        calib(sys.argv[2], W)
    elif mode == "selftest":
        selftest()
    elif mode == "dev1":
        pool_run(mkjobs(load_stores(sys.argv[2]), dev1_need()), W, sys.argv[3])
    elif mode == "dev2":
        pool_run(mkjobs(load_stores(sys.argv[2]), dev2_need(sys.argv[3])), W, sys.argv[4])
    elif mode == "choose":
        choose(sys.argv[2], sys.argv[3])
    elif mode == "plan":
        plan(sys.argv[2], sys.argv[3], sys.argv[4])
    elif mode == "extend":
        extend(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
    elif mode == "run":
        pool_run(mkjobs(load_stores(sys.argv[2]), need_of_plan(sys.argv[3])), W, sys.argv[4])
