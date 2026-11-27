"""B69 (prereg/B69_vocab_sweep_prereg.md): the equal-cell device comparison under both vocabulary accountings, and a
sweep of the vocabulary size N.

(A) Both accountings on the registered runs (B38 / B59 / B61 / B63 rows, per-store failed-key counts): the dense store
    U, the sparse block code store H-SBC, the best static coded table (min of T-rep, E-BCH, E-LDPC, E-CAM) and the best
    insertable table (min of O-BCH, O-LDPC), cells per stored pair at each noise level, with the value vocabulary
    (index -> value string) uncharged (free, shared, as registered) and charged (its strings on the same cells,
    protected by a channel code: the BSC capacity bound, and the tables' own repetition / BCH codes by B38's closed
    form), pooled over the 50 stores and per family.
(B) Vocabulary sweep: synthetic stores of 6,002 pairs (B34's mk2big2 generator: adjective-noun keys, 7-digit values,
    two keys with two values) with a vocabulary of N in NS distinct 7-digit values (every one a clean-up candidate and
    an index of ceil(log2 N) bits); cells-per-pair search (dev then test, B38 / B61 / B63's code unchanged) for U,
    H-SBC, T-rep, E-BCH, E-LDPC, O-BCH, O-LDPC at CONDS; both accountings.

python3 b69_vocab.py theory OUT.json                     # predictions from closed-form models (no store read)
python3 b69_vocab.py selftest                            # synthetic small-N stores only
python3 b69_vocab.py stores                              # writes b69_out/sweep_{dev,test}.json.gz
python3 b69_vocab.py devplan OUT_PLAN.json
python3 b69_vocab.py run STORES.json.gz PLAN.json OUT.json [--workers N]
python3 b69_vocab.py choose DEV.json[,DEV2.json] CHOICE.json
python3 b69_vocab.py devext CHOICE.json DEVPLAN.json OUT_PLAN.json      # one dev extension round
python3 b69_vocab.py testplan CHOICE.json THEORY.json PLAN.json
python3 b69_vocab.py extend CHOICE.json PLAN.json[,..] TEST.json[,..] OUT_PLAN.json
python3 b69_vocab.py partA OUT_PREFIX
python3 b69_vocab.py verdictB CHOICE.json THEORY.json TEST.json[,..] OUT_PREFIX
"""
import collections
import gzip
import json
import math
import os
import random
import sys
import time

import numpy as np
from scipy import stats

import b25_pcm as P
import b38_eq as E

GRID = E.GRID
NG = len(GRID)
TARGET = E.TARGET
NS = (256, 1024, 6000, 24000)
CONDS = ("k1", "k4", "k8", "k16")            # k4 secondary (cut first)
PRIMARY = ("k1", "k8", "k16")
NEXT = {"k1": "k4", "k4": "k8", "k8": "k16", "k16": None}
NKEYS_SWEEP, NMULTI = 6000, 2                  # 6,002 pairs per sweep store
NPROBE = 300
DEV_REPS, TEST_REPS = 3, 8
LAMS = (0.35, 0.5, 0.7, 1.0)
FS = (12, 14, 16)
RATESETS = {"k1": (5 / 6, 0.9), "k4": (2 / 3, 0.75, 5 / 6), "k8": (0.25, 1 / 3, 0.5), "k16": (0.25, 1 / 3, 0.5)}
MEMS = ("U", "H-SBC", "T-rep", "E-BCH", "E-LDPC", "O-BCH", "O-LDPC")
DEVFREE = ("U", "T-rep", "E-BCH")
STATIC = ("T-rep", "E-BCH", "E-LDPC")
INSERT = ("O-BCH", "O-LDPC")
VOCAB_PFAIL = 0.01                              # vocabulary charged: P(any of the store's strings wrong) <= 1%
# the registered workload N* (B38 / B59 / B61 / B63 test rows; reproduced by partA from the per-store counts)
WORK = {"U": {"k1": 76, "k4": 76, "k8": 128, "k16": 431},
        "H-SBC": {"k1": 32, "k4": 45, "k8": 91, "k16": 304},
        "T-rep": {"k1": 19, "k4": 91, "k8": 256, "k16": 1024},
        "E-BCH": {"k1": 19, "k4": 27, "k8": 108, "k16": 431},
        "E-LDPC": {"k1": 23, "k4": 27, "k8": 54, "k16": 215},
        "O-BCH": {"k1": 32, "k4": 45, "k8": 181, "k16": 724},
        "O-LDPC": {"k1": 32, "k4": 38, "k8": 91, "k16": 304}}


def gidx(B):
    return min(range(NG), key=lambda i: abs(GRID[i] - B))


def gceil(x):
    """Index of the smallest grid budget >= x (NG if above the grid)."""
    for i, g in enumerate(GRID):
        if g >= x - 1e-9:
            return i
    return NG


def fmt(gi):
    return ">4096" if gi is None or gi >= NG else "%.0f" % GRID[gi]


def label(d):
    return "store fewer" if d >= 2 else "table fewer" if d <= -2 else "tie"


def h_label(ds):
    s = any(d >= 2 for d in ds)
    t = any(d <= -2 for d in ds)
    return "MIXED" if s and t else "VSA WIN" if s else "TABLE WIN" if t else "TIE"


# =============================================================================================== vocabulary patch
_EXTRA = None
_orig_codes = P.codes


def _codes(pairs, d=P.D):
    """b25_pcm.codes with the sweep store's unused vocabulary values appended (after the stored values, in vocabulary
    order), so every one of the N values is a clean-up candidate and the tables index N values."""
    cd = _orig_codes(pairs, d)
    if _EXTRA:
        have = set(cd["vocab"])
        add = [v for v in _EXTRA if v not in have]
        if add:
            cd["vocab"] = cd["vocab"] + add
            cd["Vc"] = P.bip(cd["vocab"], "value", d)
    return cd


P.codes = _codes


def install(extra, nprobe):
    global _EXTRA
    _EXTRA = list(extra) if extra else None
    P.NKEYS = nprobe


# =============================================================================================== sweep stores
_WORDS = None


def words():
    global _WORDS
    if _WORDS is None:
        try:
            import wonderwords
            nouns = sorted(set(wonderwords.random_word._get_words_from_text_file("nounlist.txt")))
            adjs = sorted(set(wonderwords.random_word._get_words_from_text_file("adjectivelist.txt")))
        except ImportError:                                   # the box: the same sorted lists, exported locally
            adjs, nouns = json.load(gzip.open("b69_out/words.json.gz", "rt"))
        _WORDS = (adjs, nouns)
    return _WORDS


def make_store(N, split, rep, n_keys=NKEYS_SWEEP, nmulti=NMULTI):
    """B34's dev generator (kind 'word', as mk2big2) with an explicit vocabulary of N distinct 7-digit values.
    N >= n pairs: the stored values are a sample without replacement; N < n: every value used floor or ceil(n / N)
    times, shuffled. Returns (sid, pairs, extra) with extra = the unused vocabulary values in vocabulary order."""
    adjs, nouns = words()
    rng = random.Random("b69:%s:%d:%d:%d" % (split, N, rep, n_keys))
    keys, seen = [], set()
    while len(keys) < n_keys:
        k = "%s-%s" % (rng.choice(adjs), rng.choice(nouns))
        if k not in seen:
            seen.add(k)
            keys.append(k)
    counts = [1] * n_keys
    for i in range(nmulti):
        counts[(i + 1) * 7] = 2
    n = sum(counts)
    vocab = [str(x) for x in rng.sample(range(10 ** 6, 10 ** 7), N)]
    if N >= n:
        vals = rng.sample(vocab, n)
    else:
        vals = (vocab * (-(-n // N)))[:n]
        rng.shuffle(vals)
    pairs, it = [], iter(vals)
    for k, c in zip(keys, counts):
        for j in range(c):
            pairs.append((k.lower(), next(it), j))
    used = set(vals)
    extra = [v for v in vocab if v not in used]
    return ("b69/N%d/%s/%d" % (N, split, rep), pairs, extra)


def write_stores(out_dir="b69_out", ns=NS, reps=None):
    os.makedirs(out_dir, exist_ok=True)
    reps = reps or {"dev": DEV_REPS, "test": TEST_REPS}
    for split, R in reps.items():
        st = [make_store(N, split, r) for N in ns for r in range(R)]
        json.dump([[s, p, x] for s, p, x in st], gzip.open("%s/sweep_%s.json.gz" % (out_dir, split), "wt"))
        print(split, len(st), "stores", flush=True)


def load_sweep(path):
    return [(s, [tuple(q) for q in p], x) for s, p, x in json.load(gzip.open(path, "rt"))]


def n_of(sid):
    return int(sid.split("/")[1][1:])


# =============================================================================================== theory (closed forms)
def u_fail(B, eps, M):
    """Dense sign store (V-last) slot failure: true similarity ~ N(mu, 1), M wrong ~ N(0, 1) i.i.d.,
    mu = sqrt(2 B / pi) (1 - 2 eps) (sign of a bundle of d / B pairs per d cells, read through flips)."""
    mu = math.sqrt(2 * B / math.pi) * (1 - 2 * eps)
    x = np.linspace(mu - 10, mu + 10, 4001)
    w = stats.norm.pdf(x - mu) * np.exp(M * stats.norm.logcdf(x))
    return 1 - float(w.sum() * (x[1] - x[0]))


def sbc_fail(B, eps, lam, M):
    """Sparse block store slot failure (B61's model): nb = lam B blocks; true overlap ~ Bin(nb, 1 - eps); M wrong
    ~ Bin(nb, rho') i.i.d., rho' = rho (1 - eps) + (1 - rho) eps, rho = 1 - exp(-lam); uniform tie break."""
    nb = max(1, int(round(lam * B)))
    rho = 1 - math.exp(-lam)
    rp = rho * (1 - eps) + (1 - rho) * eps
    t = np.arange(nb + 1)
    pt = stats.binom.pmf(t, nb, 1 - eps)
    lt = stats.binom.logcdf(t - 1, nb, rp)          # P(W < t)
    le = stats.binom.logcdf(t, nb, rp)              # P(W <= t)
    pe = stats.binom.pmf(t, nb, rp)
    # P(win | t) = [P(W<=t)^(M+1) - P(W<t)^(M+1)] / ((M+1) P(W=t)), in log space (no cancellation)
    with np.errstate(divide="ignore", invalid="ignore"):
        x = np.where(pe > 0, pe / np.exp(le), 0.0)
        win = np.where(pe > 0, np.exp(M * le) * (-np.expm1((M + 1) * np.log1p(-np.minimum(x, 1 - 1e-300))))
                       / np.where(x > 0, (M + 1) * x, 1.0), np.exp(M * lt))
    return 1 - float(np.sum(pt * np.clip(win, 0, 1)))


def model_nstar(fail, target=0.005):
    for i in range(NG):
        if fail(GRID[i]) <= target:
            return i
    return NG


def theory(out_path):
    """Model N* per (memory, N, condition). Stores: closed forms above, shifted by their step change from N = 6,000
    and anchored at the workload's measured N* (WORK); tables: exact logical bits per pair of a sweep store at each N
    (built from the synthetic generator), the workload's cells per logical bit kept."""
    import b34_inmem as I34
    import b63_online as O
    out = dict(model={}, pred={}, logical={})
    for N in NS:
        sid, pairs, extra = make_store(N, "theory", 0)
        install(extra, NPROBE)
        S38 = E.Store38(sid, pairs)
        mc = E.mphf_content(S38)
        ck = O.Cuckoo(S38, 12)
        out["logical"][str(N)] = dict(mphf=len(mc["bits"]) / S38.n, cuckoo12=len(ck.bits(S38)) / S38.n,
                                      nv=S38.Nv, b=S38.s.b, n=S38.n)
    install(None, P.NKEYS)
    L6 = out["logical"]["6000"]
    for c in CONDS:
        eps = E.IID[c]
        for N in NS:
            M = 2 * N - 1
            mu = model_nstar(lambda B: u_fail(B, eps, M))
            ms = min(model_nstar(lambda B: sbc_fail(B, eps, lam, M)) for lam in LAMS)
            out["model"]["U|%d|%s" % (N, c)] = fmt(mu)
            out["model"]["H-SBC|%d|%s" % (N, c)] = fmt(ms)
        mu6 = gidx(float(out["model"]["U|6000|%s" % c]))
        ms6 = gidx(float(out["model"]["H-SBC|6000|%s" % c]))
        for N in NS:
            LN = out["logical"][str(N)]
            pr = {}
            pr["U"] = gidx(WORK["U"][c]) + gidx(float(out["model"]["U|%d|%s" % (N, c)])) - mu6
            pr["H-SBC"] = gidx(WORK["H-SBC"][c]) + gidx(float(out["model"]["H-SBC|%d|%s" % (N, c)])) - ms6
            for m in STATIC:
                if c == "k1" and m == "T-rep":
                    pr[m] = gceil(LN["mphf"])                      # no cell errors: one cell per logical bit
                else:
                    pr[m] = gceil(GRID[gidx(WORK[m][c])] * LN["mphf"] / L6["mphf"] - 1e-6)
            for m in INSERT:
                pr[m] = gceil(GRID[gidx(WORK[m][c])] * LN["cuckoo12"] / L6["cuckoo12"] - 1e-6)
            for m, gi in pr.items():
                out["pred"]["%s|%d|%s" % (m, N, c)] = int(min(max(gi, 0), NG - 1))
    json.dump(out, open(out_path, "w"), indent=1)
    return out


# =============================================================================================== vocabulary bits and cost
def vocab_bits(vocab, compact=False):
    """Bits to hold the vocabulary strings: UTF-8 bytes x 8 in the cheaper of a fixed width (the longest string, a
    design constant) and terminated strings (one 0 byte each). compact: decimal strings of one length L as integers
    of ceil(log2(9 10^(L-1))) bits (leading digit non-zero) or ceil(L log2 10) bits, version-4 UUIDs as 122 bits,
    anything else as UTF-8."""
    enc = [v.encode("utf-8") for v in vocab]
    utf = 8 * min(len(enc) * max(len(e) for e in enc), sum(len(e) + 1 for e in enc))
    if not compact:
        return utf
    if all(v.isdigit() for v in vocab) and len(set(len(v) for v in vocab)) == 1:
        L = len(vocab[0])
        lead0 = any(v[0] == "0" for v in vocab)
        per = math.ceil(L * math.log2(10)) if lead0 else math.ceil(math.log2(9 * 10 ** (L - 1)))
        return len(vocab) * per
    import uuid as U
    try:
        if all(U.UUID(v).version == 4 and str(U.UUID(v)) == v for v in vocab):
            return len(vocab) * 122
    except ValueError:
        pass
    return utf


def h2(p):
    return 0.0 if p <= 0 else float(-p * math.log2(p) - (1 - p) * math.log2(1 - p))


def cap_cells(L, eps):
    return int(math.ceil(L / (1 - h2(eps))))


_BD = None


def bch_designs():
    global _BD
    if _BD is None:
        _BD = E.bch_designs()
    return _BD


_VC = {}


def bch_cells(L, eps, pfail=VOCAB_PFAIL):
    """Fewest cells holding L bits with P(any codeword fails) <= pfail under i.i.d. flips at eps, over repetition
    (majority of r or r + 1 copies, ties half wrong) and B38's BCH designs (each coded bit r or r + 1 copies, spread
    evenly; bounded-distance decoding; B38's closed form bch_cw_logfail; every codeword counted at full length n, an
    upper bound for the shortened last one). Returns (cells, design)."""
    key = (L, eps, pfail)
    if key in _VC:
        return _VC[key]
    if eps <= 0:
        _VC[key] = (L, ("uncoded",))
        return _VC[key]
    best = None
    lpf = math.log(pfail)

    def rep_ok(r, n1):
        w1, t1 = E.q_rep(r, eps)
        w2, t2 = E.q_rep(r + 1, eps)
        pa, pb = w1 + t1 / 2, w2 + t2 / 2
        # P(any bit wrong) <= sum
        tot = (L - n1) * pa + n1 * pb
        return tot <= pfail

    for r in range(1, 400):
        if rep_ok(r, 0):
            lo, hi = 0, L                                         # n1 bits with r copies replaced: search r-1 + extra
            if r > 1:
                lo_ok = None
                a, b = 0, L
                while a < b:                                      # fewest bits with r copies on top of r - 1 copies
                    mid = (a + b) // 2
                    if rep_ok(r - 1, mid):
                        b = mid
                    else:
                        a = mid + 1
                cells = (r - 1) * L + a
            else:
                cells = L
            best = (cells, ("rep", r))
            break
    for n, k, t in bch_designs():
        ncw = -(-L // k)
        Nc = L + ncw * (n - k)
        if best is not None and Nc >= best[0]:
            continue

        def ok(r, n1c):                                           # n1c = coded bits per codeword with r + 1 copies
            lf = E.bch_cw_logfail(n, t, r, n1c, eps)
            return math.log(ncw) + lf <= lpf
        r = 1
        while r < 400 and not ok(r, 0):
            r += 1
            if best is not None and Nc * (r - 1) >= best[0]:
                break
        if r >= 400 or not ok(r, 0):
            continue
        cells = Nc * r
        if r > 1:
            a, b = 0, n
            while a < b:
                mid = (a + b) // 2
                if ok(r - 1, mid):
                    b = mid
                else:
                    a = mid + 1
            cells = Nc * (r - 1) + ncw * a
        if best is None or cells < best[0]:
            best = (cells, ("bch", n, k, t, r))
    _VC[key] = best
    return best


def vocab_cost(L, eps, prot):
    return L if eps <= 0 else (cap_cells(L, eps) if prot == "cap" else bch_cells(L, eps)[0])


def charged_nstar(ns_gi, stores, eps, prot, compact=False):
    """Smallest grid budget B with floor(B n_s) - floor(B_mem n_s) >= vocabulary cells of store s on every store in
    scope (the memory at its own N*, the vocabulary in the remaining cells). stores: [(n_pairs, vocab list)]."""
    if ns_gi is None or ns_gi >= NG:
        return NG
    b = GRID[ns_gi]
    need = [(n, vocab_cost(vocab_bits(v, compact), eps, prot)) for n, v in stores]
    for gi in range(ns_gi, NG):
        if all(int(GRID[gi] * n) - int(b * n) >= c for n, c in need):
            return gi
    return NG


# =============================================================================================== N* from per-store counts
def nstar_counts(curve, sids):
    """B38's rule on per-store (failed, n): smallest budget with share >= TARGET there and at every larger evaluated
    budget; a budget counts only if every store in sids was evaluated there."""
    share = {}
    for gi, d in curve.items():
        if all(s in d for s in sids):
            f = sum(d[s][0] for s in sids)
            n = sum(d[s][1] for s in sids)
            share[gi] = 1 - min(f, n) / n
    best = None
    for gi in sorted(share, reverse=True):
        if share[gi] >= TARGET:
            best = gi
        else:
            break
    return (NG if best is None else best), share


# =============================================================================================== sweep jobs
def run_job(job):
    """job: dict(sid, pairs, extra, mem, design, evals=[(B, [conds])], nprobe). Rows per (B, cond)."""
    t0 = time.time()
    install(job["extra"], job.get("nprobe", NPROBE))
    mem, des, sid, pairs = job["mem"], job["design"], job["sid"], job["pairs"]
    rows, meta = [], {}
    if mem in ("U", "T-rep", "E-LDPC", "E-BCH"):
        arm = {"U": "V-last", "T-rep": "T-rep", "E-LDPC": "E-LDPC", "E-BCH": "E-BCH"}[mem]
        d = "auto" if mem == "E-BCH" else des
        j = dict(sid=sid, pairs=pairs, arm=arm, evals=[(B, d, cs) for B, cs in job["evals"]])
        r = E.run_job_auto(j) if mem == "E-BCH" else E.run_job(j)
        meta = dict(nv=r["n_values"], n_pairs=r["n_pairs"], nq=r["nq"])
        par = {}
        for x in r["res"]:
            if "params" in x:
                par[round(x["B"], 4)] = dict(x["params"], cells=x["cells"], budget=x["budget"])
        for x in r["res"]:
            if "cond" not in x:
                continue
            row = dict(B=x["B"], cond=x["cond"], infeasible=bool(x.get("infeasible")))
            if not row["infeasible"]:
                row.update(exact=x["exact"], n=x["n"], failed=x["failed"], cells=x["cells"],
                           budget=int(x["B"] * r["n_pairs"]))
                p = par.get(round(x["B"], 4), {})
                if mem == "E-BCH":
                    row["bch"] = x.get("design")
                if "L" in p:
                    row["L"] = p["L"]
            rows.append(row)
    elif mem == "H-SBC":
        import b61_hd as H
        S38 = H.slim(E.Store38(sid, pairs))
        meta = dict(nv=S38.Nv, n_pairs=S38.n, nq=len(S38.want))
        for B, cs in job["evals"]:
            for c in cs:
                x = H.sbc_evaluate(S38, des, B, c, sic=False)
                if x.get("infeasible"):
                    rows.append(dict(B=B, cond=c, infeasible=True))
                    continue
                y = x["rules"]["H-SBC"]
                rows.append(dict(B=B, cond=c, infeasible=False, exact=y["exact"], n=y["n"], failed=y["failed"],
                                 cells=x["cells"], budget=x["budget"], S=x["S"], d=x["d"], Lb=x["Lb"], nb=x["nb"],
                                 density=x["density"]))
    else:
        import b63_online as O
        f, R = des if mem == "O-LDPC" else (des, None)
        r = O.run_job(dict(sid=sid, pairs=pairs, arm=mem, f=f, evals=[(R, B, cs) for B, cs in job["evals"]]))
        meta = dict(nv=r["n_values"], n_pairs=r["n_pairs"], nq=r["nq"], lost=r["lost"], m=r["m"], empty=r["empty"])
        for x in r["res"]:
            row = dict(B=x["B"], cond=x["cond"], infeasible=bool(x.get("infeasible")), L=x.get("L"))
            if not row["infeasible"]:
                row.update(exact=x["exact"], n=x["n"], failed=x["failed"], cells=x["cells"], budget=x["budget"],
                           bch=x.get("bch"))
            rows.append(row)
    return dict(sid=sid, mem=mem, design=des, rows=rows, sec=time.time() - t0, **meta)


def pool_run(jobs, workers, out_path):
    from multiprocessing import Pool
    done, last = [], time.time()
    with Pool(workers, maxtasksperchild=8) as pool:
        for r in pool.imap_unordered(run_job, jobs, chunksize=1):
            done.append(r)
            print("done %-22s %-6s %-18s %6.0fs %d/%d" % (r["sid"], r["mem"], json.dumps(r["design"]), r["sec"],
                                                          len(done), len(jobs)), flush=True)
            if time.time() - last > 120:
                json.dump(done, open(out_path + ".partial", "w"))
                last = time.time()
    json.dump(done, open(out_path, "w"))
    return done


def designs_of(mem, cond):
    if mem in DEVFREE:
        return [None]
    if mem == "H-SBC":
        return list(LAMS)
    if mem == "E-LDPC":
        return list(RATESETS[cond])
    if mem == "O-BCH":
        return list(FS)
    return [[f, R] for f in FS for R in RATESETS[cond]]


def make_jobs(stores, need, per_job=24):
    """need: {(mem, N, design_json): {gi: set(conds)}} -> jobs of at most per_job evaluations per store (U at N >=
    6,000: one budget per job, its clean-up over 2N candidates being the slowest read); longest first."""
    jobs = []
    unit = {"U": 3.0, "H-SBC": 2.0, "T-rep": 0.3, "E-BCH": 1.0, "E-LDPC": 2.0, "O-BCH": 1.0, "O-LDPC": 2.0}
    for sid, pairs, extra in stores:
        N = n_of(sid)
        for (mem, n_, dj), bud in sorted(need.items()):
            if n_ != N:
                continue
            ev = [(GRID[gi], sorted(cs)) for gi, cs in sorted(bud.items())]
            cap = 4 if (mem == "U" and N >= 6000) else per_job
            chunk = []

            def flush(ch):
                jobs.append(dict(sid=sid, pairs=pairs, extra=extra, mem=mem, design=json.loads(dj), evals=ch,
                                 nprobe=NPROBE, cost=unit[mem] * sum(b * len(c) for b, c in ch) * (1 + N / 6000)
                                 + 15 * (1 + N / 6000)))
            for e in ev:
                if chunk and sum(len(c) for _, c in chunk) + len(e[1]) > cap:
                    flush(chunk)
                    chunk = []
                chunk.append(e)
            if chunk:
                flush(chunk)
    jobs.sort(key=lambda j: -j["cost"])
    return jobs


def window(center, lo=4, hi=4):
    return range(max(0, center - lo), min(NG, center + hi + 1))


def dev_plan(theory_path, out_path, conds=CONDS):
    th = json.load(open(theory_path))
    need = collections.defaultdict(lambda: collections.defaultdict(set))
    for mem in MEMS:
        if mem in DEVFREE:
            continue
        for N in NS:
            for c in conds:
                ctr = th["pred"]["%s|%d|%s" % (mem, N, c)]
                for d in designs_of(mem, c):
                    for gi in window(ctr):
                        need[(mem, N, json.dumps(d))][gi].add(c)
    save_need(need, out_path)
    return need


def save_need(need, path):
    json.dump([[list(k), {str(g): sorted(cs) for g, cs in v.items()}] for k, v in need.items()], open(path, "w"))


def load_need(path):
    need = collections.defaultdict(lambda: collections.defaultdict(set))
    for k, v in json.load(open(path)):
        for g, cs in v.items():
            need[tuple(k)][int(g)] |= set(cs)
    return need


def jl(p):
    return json.load(gzip.open(p, "rt")) if p.endswith(".gz") else json.load(open(p))


def aggregate(paths):
    """acc[(mem, N, cond)][gi][design_json][sid] = (failed, n)."""
    acc = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(dict)))
    for p in paths:
        for r in jl(p):
            N = n_of(r["sid"])
            dj = json.dumps(r["design"])
            for x in r["rows"]:
                if x.get("infeasible"):
                    continue
                acc[(r["mem"], N, x["cond"])][gidx(x["B"])][dj][r["sid"]] = (len(x["failed"]), x["n"])
    return acc


def choose(paths, out_path):
    """Per (mem, N, cond, budget): the design with the highest pooled dev share (a design counts at a budget only where
    it was evaluated on every dev store of that N); ties: the higher share at the next harder condition at the same
    budget, then list order. Dev N* per (mem, N, cond) on the chosen designs' curve (B38's rule)."""
    acc = aggregate(paths)
    nd = collections.Counter()
    for p in paths:
        for r in jl(p):
            nd[n_of(r["sid"])] = max(nd[n_of(r["sid"])], 0)
    sids_of = collections.defaultdict(set)
    for p in paths:
        for r in jl(p):
            sids_of[n_of(r["sid"])].add(r["sid"])
    ch = dict(designs={}, dev_curve={}, dev_nstar={})

    def share(mem, N, c, gi, dj):
        if c is None:
            return -1.0
        d = acc[(mem, N, c)].get(gi, {}).get(dj)
        if d is None or len(d) < len(sids_of[N]):
            return -1.0
        f = sum(v[0] for v in d.values())
        n = sum(v[1] for v in d.values())
        return 1 - f / n
    for mem in MEMS:
        if mem in DEVFREE:
            continue
        for N in NS:
            for c in CONDS:
                dl = [json.dumps(d) for d in designs_of(mem, c)]
                key = "%s|%d|%s" % (mem, N, c)
                ch["designs"][key], ch["dev_curve"][key] = {}, {}
                for gi in sorted(acc[(mem, N, c)]):
                    cand = [(share(mem, N, c, gi, dj), share(mem, N, NEXT[c], gi, dj), -i, dj)
                            for i, dj in enumerate(dl) if share(mem, N, c, gi, dj) >= 0]
                    if not cand:
                        continue
                    best = max(cand)
                    ch["designs"][key][str(gi)] = json.loads(best[3])
                    ch["dev_curve"][key][str(gi)] = best[0]
                cur = {int(g): s for g, s in ch["dev_curve"][key].items()}
                ns = None
                for gi in sorted(cur, reverse=True):
                    if cur[gi] >= TARGET:
                        ns = gi
                    else:
                        break
                ch["dev_nstar"][key] = ns
    json.dump(ch, open(out_path, "w"), indent=0)
    return ch


def design_at(ch, mem, N, c, gi):
    if mem in DEVFREE:
        return None
    d = ch["designs"]["%s|%d|%s" % (mem, N, c)]
    if not d:
        return designs_of(mem, c)[0]
    if str(gi) in d:
        return d[str(gi)]
    near = min((abs(int(g) - gi), -int(g)) for g in d)
    return d[str(-near[1])]


def dev_ext_plan(ch, devplan_path, out_path):
    """One dev extension round (B38's rule): where the chosen curve never reaches the bar, every design 4 steps
    above the window; where the dev N* is the lowest evaluated budget, 4 steps below."""
    old = load_need(devplan_path)
    need = collections.defaultdict(lambda: collections.defaultdict(set))
    for mem in MEMS:
        if mem in DEVFREE:
            continue
        for N in NS:
            for c in CONDS:
                key = "%s|%d|%s" % (mem, N, c)
                cur = [int(g) for g in ch["dev_curve"].get(key, {})]
                if not cur:
                    continue
                ns = ch["dev_nstar"][key]
                add = []
                if ns is None:
                    add = range(max(cur) + 1, min(NG, max(cur) + 5))
                elif ns == min(cur) and ns > 0:
                    add = range(max(0, ns - 4), ns)
                for d in designs_of(mem, c):
                    dj = json.dumps(d)
                    for gi in add:
                        if c not in old.get((mem, N, dj), {}).get(gi, set()):
                            need[(mem, N, dj)][gi].add(c)
    save_need(need, out_path)
    return need


def test_plan(ch, th, out_path, conds=CONDS):
    """Per (mem, N, cond): dev N* +- 4 grid steps with the dev design per budget (the model prediction +- 4 for the
    development-free memories U, T-rep, E-BCH; the top 5 dev budgets if dev never reached the bar)."""
    need = collections.defaultdict(lambda: collections.defaultdict(set))
    for mem in MEMS:
        for N in NS:
            for c in conds:
                key = "%s|%d|%s" % (mem, N, c)
                if mem in DEVFREE:
                    gis = window(th["pred"][key])
                else:
                    ns = ch["dev_nstar"].get(key)
                    if ns is None:
                        cur = sorted(int(g) for g in ch["dev_curve"].get(key, {}))
                        gis = cur[-5:] if cur else window(th["pred"][key])
                    else:
                        gis = window(ns)
                for gi in gis:
                    need[(mem, N, json.dumps(design_at(ch, mem, N, c, gi)))][gi].add(c)
    save_need(need, out_path)
    return need


def test_curves(ch, paths, sids_of):
    """C[(mem, N, cond)] = {gi: {sid: (failed, n)}} at each budget's dev design."""
    acc = aggregate(paths)
    C = collections.defaultdict(dict)
    for (mem, N, c), byg in acc.items():
        for gi, byd in byg.items():
            dj = json.dumps(design_at(ch, mem, N, c, gi))
            if dj in byd:
                C[(mem, N, c)][gi] = byd[dj]
    return C


def extend(ch, plan_paths, test_paths, out_path, conds=CONDS):
    """B38's extension rule on the test: 4 steps up while the top evaluated budget fails, 4 down while N* is the
    lowest evaluated budget; budgets already planned are not requested again."""
    planned = set()
    for p in plan_paths:
        for k, v in json.load(open(p)):
            for g, cs in v.items():
                for c in cs:
                    planned.add((k[0], k[1], int(g), c))
    sids_of = collections.defaultdict(set)
    for p in test_paths:
        for r in jl(p):
            sids_of[n_of(r["sid"])].add(r["sid"])
    C = test_curves(ch, test_paths, sids_of)
    need = collections.defaultdict(lambda: collections.defaultdict(set))
    for mem in MEMS:
        for N in NS:
            for c in conds:
                cur = C.get((mem, N, c), {})
                if not cur:
                    continue
                ns, sh = nstar_counts(cur, sorted(sids_of[N]))
                if not sh:
                    continue
                top, low = max(sh), min(sh)
                add = []
                if sh[top] < TARGET and top < NG - 1:
                    add = range(top + 1, min(NG, top + 5))
                elif ns == low and low > 0:
                    add = range(max(0, low - 4), low)
                for gi in add:
                    if (mem, N, gi, c) in planned:
                        continue
                    need[(mem, N, json.dumps(design_at(ch, mem, N, c, gi)))][gi].add(c)
    save_need(need, out_path)
    print("extension evaluations:", sum(len(cs) for v in need.values() for cs in v.values()))
    return need


# =============================================================================================== verdicts
ALLC = list(E.IID)                                  # k1 k2 k4 e05 k8 k16 k32 (part A)
LABC = ("k1", "k2", "k4", "k8", "k16", "k32")        # labelled (e05 reported)
COMPS = (("static", "U"), ("static", "H-SBC"), ("insert", "H-SBC"), ("insert", "U"))


def family(sid):
    if "niah_multikey_2" in sid:
        return "mk2"
    if "niah_multikey_3" in sid:
        return "mk3"
    return "small"


def best_of(NSg, arms, c):
    return min(NSg[(a, c)] for a in arms)


def delta_rows(NSg, conds, labc):
    """Delta = steps(table) - steps(store); + = the store needs fewer cells (B38's sign)."""
    out = {}
    for tab, st in COMPS:
        ds = {c: NSg[(tab, c)] - NSg[(st, c)] for c in conds}
        out["%s_vs_%s" % (tab, st)] = dict(delta=ds, label=h_label([ds[c] for c in labc if c in ds]),
                                           diff_cells={c: (None if max(NSg[(tab, c)], NSg[(st, c)]) >= NG else
                                                           round(GRID[NSg[(st, c)]] - GRID[NSg[(tab, c)]], 1))
                                                       for c in conds})
    return out


def part_a(out_prefix):
    import b63_verdict as V63
    ch63 = json.load(open("results/b63_choice.json"))
    C, sids = V63.all_counts(ch63, ["results/b63_test1.json.gz"])
    stores = {s: p for s, p in E.load_stores("b34_out/test_stores.json.gz")}
    vocab = {s: list(dict.fromkeys(v for _, v, _ in p)) for s, p in stores.items()}
    npairs = {s: len(p) for s, p in stores.items()}
    scopes = collections.OrderedDict([("pooled", sids)])
    scopes["mk2"] = [s for s in sids if family(s) == "mk2"]
    scopes["mk2-6k"] = [s for s in sids if family(s) == "mk2" and npairs[s] > 5000]
    scopes["mk3"] = [s for s in sids if family(s) == "mk3"]
    scopes["small"] = [s for s in sids if family(s) == "small"]
    arms = ("U", "H-SBC", "T-rep", "E-BCH", "E-LDPC", "E-CAM", "O-BCH", "O-LDPC")
    J = dict(nstar={}, delta={}, vocab={}, check_registered={})
    L = ["# B69 part (A): cells per stored pair for >= 99.5% exact, value vocabulary uncharged / charged",
         "# (registered B38 / B59 / B61 / B63 test rows; per-store failed-key counts; 30-day i.i.d. cell errors).",
         "# charged: the store's vocabulary strings on the same cells; cap = BSC capacity bound (no code reaches it);",
         "# bch = repetition or BCH by B38's closed form, P(any string of the store wrong) <= 1%; utf8 / compact.",
         "# Delta = steps(table) - steps(store), + = the store needs fewer cells; labels over k1 k2 k4 k8 k16 k32.", ""]
    for sc, ss in scopes.items():
        vb = {s: vocab_bits(vocab[s]) for s in ss}
        vbc = {s: vocab_bits(vocab[s], True) for s in ss}
        J["vocab"][sc] = dict(max_bits_per_pair=max(vb[s] / npairs[s] for s in ss),
                              max_bits_per_pair_compact=max(vbc[s] / npairs[s] for s in ss),
                              nstores=len(ss), nq=sum(C[("U", "k1")][min(C[("U", "k1")])][s][1] for s in ss))
        L.append("== scope %s: %d stores, %d probed keys; vocabulary bits per pair (max over stores) utf8 %.1f, compact %.1f"
                 % (sc, len(ss), J["vocab"][sc]["nq"], J["vocab"][sc]["max_bits_per_pair"],
                    J["vocab"][sc]["max_bits_per_pair_compact"]))
        own = {}
        for a in arms:
            for c in ALLC:
                own[(a, c)] = nstar_counts(C[(a, c)], ss)[0] if (a, c) in C else NG
        st_list = [(npairs[s], vocab[s]) for s in ss]
        for acct in ("free", "cap-utf8", "bch-utf8", "cap-compact", "bch-compact"):
            NSg = {}
            for a in arms:
                for c in ALLC:
                    if acct == "free":
                        NSg[(a, c)] = own[(a, c)]
                    else:
                        prot, fm = acct.split("-")
                        NSg[(a, c)] = charged_nstar(own[(a, c)], st_list, E.IID[c], prot, fm == "compact")
            for c in ALLC:
                NSg[("static", c)] = best_of(NSg, ("T-rep", "E-BCH", "E-LDPC", "E-CAM"), c)
                NSg[("insert", c)] = best_of(NSg, ("O-BCH", "O-LDPC"), c)
            for (a, c), v in NSg.items():
                J["nstar"]["%s|%s|%s|%s" % (sc, acct, a, c)] = None if v >= NG else GRID[v]
            D_ = delta_rows(NSg, ALLC, LABC)
            J["delta"]["%s|%s" % (sc, acct)] = D_
            L.append("  -- accounting %s" % acct)
            L.append("    %-8s" % "" + "".join("%9s" % c for c in ALLC))
            for a in ("U", "H-SBC", "static", "insert"):
                L.append("    %-8s" % a + "".join("%9s" % fmt(NSg[(a, c)]) for c in ALLC))
            for k, v in D_.items():
                L.append("    %-17s %s  -> %s" % (k, " ".join("%+d" % v["delta"][c] for c in ALLC), v["label"]))
        L.append("")
    # the registered pooled numbers must be reproduced by the free accounting
    reg = dict(U=[76, 76, 76, 91, 128, 431, 1722], **{"H-SBC": [32, 32, 45, 54, 91, 304, 1448],
               "static": [19, 19, 27, 38, 54, 215, 724], "insert": [32, 32, 38, 64, 91, 304, 1218]})
    bad = 0
    for a, vals in reg.items():
        for c, v in zip(ALLC, vals):
            got = J["nstar"]["pooled|free|%s|%s" % (a, c)]
            ok = got is not None and abs(got - v) < 1
            bad += not ok
            J["check_registered"]["%s|%s" % (a, c)] = [got, v, ok]
    L.append("Registered pooled N* reproduced by the uncharged accounting: %d mismatches of %d" % (bad, 28))
    open(out_prefix + ".txt", "w").write("\n".join(L) + "\n")
    json.dump(J, open(out_prefix + ".json", "w"), indent=1)
    print("\n".join(L))
    return J


def verdict_b(ch, th, paths, out_prefix, conds=CONDS):
    sids_of = collections.defaultdict(set)
    meta = {}
    for p in paths:
        for r in jl(p):
            sids_of[n_of(r["sid"])].add(r["sid"])
    C = test_curves(ch, paths, sids_of)
    stores = {s: (p, x) for s, p, x in load_sweep("b69_out/sweep_test.json.gz")}
    J = dict(nstar={}, delta={}, pred={}, scaling={}, coverage={})
    L = ["# B69 part (B): vocabulary sweep. Synthetic stores of 6,002 pairs (B34's mk2big2 generator), N distinct",
         "# 7-digit values in the vocabulary; 8 test stores x 300 probed keys per N; designs from 3 dev stores per N.",
         "# N* in cells per stored pair; Delta = steps(table) - steps(store), + = the store needs fewer cells.", ""]
    NSall = {}
    for N in NS:
        ss = sorted(sids_of[N])
        own = {}
        for m in MEMS:
            for c in conds:
                cur = C.get((m, N, c), {})
                own[(m, c)] = nstar_counts(cur, ss)[0] if cur else NG
                J["coverage"]["%s|%d|%s" % (m, N, c)] = [GRID[g] for g in sorted(nstar_counts(cur, ss)[1])] if cur else []
        st_list = [(len(stores[s][0]), list(dict.fromkeys([v for _, v, _ in stores[s][0]] + stores[s][1]))) for s in ss]
        for acct in ("free", "cap-utf8", "bch-utf8", "cap-compact", "bch-compact"):
            NSg = {}
            for m in MEMS:
                for c in conds:
                    if acct == "free":
                        NSg[(m, c)] = own[(m, c)]
                    else:
                        prot, fm = acct.split("-")
                        NSg[(m, c)] = charged_nstar(own[(m, c)], st_list, E.IID[c], prot, fm == "compact")
            for c in conds:
                NSg[("static", c)] = best_of(NSg, STATIC, c)
                NSg[("insert", c)] = best_of(NSg, INSERT, c)
            for (m, c), v in NSg.items():
                J["nstar"]["%d|%s|%s|%s" % (N, acct, m, c)] = None if v >= NG else GRID[v]
                NSall[(N, acct, m, c)] = v
            D_ = delta_rows(NSg, list(conds), [c for c in conds if c in PRIMARY])
            J["delta"]["%d|%s" % (N, acct)] = D_
    for acct in ("free", "cap-utf8", "bch-utf8", "cap-compact", "bch-compact"):
        L.append("== accounting %s" % acct)
        L.append("  %-8s %-6s" % ("memory", "N") + "".join("%9s" % c for c in conds))
        for m in MEMS + ("static", "insert"):
            for N in NS:
                L.append("  %-8s %-6d" % (m, N) + "".join("%9s" % fmt(NSall[(N, acct, m, c)]) for c in conds))
        for tab, st in COMPS:
            k = "%s_vs_%s" % (tab, st)
            for N in NS:
                v = J["delta"]["%d|%s" % (N, acct)][k]
                L.append("  %-17s N %-6d %s  -> %s" % (k, N, " ".join("%s %+d" % (c, v["delta"][c]) for c in conds),
                                                       v["label"]))
        L.append("")
    # theory predictions and scaling
    L.append("== predictions (theory) vs test, uncharged: N* per (memory, N, condition), steps test - predicted")
    within2 = tot = 0
    for m in MEMS:
        for N in NS:
            for c in conds:
                p_ = th["pred"]["%s|%d|%s" % (m, N, c)]
                t_ = NSall[(N, "free", m, c)]
                J["pred"]["%s|%d|%s" % (m, N, c)] = [fmt(p_), fmt(t_), t_ - p_]
                tot += 1
                within2 += abs(t_ - p_) <= 2
        L.append("  %-7s " % m + " | ".join("N%d %s" % (N, " ".join("%s:%s/%s" % (c, fmt(th["pred"]["%s|%d|%s" % (m, N, c)]),
                                                                              fmt(NSall[(N, "free", m, c)])) for c in conds))
                                           for N in NS))
    L.append("  within 2 steps of the prediction: %d of %d" % (within2, tot))
    J["pred_within2"] = [within2, tot]
    for m in MEMS + ("static", "insert"):
        for c in conds:
            a, b = NSall[(NS[0], "free", m, c)], NSall[(NS[-1], "free", m, c)]
            J["scaling"]["%s|%s" % (m, c)] = None if max(a, b) >= NG else b - a
    L.append("  step change N = %d -> %d (uncharged): " % (NS[0], NS[-1]) + "; ".join(
        "%s %s" % (m, " ".join("%s %s" % (c, J["scaling"]["%s|%s" % (m, c)]) for c in conds)) for m in MEMS + ("static", "insert")))
    open(out_prefix + ".txt", "w").write("\n".join(L) + "\n")
    json.dump(J, open(out_prefix + ".json", "w"), indent=1)
    print("\n".join(L))
    return J


# =============================================================================================== self-test
def selftest():
    """Synthetic small-N stores only: the vocabulary patch, the sweep generator, every memory's job on a small store,
    the vocabulary cost functions and the N* rule."""
    ok = True
    for N in (64, 512):
        sid, pairs, extra = make_store(N, "selftest", 0, n_keys=400)
        install(extra, 50)
        S38 = E.Store38(sid, pairs)
        assert S38.Nv == N, (S38.Nv, N)
        assert S38.s.b == math.ceil(math.log2(N))
        assert len(S38.want) == 50
        assert set(S38.s.cd["vocab"][:len(set(v for _, v, _ in pairs))]) == set(v for _, v, _ in pairs)
        print("store N=%d n=%d Nv=%d b=%d probes=%d extra=%d" % (N, S38.n, S38.Nv, S38.s.b, len(S38.want), len(extra)))
        install(None, 150)
        for mem in MEMS:
            for c in ("k1", "k8"):
                d = designs_of(mem, c)[0] if mem not in DEVFREE else None
                B = {"U": 128.0, "H-SBC": 91.0, "T-rep": 32.0, "E-BCH": 64.0, "E-LDPC": 64.0, "O-BCH": 64.0,
                     "O-LDPC": 64.0}[mem]
                B = GRID[gidx(B if c == "k1" else 4 * B)]
                r = run_job(dict(sid=sid, pairs=pairs, extra=extra, mem=mem, design=d, evals=[(B, [c])], nprobe=50))
                x = r["rows"][0]
                over = (not x["infeasible"]) and x["cells"] > x["budget"]
                ok &= not over and r["nv"] == N
                print("  %-7s %-4s B %6.1f design %-16s exact %s/%s cells %s budget %s nv %d %.1fs" % (
                    mem, c, B, json.dumps(d), x.get("exact"), x.get("n"), x.get("cells"), x.get("budget"), r["nv"],
                    r["sec"]))
    # vocabulary costs
    for L_, eps in ((336000, 0.0), (336000, 0.0208065), (336000, 0.12263475), (489600, 0.27600645), (2000, 0.12263475)):
        cb = bch_cells(L_, eps)
        cp = cap_cells(L_, eps)
        print("vocab L %d eps %.4f: cap %.3f cells/bit, bch/rep %.3f cells/bit %s" % (L_, eps, cp / L_, cb[0] / L_, cb[1]))
        ok &= cb[0] >= cp
    vb = vocab_bits(["1234567", "7654321"])
    ok &= vb == 112 and vocab_bits(["1234567", "7654321"], True) == 48
    ok &= vocab_bits(["ab", "abcd"]) == 8 * min(2 * 4, 3 + 5)
    print("vocab bits checks", vb, vocab_bits(["1234567", "7654321"], True), vocab_bits(["ab", "abcd"]))
    # N* rule
    cur = {10: {"a": (5, 100)}, 11: {"a": (0, 100)}, 12: {"a": (1, 100)}, 13: {"a": (0, 100)}}
    ok &= nstar_counts(cur, ["a"])[0] == 13
    print("SELFTEST", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    cmd = sys.argv[1]
    arg = lambda k, d: (sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d)
    if cmd == "selftest":
        selftest()
    elif cmd == "stores":
        write_stores()
    elif cmd == "devplan":
        dev_plan(sys.argv[2], sys.argv[3])
    elif cmd == "run":
        stores = load_sweep(sys.argv[2])
        jobs = make_jobs(stores, load_need(sys.argv[3]))
        print("jobs", len(jobs), "evaluations", sum(sum(len(c) for _, c in j["evals"]) for j in jobs), flush=True)
        pool_run(jobs, int(arg("--workers", 8)), sys.argv[4])
    elif cmd == "choose":
        choose(sys.argv[2].split(","), sys.argv[3])
    elif cmd == "devext":
        dev_ext_plan(json.load(open(sys.argv[2])), sys.argv[3], sys.argv[4])
    elif cmd == "testplan":
        test_plan(json.load(open(sys.argv[2])), json.load(open(sys.argv[3])), sys.argv[4])
    elif cmd == "extend":
        extend(json.load(open(sys.argv[2])), sys.argv[3].split(","), sys.argv[4].split(","), sys.argv[5])
    elif cmd == "partA":
        part_a(sys.argv[2])
    elif cmd == "verdictB":
        verdict_b(json.load(open(sys.argv[2])), json.load(open(sys.argv[3])), sys.argv[4].split(","), sys.argv[5])
    elif cmd == "theory":
        th = theory(sys.argv[2])
        for k in sorted(th["pred"]):
            pass
        print(json.dumps(th["logical"], indent=0))
        for mem in MEMS:
            for c in CONDS:
                print("%-7s %-4s" % (mem, c), " ".join("N%-6d %6s" % (N, fmt(th["pred"]["%s|%d|%s" % (mem, N, c)]))
                                                     for N in NS),
                      "| model", " ".join(th["model"].get("%s|%d|%s" % (mem, N, c), "-") for N in NS))
