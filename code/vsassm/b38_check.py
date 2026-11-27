"""B38 checks from other angles (prereg/B38_ecc_equal_cells_prereg.md).

python3 b38_check.py closed  STORES.json.gz TEST.json[,TEST2.json] OUT.json [--workers N]
    Closed-form expected share of probed keys read exactly for every simulated i.i.d. cell of T-rep (exact binomial),
    E-BCH / E-CAM (exact errors-and-erasures bound per codeword; a failed codeword counted as a failed key, so the
    closed form is a lower bound up to sampling error), V-pub / V-Snt / V-last (Gaussian approximation of the clean-up
    similarities, candidates independent). No cell is simulated.
python3 b38_check.py decoders STORES.json.gz TEST.json OUT.txt [--workers N]
    Second implementations: the run's own BCH codec (b38_eq.BCH: generator polynomial, systematic encoder,
    Berlekamp-Massey / Chien decoder) against the galois library (generator polynomial, encoding and decoding of every
    codeword of re-simulated test cells, including the bounded-distance shortcut); an own numpy sum-product LDPC
    decoder against the ldpc package (the run's LDPC decoder) on the same channel inputs.
python3 b38_check.py device STORES.json.gz TEST.json OUT.json [--workers N]
    Re-reads a subset of test cells with B25's device model itself (programming noise, drift, read noise, ADC at k,
    30 days) instead of i.i.d. flips at the measured rate; the shares must agree within sampling error.
python3 b38_check.py handread STORES.json.gz SID ARM B DESIGN COND [N]
    Re-simulates one cell with the run's seeds and prints up to N failed probed keys with the cause.
"""
import collections
import json
import math
import sys

import numpy as np
from scipy import stats
from scipy.special import log_ndtr

import b25_pcm as P
import b34_inmem as I34
import b38_eq as E


# =============================================================================================== closed form
def q_err_tie(c, p):
    """(P(majority of c copies wrong, no tie), P(tie))."""
    if p <= 0:
        return 0.0, 0.0
    if c % 2:
        return float(stats.binom.sf(c // 2, c, p)), 0.0
    return float(stats.binom.sf(c // 2, c, p)), float(stats.binom.pmf(c // 2, c, p))


def cw_ok_ee(groups, t, p):
    """P(2 e + s <= 2t) for a codeword whose coded bits come in groups [(m, copies)], copies giving per-bit error w and
    tie (erasure) probability z."""
    # distribution of W = 2e + s, truncated at 2t, by convolution
    W = np.zeros(2 * t + 1)
    W[0] = 1.0
    for m, c in groups:
        w, z = q_err_tie(c, p)
        k = 1 - w - z
        # per-bit pmf over {0: k, 1: z, 2: w}; m-fold convolution by repeated squaring, truncated
        base = np.zeros(2 * t + 1)
        base[0] = k
        if 2 * t >= 1:
            base[1] = z
        if 2 * t >= 2:
            base[2] = w
        acc = np.zeros(2 * t + 1)
        acc[0] = 1.0
        e = m
        b = base
        while e:
            if e & 1:
                acc = np.convolve(acc, b)[:2 * t + 1]
            b = np.convolve(b, b)[:2 * t + 1]
            e >>= 1
        W = np.convolve(W, acc)[:2 * t + 1]
    return float(min(1.0, W.sum()))


def cf_ecc(S38, arm, design, B, eps):
    cont = E.mphf_content(S38) if arm == "E-BCH" else E.cam_content(S38)
    needs = E.mphf_needs(S38, cont) if arm == "E-BCH" else E.cam_needs(S38, cont)
    n, k, t = design
    lay = E.ecc_layout(len(cont["bits"]), n, k, t, int(B * S38.n))
    starts = np.r_[0, np.cumsum(lay["sizes"])]
    cst = np.r_[0, np.cumsum(lay["clen"])]
    cache = {}
    pok = []
    for i in range(len(lay["sizes"])):
        cc = collections.Counter(lay["counts"][cst[i]:cst[i + 1]].tolist())
        key = tuple(sorted(cc.items()))
        if key not in cache:
            cache[key] = cw_ok_ee([(m, c) for c, m in key], t, eps)
        pok.append(cache[key])
    pok = np.array(pok)
    out = []
    for pos in needs:
        cws = np.unique(np.searchsorted(starts, pos, side="right") - 1)
        out.append(float(np.prod(pok[cws])))
    return out


def cf_rep(S38, B, eps):
    """T-rep: every needed logical bit (displacements of the key's buckets, its rows) decoded by majority of r copies."""
    s = S38.s
    info = I34.layout(s, "T-v", int(B * S38.n), lam=E.LAM)
    if info is None:
        return None
    r = info["r"]
    w, z = q_err_tie(r, eps)
    qb = w + z / 2
    cont = E.mphf_content(S38)
    return [float((1 - qb) ** len(pos)) for pos in E.mphf_needs(S38, cont)]


def cf_vsa(S38, B, eps, kind):
    """Gaussian approximation (B34's, adapted): each clean-up similarity ~ N((1 - 2 eps) s0, 4 eps (1 - eps) / d) with
    s0 the noiseless similarity; candidates independent. kind: pub / snt (plain store) or last (V-last store)."""
    bd = E.vsa_build(S38, B, "plain" if kind in ("pub", "snt") else "last")
    if bd is None:
        return None
    S, d = bd["S"], bd["d"]
    rows = bd["cells"][:S * d].reshape(S, d).astype(np.float32)
    sh = bd["shard_k"][S38.qk]
    X = rows[sh] * S38.Kc[S38.qk, :d]
    Vf = S38.vf(d)
    Lf = S38.L[:d].astype(np.float32)
    sig = math.sqrt(4 * eps * (1 - eps) / d)
    s = S38.s
    probs = []
    for a, key in enumerate(s.keys):
        c = s.count[key]
        scale = float(bd["rms0"][sh[a]]) if kind in ("pub", "snt") else 1.0
        pk = 1.0
        for j in range(min(c + 1, S38.J)):
            Y = X[a] * S38.Pc[j, :d]
            mu = (Y @ Vf.T) / d
            if kind == "last":
                mu = np.concatenate([mu, ((Y * Lf) @ Vf.T) / d])
            mu = mu * (1 - 2 * eps) * scale
            sg = sig * scale
            if j < c:
                tgt = s.want[a][j] + (S38.Nv if (kind == "last" and j == c - 1) else 0)
                thr = -np.inf if (kind in ("snt", "last") and j == 0) or kind == "last" else P.THR
                others = np.delete(mu, tgt)
                if sg < 1e-12:
                    pk *= float(mu[tgt] >= thr and mu[tgt] > others.max())
                    continue
                lo, hi = max(thr, mu[tgt] - 8 * sg), mu[tgt] + 8 * sg
                if lo >= hi:
                    pk = 0.0
                    break
                grid = np.linspace(lo, hi, 801)
                rel = others[others > mu[tgt] - 12 * sg]
                if len(rel):
                    nb = max(1, int(np.ceil((rel.max() - rel.min()) / (sg / 20))))
                    cnt, edges = np.histogram(rel, bins=nb)
                    cen = 0.5 * (edges[1:] + edges[:-1])
                    kp = cnt > 0
                    lp = (cnt[kp][None, :] * log_ndtr((grid[:, None] - cen[kp][None, :]) / sg)).sum(1)
                else:
                    lp = np.zeros_like(grid)
                dens = stats.norm.pdf(grid, mu[tgt], sg) * np.exp(lp)
                pk *= float(np.trapezoid(dens, grid)) if hasattr(np, "trapezoid") else float(np.trapz(dens, grid))
            elif kind in ("pub", "snt"):                       # the stop read
                if sg < 1e-12:
                    pk *= float(mu.max() < P.THR)
                else:
                    pk *= float(np.exp(np.sum(log_ndtr((P.THR - mu) / sg))))
        probs.append(pk)
    return probs


def _closed_job(job):
    sid, pairs, cells = job
    S38 = E.Store38(sid, pairs)
    out = []
    for arm, rule, B, design, cond in cells:
        eps = E.IID.get(cond, E.M3_EPS.get(cond))
        if arm in ("E-BCH", "E-CAM"):
            pr = cf_ecc(S38, arm, design, B, eps)
        elif arm == "T-rep":
            pr = cf_rep(S38, B, eps)
        else:
            pr = cf_vsa(S38, B, eps, rule)
        if pr is not None:
            out.append(dict(arm=arm, rule=rule, B=B, cond=cond, design=design, expect=float(np.sum(pr)),
                            var=float(np.sum(np.array(pr) * (1 - np.array(pr)))), n=len(pr)))
    return dict(sid=sid, cells=out)


def closed(stores_f, test_fs, out_f, workers):
    stores = dict(E.load_stores(stores_f))
    res = []
    for f in test_fs.split(","):
        res += json.load(open(f))
    want = collections.defaultdict(list)
    sim = collections.defaultdict(lambda: [0, 0])
    for r in res:
        for x in r["res"]:
            if "cond" not in x or x.get("infeasible") or x["cond"] not in E.IID:
                continue
            arm = r["arm"]
            if arm == "V-plain" and x["rule"] in ("pub", "snt"):
                key = ("V-plain", x["rule"], x["B"], None, x["cond"])
            elif arm == "V-last":
                key = ("V-last", "last", x["B"], None, x["cond"])
            elif arm in ("E-BCH", "E-CAM") and (x.get("dkey") or "auto") == "auto":
                key = (arm, "", x["B"], tuple(x["design"]), x["cond"])
            elif arm == "T-rep":
                key = ("T-rep", "", x["B"], None, x["cond"])
            else:
                continue
            want[r["sid"]].append(key)
            agg = (key[0], key[1], key[2], key[4])
            sim[agg][0] += x["exact"]
            sim[agg][1] += x["n"]
    jobs = [(sid, stores[sid], sorted(set(v), key=str)) for sid, v in want.items()]
    jobs.sort(key=lambda j: -len(j[1]))
    from multiprocessing import Pool
    cf = collections.defaultdict(lambda: [0.0, 0.0, 0])
    with Pool(workers) as pool:
        for i, r in enumerate(pool.imap_unordered(_closed_job, jobs)):
            for c in r["cells"]:
                agg = (c["arm"], c["rule"], c["B"], c["cond"])
                cf[agg][0] += c["expect"]
                cf[agg][1] += c["var"]
                cf[agg][2] += c["n"]
            print("closed %d/%d %s" % (i + 1, len(jobs), r["sid"]), flush=True)
    rows = []
    flagged = 0
    for agg, (e, v, n) in sorted(cf.items(), key=lambda kv: str(kv[0])):
        s_ex, s_n = sim[agg]
        if s_n != n:
            continue
        d = 100 * (s_ex - e) / n
        se = 100 * math.sqrt(max(v, 1e-12)) / n
        flag = abs(d) > max(1.0, 4 * se)
        flagged += flag
        rows.append(dict(arm=agg[0], rule=agg[1], B=agg[2], cond=agg[3], sim=100 * s_ex / n, cf=100 * e / n, diff=d,
                         se=se, flag=bool(flag)))
    json.dump(dict(rows=rows, flagged=flagged, n_cells=len(rows)), open(out_f, "w"), indent=1)
    print("closed form: %d cells, %d flagged" % (len(rows), flagged))


# =============================================================================================== second decoders
def _sp_decode(H, llr, iters):
    """Own flooding sum-product decoder. llr: channel LLR of bit = 0. Returns (hard bits, converged)."""
    Hc = H.tocoo()
    r, c = Hc.row, Hc.col
    M, N = H.shape
    v2c = llr[c].copy()
    hard = (llr < 0).astype(np.uint8)
    for it in range(iters):
        t = np.tanh(np.clip(v2c, -40, 40) / 2)
        t = np.where(np.abs(t) < 1e-15, 1e-15, t)
        # product over each check excluding the edge
        lt = np.log(np.abs(t))
        sg = (t < 0).astype(np.int64)
        sum_lt = np.bincount(r, weights=lt, minlength=M)
        sum_sg = np.bincount(r, weights=sg, minlength=M)
        prod_abs = np.exp(sum_lt[r] - lt)
        prod_sg = 1 - 2 * ((sum_sg[r] - sg) % 2)
        x = np.clip(prod_abs, 0, 1 - 1e-12) * prod_sg
        c2v = 2 * np.arctanh(x)
        tot = llr + np.bincount(c, weights=c2v, minlength=N)
        hard = (tot < 0).astype(np.uint8)
        if not (H @ hard % 2).any():
            return hard, True
        v2c = tot[c] - c2v
    return hard, False


def decoders(stores_f, test_f, out_f, workers):
    """Re-simulates E-BCH and E-LDPC test cells near their N* and compares decoders word by word."""
    import galois
    stores = dict(E.load_stores(stores_f))
    res = json.load(open(test_f))
    picks = []
    import os
    arms = tuple(os.environ.get("B38_DEC_ARMS", "E-BCH,E-LDPC").split(","))
    npick = int(os.environ.get("B38_DEC_N", "12"))
    for r in res:
        if r["arm"] not in arms or not r["sid"].startswith("131072/niah_multikey_2"):
            continue
        for x in r["res"]:
            if "cond" in x and not x.get("infeasible") and x["cond"] in ("k8", "k16", "lines-nat-k8") and \
                    0.5 < x["exact"] / x["n"] < 1.0:
                picks.append((r["sid"], r["arm"], x["B"], x["design"], x["cond"]))
    rng0 = np.random.default_rng(38)
    rng0.shuffle(picks)
    lines = []
    tot = collections.Counter()
    for sid, arm, B, design, cond in picks[:npick]:
        S38 = E.Store38(sid, stores[sid])
        A = E.build_arm(S38, arm, B, design)
        tag = "%s|%s" % (arm, json.dumps(design))
        ch = E.Channel(sid, tag, B, len(A["cells"]), int(B * S38.n))
        sig = ch.apply(A["cells"], cond)
        sums = E.rep_sum(sig, A["lay"]["counts"])
        lay = A["lay"]
        if arm == "E-BCH":
            own = E.bch_code(lay["n"], lay["k"], lay["t"])
            lib = galois.BCH(lay["n"], d=2 * lay["t"] + 1)
            tot["bch_codes"] += 1
            tot["bch_generator_same"] += np.array_equal(np.array(lib.generator_poly.coeffs, np.uint8),
                                                        own.g_low[::-1])
            o = 0
            for s_, cl in zip(lay["sizes"], lay["clen"]):
                sm = sums[o:o + cl]
                tr = A["coded"][o:o + cl]
                tot["bch_encode_same"] += np.array_equal(
                    np.asarray(lib.encode(galois.GF2(tr[:s_]))).astype(np.uint8), tr)
                hard = (sm > 0).astype(np.uint8)
                era = sm == 0
                e = int(((hard != tr) & ~era).sum())
                se = int(era.sum())
                bound = 2 * e + se <= 2 * lay["t"]
                # library and own decoder on both fillings; keep the decodable one with fewer corrections
                res_l, res_o = [], []
                for fill in ((0, 1) if se else (0,)):
                    w = hard.copy()
                    w[era] = fill
                    dm, ne = lib.decode(galois.GF2(w), errors=True)
                    if ne >= 0:
                        res_l.append((int(ne), fill, np.asarray(dm).astype(np.uint8)))
                    ok, mo, no = own.decode(w)
                    if ok:
                        res_o.append((no, fill, mo))
                res_l.sort(key=lambda z: (z[0], z[1]))
                res_o.sort(key=lambda z: (z[0], z[1]))
                ml = res_l[0][2] if res_l else None
                mo = res_o[0][2] if res_o else None
                agree = (ml is None and mo is None) or (ml is not None and mo is not None and np.array_equal(ml, mo))
                tot["bch_words"] += 1
                tot["bch_agree"] += agree
                tot["bch_overloaded"] += not bound
                tot["bch_overloaded_agree"] += (not bound) and agree
                if bound:
                    tot["bch_bound_words"] += 1
                    tot["bch_bound_lib_correct"] += ml is not None and np.array_equal(ml, tr[:s_])
                    tot["bch_bound_own_correct"] += mo is not None and np.array_equal(mo, tr[:s_])
                o += cl
            lines.append("BCH %s B=%g %s (n,k,t)=%s: words %d" % (sid, B, cond, design, len(lay["sizes"])))
        else:
            from ldpc import BpDecoder
            code = E.ira_code(lay["K"], lay["R"])
            c = E.parse_cond(cond)
            e_ = min(max(c["eps"], 1e-6), 0.4999)
            Le = math.log((1 - e_) / e_)
            full = np.zeros(lay["ncw"] * lay["N"])
            full[A["keep"]] = sums * Le
            full[~A["keep"]] = -60.0
            pr = np.clip(1.0 / (1.0 + np.exp(np.minimum(np.abs(full), 60.0))), 1e-15, 0.5)
            hard = (full > 0).astype(np.uint8)
            ties = full == 0
            hard[ties] = np.random.default_rng(1).integers(0, 2, int(ties.sum()))
            bpd = BpDecoder(code["H"], error_rate=0.1, max_iter=E.LDPC_ITERS, bp_method="product_sum",
                            input_vector_type="received_vector")
            coded = np.zeros(lay["ncw"] * lay["N"], np.uint8)
            coded[A["keep"]] = A["coded"]
            for w in range(lay["ncw"]):
                sl = slice(w * lay["N"], (w + 1) * lay["N"])
                bpd.update_channel_probs(pr[sl])
                dl = np.asarray(bpd.decode(hard[sl]), np.uint8)
                llr = np.where(hard[sl] == 0, 1, -1) * np.log((1 - pr[sl]) / pr[sl])
                do, conv = _sp_decode(code["H"], llr, E.LDPC_ITERS)
                tot["ldpc_words"] += 1
                tot["ldpc_info_agree"] += np.array_equal(dl[:lay["K"]], do[:lay["K"]])
                tot["ldpc_lib_correct"] += np.array_equal(dl[:lay["K"]], coded[sl][:lay["K"]])
                tot["ldpc_own_correct"] += np.array_equal(do[:lay["K"]], coded[sl][:lay["K"]])
            lines.append("LDPC %s B=%g %s R=%s: words %d" % (sid, B, cond, design, lay["ncw"]))
        print(lines[-1], dict(tot), flush=True)
    lines.append("totals: " + json.dumps(dict(tot)))
    open(out_f, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines[-1:]))


# =============================================================================================== device model re-read
def _device_job(job):
    sid, pairs, arm, B, design, cond, k = job
    S38 = E.Store38(sid, pairs)
    A = E.build_arm(S38, arm, B, design)
    tag = "%s|%s" % (arm, json.dumps(design))
    sd = E.seed_of(sid, tag, "%.4f" % B, "device")
    sig = I34.signs_over_times(A["cells"].reshape(1, -1).astype(np.float32), sd, float(k), [("30 d", 30 * 86400.0)])
    sig = sig["30 d"].reshape(-1).astype(np.int8)
    rng = np.random.default_rng(E.seed_of(sid, tag, "%.4f" % B, cond, "dec-device"))
    res, _ = E.read_arm(S38, arm, A, sig, cond, rng)
    flips = float((sig != A["cells"]).mean())
    return dict(sid=sid, arm=arm, B=B, cond=cond, k=k, flips=flips,
                res={rule: len(S38.want) - len(E.exact(got, S38.want)) for rule, got in res.items()}, n=len(S38.want))


def device(stores_f, test_f, out_f, workers):
    stores = dict(E.load_stores(stores_f))
    res = json.load(open(test_f))
    kof = {"k4": 4, "k8": 8, "k16": 16, "k32": 32}
    jobs, simtot = [], collections.defaultdict(lambda: [0, 0])
    for r in res:
        if not r["sid"].startswith("131072/niah_multikey_2"):
            continue
        for x in r["res"]:
            if "cond" not in x or x.get("infeasible") or x["cond"] not in ("k8", "k16") or x["B"] not in (
                    E.GRID[E.GRID.index(min(E.GRID, key=lambda g: abs(g - 256.0)))],):
                continue
            if r["arm"] == "V-plain" and x["rule"] != "snt":
                continue
            if r["arm"] in ("E-BCH", "E-CAM") and (x.get("dkey") or "auto") != "auto":
                continue
            jobs.append((r["sid"], stores[r["sid"]], r["arm"], x["B"], x["design"], x["cond"], kof[x["cond"]]))
            simtot[(r["arm"], x["B"], x["cond"])][0] += x["exact"]
            simtot[(r["arm"], x["B"], x["cond"])][1] += x["n"]
    jobs = list(dict.fromkeys((j[0], j[2], j[3], json.dumps(j[4]), j[5]) for j in jobs))
    jobs = [(s, stores[s], a, B, json.loads(d), c, kof[c]) for s, a, B, d, c in jobs]
    from multiprocessing import Pool
    dev = collections.defaultdict(lambda: [0, 0, []])
    with Pool(workers) as pool:
        for r in pool.imap_unordered(_device_job, jobs):
            for rule, ex in r["res"].items():
                if r["arm"] == "V-plain" and rule != "snt":
                    continue
                key = (r["arm"], r["B"], r["cond"])
                dev[key][0] += ex
                dev[key][1] += r["n"]
                dev[key][2].append(r["flips"])
            print("device", r["sid"], r["arm"], r["cond"], r["res"], "flip rate %.4f" % r["flips"], flush=True)
    rows = []
    for key, (ex, n, fl) in sorted(dev.items(), key=lambda kv: str(kv[0])):
        s_ex, s_n = simtot[key]
        p1, p2 = s_ex / s_n, ex / n
        se = 100 * math.sqrt(max(p1 * (1 - p1) / s_n + p2 * (1 - p2) / n, 1e-12))
        rows.append(dict(arm=key[0], B=key[1], cond=key[2], bernoulli=100 * p1, device=100 * p2, diff=100 * (p2 - p1),
                         se=se, flip_rate=float(np.mean(fl)), flag=bool(abs(100 * (p2 - p1)) > max(1.0, 4 * se))))
    json.dump(rows, open(out_f, "w"), indent=1)
    for r_ in rows:
        print(r_)


# =============================================================================================== hand reading
def handread(stores_f, sid, arm, B, design, cond, nshow=4):
    stores = dict(E.load_stores(stores_f))
    S38 = E.Store38(sid, stores[sid])
    design = json.loads(design)
    if arm in ("E-BCH", "E-CAM") and isinstance(design, str):
        cont = E.mphf_content(S38) if arm == "E-BCH" else E.cam_content(S38)
        allk = list(S38.s.count)
        needs = E.mphf_needs(S38, cont, allk) if arm == "E-BCH" else E.cam_needs(S38, cont, allk)
        m = design.split("@")[1] if "@" in design else E.parse_cond(cond)["match"]
        nh = E.NEXT_HARDER.get(m)
        design = list(E.bch_choose(len(cont["bits"]), needs, int(B * S38.n), E.parse_cond(m)["eps"],
                                   E.IID[nh] if nh else None))
    A = E.build_arm(S38, arm, B, design)
    tag = "%s|%s" % (arm, json.dumps(design))
    ch = E.Channel(sid, tag, B, len(A["cells"]), int(B * S38.n))
    sig = ch.apply(A["cells"], cond)
    rng = np.random.default_rng(E.seed_of(sid, tag, "%.4f" % B, cond, "dec"))
    res, diag = E.read_arm(S38, arm, A, sig, cond, rng)
    s = S38.s
    print("== %s %s B=%g design=%s %s: cells %d, cell error rate %.4f, %s" % (
        sid, arm, B, design, cond, len(A["cells"]), float(((sig != A["cells"]) & (sig != 0)).mean()), diag))
    for rule, got in res.items():
        f = E.exact(got, S38.want)
        print("  rule %-7s failed %d/%d" % (rule, len(f), len(S38.want)))
        for a in f[:nshow]:
            k = s.keys[a]
            vocab = s.cd["vocab"]
            line = "    key %r want %s got %s" % (k, [vocab[v] for v in S38.want[a]],
                                                [vocab[v] if v < len(vocab) else v for v in got[a]])
            if arm in ("V-plain", "V-last", "V-cb"):
                rd = E.vsa_read(S38, A["bd"], sig)
                line += " | best sims per j %s, tags %s" % (np.round(rd["sim"][a], 3).tolist(), rd["tag"][a].tolist())
                # true value's similarity at j = 0
                bd = A["bd"]
                d = bd["d"]
                rows = sig[:bd["S"] * d].reshape(bd["S"], d).astype(np.float32)
                X = rows[bd["shard_k"][S38.qk[a]]] * S38.Kc[S38.qk[a], :d] * S38.Pc[0, :d]
                if arm != "V-plain" and s.count[k] == 1:
                    X = X * S38.L[:d]
                tv = float(X @ S38.Vc[S38.want[a][0], :d].astype(np.float32) / d)
                line += " | true value sim at j=0 %.3f" % tv
                # damage on this key's shard vector
                dead = (sig[bd["shard_k"][S38.qk[a]] * d:(bd["shard_k"][S38.qk[a]] + 1) * d] !=
                        A["cells"][bd["shard_k"][S38.qk[a]] * d:(bd["shard_k"][S38.qk[a]] + 1) * d]).mean()
                line += " | its shard's cells read wrong %.3f" % dead
            elif arm in ("E-BCH", "E-CAM", "E-LDPC"):
                cont = A["cont"]
                needs = (E.mphf_needs(S38, cont) if arm != "E-CAM" else E.cam_needs(S38, cont))[a]
                lay = A["lay"]
                if arm == "E-LDPC":
                    cws = np.unique(needs // lay["K"])
                    line += " | needs LDPC codewords %s of %d" % (cws.tolist(), lay["ncw"])
                else:
                    starts = np.r_[0, np.cumsum(lay["sizes"])]
                    cst = np.r_[0, np.cumsum(lay["clen"])]
                    cws = np.unique(np.searchsorted(starts, needs, side="right") - 1)
                    sums = E.rep_sum(sig, lay["counts"])
                    info = []
                    for cw in cws:
                        sm = sums[cst[cw]:cst[cw + 1]]
                        tr = A["coded"][cst[cw]:cst[cw + 1]]
                        era = sm == 0
                        e = int((((sm > 0).astype(np.uint8) != tr) & ~era).sum())
                        info.append("cw%d: %d errors + %d erasures (2e+s=%d vs 2t=%d)" % (
                            cw, e, int(era.sum()), 2 * e + int(era.sum()), 2 * lay["t"]))
                    line += " | " + "; ".join(info)
            print(line)


if __name__ == "__main__":
    mode = sys.argv[1]
    workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 4
    if mode == "closed":
        closed(sys.argv[2], sys.argv[3], sys.argv[4], workers)
    elif mode == "decoders":
        decoders(sys.argv[2], sys.argv[3], sys.argv[4], workers)
    elif mode == "device":
        device(sys.argv[2], sys.argv[3], sys.argv[4], workers)
    elif mode == "handread":
        handread(sys.argv[2], sys.argv[3], sys.argv[4], float(sys.argv[5]), sys.argv[6], sys.argv[7],
                 int(sys.argv[8]) if len(sys.argv) > 8 else 4)
