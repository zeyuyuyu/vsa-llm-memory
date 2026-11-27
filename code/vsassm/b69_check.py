"""B69 checks (prereg/B69_vocab_sweep_prereg.md, 'Checks').

python3 b69_check.py c0 OUT.json                                   # x1 has no cell errors (device model, 2e8 cells)
python3 b69_check.py audit CHOICE.json DEV.json[,..] TEST.json[,..] OUT.json        # C1 bits audit (sweep rows)
python3 b69_check.py recount CHOICE.json TEST.json[,..] PARTA.json VERDICTB.json OUT.json   # C2 + C3
python3 b69_check.py cross TEST.json[,..] OUT.json [--n 6]         # C4: re-run sampled test jobs locally
"""
import collections
import gzip
import json
import math
import random
import sys

import numpy as np

GRID = [float(2 ** (i / 4)) for i in range(12, 49)]
NG = len(GRID)
EPS = {"k1": 0.0, "k2": 0.00013175, "k4": 0.0208065, "e05": 0.05, "k8": 0.12263475, "k16": 0.27600645,
       "k32": 0.374429}


def gi_of(B):
    return min(range(NG), key=lambda i: abs(GRID[i] - B))


def jl(p):
    return json.load(gzip.open(p, "rt")) if p.endswith(".gz") else json.load(open(p))


# ------------------------------------------------------------------------------------------------ C0
def _c0_chunk(seed):
    import b34_inmem as I34
    import b25_pcm as P
    rows, cols = 1000, 20000
    bits = np.where(np.random.default_rng(seed).random((rows, cols)) < 0.5, 1.0, -1.0).astype(np.float32)
    times = [("30 d", 30 * 86400.0), ("1 y", 365 * 86400.0)]
    s = I34.signs_over_times(bits, seed, 1.0, times)
    out = {tl: int((s[tl] != bits).sum()) for tl, _ in times}
    # margin: the differential read |G+ - G-| relative to G_ON at the same draws (smallest over the chunk)
    r = np.random.default_rng([seed, 3])
    gp = np.where(bits > 0, P.G_ON, 0.0).astype(np.float32)
    gm = np.where(bits < 0, P.G_ON, 0.0).astype(np.float32)
    nu_p, g1p = P.program(gp, gp.shape, r, 1.0)
    nu_m, g1m = P.program(gm, gm.shape, r, 1.0)
    marg = {}
    for tl, t in times:
        rr = np.random.default_rng([seed, 4, int(t)])
        Gp, _, _ = P.at_time(gp, nu_p, g1p, t, rr, 1.0)
        Gm, _, _ = P.at_time(gm, nu_m, g1m, t, rr, 1.0)
        x = (Gp - Gm) * bits
        marg[tl] = dict(min=float(x.min() / P.G_ON), mean=float(x.mean() / P.G_ON), sd=float(x.std() / P.G_ON))
    return dict(seed=seed, cells=rows * cols, flips=out, margin=marg)


def c0(out_path):
    from multiprocessing import Pool
    with Pool(10) as pool:
        res = pool.map(_c0_chunk, list(range(69000, 69010)))
    tot = sum(r["cells"] for r in res)
    fl = {tl: sum(r["flips"][tl] for r in res) for tl in ("30 d", "1 y")}
    J = dict(chunks=res, cells=tot, flips=fl, upper95={tl: (3.0 / tot if v == 0 else None) for tl, v in fl.items()},
             min_margin={tl: min(r["margin"][tl]["min"] for r in res) for tl in ("30 d", "1 y")},
             z_margin={tl: float(np.mean([r["margin"][tl]["mean"] / r["margin"][tl]["sd"] for r in res]))
                       for tl in ("30 d", "1 y")})
    json.dump(J, open(out_path, "w"), indent=1)
    print(json.dumps({k: v for k, v in J.items() if k != "chunks"}, indent=1))


# ------------------------------------------------------------------------------------------------ C1
def audit(choice_path, dev_paths, test_paths, out_path):
    import b69_vocab as B
    J = dict(over_budget=0, evaluations=0, stores={}, logical={}, vocab_bits={})
    for p in dev_paths + test_paths:
        for r in jl(p):
            N = B.n_of(r["sid"])
            st = J["stores"].setdefault(r["sid"], dict(N=N, nv=set(), nq=set(), n_pairs=set()))
            st["nv"].add(r["nv"])
            st["nq"].add(r["nq"])
            st["n_pairs"].add(r["n_pairs"])
            for x in r["rows"]:
                if x.get("infeasible"):
                    continue
                J["evaluations"] += 1
                if x["cells"] > x["budget"] or x["budget"] > int(x["B"] * r["n_pairs"]):
                    J["over_budget"] += 1
                if x.get("L") is not None and r["mem"] in ("T-rep", "E-BCH", "E-LDPC", "O-BCH", "O-LDPC"):
                    J["logical"].setdefault("%s|%d" % (r["mem"], N), set()).add(round(x["L"] / r["n_pairs"], 3))
                if r["mem"] in ("O-BCH", "O-LDPC") and x.get("L") is not None:
                    f = r["design"][0] if r["mem"] == "O-LDPC" else r["design"]
                    b = math.ceil(math.log2(N))
                    want = r["m"] * 4 * (f + 1 + b)
                    if x["L"] != want:
                        J.setdefault("cuckoo_bits_mismatch", 0)
                        J["cuckoo_bits_mismatch"] = J.get("cuckoo_bits_mismatch", 0) + 1
    bad = []
    for sid, st in J["stores"].items():
        ok = st["nv"] == {st["N"]} and st["nq"] == {300} and st["n_pairs"] == {6002}
        if not ok:
            bad.append(sid)
        for k in ("nv", "nq", "n_pairs"):
            st[k] = sorted(st[k])
    J["stores_bad"] = bad
    J["logical"] = {k: sorted(v) for k, v in J["logical"].items()}
    # vocabulary bits by a second method: characters counted directly (every sweep value is 7 ASCII digits)
    stores = B.load_sweep("b69_out/sweep_test.json.gz") + B.load_sweep("b69_out/sweep_dev.json.gz")
    vb_bad = 0
    for sid, pairs, extra in stores:
        voc = list(dict.fromkeys([v for _, v, _ in pairs] + extra))
        second = sum(len(v) for v in voc) * 8
        first = B.vocab_bits(voc)
        if len(voc) != B.n_of(sid) or first != second or B.vocab_bits(voc, True) != 24 * len(voc):
            vb_bad += 1
    J["vocab_bits_sweep_bad"] = vb_bad
    import b38_eq as E
    tv_bad = 0
    for sid, pairs in E.load_stores("b34_out/test_stores.json.gz"):
        voc = list(dict.fromkeys(v for _, v, _ in pairs))
        lens = [len(v.encode("utf-8")) for v in voc]
        second = 8 * min(len(voc) * max(lens), sum(l + 1 for l in lens))
        J["vocab_bits"][sid] = [B.vocab_bits(voc), second, B.vocab_bits(voc, True), len(pairs)]
        tv_bad += B.vocab_bits(voc) != second
    J["vocab_bits_workload_bad"] = tv_bad
    json.dump(J, open(out_path, "w"), indent=1)
    print("evaluations", J["evaluations"], "over budget", J["over_budget"], "stores bad", bad,
          "cuckoo bits mismatch", J.get("cuckoo_bits_mismatch", 0), "vocab bits bad (sweep / workload)", vb_bad, tv_bad)
    for k in sorted(J["logical"]):
        print("  logical bits per pair %-12s %s" % (k, J["logical"][k][:6]))


# ------------------------------------------------------------------------------------------------ C2 / C3
def own_h2(p):
    return 0.0 if p <= 0 else -p * math.log2(p) - (1 - p) * math.log2(1 - p)


_TAIL = {}


def own_tail(m, p, t):
    """P(Bin(m, p) > t) by explicit summation of the lower part (math.comb, log space)."""
    key = (m, p, t)
    if key in _TAIL:
        return _TAIL[key]
    if p <= 0:
        return 0.0
    lp, lq = math.log(p), math.log1p(-p)
    s = 0.0
    for e in range(0, t + 1):
        s += math.exp(math.lgamma(m + 1) - math.lgamma(e + 1) - math.lgamma(m - e + 1) + e * lp + (m - e) * lq)
    v = max(0.0, 1.0 - s)
    if v < 1e-12:                                         # 1 - s loses precision: sum the upper tail directly
        v = 0.0
        for e in range(t + 1, min(m, t + 400) + 1):
            v += math.exp(math.lgamma(m + 1) - math.lgamma(e + 1) - math.lgamma(m - e + 1) + e * lp + (m - e) * lq)
    _TAIL[key] = v
    return v


def own_rep_err(r, p):
    """P(majority of r copies wrong) + P(tie) / 2."""
    if p <= 0:
        return 0.0
    w = sum(math.comb(r, e) * p ** e * (1 - p) ** (r - e) for e in range(r // 2 + 1, r + 1))
    tie = math.comb(r, r // 2) * p ** (r // 2) * (1 - p) ** (r // 2) if r % 2 == 0 else 0.0
    return w + tie / 2


def own_design_ok(L, eps, des, cells, pfail=0.01):
    """Re-evaluates a vocabulary design independently: P(any string wrong) <= pfail at the given cell count."""
    if des[0] == "uncoded":
        return cells >= L
    if des[0] == "rep":
        r = des[1]
        n1 = cells - (r - 1) * L
        if n1 < 0 or n1 > L:
            return False
        return (L - n1) * own_rep_err(r - 1, eps) * (1 if r > 1 else 0) + n1 * own_rep_err(r, eps) <= pfail + 1e-12 \
            if r > 1 else L * own_rep_err(1, eps) <= pfail
    _, n, k, t, r = des
    ncw = -(-L // k)
    Nc = L + ncw * (n - k)
    extra = cells - Nc * (r - 1)
    a = extra // ncw if r > 1 else 0
    # coded bits: a per codeword with r copies, n - a with r - 1 (r = 1: all one copy); error per coded bit with ties half
    if r == 1:
        pe = own_rep_err(1, eps)
        return cells >= Nc and ncw * own_tail(n, pe, t) <= pfail * (1 + 1e-6)
    p_lo, p_hi = own_rep_err(r - 1, eps), own_rep_err(r, eps)
    # P(fail) = P(E_lo + E_hi > t), E_lo ~ Bin(n - a, p_lo), E_hi ~ Bin(a, p_hi)
    m0, m1 = n - a, a
    ok = 0.0
    for e in range(0, t + 1):
        pe0 = math.exp(math.lgamma(m0 + 1) - math.lgamma(e + 1) - math.lgamma(m0 - e + 1) + e * math.log(max(p_lo, 1e-300))
                       + (m0 - e) * math.log1p(-p_lo)) if m0 >= e else 0.0
        ok += pe0 * (1 - own_tail(m1, p_hi, t - e)) if m1 > 0 else pe0
    return ncw * max(0.0, 1 - ok) <= pfail * (1 + 1e-6)


def own_charged(ns_gi, stores, eps, prot, compact, B):
    """Independent charged requirement: capacity by own formula; bch: B's design re-verified by own tail sums."""
    if ns_gi >= NG:
        return NG
    b = GRID[ns_gi]
    need = []
    for n, voc in stores:
        L = B.vocab_bits(voc, compact)
        if eps <= 0:
            c = L
        elif prot == "cap":
            c = math.ceil(L / (1 - own_h2(eps)))
        else:
            c, des = B.bch_cells(L, eps)
            assert own_design_ok(L, eps, des, c), ("design fails own check", L, eps, des, c)
        need.append((n, c))
    for gi in range(ns_gi, NG):
        if all(int(GRID[gi] * n) - int(b * n) >= c for n, c in need):
            return gi
    return NG


def own_nstar(rows_by_gi, sids):
    """Explicit loops: pooled share per budget evaluated on every store; the smallest budget with >= 99.5% there and
    at every larger evaluated budget."""
    shares = []
    for gi in sorted(rows_by_gi):
        d = rows_by_gi[gi]
        if not all(s in d for s in sids):
            continue
        f = 0
        n = 0
        for s in sids:
            f += d[s][0]
            n += d[s][1]
        shares.append((gi, 1 - f / n))
    ans = NG
    for gi, sh in reversed(shares):
        if sh >= 0.995:
            ans = gi
        else:
            break
    return ans, shares


def recount(choice_path, test_paths, parta_path, verdictb_path, out_path):
    import b69_vocab as B
    ch = json.load(open(choice_path))
    VB = json.load(open(verdictb_path))
    PA = json.load(open(parta_path))
    L = ["# B69 checks C2 (independent recount) and C3 (sample recount)", ""]
    # ---- (B): raw rows -> own selection of the dev design per budget -> own N*
    raw = collections.defaultdict(lambda: collections.defaultdict(dict))
    sids = collections.defaultdict(set)
    for p in test_paths:
        for r in jl(p):
            N = int(r["sid"].split("/")[1][1:])
            sids[N].add(r["sid"])
            for x in r["rows"]:
                if x.get("infeasible"):
                    continue
                gi = gi_of(x["B"])
                key = "%s|%d|%s" % (r["mem"], N, x["cond"])
                if r["mem"] in ("U", "T-rep", "E-BCH"):
                    want = None
                else:
                    d = ch["designs"][key]
                    if str(gi) in d:
                        want = d[str(gi)]
                    else:
                        gs = sorted(int(g) for g in d)
                        best = sorted(gs, key=lambda g: (abs(g - gi), -g))[0]
                        want = d[str(best)]
                if json.dumps(r["design"]) != json.dumps(want):
                    continue
                raw[key][gi][r["sid"]] = (len(x["failed"]), x["n"])
    stores = {s: (p, x) for s, p, x in B.load_sweep("b69_out/sweep_test.json.gz")}
    mism = 0
    tot = 0
    own = {}
    for key, rows in raw.items():
        mem, N, c = key.split("|")
        N = int(N)
        ns, _ = own_nstar(rows, sorted(sids[N]))
        own[(mem, N, c)] = ns
        got = VB["nstar"].get("%d|free|%s|%s" % (N, mem, c))
        tot += 1
        if (got is None) != (ns >= NG) or (got is not None and abs(GRID[ns] - got) > 0.5):
            mism += 1
            L.append("  MISMATCH (B) free %s: own %s verdict %s" % (key, ns, got))
    # charged (B)
    cm = ct = 0
    for (mem, N, c), ns in own.items():
        st = [(len(stores[s][0]), list(dict.fromkeys([v for _, v, _ in stores[s][0]] + stores[s][1])))
              for s in sorted(sids[N])]
        for acct in ("cap-utf8", "bch-utf8", "cap-compact", "bch-compact"):
            prot, fm = acct.split("-")
            g = own_charged(ns, st, EPS[c], prot, fm == "compact", B)
            got = VB["nstar"].get("%d|%s|%s|%s" % (N, acct, mem, c))
            ct += 1
            if (got is None) != (g >= NG) or (got is not None and abs(GRID[g] - got) > 0.5):
                cm += 1
                L.append("  MISMATCH (B) %s %s|%d|%s: own %s verdict %s" % (acct, mem, N, c, g, got))
    L.append("C2 (B): uncharged requirements recounted %d, mismatches %d; charged %d, mismatches %d" % (tot, mism, ct, cm))
    # ---- (A): charged requirements recomputed from the verdict's own (free) N* with own charge code
    import b38_eq as E
    tst = {s: p for s, p in E.load_stores("b34_out/test_stores.json.gz")}
    fam = lambda s: "mk2" if "niah_multikey_2" in s else "mk3" if "niah_multikey_3" in s else "small"
    scopes = {"pooled": list(tst), "mk2": [s for s in tst if fam(s) == "mk2"],
              "mk2-6k": [s for s in tst if fam(s) == "mk2" and len(tst[s]) > 5000],
              "mk3": [s for s in tst if fam(s) == "mk3"], "small": [s for s in tst if fam(s) == "small"]}
    am = at = 0
    for k, v in PA["nstar"].items():
        sc, acct, mem, c = k.split("|")
        if acct == "free" or mem in ("static", "insert"):
            continue
        free = PA["nstar"]["%s|free|%s|%s" % (sc, mem, c)]
        fgi = NG if free is None else gi_of(free)
        prot, fm = acct.split("-")
        st = [(len(tst[s]), list(dict.fromkeys(x[1] for x in tst[s]))) for s in scopes[sc]]
        g = own_charged(fgi, st, EPS[c], prot, fm == "compact", B)
        at += 1
        if (v is None) != (g >= NG) or (v is not None and abs(GRID[g] - v) > 0.5):
            am += 1
            L.append("  MISMATCH (A) %s: own %s part A %s" % (k, g, v))
    L.append("C2 (A): charged requirements recounted %d, mismatches %d" % (at, am))
    # ---- C3: sample recount listing
    rng = random.Random("b69-c3")
    keys = sorted(raw)
    pick = rng.sample(keys, min(8, len(keys)))
    L.append("")
    L.append("C3 sample recount (B, uncharged): pooled failed keys / probed keys per budget around N*")
    for key in pick:
        mem, N, c = key.split("|")
        ns, shares = own_nstar(raw[key], sorted(sids[int(N)]))
        lines = []
        for gi in sorted(raw[key]):
            d = raw[key][gi]
            if all(s in d for s in sids[int(N)]):
                f = sum(d[s][0] for s in sids[int(N)])
                n = sum(d[s][1] for s in sids[int(N)])
                if abs(gi - ns) <= 2 or ns >= NG:
                    lines.append("%.0f:%d/%d" % (GRID[gi], f, n))
        L.append("  %-22s N* %s  %s" % (key, ">4096" if ns >= NG else "%.0f" % GRID[ns], "  ".join(lines)))
    J = dict(B_free=[tot, mism], B_charged=[ct, cm], A_charged=[at, am], sample=pick)
    open(out_path.replace(".json", ".txt"), "w").write("\n".join(L) + "\n")
    json.dump(J, open(out_path, "w"), indent=1)
    print("\n".join(L))


# ------------------------------------------------------------------------------------------------ C4
def cross(test_paths, out_path, n=6):
    import b69_vocab as B
    stores = {s: (p, x) for s, p, x in B.load_sweep("b69_out/sweep_test.json.gz")}
    rows = []
    for p in test_paths:
        rows += jl(p)
    rng = random.Random("b69-c4")
    pick = rng.sample(range(len(rows)), n)
    same = tot = 0
    out = []
    for i in pick:
        r = rows[i]
        ev = collections.OrderedDict()
        for x in r["rows"]:
            ev.setdefault(x["B"], []).append(x["cond"])
        pairs, extra = stores[r["sid"]]
        loc = B.run_job(dict(sid=r["sid"], pairs=pairs, extra=extra, mem=r["mem"], design=r["design"],
                             evals=list(ev.items()), nprobe=300))
        a = {(x["B"], x["cond"]): x.get("failed") for x in r["rows"]}
        b = {(x["B"], x["cond"]): x.get("failed") for x in loc["rows"]}
        for k in a:
            tot += 1
            same += a[k] == b.get(k)
        out.append(dict(sid=r["sid"], mem=r["mem"], design=r["design"], cells=len(a),
                        identical=sum(a[k] == b.get(k) for k in a)))
        print(out[-1], flush=True)
    json.dump(dict(jobs=out, identical=same, total=tot), open(out_path, "w"), indent=1)
    print("C4 identical failed-key lists: %d / %d" % (same, tot))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "c0":
        c0(sys.argv[2])
    elif cmd == "audit":
        audit(sys.argv[2], sys.argv[3].split(","), sys.argv[4].split(","), sys.argv[5])
    elif cmd == "recount":
        recount(sys.argv[2], sys.argv[3].split(","), sys.argv[4], sys.argv[5], sys.argv[6])
    elif cmd == "cross":
        cross(sys.argv[2].split(","), sys.argv[3], int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 6)
