"""B58 checks (prereg/B58_protected_store_prereg.md).

python3 b58_check.py scorer  TEST.json[,...] OUT.json      # second scorer by plain loops on a sample of test cells
python3 b58_check.py bits    TEST.json[,...] OUT.json      # bit and cell accounting of every test evaluation
python3 b58_check.py nstar   TEST.json[,...] OUT.json      # N* recomputed by a separately written rule
python3 b58_check.py decagree TEST.json[,...] OUT.json     # PKG vs OWN on P-LDPC test words; P-BCH vs galois
python3 b58_check.py sample  TEST.json[,...] OUT.txt       # sample reading by the AI-assisted process
"""
import collections
import json
import math
import sys

import numpy as np

import b38_eq as E
import b58_bp as NB
import b58_store as P


def load(paths):
    out = []
    for p in paths.split(","):
        out += json.load(open(p))
    return out


def stores():
    return dict(E.load_stores("b34_out/test_stores.json.gz"))


def full_decode(S38, var, b, R, B, cond):
    """The test cell re-run without the shard cache or censoring: every codeword decoded (P-BCH by
    b38_eq.bch_decode_stream itself), returns (decoded bits, PStore, design info)."""
    ps = P.PStore(S38, b)                                       # fresh, not the run's cache
    total = int(B * S38.n)
    pc = E.parse_cond(cond)
    tag = "%s|%s" % (var, json.dumps([round(b, 4), R]))
    if var == "P-LDPC":
        lay = E.ldpc_layout(ps.L, R, total)
        coded, keep = E.ldpc_encode_stream(ps.bits, lay)
    else:
        nh = E.NEXT_HARDER.get(pc["match"])
        des = P.store_bch_choose(ps.L, ps.d, ps.w, total, E.IID[pc["match"]], E.IID[nh] if nh else None)
        lay = E.ecc_layout(ps.L, des[0], des[1], des[2], total)
        coded = E.bch_encode_stream(ps.bits, lay)
    cells = np.repeat((coded.astype(np.int8) * 2 - 1), lay["counts"])
    ch = E.Channel(S38.sid, tag, B, len(cells), total)
    sig = ch.apply(cells, cond)
    sums = E.rep_sum(sig, lay["counts"])
    rng = np.random.default_rng(E.seed_of(S38.sid, tag, "%.4f" % B, cond, "dec"))
    if var == "P-LDPC":
        llr = P.llr0_from_sums(sums, keep, pc["eps"], lay["ncw"], lay["N"]).reshape(lay["ncw"], lay["N"])
        out, conv, _ = NB.sp_decode_batch(E.ira_code(lay["K"], R)["H"], llr, E.LDPC_ITERS)
        dec = out[:, :lay["K"]].ravel()[:ps.L]
    else:
        dec, _ = E.bch_decode_stream(coded, sums, lay, rng)
    return dec.astype(np.uint8), ps, lay, cells, sig


def plain_read(S38, ps, dec):
    """V-last read by explicit loops: per probed key, per position, per candidate value one dot product; the exact
    test from the store's own pairs (value strings in j order), not Store38.want."""
    d = ps.d
    rows = (dec.astype(np.int8) * 2 - 1).reshape(ps.S, d).astype(np.float64)
    vocab = S38.s.cd["vocab"]
    table = collections.defaultdict(dict)
    for k, v, j in S38.s.pairs:
        table[k][j] = v
    Vc = S38.Vc[:, :d].astype(np.float64)
    Lf = S38.L[:d].astype(np.float64)
    failed = []
    for a, key in enumerate(S38.s.keys):
        sh = int(ps.bd["shard_k"][S38.qk[a]])
        X = rows[sh] * S38.Kc[S38.qk[a], :d]
        got = []
        for j in range(S38.J):
            Y = X * S38.Pc[j, :d]
            YL = Y * Lf
            b0, v0, b1, v1 = -np.inf, -1, -np.inf, -1
            for v in range(S38.Nv):
                s0 = float(np.dot(Y, Vc[v])) / d
                s1 = float(np.dot(YL, Vc[v])) / d
                if s0 > b0:
                    b0, v0 = s0, v
                if s1 > b1:
                    b1, v1 = s1, v
            if b1 > b0:
                got.append(vocab[v1])
                break
            got.append(vocab[v0])
        want = [table[key][j] for j in sorted(table[key])]
        if got != want:
            failed.append(a)
    return failed


def pick_cells(res, n_per=2):
    """Uncensored test cells with some failures (preferred) or exact, every variant, i.i.d. and damage, stores of
    every family."""
    fam = lambda s: s.rsplit("/", 1)[0]
    by = collections.defaultdict(list)
    for r in res:
        for x in r["res"]:
            if x.get("infeasible") or x["censored"]:
                continue
            by[(x["var"], fam(r["sid"]))].append((len(x["failed"]) == 0, r["sid"], x))
    rng = np.random.default_rng(58)
    picks = []
    for key, lst in sorted(by.items()):
        lst.sort(key=lambda z: z[0])
        withf = [z for z in lst if not z[0]]
        pool = withf if withf else lst
        for i in rng.permutation(len(pool))[:n_per]:
            picks.append(pool[i])
    return picks


def scorer(test_paths, out_f):
    res = load(test_paths)
    st = stores()
    picks = pick_cells(res, n_per=1)
    rows, ndiff = [], 0
    for _, sid, x in picks:
        S38 = E.Store38(sid, st[sid])
        dec, ps, _, _, _ = full_decode(S38, x["var"], x["b"], x["R"], x["B"], x["cond"])
        f2 = plain_read(S38, ps, dec)
        same = sorted(f2) == sorted(x["failed"])
        ndiff += not same
        rows.append(dict(sid=sid, var=x["var"], b=x["b"], R=x["R"], B=x["B"], cond=x["cond"], run=len(x["failed"]),
                         second=len(f2), same=same))
        print(rows[-1], flush=True)
    json.dump(dict(rows=rows, cells=len(rows), differing=ndiff), open(out_f, "w"), indent=1)
    print("scorer: %d cells, %d differing" % (len(rows), ndiff))


def bits(test_paths, out_f):
    res = load(test_paths)
    st = {s: len(p) for s, p in stores().items()}
    over = info_over = 0
    used = []
    n = 0
    for r in res:
        npairs = st[r["sid"]]
        for x in r["res"]:
            if x.get("infeasible"):
                continue
            n += 1
            over += x["cells"] > int(x["B"] * npairs)
            info_over += x["L"] > int(x["b"] * npairs)
            used.append(x["cells"] / int(x["B"] * npairs))
            assert x["budget"] == int(x["B"] * npairs)
    u = np.array(used)
    out = dict(evaluations=n, cells_over_budget=int(over), info_bits_over_b=int(info_over),
               budget_used_min=float(u.min()), budget_used_median=float(np.median(u)))
    json.dump(out, open(out_f, "w"), indent=1)
    print(out)


def nstar_second(curve):
    """Separately written: ascending scan; N* = first budget from which every evaluated budget is >= 99.5%."""
    keys = sorted(curve)
    for i, g in enumerate(keys):
        if all(curve[h] is not None and curve[h] >= 0.995 for h in keys[i:]):
            return g
    return None


def nstar(test_paths, out_f):
    res = load(test_paths)
    agg = collections.defaultdict(lambda: [0, 0, False])
    for r in res:
        for x in r["res"]:
            if x.get("infeasible"):
                continue
            a = agg[(x["var"], x["cond"], round(x["B"], 3))]
            a[1] += x["n"]
            a[2] |= x["censored"]
            a[0] += 0 if x["censored"] else x["exact"]
    curves = collections.defaultdict(dict)
    for (v, c, B), (e, n, cen) in agg.items():
        curves[(v, c)][B] = None if cen else e / n
    run = P.test_curves(test_paths.split(","))
    mism = 0
    rows = []
    for (v, c), cur in sorted(curves.items()):
        a = nstar_second(cur)
        bgi = P.nstar(run[v][c])
        b = None if bgi is None else round(E.GRID[bgi], 3)
        mism += a != b
        rows.append(dict(var=v, cond=c, second=a, run=b))
    json.dump(dict(rows=rows, mismatches=mism), open(out_f, "w"), indent=1)
    print("N* second rule: %d (variant, condition), %d mismatches" % (len(rows), mism))


def decagree(test_paths, out_f):
    """PKG vs OWN on every codeword of a sample of P-LDPC i.i.d. test cells; galois BCH vs ours on P-BCH words."""
    import galois
    import b58_dec as Dd
    res = load(test_paths)
    st = stores()
    rng0 = np.random.default_rng(58)
    picksL = [(r["sid"], x) for r in res for x in r["res"] if not x.get("infeasible") and x["var"] == "P-LDPC"
              and x["cond"] in ("k8", "k16", "e05") and r["sid"].startswith("32768/")]
    picksB = [(r["sid"], x) for r in res for x in r["res"] if not x.get("infeasible") and x["var"] == "P-BCH"
              and x["cond"] in ("k4", "e05", "k8") and r["sid"].startswith("32768/")]
    tot = collections.Counter()
    for sid, x in [picksL[i] for i in rng0.permutation(len(picksL))[:2]]:
        S38 = E.Store38(sid, st[sid])
        ps = P.PStore(S38, x["b"])
        total = int(x["B"] * S38.n)
        lay = E.ldpc_layout(ps.L, x["R"], total)
        coded, keep = E.ldpc_encode_stream(ps.bits, lay)
        cells = np.repeat((coded.astype(np.int8) * 2 - 1), lay["counts"])
        tag = "%s|%s" % ("P-LDPC", json.dumps([round(x["b"], 4), x["R"]]))
        sig = E.Channel(sid, tag, x["B"], len(cells), total).apply(cells, x["cond"])
        sums = E.rep_sum(sig, lay["counts"])
        e_ = min(max(E.parse_cond(x["cond"])["eps"], 1e-6), 0.4999)
        full = np.zeros(lay["ncw"] * lay["N"])
        full[keep] = sums * math.log((1 - e_) / e_)
        full[~keep] = -60.0
        pr, hard, llr0 = Dd.inputs_from_full(full, np.random.default_rng(1))
        H = E.ira_code(lay["K"], x["R"])["H"]
        bpd = Dd.pkg_decoder(H)
        ob, _, _ = NB.sp_decode_batch(H, llr0.reshape(lay["ncw"], lay["N"]), E.LDPC_ITERS)
        for w in range(lay["ncw"]):
            sl = slice(w * lay["N"], (w + 1) * lay["N"])
            dp, _, _, _ = Dd.pkg_decode(bpd, pr[sl], hard[sl])
            tot["ldpc_words"] += 1
            tot["ldpc_info_agree"] += np.array_equal(dp[:lay["K"]], ob[w, :lay["K"]])
        print("LDPC", sid, x["B"], x["cond"], dict(tot), flush=True)
    for sid, x in [picksB[i] for i in rng0.permutation(len(picksB))[:1]]:
        S38 = E.Store38(sid, st[sid])
        ps = P.PStore(S38, x["b"])
        total = int(x["B"] * S38.n)
        pc = E.parse_cond(x["cond"])
        nh = E.NEXT_HARDER.get(pc["match"])
        des = P.store_bch_choose(ps.L, ps.d, ps.w, total, E.IID[pc["match"]], E.IID[nh] if nh else None)
        lay = E.ecc_layout(ps.L, des[0], des[1], des[2], total)
        coded = E.bch_encode_stream(ps.bits, lay)
        cells = np.repeat((coded.astype(np.int8) * 2 - 1), lay["counts"])
        tag = "%s|%s" % ("P-BCH", json.dumps([round(x["b"], 4), None]))
        sig = E.Channel(sid, tag, x["B"], len(cells), total).apply(cells, x["cond"])
        sums = E.rep_sum(sig, lay["counts"])
        own = E.bch_code(lay["n"], lay["k"], lay["t"])
        lib = galois.BCH(lay["n"], d=2 * lay["t"] + 1)
        o = 0
        for s_, cl in zip(lay["sizes"], lay["clen"]):
            sm, tr = sums[o:o + cl], coded[o:o + cl]
            hard = (sm > 0).astype(np.uint8)
            era = sm == 0
            e = int(((hard != tr) & ~era).sum())
            se = int(era.sum())
            if 2 * e + se <= 2 * lay["t"]:
                tot["bch_bound_words"] += 1
                w = hard.copy()
                w[era] = 0
                ok, mo, _ = own.decode(w)
                dm, ne = lib.decode(galois.GF2(w), errors=True)
                fill_ok = (ne >= 0 and np.array_equal(np.asarray(dm, np.uint8), tr[:s_]))
                if not fill_ok and se:
                    w[era] = 1
                    dm, ne = lib.decode(galois.GF2(w), errors=True)
                    fill_ok = ne >= 0 and np.array_equal(np.asarray(dm, np.uint8), tr[:s_])
                tot["bch_bound_lib_correct"] += fill_ok
            elif tot["bch_over_words"] < 400:
                tot["bch_over_words"] += 1
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
                tot["bch_over_agree"] += (ml is None and mo is None) or (
                    ml is not None and mo is not None and np.array_equal(ml, mo))
            o += cl
        print("BCH", sid, x["B"], x["cond"], des, dict(tot), flush=True)
    json.dump(dict(tot), open(out_f, "w"), indent=1)


def sample(test_paths, out_f):
    """Failed keys one grid step below N* (k8, k16, lines-nat-k8) for P-LDPC and P-BCH, with the cause."""
    res = load(test_paths)
    st = stores()
    run = P.test_curves(test_paths.split(","))
    lines = ["# B58 sample reading by the AI-assisted process (not independent): failed probed keys of the protected "
             "store one grid step below its test N*, with the cause read from the decoded bits and the similarities."]
    for var in P.VARS:
        for cond in ("k8", "k16", "lines-nat-k8"):
            cur = run.get(var, {}).get(cond, {})
            ns = P.nstar(cur)
            below = [g for g in cur if (ns is None or g < ns) and cur[g] is not None]
            if not below:
                lines.append("%s %s: no uncensored budget below N*" % (var, cond))
                continue
            g = max(below)
            B = E.GRID[g]
            cells = [(r["sid"], x) for r in res for x in r["res"] if not x.get("infeasible") and x["var"] == var
                     and x["cond"] == cond and P.gidx(x["B"]) == g and x["failed"] and not x["censored"]]
            lines.append("== %s %s B=%.0f (N* %s): share %.4f, %d stores with failures" % (
                var, cond, B, "-" if ns is None else "%.0f" % E.GRID[ns], cur[g], len(cells)))
            for sid, x in cells[:2]:
                S38 = E.Store38(sid, st[sid])
                dec, ps, lay, _, _ = full_decode(S38, var, x["b"], x["R"], x["B"], cond)
                vocab = S38.s.cd["vocab"]
                d = ps.d
                for a in x["failed"][:2]:
                    sh = int(ps.bd["shard_k"][S38.qk[a]])
                    db = dec[sh * d:(sh + 1) * d]
                    nerr = int((db != ps.bits[sh * d:(sh + 1) * d]).sum())
                    got = ps.read_shard(sh, db, [a])[0]
                    row = (db.astype(np.int8) * 2 - 1).astype(np.float32)
                    X = row * S38.Kc[S38.qk[a], :d] * S38.Pc[0, :d]
                    if S38.s.count[S38.s.keys[a]] == 1:
                        X = X * S38.L[:d]
                    sims = X @ S38.vf(d).T / d
                    tv = float(sims[S38.want[a][0]])
                    win = int(sims.argmax())
                    lines.append("  %s key %r (design b=%.0f %s): want %s got %s | decoded-bit errors in its shard %d "
                                 "of %d | j=0 similarity true value %.3f, best other %.3f (%s)" % (
                                     sid, S38.s.keys[a], x["b"], x["R"] if var == "P-LDPC" else x.get("bch_t"),
                                     [vocab[v] for v in S38.want[a]], [vocab[v] for v in got], nerr, d, tv,
                                     float(np.sort(np.delete(sims, S38.want[a][0]))[-1]),
                                     "true value still best" if win == S38.want[a][0] else "another value wins"))
    open(out_f, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    {"scorer": scorer, "bits": bits, "nstar": nstar, "decagree": decagree, "sample": sample}[sys.argv[1]](
        sys.argv[2], sys.argv[3])
