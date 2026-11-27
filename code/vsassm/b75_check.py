"""B75 checks (prereg/B75_strongest_options_prereg.md, 'Checks'). Independent of b69_vocab / b75_verdict scoring code.

python3 b75_check.py recount CHOICE.json VERDICT.json TEST.json[,..] STORES.json.gz PANEL OUT.json   # plain loops + cells
python3 b75_check.py detjobs CHOICE.json PLAN.json[,..] STORES.json.gz OUT_JOBS.json [--frac 0.05]
python3 b75_check.py detrun STORES.json.gz JOBS.json OUT.json [--workers N]                           # (box)
python3 b75_check.py detcmp DET.json TEST.json[,..] OUT.json
python3 b75_check.py leak CHOICE.json TEST.json[,..] OUT.json [--scramble STORES.json.gz]
python3 b75_check.py sample CHOICE.json VERDICT.json TEST.json[,..] STORES.json.gz OUT.txt
"""
import collections
import gzip
import inspect
import json
import math
import random
import re
import sys

GRID = [float(2 ** (i / 4)) for i in range(12, 49)]
NG = len(GRID)
TARGET = 0.995


def jl(p):
    return json.load(gzip.open(p, "rt")) if p.endswith(".gz") else json.load(open(p))


def gi_of(B):
    best, bi = None, None
    for i, g in enumerate(GRID):
        if best is None or abs(g - B) < best:
            best, bi = abs(g - B), i
    return bi


def design_at(ch, nm, c, gi):
    des = ch["designs"][nm][c]
    if str(gi) in des:
        return des[str(gi)]
    ks = [int(g) for g in des]
    best = None
    for g in ks:
        key = (abs(g - gi), -g)
        if best is None or key < best[0]:
            best = (key, g)
    return des[str(best[1])]


def stores_meta(path):
    meta = {}
    for sid, pairs in jl(path):
        vals = []
        seen = set()
        for k, v, j in pairs:
            if v not in seen:
                seen.add(v)
                vals.append(v)
        nv = len(vals)
        maxj = max(int(j) for _, _, j in pairs)
        b = max(1, math.ceil(math.log2(nv))) if nv > 1 else 1
        meta[sid] = dict(n=len(pairs), nv=nv, maxj=maxj, P=(1 if maxj > 0 else 0) + b)
    return meta


def recount(choice_path, verdict_path, test_paths, stores_path, panel, out_path):
    ch = json.load(open(choice_path))
    V = json.load(open(verdict_path))["panels"][panel]["nstar"]
    meta = stores_meta(stores_path)
    sids = sorted(meta)
    cur = collections.defaultdict(lambda: collections.defaultdict(dict))
    nrows = bad_exact = over = offgrid = feas = infeas = bad_L = nO2 = 0
    for p in test_paths:
        for r in jl(p):
            n = r["n_pairs"]
            for x in r["rows"]:
                nrows += 1
                if x.get("infeasible"):
                    infeas += 1
                    continue
                feas += 1
                gi = gi_of(x["B"])
                if abs(GRID[gi] - x["B"]) > 1e-9:
                    offgrid += 1
                if x["cells"] > int(x["B"] * n):
                    over += 1
                if x["exact"] != x["n"] - len(x["failed"]):
                    bad_exact += 1
                if x["mem"] in ("O2-BCH", "O2-LDPC"):
                    nO2 += 1
                    g, ss = x["design"][0], x["design"][1]
                    m = meta[r["sid"]]
                    P = m["P"]
                    w = (12 + 4 * (g - 4) if ss else 4 * g) + 4 * P
                    hb = (math.ceil(math.log2(m["n"])) if m["n"] > 1 else 0) + 12
                    mm = max(1, math.ceil(m["n"] / (4 * 0.95)))
                    mu = 4 * 0.95 * m["n"] / ((1 << g) - 1)
                    cap = int(math.ceil(mu + 3 * math.sqrt(mu))) + 1
                    if x["L"] != mm * w + cap * (hb + P) or x["m"] != mm or x["cap"] != cap:
                        bad_L += 1
                nm = "%s|%s" % (x["mem"], x["acct"])
                if x["design"] != design_at(ch, nm, x["cond"], gi):
                    continue
                cur[(nm, x["cond"])][gi][r["sid"]] = (len(x["failed"]), x["n"])
    res, diffs = {}, 0
    for (nm, c), d in sorted(cur.items()):
        share = {}
        for gi in d:
            ok = True
            f = t = 0
            for s in sids:
                if s not in d[gi]:
                    ok = False
                    break
                f += d[gi][s][0]
                t += d[gi][s][1]
            if ok:
                share[gi] = 1 - min(f, t) / t
        ns = None
        for gi in sorted(share, reverse=True):
            if share[gi] >= TARGET:
                ns = gi
            else:
                break
        mine = None if ns is None else GRID[ns]
        theirs = V.get("%s|%s" % (nm, c))
        same = (mine is None and theirs is None) or (mine is not None and theirs is not None and abs(mine - theirs) < 1e-6)
        diffs += not same
        res["%s|%s" % (nm, c)] = [mine, theirs, same]
    out = dict(rows=nrows, feasible=feas, infeasible=infeas, exact_mismatch=bad_exact, over_budget=over,
               off_grid=offgrid, o2_rows=nO2, o2_bits_mismatch=bad_L, nstar=res, nstar_differences=diffs)
    json.dump(out, open(out_path, "w"), indent=1)
    print("recount (%s): %d rows (%d feasible, %d infeasible); exact-count mismatches %d; over budget %d; off grid %d;"
          " O2 rows %d with logical-bit mismatches %d; %d pooled N* recomputed, %d differences" % (
              panel, nrows, feas, infeas, bad_exact, over, offgrid, nO2, bad_L, len(res), diffs))
    for k, v in res.items():
        if not v[2]:
            print("  DIFF", k, v)
    return out


# ------------------------------------------------------------------------------------------------ determinism
def detjobs(choice_path, plan_paths, stores_path, out_path, frac=0.05):
    import b38_eq as E
    import b75_strong as S
    stores = E.load_stores(stores_path)
    jobs = []
    for p in plan_paths:
        jobs += S.make_jobs(stores, S.load_need(p))
    rng = random.Random("b75-det")
    by = collections.defaultdict(list)
    for j in jobs:
        by[j["mem"]].append(j)
    k = max(20, int(math.ceil(frac * len(jobs))))
    pick = []
    for mem, js in sorted(by.items()):                                   # every memory, then proportional
        pick.append(rng.choice(js))
    rest = [j for j in jobs if j not in pick]
    pick += rng.sample(rest, max(0, k - len(pick)))
    spec = [dict(sid=j["sid"], mem=j["mem"], par=j["par"], evals=j["evals"], accts=j["accts"]) for j in pick]
    json.dump(spec, open(out_path, "w"))
    print("determinism sample: %d of %d jobs (%s)" % (len(spec), len(jobs), dict(collections.Counter(
        s["mem"] for s in spec))))


def _run(job):
    import b75_strong as S
    return S.run_job(job)


def detrun(stores_path, jobs_path, out_path, workers=6):
    import b38_eq as E
    from multiprocessing import Pool
    st = dict(E.load_stores(stores_path))
    spec = json.load(open(jobs_path))
    jobs = [dict(s, pairs=st[s["sid"]]) for s in spec]
    with Pool(workers, maxtasksperchild=4) as pool:
        done = list(pool.imap_unordered(_run, jobs, chunksize=1))
    json.dump(done, open(out_path, "w"))
    print("determinism re-run: %d jobs" % len(done))


def _key(r, x):
    return (r["sid"], x["mem"], x["acct"], json.dumps(x["design"]), round(x["B"], 6), x["cond"])


def detcmp(det_path, test_paths, out_path):
    T = {}
    for p in test_paths:
        for r in jl(p):
            for x in r["rows"]:
                T[_key(r, x)] = None if x.get("infeasible") else x["failed"]
    same = diff = missing = 0
    bad = []
    for r in jl(det_path):
        for x in r["rows"]:
            k = _key(r, x)
            if k not in T:
                missing += 1
                continue
            a = None if x.get("infeasible") else x["failed"]
            if a == T[k]:
                same += 1
            else:
                diff += 1
                bad.append(list(k))
    out = dict(identical=same, different=diff, missing=missing, examples=bad[:20])
    json.dump(out, open(out_path, "w"), indent=1)
    print("determinism: %d rows identical, %d different, %d not in the test output" % (same, diff, missing))
    return out


# ------------------------------------------------------------------------------------------------ leakage
DECODERS = ("uj_decode", "_decode_slots", "_sigma_shard", "_gtable", "sbcj_decode", "_sbc_probe", "_sbc_items",
            "_sbc_ac")


def leak(choice_path, test_paths, out_path, stores_path=None):
    import b75_strong as S
    out = {}
    # (iii) source audit: decode functions reference no stored value
    pat = re.compile(r"\b(pv|rv|want|pairs|S38)\b")
    src = {}
    for nm in DECODERS:
        f = getattr(S, nm)
        src[nm] = inspect.getsource(f)
    src["_all_slots"] = inspect.getsource(S._all_slots.py_func)
    src["CuckooStash.read"] = inspect.getsource(S.CuckooStash.read)
    src["CuckooStash.parse"] = inspect.getsource(S.CuckooStash.parse)
    hits = {}
    for nm, s in src.items():
        body = s.split(":", 1)[1] if nm.startswith("CuckooStash") else s
        h = sorted(set(pat.findall(body)))
        if nm.startswith("CuckooStash"):
            h = [t for t in h if t not in ("S38",)]            # read(S38, bits): uses S38.s.keys (queries) and decode_payload
            if "want" in body or ".pv" in body or ".rv" in body:
                h.append("stored-value reference")
        hits[nm] = h
    out["source_audit"] = hits
    # (i) T = 0 / (inf, 0) rows equal the comparators' registered counts at the same (store, budget, condition)
    import b75_verdict as VV
    C, sids = VV.ruler_comparators()
    ch61 = json.load(open("results/b61_choice.json"))
    import b61_hd as H
    eq = collections.Counter()
    for p in test_paths:
        for r in jl(p):
            for x in r["rows"]:
                if x.get("infeasible") or x["acct"] != "free":
                    continue
                gi = gi_of(x["B"])
                if (x["mem"] == "U-J" and x["design"] == [0]) or (x["mem"] == "U-LS" and x["design"] == [None, 0]):
                    ref = C.get(("U", x["cond"]), {}).get(gi, {}).get(r["sid"])
                    if ref is not None:
                        eq["%s same" % x["mem"] if ref[0] == len(x["failed"]) else "%s DIFFERENT" % x["mem"]] += 1
                if x["mem"] == "H-SBC-J" and x["design"][1] == 0:
                    if H.design_at(ch61, "H-SBC", x["cond"], gi) == x["design"][0]:
                        ref = C.get(("H-SBC", x["cond"]), {}).get(gi, {}).get(r["sid"])
                        if ref is not None:
                            eq["H-SBC-J same" if ref[0] == len(x["failed"]) else "H-SBC-J DIFFERENT"] += 1
    out["incumbent_rows_vs_registered"] = dict(eq)
    # (ii) scrambled stored values after the write: identical decoded lists
    if stores_path:
        import numpy as np
        import b38_eq as E
        st = E.load_stores(stores_path)
        rng = random.Random("b75-leak")
        cases = []
        small = [s for s in st if 1000 < len(s[1]) < 2000]
        for sid, pairs in rng.sample(small, 2):
            for B, c in ((32, "k1"), (64, "k8")):
                S38 = E.Store38(sid, pairs)
                bd = E.vsa_build(S38, B, "last")
                chn = E.Channel(S38.sid, "V-last|null", B, len(bd["cells"]), int(B * S38.n))
                sig = chn.apply(bd["cells"], c)
                Sx, d = bd["S"], bd["d"]
                args = lambda: (sig[:Sx * d].reshape(Sx, d), bd["shard_k"], S38.pk, S38.pj, S38.plast, S38.Kc,  # noqa
                                S38.Pc, S38.Vc, S38.L, S38.qk, S38.J, E.IID[c], [1, 2], S38.vf(d))
                a1 = S.uj_decode(*args())
                S38.pv = np.random.default_rng(1).permutation(S38.pv)
                S38.s.rv = np.random.default_rng(2).permutation(np.asarray(S38.s.rv))
                a2 = S.uj_decode(*args())
                st2 = H.SBCStore(E.Store38(sid, pairs), B / 2, 0.7)
                chs = E.Channel(sid, "H-SBC|0.7000", B / 2, len(st2.cells), st2.total)
                yb = (chs.apply(st2.cells, c) > 0).astype(np.uint8)
                S2 = E.Store38(sid, pairs)
                b1 = S.sbcj_decode(st2, yb, S2.pk, S2.pj, S2.plast, S2.qk, S2.J, E.IID[c], [1, 2])
                S2.pv = np.random.default_rng(3).permutation(S2.pv)
                b2 = S.sbcj_decode(st2, yb, S2.pk, S2.pj, S2.plast, S2.qk, S2.J, E.IID[c], [1, 2])
                cases.append([sid, B, c, a1 == a2, b1 == b2])
        out["scrambled_values"] = cases
    json.dump(out, open(out_path, "w"), indent=1)
    print("leakage:", json.dumps(out)[:2000])
    return out


# ------------------------------------------------------------------------------------------------ sample reading
def sample(choice_path, verdict_path, test_paths, stores_path, out_txt):
    """Failed probed keys one grid step below the k8 N* (k1 for the joint decoders), with the rows' diagnostics."""
    ch = json.load(open(choice_path))
    V = json.load(open(verdict_path))["panels"]["ruler"]["nstar"]
    st = dict((s, p) for s, p in jl(stores_path))
    L = ["# B75 sample reading by the AI-assisted process (not independent): failed probed keys one grid step below",
         "# N* (k8 for U-LS / O2, k1 for the joint decoders); per row its diagnostics.", ""]
    targets = [("U-LS|free", "k8"), ("O2-BCH|free", "k8"), ("O2-LDPC|free", "k8"), ("U-J|free", "k1"),
               ("H-SBC-J|free", "k1"), ("U-J|chg", "k1"), ("H-SBC-J|chg", "k1")]
    rows = collections.defaultdict(list)
    for p in test_paths:
        for r in jl(p):
            for x in r["rows"]:
                if x.get("infeasible"):
                    continue
                nm = "%s|%s" % (x["mem"], x["acct"])
                rows[(nm, x["cond"], gi_of(x["B"]))].append((r, x))
    for nm, c in targets:
        ns = V.get("%s|%s" % (nm, c))
        if ns is None:
            L.append("%s %s: N* not reached; nothing to read" % (nm, c))
            continue
        gi = gi_of(ns) - 1
        d = design_at(ch, nm, c, gi) if gi >= 0 else None
        sel = [(r, x) for r, x in rows.get((nm, c, gi), []) if x["design"] == d]
        nf = sum(len(x["failed"]) for _, x in sel)
        nn = sum(x["n"] for _, x in sel)
        L.append("== %s %s: one step below N* %.0f -> B %.0f, design %s: %d / %d probed keys failed on %d stores" % (
            nm, c, ns, GRID[gi] if gi >= 0 else -1, json.dumps(d), nf, nn, len(sel)))
        shown = 0
        for r, x in sorted(sel, key=lambda t: -len(t[1]["failed"])):
            if not x["failed"]:
                continue
            pairs = st[r["sid"]]
            keys = list(dict.fromkeys(k for k, _, _ in pairs))
            cnt = collections.Counter(k for k, _, _ in pairs)
            L.append("  %-40s n %5d failed %3d diag %s key_cells %s" % (r["sid"][-40:], r["n_pairs"], len(x["failed"]),
                                                                       json.dumps(x.get("diag")), x.get("key_cells")))
            shown += 1
            if shown >= 6:
                break
        L.append("")
    open(out_txt, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


def classify(got, want):
    """'wrong value at j' / 'stopped early' (LAST decided too soon) / 'did not stop' (LAST missed) / 'other'."""
    for j in range(min(len(got), len(want))):
        if got[j] != want[j]:
            return "wrong value at j=%d" % j
    if len(got) < len(want):
        return "stopped early"
    if len(got) > len(want):
        return "did not stop"
    return "other"


def detail(spec_json, stores_path, out_txt):
    """Re-run a few failing evaluations locally and classify each failed probed key (sample reading aid).
    spec: [[sid, mem, acct, design, B, cond], ...]."""
    import b38_eq as E
    import b75_strong as S
    st = dict(E.load_stores(stores_path))
    L = []
    for sid, mem, acct, des, B, c in json.loads(spec_json):
        S38 = E.Store38(sid, st[sid])
        if mem == "U-LS":
            bd = S.uls_build(S38, B, des[0], des[1])
            chn = E.Channel(sid, "V-last|null", B, len(bd["cells"]), int(B * S38.n))
            got = E.vsa_rule(E.vsa_read(S38, bd, chn.apply(bd["cells"], c)), "last")
        elif mem in ("U-J", "H-SBC-J"):
            import numpy as np
            tot, kc, coll, kl = S.charged_total(S38, B, c, acct)
            Bs = B if acct == "free" else (tot + 0.5) / S38.n
            T = des[-1]
            if mem == "U-J":
                bd = E.vsa_build(S38, Bs, "last")
                chn = E.Channel(sid, "V-last|null", Bs, len(bd["cells"]), tot)
                sig = chn.apply(bd["cells"], c)
                if T == 0:
                    got = E.vsa_rule(E.vsa_read(S38, bd, sig), "last")
                else:
                    Sx, d = bd["S"], bd["d"]
                    got = S.uj_decode(sig[:Sx * d].reshape(Sx, d), bd["shard_k"], S38.pk, S38.pj, S38.plast, S38.Kc,
                                      S38.Pc, S38.Vc, S38.L, S38.qk, S38.J, E.IID[c], [T], S38.vf(d))[T]
            else:
                import b61_hd as H
                stt = H.SBCStore(S38, Bs, des[0])
                chn = E.Channel(sid, "H-SBC|%.4f" % des[0], Bs, len(stt.cells), stt.total)
                yb = (chn.apply(stt.cells, c) > 0).astype(np.uint8)
                if T == 0:
                    g0, _ = H.sbc_read(S38, stt, yb, list(range(len(S38.want))))
                    got = [g0[a] for a in range(len(S38.want))]
                else:
                    got = S.sbcj_decode(stt, yb, S38.pk, S38.pj, S38.plast, S38.qk, S38.J, E.IID[c], [T])[T]
            got = [g if a not in set(coll) else g for a, g in enumerate(got)]
        else:
            continue
        fl = [a for a in range(len(S38.want)) if got[a] != S38.want[a]]
        cnt = collections.Counter(classify(got[a], S38.want[a]) for a in fl)
        L.append("%s %s %s %s B %.0f %s: %d failed of %d; %s" % (sid, mem, acct, json.dumps(des), B, c, len(fl),
                                                                len(S38.want), dict(cnt)))
        for a in fl[:5]:
            L.append("    key %r: %d value(s); wanted %s, read %s (%s)" % (S38.s.keys[a], len(S38.want[a]),
                                                                       S38.want[a], got[a], classify(got[a], S38.want[a])))
    open(out_txt, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    a = sys.argv
    cmd = a[1]
    if cmd == "recount":
        recount(a[2], a[3], a[4].split(","), a[5], a[6], a[7])
    elif cmd == "detjobs":
        detjobs(a[2], a[3].split(","), a[4], a[5], float(a[a.index("--frac") + 1]) if "--frac" in a else 0.05)
    elif cmd == "detrun":
        detrun(a[2], a[3], a[4], int(a[a.index("--workers") + 1]) if "--workers" in a else 6)
    elif cmd == "detcmp":
        detcmp(a[2], a[3].split(","), a[4])
    elif cmd == "leak":
        leak(a[2], a[3].split(","), a[4], a[a.index("--scramble") + 1] if "--scramble" in a else None)
    elif cmd == "sample":
        sample(a[2], a[3], a[4].split(","), a[5], a[6])
    elif cmd == "detail":
        detail(a[2], a[3], a[4])
    else:
        raise SystemExit(cmd)
