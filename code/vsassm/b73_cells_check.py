"""B73 equal-cell checks (prereg/B73_cells_prereg.md).

python3 b73_cells_check.py recount STORES FROZEN TEST[,EXT] VERDICT.json OUT.json   # plain-loop scorer + cells + design audit
python3 b73_cells_check.py leak STORES OUT.json                                       # fresh strings vs RULER / dev / B69 sweep
python3 b73_cells_check.py det STORES PLAN[,PLAN2] TEST[,EXT] OUT.json [--workers N]  # determinism re-run of a sample
python3 b73_cells_check.py sample STORES FROZEN TEST[,EXT] VERDICT.json OUT.txt        # sample reading material
"""
import collections
import gzip
import hashlib
import json
import random
import re
import sys

import b38_eq as E
import b73_cells as T
import ruler_vsa as V

GRID, NG = T.GRID, T.NG


def jl(p):
    return T.jl(p)


def rows_of(paths):
    for p in paths:
        for r in jl(p):
            for x in r["rows"]:
                yield r, x


# ----------------------------------------------------------------------------------------------- plain-loop scorer
def plain_nstar(byg, sids):
    """byg: {gi: {sid: (failed, n)}}; plain loops, no shared code."""
    shares = {}
    for gi in byg:
        complete = True
        f = 0
        n = 0
        for s in sids:
            if s not in byg[gi]:
                complete = False
                break
            f = f + byg[gi][s][0]
            n = n + byg[gi][s][1]
        if complete and n > 0:
            if f > n:
                f = n
            shares[gi] = 1.0 - f / n
    gis = sorted(shares)
    best = NG
    i = len(gis) - 1
    while i >= 0:
        if shares[gis[i]] >= 0.995:
            best = gis[i]
            i = i - 1
        else:
            break
    return best, shares


def recount(stores_path, frozen_path, test_paths, verdict_path, out_path):
    fz = json.load(open(frozen_path))
    stores = E.load_stores(stores_path)
    npairs = {s: len(p) for s, p in stores}
    sids = [s for s, _ in stores]
    J = json.load(open(verdict_path))
    md5 = lambda p: hashlib.md5(open(p, "rb").read()).hexdigest()  # noqa: E731
    out = dict(code_md5_equal={p: md5(p) == h for p, h in fz["code"].items()}, rows=0, infeasible=0, over=0,
               off_grid=0, exact_mismatch=0, design_mismatch=0, budget_mismatch=0, nstar={}, nstar_diff=0)
    byg = collections.defaultdict(lambda: collections.defaultdict(dict))
    for r, x in rows_of(test_paths):
        out["rows"] += 1
        if x.get("infeasible"):
            out["infeasible"] += 1
            continue
        B = x["B"]
        gi = min(range(NG), key=lambda i: abs(GRID[i] - B))
        if abs(GRID[gi] - B) > 1e-9:
            out["off_grid"] += 1
        cap = int(B * npairs[r["sid"]])
        if x["cells"] > cap:
            out["over"] += 1
        if x.get("budget") is not None and x["budget"] != cap:
            out["budget_mismatch"] += 1
        if x["exact"] != x["n"] - len(x["failed"]):
            out["exact_mismatch"] += 1
        fd = fz["mems"][r["mem"]][x["cond"]]["designs"][str(gi)]
        want = fd if (fd is None or fd == "auto") else fd["design"]
        if json.dumps(r["design"]) != json.dumps(want):
            out["design_mismatch"] += 1
            continue
        byg[(r["mem"], x["cond"])][gi][r["sid"]] = (len(x["failed"]), x["n"])
    for m in T.MEMS:
        for c in T.CONDS:
            ns, sh = plain_nstar(byg.get((m, c), {}), sids)
            mine = None if ns >= NG else GRID[ns]
            theirs = J["nstar"]["%s|%s" % (m, c)]
            same = (mine is None and theirs is None) or (mine is not None and theirs is not None and
                                                         abs(mine - theirs) < 1e-9)
            out["nstar"]["%s|%s" % (m, c)] = [mine, theirs, same]
            out["nstar_diff"] += not same
    for agg, arms in (("static", T.STATIC), ("insert", T.INSERT)):
        for c in T.CONDS:
            vals = [out["nstar"]["%s|%s" % (a, c)][0] for a in arms]
            vals = [v for v in vals if v is not None]
            mine = min(vals) if vals else None
            theirs = J["nstar"]["%s|%s" % (agg, c)]
            same = mine == theirs or (mine is not None and theirs is not None and abs(mine - theirs) < 1e-9)
            out["nstar"]["%s|%s" % (agg, c)] = [mine, theirs, same]
            out["nstar_diff"] += not same
    json.dump(out, open(out_path, "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "nstar"}, indent=0))
    print("N* recount differences:", out["nstar_diff"], "of", len(out["nstar"]))


# ----------------------------------------------------------------------------------------------- leakage
def norm(s):
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def leak(stores_path, out_path):
    fresh = E.load_stores(stores_path)
    fk = {k for _, p in fresh for k, _, _ in p}
    fv = {v for _, p in fresh for _, v, _ in p}
    ref = {}
    for name, path in (("ruler_test", "b34_out/test_stores.json.gz"), ("dev", "b34_out/dev_stores.json.gz")):
        st = E.load_stores(path)
        ref[name] = ({k for _, p in st for k, _, _ in p}, {v for _, p in st for _, v, _ in p})
    import b69_vocab as B69
    for split in ("dev", "test"):
        path = "b69_out/sweep_%s.json.gz" % split
        try:
            st = B69.load_sweep(path)
        except FileNotFoundError:
            continue
        ref["b69_sweep_" + split] = ({k for _, p, _ in st for k, _, _ in p},
                                     {v for _, p, x in st for v in [q[1] for q in p] + list(x)})
    out = dict(fresh_keys=len(fk), fresh_values=len(fv), overlaps={}, normalized_overlaps={})
    fkn, fvn = {norm(k) for k in fk}, {norm(v) for v in fv}
    for name, (rk, rv) in ref.items():
        allr = rk | rv
        out["overlaps"][name] = dict(ref_keys=len(rk), ref_values=len(rv), fresh_keys_in_ref=len(fk & allr),
                                     fresh_values_in_ref=len(fv & allr),
                                     examples=sorted((fk | fv) & allr)[:10])
        alln = {norm(s) for s in allr}
        out["normalized_overlaps"][name] = dict(fresh_keys=len(fkn & alln), fresh_values=len(fvn & alln),
                                                examples=sorted((fkn | fvn) & alln)[:10])
    # within the fresh set: keys of different stores may coincide (independent draws); report
    per = collections.Counter(k for _, p in fresh for k in {q[0] for q in p})
    out["fresh_keys_in_several_stores"] = sum(1 for v in per.values() if v > 1)
    json.dump(out, open(out_path, "w"), indent=1)
    print(json.dumps(out, indent=1))


# ----------------------------------------------------------------------------------------------- determinism
def det(stores_path, plan_paths, test_paths, out_path, workers):
    stores = E.load_stores(stores_path)
    jobs = []
    for p in plan_paths:
        jobs += T.make_jobs(stores, T.load_plan(p))
    rng = random.Random("b73cells-det")
    k = max(20, -(-len(jobs) * 5 // 100))
    pick = rng.sample(range(len(jobs)), k)
    have = {j["mem"] for j in (jobs[i] for i in pick)}
    for m in T.MEMS:                                             # every memory covered
        if m not in have:
            pick.append(rng.choice([i for i, j in enumerate(jobs) if j["mem"] == m]))
    sample = [jobs[i] for i in pick]
    orig = {}
    for r, x in rows_of(test_paths):
        orig[(r["sid"], r["mem"], json.dumps(r["design"]), round(x["B"], 4), x["cond"])] = x
    from multiprocessing import Pool
    with Pool(workers) as pool:
        res = list(pool.imap_unordered(T.run_one, sample))
    n = same = missing = 0
    diffs = []
    for r in res:
        for x in r["rows"]:
            key = (r["sid"], r["mem"], json.dumps(r["design"]), round(x["B"], 4), x["cond"])
            o = orig.get(key)
            n += 1
            if o is None:
                missing += 1
                continue
            a = (bool(x.get("infeasible")), x.get("failed"), x.get("cells"))
            b = (bool(o.get("infeasible")), o.get("failed"), o.get("cells"))
            if a == b:
                same += 1
            else:
                diffs.append(list(key))
    out = dict(jobs_sampled=len(sample), jobs_total=len(jobs), evaluations=n, identical=same, missing=missing,
               differing=diffs[:50], mems=dict(collections.Counter(j["mem"] for j in sample)))
    json.dump(out, open(out_path, "w"), indent=1)
    print(json.dumps(out, indent=1))


# ----------------------------------------------------------------------------------------------- sample reading
def probed(pairs, nk=T.NPROBE):
    table = V.table_of(pairs)
    keys = list(table)
    if len(keys) > nk:
        keys = keys[:: max(1, len(keys) // nk)][:nk]
    return keys, table


def sample(stores_path, frozen_path, test_paths, verdict_path, out_path):
    fz = json.load(open(frozen_path))
    stores = dict(E.load_stores(stores_path))
    J = json.load(open(verdict_path))
    C = collections.defaultdict(lambda: collections.defaultdict(dict))
    for r, x in rows_of(test_paths):
        if x.get("infeasible"):
            continue
        gi = T.gidx(x["B"])
        if json.dumps(r["design"]) != json.dumps(T.design_of(fz, r["mem"], x["cond"], gi)):
            continue
        C[(r["mem"], x["cond"])][gi][r["sid"]] = x["failed"]
    L = ["# B73 equal-cell sample reading material: at k8, the grid step below each memory's fresh N*, which probed keys",
         "# fail (family, store, key, number of stored values, the values).", ""]
    rng = random.Random("b73cells-sample")
    for m in T.MEMS:
        ns = J["nstar"]["%s|k8" % m]
        if ns is None:
            L.append("%s: N* not reached at k8" % m)
            continue
        gi = T.gidx(ns) - 1
        d = C[(m, "k8")].get(gi)
        if not d:
            L.append("%s: budget %s not evaluated" % (m, T.fmt(gi)))
            continue
        fam = collections.Counter()
        multi = collections.Counter()
        ex = []
        tot = 0
        for sid, failed in sorted(d.items()):
            keys, table = probed(stores[sid])
            tot += len(keys)
            for i in failed:
                k = keys[i]
                fam[T.family(sid)] += 1
                multi[len(table[k])] += 1
                ex.append((sid, k, len(table[k]), table[k]))
        L.append("%s at %s cells per pair (one step below N* %s): %d failed of %d probed; by family %s; by number of "
                 "values %s" % (m, T.fmt(gi), T.fmt(T.gidx(ns)), sum(fam.values()), tot, dict(fam), dict(multi)))
        for e in rng.sample(ex, min(8, len(ex))):
            L.append("    %-22s key %-30r values %d %r" % e)
        L.append("")
    open(out_path, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    a = sys.argv
    w = int(a[a.index("--workers") + 1]) if "--workers" in a else 8
    if a[1] == "recount":
        recount(a[2], a[3], a[4].split(","), a[5], a[6])
    elif a[1] == "leak":
        leak(a[2], a[3])
    elif a[1] == "det":
        det(a[2], a[3].split(","), a[4].split(","), a[5], w)
    elif a[1] == "sample":
        sample(a[2], a[3], a[4].split(","), a[5], a[6])
    else:
        raise SystemExit(a[1])
