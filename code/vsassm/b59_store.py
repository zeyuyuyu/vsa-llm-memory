"""B59 part 2 (prereg/B59_maps_all_encoders_protected_full_prereg.md): the protected-store search of B58 on the full
budget grid, with the uncoded store (U = B38's V-last) inside the search. b58_store's evaluation, design lists, choice
rule, censoring and runner are used unchanged.

python3 b59_store.py dev2ldpc OUT.json [--workers N]       # P-LDPC dev pass 2 by B58's registered rule (+-3 steps)
python3 b59_store.py choose OUT.json                       # per-family dev choices + U (B38 dev) + dev union P-dev
python3 b59_store.py plan CHOICE.json PLAN.json            # full-grid test plan, tiers T1-T4, B58 reuse list
python3 b59_store.py run PLAN.json TIER OUT.json [--workers N]
python3 b59_store.py verdict OUT_PREFIX TEST1.json[,...]   # curves, N*, H1-P, G, bootstrap
"""
import collections
import gzip
import json
import os
import sys

import numpy as np

import b38_eq as E
import b58_store as S

GRID = E.GRID
NG = len(GRID)
IIDC = list(E.IID)
DEV_STORES = "b34_out/dev_stores.json.gz"
TEST_STORES = "b34_out/test_stores.json.gz"
B58_DEV = ["results/b58_dev1a.json", "results/b58_dev1b.json", "results/b58_dev2.json"]
B58_TEST = ["results/b58_test_iid1p.json.gz", "results/b58_test_iid2.json.gz"]
FAMS = ("U", "P-BCH", "P-LDPC")
CODE_MIN_GI = 14                                               # 90.5: the smallest budget at which a code fits
FIT_TOL = 1e-5        # compute note 5: b is stored rounded to 4 decimals; b / R == B exactly must count as fitting


def fits(gi, b, R):
    return GRID[gi] * (1 + FIT_TOL) >= b / (R if R else 0.99)


def jload(p):
    return json.load(gzip.open(p, "rt")) if p.endswith(".gz") else json.load(open(p))


def shard(jobs):
    """--shard i/n: every n-th job from i (jobs are sorted by store size, so shards are balanced)."""
    if "--shard" not in sys.argv:
        return jobs
    i, n = map(int, sys.argv[sys.argv.index("--shard") + 1].split("/"))
    return jobs[i::n]


def workers():
    return int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 6


# =============================================================================================== dev pass 2 (P-LDPC)
def dev2ldpc(out):
    ch = S.choose(B58_DEV[:2], out + ".pass1_choice.json")          # pass-1 N* (pass-1 files only)
    only = collections.defaultdict(set)
    for c in IIDC:
        ns = ch["dev_nstar"]["P-LDPC"][c]
        if ns is None:
            gis = range(NG - 4, NG)
        else:
            g = S.gidx(ns)
            gis = range(max(0, g - 3), min(NG, g + 4))
        for gi in gis:
            if gi in S.DEV1_GI:
                continue
            only[("P-LDPC", gi)].add(c)
    need = S.dev_need(sorted({gi for (_, gi) in only}), only=only)
    need = {k: v for k, v in need.items() if k[0] == "P-LDPC"}
    print(len(need), "(design, budget) entries;", sum(len(v) for v in need.values()), "evaluations per store")
    jobs = S.make_jobs(E.load_stores(DEV_STORES), need, S.DEV_CAP)
    jobs = shard(jobs)
    print(len(jobs), "jobs", flush=True)
    S.pool_run(jobs, workers(), out)


# =============================================================================================== choose
def u_dev_curve():
    """U's dev share per (cond, gi) from B38's dev V-last rows (12 dev stores)."""
    acc = collections.defaultdict(lambda: [0, 0])
    sids = collections.defaultdict(set)
    for r in jload("results/b38_dev.json.gz"):
        if r["arm"] != "V-last":
            continue
        for x in r["res"]:
            if "cond" not in x or x.get("infeasible") or x.get("rule") != "last":
                continue
            k = (x["cond"], S.gidx(x["B"]))
            if r["sid"] in sids[k]:
                continue
            sids[k].add(r["sid"])
            acc[k][0] += x["exact"]
            acc[k][1] += x["n"]
    return {k: v[0] / v[1] for k, v in acc.items() if len(sids[k]) == 12}


def choose(out, dev_paths):
    fam = S.choose(dev_paths, out + ".families_tmp.json")             # B58's rule per code family
    agg = S.aggregate(dev_paths)
    U = u_dev_curve()
    ch = dict(fam)
    ch["U_dev_curve"] = {c: {str(gi): round(U[(c, gi)], 6) for gi in range(NG) if (c, gi) in U} for c in IIDC}
    ch["U_dev_nstar"] = {}
    for c in IIDC:
        ns = S.nstar({gi: U[(c, gi)] for gi in range(NG) if (c, gi) in U})
        ch["U_dev_nstar"][c] = None if ns is None else GRID[ns]
    # dev union P-dev: per (cond, gi) the dev-best of U and the family choices dev evaluated at gi
    order = {"U": 0, "P-BCH": 1, "P-LDPC": 2}
    ch["union"] = {}
    ch["union_curve"] = {}
    for c in IIDC:
        nh = E.NEXT_HARDER[c]
        ch["union"][c], ch["union_curve"][c] = {}, {}
        for gi in range(NG):
            cands = []
            if (c, gi) in U:
                f2 = U.get((nh, gi), 0.0) if nh else 0.0
                cands.append((0, -U[(c, gi)], -f2, 0, 0, "U", None, U[(c, gi)]))
            for var in ("P-BCH", "P-LDPC"):
                if str(gi) not in fam["dev_curve"][var][c]:
                    continue
                des = fam["designs"][var][c].get(str(gi))
                if des is None:
                    continue
                dk = json.dumps([round(des[0], 4), des[1]])
                a = agg[var][dk][c].get(gi)
                if a is None or a[1] == 0:
                    continue
                if a[2]:
                    cands.append((1, 0.0, 0.0, order[var], 0, var, des, None))
                    continue
                fr = a[0] / a[1]
                a2 = agg[var][dk][nh].get(gi) if nh else None
                f2 = (a2[0] / a2[1]) if (a2 and a2[1] and not a2[2]) else 0.0
                cands.append((0, -fr, -f2, order[var], 0, var, des, fr))
            if not cands:
                continue
            cands.sort(key=lambda t: t[:5])
            ch["union"][c][str(gi)] = [cands[0][5], cands[0][6]]
            ch["union_curve"][c][str(gi)] = None if cands[0][7] is None else round(cands[0][7], 6)
    ch["union_dev_nstar"] = {}
    for c in IIDC:
        cur = {int(g): v for g, v in ch["union_curve"][c].items()}
        ns = S.nstar(cur)
        ch["union_dev_nstar"][c] = None if ns is None else GRID[ns]
    json.dump(ch, open(out, "w"), indent=1)
    print("dev N*:")
    for var in ("P-BCH", "P-LDPC"):
        print("  %-7s" % var, " ".join("%8s" % ("%.0f" % x if x else "-") for x in
                                       (ch["dev_nstar"][var].get(c) for c in IIDC)))
    print("  %-7s" % "U", " ".join("%8s" % ("%.0f" % x if x else "-") for x in (ch["U_dev_nstar"][c] for c in IIDC)))
    print("  %-7s" % "P-dev", " ".join("%8s" % ("%.0f" % x if x else "-") for x in
                                       (ch["union_dev_nstar"][c] for c in IIDC)))


# =============================================================================================== plan
def b38_u_done():
    """(sid, cond, gi) at which B38's test evaluated U (V-last, rule 'last')."""
    out = set()
    for r in jload("results/b38_test.json.gz"):
        if r["arm"] != "V-last":
            continue
        for x in r["res"]:
            if "cond" in x and not x.get("infeasible") and x.get("rule") == "last":
                out.add((r["sid"], x["cond"], S.gidx(x["B"])))
    return out


def b58_done():
    """Coded test evaluations in B58's test files keyed (sid, var, b, R, gi, cond) -> (row, sid meta)."""
    out = {}
    for p in B58_TEST:
        for r in jload(p):
            for x in r["res"]:
                k = (r["sid"], x["var"], round(x["b"], 4), x["R"], S.gidx(x["B"]), x["cond"])
                out[k] = x
    return out


def plan(choice_path, out):
    ch = json.load(open(choice_path))
    fams = sys.argv[sys.argv.index("--fams") + 1].split(",") if "--fams" in sys.argv else list(FAMS)
    stores = E.load_stores(TEST_STORES)
    sids = [s for s, _ in stores]
    tiers = {"T1": collections.defaultdict(set), "T2": collections.defaultdict(set),
             "T3": collections.defaultdict(set), "T4": collections.defaultdict(set)}
    # T1: U budgets B38's test did not evaluate (any store missing -> run that budget for the missing stores)
    ud = b38_u_done()
    u_need = collections.defaultdict(set)                         # (cond, gi) -> missing sids
    for c in IIDC:
        for gi in range(NG):
            miss = [s for s in sids if (s, c, gi) not in ud]
            if miss and "U" in fams:
                u_need[(c, gi)] = set(miss)
    # T2-T4: both code families, every budget 90.5 ... 4,096 with the family's dev design there
    for c in IIDC:
        for var in ("P-BCH", "P-LDPC"):
            if var not in fams:
                continue
            win = set(S.window(ch, var, c))
            lo = min(win)
            for gi in range(CODE_MIN_GI, NG):
                b, R = S.design_at(ch, var, c, gi)
                if not fits(gi, b, R):
                    continue                                       # no design fits (infeasible; not a result)
                t = "T2" if gi in win else ("T3" if gi < lo else "T4")
                tiers[t][(var, b, R, gi)].add(c)
    done = b58_done()
    reuse = []
    for t, need in tiers.items():
        for (var, b, R, gi), cs in need.items():
            for c in cs:
                for s in sids:
                    k = (s, var, round(b, 4), R, gi, c)
                    if k in done:
                        reuse.append(list(k))
    J = dict(U=[[c, gi, sorted(v)] for (c, gi), v in u_need.items()],
             tiers={t: [[list(k), sorted(v)] for k, v in need.items()] for t, need in tiers.items()},
             reuse=reuse)
    json.dump(J, open(out, "w"))
    print("U missing (cond, budget):", len(u_need), "store-evaluations", sum(len(v) for v in u_need.values()))
    for t, need in tiers.items():
        print(t, len(need), "(family, design, budget) entries;", sum(len(v) for v in need.values()) * len(sids),
              "store-evaluations")
    print("reused from B58's test:", len(reuse))


DMG4 = ["lines-nat-k4", "clust-nat-k4", "lines-nat-k8", "clust-nat-k8"]


def plan_dmg(choice_path, out):
    """T5 (registered 'Damage'): failed lines / stuck clusters, natural layout, k4 / k8; each code family with its
    matched i.i.d. dev designs at the matched i.i.d. dev N* +- 4 grid steps (B58's window); U = B38's V-last rows."""
    ch = json.load(open(choice_path))
    need = collections.defaultdict(set)
    for c in DMG4:
        m = E.parse_cond(c)["match"]
        for var in ("P-BCH", "P-LDPC"):
            for gi in S.window(ch, var, m):
                b, R = S.design_at(ch, var, m, gi)
                if not fits(gi, b, R):
                    continue
                need[(var, b, R, gi)].add(c)
    J = dict(U=[], tiers={"T5": [[list(k), sorted(v)] for k, v in need.items()]}, reuse=[])
    json.dump(J, open(out, "w"))
    print("T5", len(need), "entries;", sum(len(v) for v in need.values()) * 50, "store-evaluations")


def extend_dmg(choice_path, test_paths, out):
    """B38's extension rule (b58_store.extend's code) for the T5 damage conditions, with fits() (compute note 5)."""
    ch = json.load(open(choice_path))
    cur = S.test_curves(test_paths)
    tried = set()
    for p in test_paths:
        for r in json.load(open(p)):
            for x in r["res"]:
                tried.add((x["var"], x["cond"], S.gidx(x["B"])))
    need = collections.defaultdict(set)
    for c in DMG4:
        m = E.parse_cond(c)["match"]
        for var in ("P-BCH", "P-LDPC"):
            curve = cur.get(var, {}).get(c, {})
            if not curve:
                continue
            ev = sorted(curve)
            ns = S.nstar(curve)
            top_ok = curve[max(ev)] is not None and curve[max(ev)] >= E.TARGET
            add = []
            if not top_ok and max(ev) < NG - 1:
                add = list(range(max(ev) + 1, min(NG, max(ev) + 5)))
            elif ns is not None and ns == min(ev) and ns > 0 and GRID[ns - 1] >= S.INFO[0]:
                add = list(range(max(0, ns - 4), ns))
            elif ns is not None and ns > 0 and (ns - 1) not in curve and GRID[ns - 1] >= S.INFO[0]:
                below = [i for i in ev if i < ns]
                add = list(range((max(below) + 1) if below else max(0, ns - 4), ns))
            for gi in add:
                b, R = S.design_at(ch, var, m, gi)
                if not fits(gi, b, R) or (var, c, gi) in tried:
                    continue
                need[(var, b, R, gi)].add(c)
    need = [[list(k), sorted(v)] for k, v in need.items()]
    json.dump(dict(U=[], tiers={"T5": need}, reuse=[]), open(out, "w"))
    print(sum(len(cs) for _, cs in need), "T5 extension (family, condition, budget) additions")


def fixplan(old_full, new_full, old_dmg, new_dmg, out):
    """Entries of the corrected plans (fits()) that the first plans dropped (compute note 5)."""
    def keys(p, tiers):
        J = json.load(open(p))
        return {(k[0], round(k[1], 4), k[2], k[3], c) for t in tiers for k, cs in J["tiers"].get(t, []) for c in cs}
    fix = collections.defaultdict(set)
    for o, n_, tiers in ((old_full, new_full, ("T2", "T3", "T4")), (old_dmg, new_dmg, ("T5",))):
        have = keys(o, tiers)
        for t in tiers:
            for k, cs in json.load(open(n_))["tiers"].get(t, []):
                for c in cs:
                    if (k[0], round(k[1], 4), k[2], k[3], c) not in have:
                        fix[tuple(k)].add(c)
    json.dump(dict(U=[], tiers={"T6": [[list(k), sorted(v)] for k, v in fix.items()]}, reuse=[]), open(out, "w"))
    print("T6", len(fix), "entries;", sum(len(v) for v in fix.values()) * 50, "store-evaluations")


# =============================================================================================== run
def u_job(job):
    return E.run_job(job)


def run(plan_path, tier, out):
    J = json.load(open(plan_path))
    stores = dict(E.load_stores(TEST_STORES))
    if tier == "T1":
        per = collections.defaultdict(lambda: collections.defaultdict(list))
        for c, gi, ss in J["U"]:
            for s in ss:
                per[s][gi].append(c)
        jobs = []
        for s, d in per.items():
            ev = [(GRID[gi], None, sorted(cs)) for gi, cs in sorted(d.items())]
            for i in range(0, len(ev), 6):
                jobs.append(dict(sid=s, pairs=stores[s], arm="V-last", evals=ev[i:i + 6]))
        jobs.sort(key=lambda j: -len(j["pairs"]))
        jobs = shard(jobs)
        print(len(jobs), "U jobs", flush=True)
        from multiprocessing import Pool
        res = []
        with Pool(workers(), maxtasksperchild=4) as pool:
            for r in pool.imap_unordered(u_job, jobs):
                res.append(r)
                print("done U %-28s %5.0fs %d/%d" % (r["sid"], r["sec"], len(res), len(jobs)), flush=True)
        json.dump(res, open(out, "w"))
        return
    need = {tuple(k): set(v) for k, v in J["tiers"][tier]}
    if "--only" in sys.argv:                                     # one family (the other may run from another plan)
        fam = sys.argv[sys.argv.index("--only") + 1]
        need = {k: v for k, v in need.items() if k[0] == fam}
    done = {tuple(k) for k in J["reuse"]}
    jobs = shard(S.make_jobs(list(stores.items()), need, S.TEST_CAP, done=done))
    print(len(jobs), tier, "jobs", flush=True)
    S.pool_run(jobs, workers(), out)


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "dev2ldpc":
        dev2ldpc(a[1])
    elif a[0] == "choose":
        choose(a[1], B58_DEV + ["results/b59_dev2_ldpc.json"])
    elif a[0] == "plan":
        plan(a[1], a[2])
    elif a[0] == "plan_dmg":
        plan_dmg(a[1], a[2])
    elif a[0] == "fixplan":
        fixplan(*a[1:6])
    elif a[0] == "extend_dmg":
        extend_dmg(a[1], [p for p in a[2].split(",") if p], a[3])
    elif a[0] == "run":
        run(a[1], a[2], a[3])
    else:
        raise SystemExit(a[0])
