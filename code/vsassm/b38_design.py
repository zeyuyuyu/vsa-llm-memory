"""B38 design choice on the synthetic dev stores, and the test plan (prereg/B38_ecc_equal_cells_prereg.md).

python3 b38_design.py choose DEV.json CHOICE.json          # designs per (arm, i.i.d. condition, budget) + dev N*
python3 b38_design.py plan STORES.json.gz CHOICE.json PLAN.json [extra-plan...]   # test plan (pass 1)
python3 b38_design.py extend STORES.json.gz CHOICE.json TEST.json PLAN_n.json      # extension pass (unbracketed N*)

Rules (registered):
- Per (arm, i.i.d. condition, budget B) the design with the highest dev accuracy; ties: higher dev accuracy at the next
  harder i.i.d. condition at the same B; then the first in the arm's design list. E-BCH / E-CAM are chosen per store
  by the closed form inside the job (no dev choice). V-plain computes every read rule; V-am's c is chosen like a design.
- Damage conditions use the design of the matched i.i.d. condition (eps = f / 2): the designer knows the overall
  cell-error rate, not where the damage falls.
- N*(arm, condition) = the smallest grid budget B at which the share of probed keys read exactly is >= 99.5% at B and
  at every larger budget evaluated for that arm and condition.
- Test window: dev N* of the matched i.i.d. condition +- 4 grid steps (one octave each way), plus B = 512 (B34's
  budget); if dev never reaches 99.5% (to 4096), the top 5 grid budgets. Extension: if the test N* is not bracketed
  (below 99.5% at the largest budget evaluated, or >= 99.5% at a window bottom above the grid minimum), 4 more grid
  steps in that direction, repeated (b38_box.sh extend-loop) until every N* is bracketed or the grid ends.
- M2b (equal cells across budgets, i.i.d.): every arm at every budget from 8 grid steps below the smallest dev N*
  to 4 steps above the largest (VSA reads, non-HD tables, T-vhdd), so accuracies can be compared at equal cells where
  both sides are under-provisioned.
- M3 (graceful degradation at equal cells, fixed designs): budget B_g = dev N* of the best VSA read at k8; every arm at
  B_g with its dev design for k8, evaluated at the i.i.d. rates M3_EPS and at lines/contiguous damage in the natural
  layout with f = 2 eps.
"""
import collections
import gzip
import json
import math
import sys

import b38_eq as E

GRID = E.GRID
VARIANTS = ["V-pub", "V-Snt", "V-am", "V-last", "V-cb", "T-rep", "T-vhdd", "E-BCH", "E-LDPC", "E-CAM"]
FAMILY = {"V-pub": "V-plain", "V-Snt": "V-plain", "V-am": "V-plain", "V-last": "V-last", "V-cb": "V-cb",
          "T-rep": "T-rep", "T-vhdd": "T-vhdd", "E-BCH": "E-BCH", "E-LDPC": "E-LDPC", "E-CAM": "E-CAM"}
VSA_FREE = ("V-pub", "V-Snt", "V-am", "V-last")              # codebook regenerated from strings (0 cells)
TABLES = ("T-rep", "E-BCH", "E-LDPC", "E-CAM")               # non-HD tables (repetition / BCH / LDPC)


def gidx(B):
    return min(range(len(GRID)), key=lambda i: abs(GRID[i] - B))


def variant_rows(r):
    """Map one result row to (variant, design-key) pairs."""
    arm = r.get("arm")
    for x in r["res"]:
        if "cond" not in x:
            continue
        if arm == "V-plain":
            rule = x["rule"]
            if x.get("infeasible"):
                for v in ("V-pub", "V-Snt"):
                    yield v, None, x
                for c in E.AM_C:
                    yield "V-am", "%.1f" % c, x
                continue
            if rule == "pub":
                yield "V-pub", None, x
            elif rule == "snt":
                yield "V-Snt", None, x
            else:
                yield "V-am", rule.split(":")[1], x
        elif arm in ("E-BCH", "E-CAM"):
            yield arm, x.get("dkey") or "auto", x
        else:
            yield arm, None if x["design"] is None and arm in ("V-last", "T-rep") else json.dumps(x["design"]), x


def aggregate(results):
    """acc[variant][design][cond][grid index] = [exact, n] summed over stores."""
    acc = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(
        lambda: collections.defaultdict(lambda: [0, 0]))))
    for r in results:
        for v, d, x in variant_rows(r):
            a = acc[v][d][x["cond"]][gidx(x["B"])]
            a[0] += x["exact"]
            a[1] += x["n"]
    return acc


def frac(a):
    return a[0] / a[1] if a[1] else 0.0


def nstar(curve):
    """curve: {grid index: share}. Smallest index i with share >= target at i and every larger evaluated index."""
    idx = sorted(curve)
    best = None
    for i in reversed(idx):
        if curve[i] >= E.TARGET:
            best = i
        else:
            break
    return best


def merge_dev(paths, out_path):
    """Merge dev result files: rows of a later file replace every earlier row of the same (store, arm, design rule)
    (E-BCH / E-CAM: the rule "auto"; other arms: the design value) - used for the E-BCH / E-CAM rerun with the
    galois-matched field polynomials - and add new designs (V-cb loads 1, 2)."""
    def dk(arm, x):
        return (x.get("dkey") or "auto") if arm in ("E-BCH", "E-CAM") else json.dumps(x.get("design"))
    groups = collections.OrderedDict()
    meta = {}
    for p in paths:
        fresh = collections.defaultdict(list)
        for r in json.load(open(p)):
            meta[(r["sid"], r["arm"])] = r
            for x in r["res"]:
                fresh[(r["sid"], r["arm"], dk(r["arm"], x))].append(x)
        for key, xs in fresh.items():
            groups[key] = xs                                    # replaces earlier rows of the same key
    by = collections.OrderedDict()
    for (sid, arm, _), xs in groups.items():
        r = meta[(sid, arm)]
        if (sid, arm) not in by:
            by[(sid, arm)] = dict(sid=sid, arm=arm, n_pairs=r["n_pairs"], n_values=r["n_values"], nq=r["nq"], res=[],
                                  sec=0.0)
        by[(sid, arm)]["res"].extend(xs)
    json.dump(list(by.values()), open(out_path, "w"))
    print("merged", sum(len(v) for v in groups.values()), "rows in", len(groups), "groups from", paths)


def choose(dev_path, out_path):
    res = json.load(open(dev_path))
    acc = aggregate(res)
    choice = {"designs": {}, "dev_curve": {}, "dev_nstar": {}}
    for v in VARIANTS:
        if v not in acc:
            continue
        designs = list(acc[v])
        choice["designs"][v] = {}
        choice["dev_curve"][v] = {}
        choice["dev_nstar"][v] = {}
        for c in E.IID:
            nh = E.NEXT_HARDER[c]
            ch, curve = {}, {}
            for gi in range(len(GRID)):
                cands = []
                for o, d in enumerate(designs):
                    a = acc[v][d][c].get(gi)
                    if a is None or a[1] == 0:
                        continue
                    a2 = acc[v][d][nh].get(gi) if nh else None
                    cands.append((-frac(a), -(frac(a2) if a2 else 0.0), o, d, frac(a)))
                if not cands:
                    continue
                cands.sort()
                ch[gi] = cands[0][3]
                curve[gi] = cands[0][4]
            choice["designs"][v][c] = {str(gi): d for gi, d in ch.items()}
            choice["dev_curve"][v][c] = {str(gi): round(s, 6) for gi, s in curve.items()}
            ns = nstar(curve)
            choice["dev_nstar"][v][c] = None if ns is None else GRID[ns]
    # M3 budget: dev N* of the best VSA read at k8
    best = [choice["dev_nstar"][v]["k8"] for v in VSA_FREE if v in choice["dev_nstar"] and
            choice["dev_nstar"][v].get("k8")]
    choice["B_g"] = min(best) if best else None
    json.dump(choice, open(out_path, "w"), indent=1)
    # print a compact dev table
    print("dev N* (cells per stored pair; None = not reached by 4096):")
    print("%-7s" % "", " ".join("%8s" % c for c in E.IID))
    for v in VARIANTS:
        if v in choice["dev_nstar"]:
            print("%-7s" % v, " ".join("%8s" % ("%.0f" % x if x else "-") for x in
                                       (choice["dev_nstar"][v].get(c) for c in E.IID)))
    print("B_g (M3) =", choice["B_g"])


def design_for(choice, v, cond_match, gi):
    d = choice["designs"].get(v, {}).get(cond_match, {}).get(str(gi))
    return d


def window(choice, v, match):
    ns = choice["dev_nstar"].get(v, {}).get(match)
    if ns is None:
        return list(range(len(GRID) - 5, len(GRID)))
    g = gidx(ns)
    return list(range(max(0, g - 4), min(len(GRID), g + 5)))


def common_window(choice, match):
    """M2b (equal cells across budgets, i.i.d. only): every arm at every grid budget from 8 steps below the smallest
    dev N* to 4 steps above the largest dev N* among the VSA reads, the non-HD tables and T-vhdd."""
    ns = [choice["dev_nstar"].get(v, {}).get(match) for v in VSA_FREE + TABLES + ("T-vhdd",)]
    gs = [gidx(x) for x in ns if x]
    lo = max(0, min(gs) - 8) if gs else len(GRID) - 9
    hi = min(len(GRID) - 1, max(gs) + 4) if gs and len(gs) == len(ns) else len(GRID) - 1
    return list(range(lo, hi + 1))


def to_design(v, d):
    """Design key from the choice file -> the job's design value."""
    if v in ("E-BCH", "E-CAM"):
        return "auto"
    if d is None:
        return None
    return json.loads(d) if v not in ("V-am",) else None


def make_jobs(stores, need):
    """Jobs per store and family: 4 (budget, design) entries per job; E-BCH / E-CAM (decoding-heavy at small budgets)
    one entry of at most 8 conditions per job."""
    jobs = []
    for sid, pairs in stores:
        byfam = collections.defaultdict(list)
        for (fam, gi, dj), cs in sorted(need.items()):
            cs = sorted(cs)
            if fam in ("E-BCH", "E-CAM"):
                for i in range(0, len(cs), 8):
                    byfam[fam].append([GRID[gi], json.loads(dj), cs[i:i + 8]])
            else:
                byfam[fam].append([GRID[gi], json.loads(dj), cs])
        for fam, ev in byfam.items():
            step = 1 if fam in ("E-BCH", "E-CAM") else 4
            for i in range(0, len(ev), step):
                jobs.append(dict(sid=sid, arm=fam, evals=ev[i:i + step]))
    return jobs


def plan(stores_path, choice_path, out_path):
    stores = E.load_stores(stores_path)
    choice = json.load(open(choice_path))
    conds = E.all_conditions()
    need = collections.defaultdict(set)                         # (family, gi, design json) -> conds
    for c in conds:
        m = E.parse_cond(c)["match"]
        for v in VARIANTS:
            fam = FAMILY[v]
            extra = set(common_window(choice, m)) if c in E.IID else set()
            for gi in sorted(set(window(choice, v, m)) | {gidx(512.0)} | extra):
                d = design_for(choice, v, m, gi)
                if fam == "V-plain":
                    dj = "null"
                else:
                    dj = json.dumps(to_design(v, d))
                need[(fam, gi, dj)].add(c)
    # M3
    if choice.get("B_g"):
        gi = gidx(choice["B_g"])
        m3 = list(E.M3_EPS) + ["%s-nat-%s" % (g, m) for g in ("lines", "contig") for m in E.M3_EPS]
        for v in VARIANTS:
            fam = FAMILY[v]
            d = design_for(choice, v, "k8", gi)
            dj = "null" if fam == "V-plain" else json.dumps("auto@k8" if v in ("E-BCH", "E-CAM") else to_design(v, d))
            for c in m3:
                need[(fam, gi, dj)].add(c)
    jobs = make_jobs(stores, need)
    json.dump({"jobs": jobs, "B_g": choice.get("B_g")}, open(out_path, "w"))
    print(len(jobs), "jobs;", sum(len(e[2]) for j in jobs for e in j["evals"]), "evaluations")


def test_curves(test_paths):
    res = []
    for p in test_paths:
        res += json.load(open(p))
    return aggregate(res), res


def test_nstar(acc, choice, v, c):
    """Curve of variant v at condition c with the registered designs (V-am: the dev-chosen c per budget)."""
    m = E.parse_cond(c)["match"]
    curve = {}
    for gi in range(len(GRID)):
        if v == "V-am":
            d = design_for(choice, v, m, gi)
            a = acc[v].get(d, {}).get(c, {}).get(gi) if d else None
        elif v in ("E-BCH", "E-CAM"):
            a = acc[v]["auto"].get(c, {}).get(gi)
        elif v in ("V-pub", "V-Snt", "V-last", "T-rep"):
            a = acc[v][None].get(c, {}).get(gi)
        else:
            d = design_for(choice, v, m, gi)
            a = acc[v].get(d, {}).get(c, {}).get(gi) if d else None
        if a and a[1]:
            curve[gi] = frac(a)
    return curve, nstar(curve)


def extend(stores_path, choice_path, test_paths, out_path):
    stores = E.load_stores(stores_path)
    choice = json.load(open(choice_path))
    acc, _ = test_curves(test_paths)
    need = collections.defaultdict(set)
    nadd = 0
    for c in E.all_conditions():
        m = E.parse_cond(c)["match"]
        for v in VARIANTS:
            curve, ns = test_nstar(acc, choice, v, c)
            ev = sorted(curve)
            if not ev:
                continue
            win = sorted(set(window(choice, v, m)))
            top, bot = max(i for i in ev if i in win or True), min(ev)
            add = []
            if curve[max(ev)] < E.TARGET and max(ev) < len(GRID) - 1:
                add = list(range(max(ev) + 1, min(len(GRID), max(ev) + 5)))
            elif ns is not None and ns == bot and bot > 0:
                add = list(range(max(0, bot - 4), bot))
            elif ns is not None and ns > 0 and (ns - 1) not in curve:
                # gap: N* sits above unevaluated budgets (bracketing needs the budget one step below N*); fill every
                # grid budget between the highest evaluated failing budget below N* and N*
                below = [i for i in ev if i < ns]
                add = list(range((max(below) + 1) if below else max(0, ns - 4), ns))
            for gi in add:
                d = design_for(choice, v, m, gi)
                fam = FAMILY[v]
                dj = "null" if fam == "V-plain" else json.dumps(to_design(v, d))
                need[(fam, gi, dj)].add(c)
                nadd += 1
    jobs = make_jobs(stores, need)
    json.dump({"jobs": jobs}, open(out_path, "w"))
    print(nadd, "(variant, condition, budget) additions;", len(jobs), "jobs")


def missing(stores_path, plan_path, test_paths, out_path):
    """Jobs for the (store, family, budget, design, condition) evaluations of a plan that no result file holds yet
    (used when the plan grew, e.g. the M2b budgets, after a pass had started)."""
    stores = E.load_stores(stores_path)
    pl = json.load(open(plan_path))
    have = set()
    for p in test_paths:
        for r in json.load(open(p)):
            for x in r["res"]:
                if "cond" in x:
                    have.add((r["sid"], r["arm"], gidx(x["B"]), x["cond"]))
    need = collections.defaultdict(set)
    n = 0
    for j in pl["jobs"]:
        for B, d, cs in j["evals"]:
            for c in cs:
                if (j["sid"], j["arm"], gidx(B), c) not in have:
                    need[(j["sid"], j["arm"], gidx(B), json.dumps(d))].add(c)
                    n += 1
    jobs = []
    byk = collections.defaultdict(list)
    for (sid, fam, gi, dj), cs in sorted(need.items()):
        byk[(sid, fam)].append([GRID[gi], json.loads(dj), sorted(cs)])
    for (sid, fam), ev in byk.items():
        step = 1 if fam in ("E-BCH", "E-CAM") else 4
        for i in range(0, len(ev), step):
            jobs.append(dict(sid=sid, arm=fam, evals=ev[i:i + step]))
    json.dump({"jobs": jobs}, open(out_path, "w"))
    print(n, "missing evaluations;", len(jobs), "jobs")


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "merge":
        merge_dev(sys.argv[2].split(","), sys.argv[3])
    elif mode == "choose":
        choose(sys.argv[2], sys.argv[3])
    elif mode == "plan":
        plan(sys.argv[2], sys.argv[3], sys.argv[4])
    elif mode == "missing":
        missing(sys.argv[2], sys.argv[3], sys.argv[4].split(","), sys.argv[5])
    elif mode == "extend":
        extend(sys.argv[2], sys.argv[3], sys.argv[4].split(","), sys.argv[5])
