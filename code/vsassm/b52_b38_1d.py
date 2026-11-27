"""B52 part (c) (prereg/B52_robustness_layers_maps_1day_prereg.md): B38's cells-per-fact comparison (N* = the smallest
budget with >= 99.5% of the 4,776 probed RULER keys read exactly, every stored bit on a cell) re-run with the device
model's ONE-DAY bit-error rates instead of its 30-day rates (the model was fitted to about one day of data).

Everything is B38's code (b38_eq.py, b38_design.py, b38_check.py, unchanged); this wrapper replaces, at import time and
before any worker is forked, the rate table and restricts the conditions to the i.i.d. ones:
  E.P30 / E.IID  <- results/b34_ber.json "k|1 d": k1 0, k2 8.2e-6, k4 0.0097645, k8 0.1017177, k16 0.25640955,
                    k32 0.36504285; e05 stays at 5% (a control, not a device rate; it is also k4's "next harder" rate
                    in B38's tie rule and BCH design)
  E.all_conditions <- the i.i.d. conditions only (no localized damage; H1 is an i.i.d. comparison)
Arms: the VSA reads with regenerated codebooks (V-plain = V-pub / V-Snt / V-am, V-last) and the non-HD tables (T-rep,
E-BCH, E-LDPC, E-CAM): H1's two sides. V-cb and T-vhdd are not rerun.
Pipeline (B38's registered rules): dev grid on B34's 12 synthetic dev stores -> designs and dev N* (b38_design.choose)
-> test window = dev N* +- 4 grid steps plus B = 512, i.i.d. conditions only -> test on the 50 RULER stores -> B38's
extension rule repeated until every N* is bracketed (at most 6 passes) -> N*.

  python b52_b38_1d.py dev    --workers W
  python b52_b38_1d.py choose
  python b52_b38_1d.py plan
  python b52_b38_1d.py test   --workers W
  python b52_b38_1d.py extend --workers W        (loop)
  python b52_b38_1d.py closed --workers W        (second angle: b38_check closed form at the one-day rates)
  python b52_b38_1d.py verdict                   (-> results/b52c_verdict.{txt,json}; 30-day N* from results/b38_verdict.txt)
"""
import collections, json, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import b38_eq as E                                                          # noqa: E402
import b38_design as D                                                      # noqa: E402

BER = json.load(open(os.path.join(HERE, "results", "b34_ber.json")))
P1D = {k: BER["%d|1 d" % k] for k in (1, 2, 4, 8, 16, 32)}
E.P30 = P1D                                                                  # name kept: the module reads E.IID below
E.IID = collections.OrderedDict([("k1", P1D[1]), ("k2", P1D[2]), ("k4", P1D[4]), ("e05", 0.05), ("k8", P1D[8]),
                                 ("k16", P1D[16]), ("k32", P1D[32])])
E.all_conditions = lambda: list(E.IID)
ARMS = ["V-plain", "V-last", "T-rep", "E-BCH", "E-LDPC", "E-CAM"]
VARS = list(D.VSA_FREE) + list(D.TABLES)
S38 = os.environ.get("B38_OUT", "/path/to/work/b38/b38_out")
O = os.environ.get("B52C_OUT", "out_c")
DEV, TEST = os.path.join(S38, "dev_stores.json.gz"), os.path.join(S38, "test_stores.json.gz")


def run_jobs(stores_path, jobs, out):
    stores = dict(E.load_stores(stores_path))
    js = [dict(sid=j["sid"], pairs=stores[j["sid"]], arm=j["arm"], evals=[(e[0], e[1], e[2]) for e in j["evals"]])
          for j in jobs]
    js.sort(key=lambda j: -len(j["pairs"]))
    print(len(js), "jobs ->", out, flush=True)
    res = E.pool_run_auto(js, W, out)
    json.dump(res, open(out, "w"))


def plan():
    """B38's plan restricted to H1: per i.i.d. condition and variant, the window around dev N* plus B = 512."""
    stores = E.load_stores(TEST)
    choice = json.load(open(os.path.join(O, "choice.json")))
    need = collections.defaultdict(set)
    for c in E.IID:
        for v in VARS:
            fam = D.FAMILY[v]
            for gi in sorted(set(D.window(choice, v, c)) | {D.gidx(512.0)}):
                d = D.design_for(choice, v, c, gi)
                dj = "null" if fam == "V-plain" else json.dumps(D.to_design(v, d))
                need[(fam, gi, dj)].add(c)
    jobs = D.make_jobs(stores, need)
    json.dump({"jobs": jobs}, open(os.path.join(O, "plan_test.json"), "w"))
    print(len(jobs), "jobs;", sum(len(e[2]) for j in jobs for e in j["evals"]), "evaluations")


def test_files():
    fs = [os.path.join(O, "test.json")]
    n = 1
    while os.path.exists(os.path.join(O, "test_x%d.json" % n)):
        fs.append(os.path.join(O, "test_x%d.json" % n)); n += 1
    return fs


def extend_loop():
    for n in range(1, 7):
        fs = test_files()
        p = os.path.join(O, "plan_x%d.json" % n)
        if os.path.exists(os.path.join(O, "test_x%d.json" % n)):
            continue
        D.extend(TEST, os.path.join(O, "choice.json"), [f for f in fs], p)
        jobs = json.load(open(p))["jobs"]
        if not jobs:
            print("all N* bracketed after", n - 1, "extension passes"); return
        run_jobs(TEST, jobs, os.path.join(O, "test_x%d.json" % n))
    print("stopped after 6 extension passes")


def b38_30d():
    """Table 1 of results/b38_verdict.txt (30-day rates): {variant: {cond: N* or None}}."""
    out, conds = {}, None
    for l in open(os.path.join(HERE, "results", "b38_verdict.txt")):
        p = l.split()
        if p[:7] == ["k1", "k2", "k4", "e05", "k8", "k16", "k32"]:
            conds = p[:7]; continue
        if conds and p and p[0] in VARS + ["V-cb", "T-vhdd"] and len(p) == 8 and p[0] not in out:
            out[p[0]] = {c: (None if x.startswith(">") else float(x)) for c, x in zip(conds, p[1:])}
        if conds and p and p[0] == "capacity":
            break
    return out


def verdict():
    choice = json.load(open(os.path.join(O, "choice.json")))
    acc, _ = D.test_curves(test_files())
    ref = b38_30d()
    G = E.GRID
    L, J = [], {"rates_1d": dict(E.IID), "nstar_1d": {}, "nstar_30d": ref, "bars": {}}

    def P(s=""):
        L.append(s); print(s)

    def fmt(x):
        return ">4096" if x is None else "%.0f" % x
    P("# B52 (c) verdict (prereg/B52_robustness_layers_maps_1day_prereg.md): B38's N* (cells per stored pair for >= 99.5%")
    P("# exact reads of the 4,776 probed RULER keys) at the device model's ONE-DAY bit-error rates; generated %s UTC"
      % time.strftime("%Y-%m-%d %H:%M", time.gmtime()))
    P("\n  rates (1 day): " + ", ".join("%s %.6g" % (c, e) for c, e in E.IID.items()) + "   (30 days: " +
      ", ".join("%s %.6g" % (c, e) for c, e in zip(E.IID, (0.0, 0.00013175, 0.0208065, 0.05, 0.12263475, 0.27600645, 0.374429))) + ")")
    P("\n  N* at 1 day (30 days in parentheses)")
    P("  %-8s" % "" + "".join("%17s" % c for c in E.IID))
    for v in VARS:
        J["nstar_1d"][v] = {}
        row = []
        for c in E.IID:
            curve, ns = D.test_nstar(acc, choice, v, c)
            x = None if ns is None else G[ns]
            unb = bool(curve) and ns is None and max(curve) < len(G) - 1
            J["nstar_1d"][v][c] = x
            row.append("%9s%s (%5s)" % (fmt(x), "?" if unb else " ", fmt(ref.get(v, {}).get(c))))
        P("  %-8s" % v + "".join("%17s" % r for r in row))
    P("\n  H1 at 1 day: Delta = grid steps of the best non-HD table's N* minus the best VSA read's N* (never reached = one")
    P("  step beyond 4,096); 'VSA fewer cells' if Delta >= +2, 'table fewer cells' if <= -2, 'tie' if |Delta| <= 1")

    def steps(x):
        return len(G) if x is None else D.gidx(x)
    labels = {}
    for c in E.IID:
        bv = min(VSA_ := [(steps(J["nstar_1d"][v][c]), v) for v in D.VSA_FREE])
        bt = min([(steps(J["nstar_1d"][v][c]), v) for v in D.TABLES])
        d = bt[0] - bv[0]
        lab = "VSA fewer cells" if d >= 2 else ("table fewer cells" if d <= -2 else "tie")
        rv = min([(steps(ref[v][c]), v) for v in D.VSA_FREE]); rt = min([(steps(ref[v][c]), v) for v in D.TABLES])
        d30 = rt[0] - rv[0]
        labels[c] = lab
        J["bars"][c] = {"best_vsa": bv[1], "best_vsa_nstar": J["nstar_1d"][bv[1]][c], "best_table": bt[1],
                        "best_table_nstar": J["nstar_1d"][bt[1]][c], "delta": d, "label": lab, "delta_30d": d30,
                        "best_vsa_30d": rv[1], "best_table_30d": rt[1]}
        P("    %-4s best VSA %-6s %6s | best table %-6s %6s | Delta %+d -> %-17s (30 d: %s %s vs %s %s, Delta %+d)" % (
            c, bv[1], fmt(J["nstar_1d"][bv[1]][c]), bt[1], fmt(J["nstar_1d"][bt[1]][c]), d, lab,
            rv[1], fmt(ref[rv[1]][c]), rt[1], fmt(ref[rt[1]][c]), d30))
    same = all(J["bars"][c]["label"] == "table fewer cells" for c in E.IID)
    order_same = {c: (J["bars"][c]["best_table"], J["bars"][c]["best_vsa"]) == (J["bars"][c]["best_table_30d"], J["bars"][c]["best_vsa_30d"])
                  for c in E.IID}
    J["H1_1d"] = "TABLE WIN" if same else ("VSA WIN" if all(l == "VSA fewer cells" for l in labels.values()) else "MIXED")
    J["ranking_unchanged"] = same
    P("\n  H1 at 1 day: %s; the 30-day ranking (table fewer cells at every k) %s" % (J["H1_1d"], "HOLDS" if same else "CHANGES"))
    P("  best arms identical to 30 days per condition: %s" % order_same)
    # compared in grid steps (the 30-day table prints rounded budgets, e.g. 76 for 2^(25/4) = 76.1)
    mono = {(v, c): steps(J["nstar_1d"][v][c]) > steps(ref[v][c]) for v in VARS for c in E.IID}
    J["monotone_violations"] = [list(k) for k, bad in mono.items() if bad]
    P("  check: N*(1 day) <= N*(30 days) for every arm and condition (rates are lower at 1 day): violations %s"
      % (J["monotone_violations"] or "none"))
    cl = os.path.join(O, "closed.json")
    if os.path.exists(cl):
        c = json.load(open(cl))
        J["closed"] = {"n_cells": c["n_cells"], "flagged": c["flagged"]}
        P("  closed-form check (b38_check closed at the one-day rates; T-rep, V-pub, V-Snt, V-last, E-BCH/E-CAM where "
          "available): %d cells, %d flagged (|sim - closed form| > max(1 point, 4 SE))" % (c["n_cells"], c["flagged"]))
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    open(os.path.join(HERE, "results", "b52c_verdict.txt"), "w").write("\n".join(L) + "\n")
    json.dump(J, open(os.path.join(HERE, "results", "b52c_verdict.json"), "w"), indent=1)


if __name__ == "__main__":
    mode = sys.argv[1]
    W = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 8
    os.makedirs(O, exist_ok=True)
    if mode == "dev":
        out = os.path.join(O, "dev.json")
        jobs = E.dev_jobs(E.load_stores(DEV), ARMS)
        print(len(jobs), "jobs", flush=True)
        res = E.pool_run_auto(jobs, W, out)
        json.dump(res, open(out, "w"))
    elif mode == "choose":
        D.choose(os.path.join(O, "dev.json"), os.path.join(O, "choice.json"))
    elif mode == "plan":
        plan()
    elif mode == "test":
        run_jobs(TEST, json.load(open(os.path.join(O, "plan_test.json")))["jobs"], os.path.join(O, "test.json"))
    elif mode == "extend":
        extend_loop()
    elif mode == "closed":
        import b38_check as CK
        CK.closed(TEST, ",".join(test_files()), os.path.join(O, "closed.json"), W)
    elif mode == "verdict":
        verdict()
