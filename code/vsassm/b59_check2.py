"""B59 part 2 checks (prereg/B59_maps_all_encoders_protected_full_prereg.md, 'Checks', part 2). B58's scorer / bits /
sample checks are run by b58_check.py on the B59 test files; this file adds:

python3 b59_check2.py reuse PLAN.json N      # N reused B58 test evaluations recomputed by b58_store.evaluate (identity)
python3 b59_check2.py u N                    # N U evaluations recomputed by b38_eq.run_job vs B38's test rows
python3 b59_check2.py nstar CHOICE PLAN TESTS  # P-best / P-dev / family N* recomputed by a separately written rule
Appends to results/b59_check.txt.
"""
import collections
import gzip
import json
import sys

import numpy as np

import b38_eq as E
import b58_store as S


def log(s):
    print(s, flush=True)
    open("results/b59_check.txt", "a").write(s + "\n")


def jload(p):
    return json.load(gzip.open(p, "rt")) if p.endswith(".gz") else json.load(open(p))


def reuse(plan_path, n):
    J = json.load(open(plan_path))
    keys = [tuple(k) for k in J["reuse"]]
    rng = np.random.default_rng(591)
    pick = [keys[i] for i in rng.choice(len(keys), min(n, len(keys)), replace=False)]
    old = {}
    for p in ("results/b58_test_iid1p.json.gz", "results/b58_test_iid2.json.gz"):
        for r in jload(p):
            for x in r["res"]:
                old[(r["sid"], x["var"], round(x["b"], 4), x["R"], S.gidx(x["B"]), x["cond"])] = x
    st = dict(E.load_stores("b34_out/test_stores.json.gz"))
    same = 0
    for k in pick:
        sid, var, b, R, gi, c = k
        S38 = S.slim(E.Store38(sid, st[sid]))
        x = S.evaluate(S38, var, b, R, E.GRID[gi], c, S.TEST_CAP)
        o = old[k]
        ok = x["failed"] == o["failed"] and x["censored"] == o["censored"]
        same += ok
    log(f"- reuse: {len(pick)} reused B58 test evaluations recomputed by b58_store.evaluate: {same} identical "
        f"(failed-key lists and censoring)")


def u(n):
    res = jload("results/b38_test.json.gz")
    rows = []
    for r in res:
        if r["arm"] != "V-last":
            continue
        for x in r["res"]:
            if "cond" in x and x["cond"] in E.IID and not x.get("infeasible") and x.get("rule") == "last":
                rows.append((r["sid"], x["B"], x["cond"], x["failed"]))
    rng = np.random.default_rng(592)
    pick = [rows[i] for i in rng.choice(len(rows), min(n, len(rows)), replace=False)]
    st = dict(E.load_stores("b34_out/test_stores.json.gz"))
    same = 0
    for sid, B, c, f in pick:
        out = E.run_job(dict(sid=sid, pairs=st[sid], arm="V-last", evals=[(B, None, [c])]))
        g = [x for x in out["res"] if "cond" in x and x.get("rule") == "last"][0]["failed"]
        same += g == f
    log(f"- U identity: {len(pick)} B38 V-last test evaluations recomputed by b38_eq.run_job (the T1 code path): "
        f"{same} identical failed-key lists")


def nstar(choice_path, plan_path, tests):
    """Second rule: per family the pooled share at every budget evaluated on all 50 stores; N* = the smallest budget
    from which every evaluated budget upwards passes, scanning upwards (written separately from V58.nstar_counts)."""
    import b59_store_verdict as SV
    ch = json.load(open(choice_path))
    plan = SV.merged_plan(plan_path.split(","))
    C, _ = SV.counts(ch, plan, tests)
    J = json.load(open("results/b59_store_verdict.json"))
    sids = [s for s, _ in E.load_stores("b34_out/test_stores.json.gz")]
    bad = tot = 0
    for c in E.IID:
        share = {}
        for f in SV.FAMS:
            for gi, d in C.get((f, c), {}).items():
                if all(s in d for s in sids):
                    fl = sum(min(d[s][0], d[s][1]) for s in sids)
                    nn = sum(d[s][1] for s in sids)
                    share.setdefault(f, {})[gi] = 1 - fl / nn
        best = collections.defaultdict(float)
        for f, sh in share.items():
            for gi, v in sh.items():
                best[gi] = max(best[gi], v)
        for name, sh in list(share.items()) + [("P-best", best)]:
            gis = sorted(sh)
            ns = None
            for i, gi in enumerate(gis):
                if all(sh[g] >= E.TARGET for g in gis[i:]):
                    ns = E.GRID[gi]
                    break
            tot += 1
            bad += ns != J["nstar"][f"{name}|{c}"]
    log(f"- N* second rule (families U / P-BCH / P-LDPC and P-best, 7 conditions): {tot} recomputed, {bad} mismatches")


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "reuse":
        reuse(a[1], int(a[2]))
    elif a[0] == "u":
        u(int(a[1]))
    elif a[0] == "nstar":
        nstar(a[1], a[2], [p for p in a[3].split(",") if p])
