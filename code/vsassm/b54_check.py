"""B54 checks from other angles (prereg/B54_mlc_cells_prereg.md).

python3 b54_check.py closed  STORES TEST1.json[,..] OUT.json [--workers N]
    Closed-form expected exact keys of every simulated MLC test cell of T-rep (binomial majority) and E-BCH
    (errors-and-erasures bound per codeword) within +-2 grid steps of its test N*, at the condition's calibrated mean
    bit-error rate (B38's closed forms, b38_check.py). No cell is simulated.
python3 b54_check.py device  STORES TEST1.json[,..] OUT.json [--workers N]
    Re-reads a subset (the 10 131072/niah_multikey_2 stores; every arm; b = 2, 3; k1, k2, k4 at 30 days; at one grid
    step below the test N*) with B25's device model itself on the actual cells instead of the calibrated i.i.d.
    channel; shares must agree within max(1 point, 4 SE).
python3 b54_check.py second  STORES TEST1.json[,..] OUT.txt [--workers N]
    (a) a second level demapper (scalar loop, bisect + explicit Gray table) against MLCChannel.packed on 200,000
    random cells per condition; (b) a second exactness scorer (decoded value strings against the store's own pairs,
    not Store38.want) on re-run test cells; (c) N* recomputed independently from the per-store failed-key lists.
python3 b54_check.py handread STORES TEST1.json[,..] OUT.txt
"""
import bisect
import collections
import json
import math
import sys

import numpy as np

import b38_check as K
import b38_eq as E
import b54_mlc as M


def load(test_fs):
    res = []
    for f in test_fs.split(","):
        res += json.load(open(f))
    return res


def test_nstar(res):
    acc = M.aggregate(res)
    ns = {}
    for (arm, b), dd in acc.items():
        for c in M.conditions(b):
            curve = {}
            for dk in dd:
                for gi, a in dd[dk][c].items():
                    curve[gi] = a[0] / a[1]
            ns[(arm, b, c)] = M.nstar(curve) if curve else None
    return ns


# =============================================================================================== closed form
def _closed_job(job):
    sid, pairs, cells = job
    S38 = E.Store38(sid, pairs)
    out = []
    for arm, b, B, design, cond, sim in cells:
        eps = M.ber(cond)
        Bb = M.bits_budget(S38, B, b)
        pr = K.cf_rep(S38, Bb, eps) if arm == "T-rep" else K.cf_ecc(S38, "E-BCH", tuple(design), Bb, eps)
        if pr is not None:
            pr = np.array(pr)
            out.append(dict(arm=arm, b=b, B=B, cond=cond, expect=float(pr.sum()), var=float((pr * (1 - pr)).sum()),
                            n=len(pr), sim=sim))
    return out


def closed(stores_f, test_fs, out_f, workers):
    stores = dict(M.load_stores(stores_f))
    res = load(test_fs)
    ns = test_nstar(res)
    per = collections.defaultdict(list)
    for r in res:
        if r["arm"] not in ("T-rep", "E-BCH") or r["b"] == 1:
            continue
        for x in r["res"]:
            if x.get("infeasible") or M.parse(x["cond"])["ideal"]:
                continue
            g = ns.get((r["arm"], r["b"], x["cond"]))
            gi = M.gidx(x["B"])
            if g is None or abs(gi - g) > 2:
                continue
            per[r["sid"]].append((r["arm"], r["b"], x["B"], x["design"], x["cond"], x["exact"]))
    jobs = [(sid, stores[sid], cl) for sid, cl in per.items()]
    from multiprocessing import Pool
    tot = collections.defaultdict(lambda: [0.0, 0.0, 0, 0])
    with Pool(workers) as pool:
        for out in pool.imap_unordered(_closed_job, jobs):
            for o in out:
                t = tot[(o["arm"], o["b"], o["cond"], o["B"])]
                t[0] += o["expect"]
                t[1] += o["var"]
                t[2] += o["sim"]
                t[3] += o["n"]
    rows = []
    for (arm, b, cond, B), (ex, var, sim, n) in sorted(tot.items(), key=lambda kv: str(kv[0])):
        se = 100 * math.sqrt(max(var, 1e-12)) / n
        d = 100 * (sim - ex) / n
        rows.append(dict(arm=arm, b=b, cond=cond, B=B, closed=100 * ex / n, sim=100 * sim / n, diff=d, se=se,
                         flag=bool(abs(d) > max(1.0, 4 * se))))
    json.dump(rows, open(out_f, "w"), indent=1)
    fl = [r for r in rows if r["flag"]]
    print("closed form: %d cells, %d flagged (sim above closed form: %d)" % (len(rows), len(fl),
                                                                            sum(r["diff"] > 0 for r in fl)))
    for r in fl[:40]:
        print("  %(arm)s m%(b)d %(cond)s B=%(B).1f closed %(closed).2f sim %(sim).2f diff %(diff)+.2f se %(se).2f" % r)


# =============================================================================================== device model
def run_cell(S38, arm, b, B, design, cond, device_seed=None):
    A = M.build(S38, arm, B, b, design)
    A, nu, mode = A
    tag = "%s|%s|%s" % (arm, json.dumps(design), mode)
    ch = M.MLCChannel(S38.sid, tag, B, b, nu)
    ch.device_seed = device_seed
    rng = np.random.default_rng(E.seed_of("b54", S38.sid, tag, "%.4f" % B, cond, "dec"))
    got, diag = M.read(S38, arm, A, mode, ch, cond, rng)
    return got, diag


def _device_job(job):
    sid, pairs, arm, b, B, design, cond = job
    S38 = E.Store38(sid, pairs)
    got, _ = run_cell(S38, arm, b, B, design, cond, device_seed=E.seed_of("b54dev", sid, arm, B, cond) % (2 ** 31))
    return dict(sid=sid, arm=arm, b=b, B=B, cond=cond, exact=len(S38.want) - len(E.exact(got, S38.want)),
                n=len(S38.want))


def device(stores_f, test_fs, out_f, workers):
    stores = dict(M.load_stores(stores_f))
    res = load(test_fs)
    ns = test_nstar(res)
    jobs, sim = [], collections.defaultdict(lambda: [0, 0])
    for r in res:
        if not r["sid"].startswith("131072/niah_multikey_2") or r["b"] == 1:
            continue
        for x in r["res"]:
            if x.get("infeasible") or x["cond"] not in ("m%d-k%d-30d" % (r["b"], k) for k in (1, 2, 4)):
                continue
            g = ns.get((r["arm"], r["b"], x["cond"]))
            if g is None or M.gidx(x["B"]) != g - 1:
                continue
            jobs.append((r["sid"], stores[r["sid"]], r["arm"], r["b"], x["B"], x["design"], x["cond"]))
            s = sim[(r["arm"], r["b"], x["cond"], x["B"])]
            s[0] += x["exact"]
            s[1] += x["n"]
    from multiprocessing import Pool
    dev = collections.defaultdict(lambda: [0, 0])
    with Pool(workers) as pool:
        for o in pool.imap_unordered(_device_job, jobs):
            d = dev[(o["arm"], o["b"], o["cond"], o["B"])]
            d[0] += o["exact"]
            d[1] += o["n"]
    rows = []
    for key, (ex, n) in sorted(dev.items(), key=lambda kv: str(kv[0])):
        s_ex, s_n = sim[key]
        p1, p2 = s_ex / s_n, ex / n
        se = 100 * math.sqrt(max(p1 * (1 - p1) / s_n + p2 * (1 - p2) / n, 1e-12))
        rows.append(dict(arm=key[0], b=key[1], cond=key[2], B=key[3], channel=100 * p1, device=100 * p2,
                         diff=100 * (p2 - p1), se=se, flag=bool(abs(100 * (p2 - p1)) > max(1.0, 4 * se))))
    json.dump(rows, open(out_f, "w"), indent=1)
    print("device model re-read: %d cells, %d flagged" % (len(rows), sum(r["flag"] for r in rows)))
    for r in rows:
        print("  %(arm)-7s m%(b)d %(cond)-11s B=%(B)7.1f channel %(channel)6.2f device %(device)6.2f diff %(diff)+.2f "
              "se %(se).2f %(flag)s" % r)


# =============================================================================================== second implementations
def demap2(r, thr, b):
    """Scalar second demapper: level by bisect over the thresholds, Gray label by an explicit table."""
    table = {l: format(l ^ (l >> 1), "0%db" % b) for l in range(2 ** b)}
    return [table[bisect.bisect_left(list(thr), float(v))] for v in r]


def _second_job(job):
    sid, pairs, cells = job
    S38 = E.Store38(sid, pairs)
    truth = collections.defaultdict(list)
    for k, v, j in pairs:
        truth[k].append((j, v))
    vocab = S38.s.cd["vocab"]
    out = []
    for arm, b, B, design, cond, failed in cells:
        got, _ = run_cell(S38, arm, b, B, design, cond)
        agree = 0
        for a, k in enumerate(S38.s.keys):
            want = [v for _, v in sorted(truth[k])]
            ok2 = [vocab[i] if 0 <= i < len(vocab) else "<invalid index %d>" % i for i in got[a]] == want
            agree += ok2 == (a not in set(failed))
        out.append(dict(arm=arm, b=b, cond=cond, B=B, agree=agree, n=len(S38.s.keys)))
    return out


def second(stores_f, test_fs, out_f, workers):
    lines = []
    # (a) demapper
    rng = np.random.default_rng(5)
    for b in M.BITS:
        for c in M.conditions(b)[1:]:
            nb = 200000 * b
            bits = (rng.integers(0, 2, nb) * 2 - 1).astype(np.int8)
            ch = M.MLCChannel("second", "demap", 1.0, b, nb)
            hard, _ = ch.packed(bits, c)
            ncell = nb // b
            lab = np.zeros(ncell * b, np.int64)
            lab[ch._perm[:nb]] = bits > 0
            lab = lab.reshape(ncell, b)
            inv = np.argsort(M.gray(np.arange(2 ** b)))
            level = inv[(lab << (b - 1 - np.arange(b))[None, :]).sum(1)]
            cc = M.cal(c)
            r = M.sample_read(level, ch.u(ncell), cc)
            s2 = demap2(r, cc["thr"], b)
            rb = np.array([[int(ch_) for ch_ in w] for w in s2]).reshape(-1)
            hard2 = (rb[ch._perm[:nb]] * 2 - 1).astype(np.int8)
            lines.append("demapper %-12s cells %d  bits agreeing %d / %d" % (c, ncell, int((hard2 == hard).sum()), nb))
    # (b) second exactness scorer on re-run test cells (10 stores of each task family at k4 / 30 d, b = 2, all arms,
    # every evaluated budget within 1 step of N*)
    stores = dict(M.load_stores(stores_f))
    res = load(test_fs)
    ns = test_nstar(res)
    per = collections.defaultdict(list)
    seen = collections.Counter()
    for r in res:
        if r["b"] != 2:
            continue
        fam = r["sid"].rsplit("/", 1)[0]
        for x in r["res"]:
            if x.get("infeasible") or x["cond"] != "m2-k4-30d":
                continue
            g = ns.get((r["arm"], 2, x["cond"]))
            if g is None or abs(M.gidx(x["B"]) - g) > 1:
                continue
            if seen[(fam, r["arm"])] >= 6:
                continue
            seen[(fam, r["arm"])] += 1
            per[r["sid"]].append((r["arm"], 2, x["B"], x["design"], x["cond"], x["failed"]))
    from multiprocessing import Pool
    tot = collections.defaultdict(lambda: [0, 0])
    with Pool(workers) as pool:
        for out in pool.imap_unordered(_second_job, [(s, stores[s], cl) for s, cl in per.items()]):
            for o in out:
                tot[o["arm"]][0] += o["agree"]
                tot[o["arm"]][1] += o["n"]
    for arm, (a, n) in sorted(tot.items()):
        lines.append("second scorer %-7s probed keys agreeing with the run's exact/failed label: %d / %d" % (arm, a, n))
    # (c) N* from the failed-key lists, recomputed independently
    cur = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
    bad = 0
    for r in res:
        for x in r["res"]:
            if x.get("infeasible"):
                n_ex = 0
            else:
                n_ex = x["n"] - len(x["failed"])
                bad += n_ex != x["exact"]
            cur[(r["arm"], r["b"], x["cond"])][x["B"]][0] += n_ex
            cur[(r["arm"], r["b"], x["cond"])][x["B"]][1] += x["n"]
    mism = 0
    for key, cv in cur.items():
        Bs = np.array(sorted(cv))
        sh = np.array([cv[B][0] / cv[B][1] for B in Bs])
        ok = sh >= M.TARGET
        tail = np.flip(np.cumprod(np.flip(ok)))               # True where this and every larger budget pass
        n2 = float(Bs[tail.astype(bool)].min()) if tail.any() else None
        n1 = ns.get(key)
        n1 = None if n1 is None else M.GRID[n1]
        mism += (n1 is None) != (n2 is None) or (n1 is not None and abs(n1 - n2) > 1e-6)
    lines.append("N* recomputed from failed-key lists: %d (arm, b, condition) curves, %d mismatches; exact counts "
                 "inconsistent with failed lists: %d" % (len(cur), mism, bad))
    open(out_f, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


# =============================================================================================== hand reading
def handread(stores_f, test_fs, out_f):
    stores = dict(M.load_stores(stores_f))
    res = load(test_fs)
    ns = test_nstar(res)
    lines = []
    want = [("V-last", 2, "m2-k2-30d"), ("V-last", 3, "m3-k1-30d"), ("E-BCH", 2, "m2-k4-30d"),
            ("E-LDPC", 2, "m2-k4-30d"), ("T-rep", 2, "m2-k2-30d"), ("T-vhdd", 2, "m2-k4-30d"),
            ("V-last", 2, "m2-k8-30d"), ("E-LDPC", 3, "m3-k2-30d")]
    for arm, b, cond in want:
        g = ns.get((arm, b, cond))
        if g is None:
            lines.append("%s m%d %s: N* not reached; skipped" % (arm, b, cond))
            continue
        shown = 0
        for r in res:
            if r["arm"] != arm or r["b"] != b or shown >= 3:
                continue
            for x in r["res"]:
                if x["cond"] != cond or x.get("infeasible") or M.gidx(x["B"]) != g - 1 or not x["failed"]:
                    continue
                S38 = E.Store38(r["sid"], stores[r["sid"]])
                got, diag = run_cell(S38, arm, b, x["B"], x["design"], cond)
                for a in x["failed"][:2]:
                    lines.append("%s m%d %s B=%.1f design %s store %s key %s: want %s got %s  diag %s" % (
                        arm, b, cond, x["B"], x["design"], r["sid"], S38.s.keys[a], S38.want[a], got[a], diag))
                    shown += 1
                break
    open(out_f, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    mode = sys.argv[1]
    W = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 4
    if mode == "closed":
        closed(sys.argv[2], sys.argv[3], sys.argv[4], W)
    elif mode == "device":
        device(sys.argv[2], sys.argv[3], sys.argv[4], W)
    elif mode == "second":
        second(sys.argv[2], sys.argv[3], sys.argv[4], W)
    elif mode == "handread":
        handread(sys.argv[2], sys.argv[3], sys.argv[4])
