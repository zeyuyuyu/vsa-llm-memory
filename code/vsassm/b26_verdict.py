"""B26 verdict generator (prereg/B26_vector_form_prereg.md), written before any test output exists.

python3 b26_verdict.py QUERY.jsonl [WRITE.jsonl] [--choice results/b26_dev_choice.json] [--pcm results/b26_pcm_table.json]
                       [--gpu results/b26_gpu_ledger.json] [--out results/b26_verdict.txt] [--hand results/b26_handread.txt]
"""
import argparse
import collections
import json

import numpy as np
from scipy.stats import binomtest

ARMS = ["E", "Ev", "En", "L", "W", "N", "V"]
VAR = ["T1", "T2", "T3", "P", "A"]
TASKS = ["niah_single_1", "niah_single_2", "niah_single_3", "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
         "niah_multivalue", "niah_multiquery"]
LENGTHS = [4096, 8192, 16384, 32768, 65536, 131072]
NBOOT = 20000


def load(path):
    rows = [json.loads(l) for l in open(path)]
    by = {}
    for r in rows:
        by[(r["cell"][0], r["cell"][1], r["pos"], r["cond"])] = r
    samples = sorted({k[:3] for k in by})
    conds = sorted({k[3] for k in by}, key=lambda c: ["C0", "T1", "T2", "T3", "P", "A"].index(c))
    return by, samples, conds


def S(by, samples, cond, arm):
    return np.array([by[s + (cond,)]["arms"][arm]["score"] for s in samples], float)


def boot(d, seed=0):
    """95% paired bootstrap interval of the mean of per-sample differences d (n,) or (n, c) (rows resampled)."""
    d = np.asarray(d, float)
    if d.ndim == 2:
        d = d.mean(1)
    rng = np.random.default_rng(seed)
    n = len(d)
    means = np.empty(NBOOT)
    for a in range(0, NBOOT, 1000):
        idx = rng.integers(0, n, size=(min(1000, NBOOT - a), n))
        means[a:a + len(idx)] = d[idx].mean(1)
    return 100 * d.mean(), 100 * np.percentile(means, 2.5), 100 * np.percentile(means, 97.5)


def sign(d):
    d = np.asarray(d, float)
    if d.ndim == 2:
        d = d.mean(1)
    up, dn = int((d > 1e-12).sum()), int((d < -1e-12).sum())
    p = binomtest(up, up + dn, 0.5).pvalue if up + dn else 1.0
    return up, dn, p


def fmt_cmp(d):
    m, lo, hi = boot(d)
    up, dn, p = sign(d)
    return "%+6.2f [%+6.2f, %+6.2f]  sign %d/%d p=%.2g" % (m, lo, hi, up, dn, p), (m, lo, hi, up, dn, p)


def side_report(by, samples, conds, label, out, hand=None, write=False):
    P = lambda *a: out.append(" ".join(str(x) for x in a))
    res = {}
    n = len(samples)
    P("## %s side (n = %d samples per condition; string_match_all, points)" % (label, n))
    P("arm   " + "".join(c.rjust(8) for c in conds))
    acc = {}
    for a in ARMS:
        acc[a] = {c: 100 * S(by, samples, c, a).mean() for c in conds}
        P(a.ljust(6) + "".join("%8.2f" % acc[a][c] for c in conds))
    res["acc"] = acc
    vconds = [c for c in VAR if c in conds]
    # paired comparisons
    P("")
    P("Paired: V minus each arm, mean difference in points [95% bootstrap], sign test (V better / other better)")
    cmp = {}
    for other in ["E", "En", "L", "W", "N"]:
        for c in vconds + ["pooled"]:
            if c == "pooled":
                d = np.stack([S(by, samples, cc, "V") - S(by, samples, cc, other) for cc in vconds], 1)
            else:
                d = S(by, samples, c, "V") - S(by, samples, c, other)
            txt, vals = fmt_cmp(d)
            cmp[(other, c)] = vals
            P("  V-%-2s %-7s %s" % (other, c, txt))
    res["cmp"] = {"%s|%s" % k: v for k, v in cmp.items()}
    # by length and by task
    for c in vconds:
        P("")
        P("%s by length (n = 800 per length): arm accuracy; V-L and V-W [95%%]" % c)
        for L in LENGTHS:
            idx = [i for i, s in enumerate(samples) if s[0] == L]
            if not idx:
                continue
            ss = [samples[i] for i in idx]
            line = "  %6dk " % (L // 1024) + " ".join("%s %6.2f" % (a, 100 * S(by, ss, c, a).mean()) for a in ["E", "En", "L", "W", "N", "V"])
            for o in ["L", "W"]:
                m, lo, hi = boot(S(by, ss, c, "V") - S(by, ss, c, o))
                line += "  V-%s %+.2f [%+.2f, %+.2f]" % (o, m, lo, hi)
            P(line)
        P("%s by task (n = 600 per task)" % c)
        for t in TASKS:
            ss = [s for s in samples if s[1] == t]
            if not ss:
                continue
            P("  %-17s " % t + " ".join("%s %6.2f" % (a, 100 * S(by, ss, c, a).mean()) for a in ["E", "En", "L", "W", "N", "V"]))
    # niah_multikey_2 by length (prediction on aliases)
    mk2 = {}
    for c in conds:
        for L in LENGTHS:
            ss = [s for s in samples if s[0] == L and s[1] == "niah_multikey_2"]
            if ss:
                mk2[(c, L)] = {a: 100 * S(by, ss, c, a).mean() for a in ARMS}
    res["mk2"] = {"%s|%d" % k: v for k, v in mk2.items()}
    # largest per-cell typo gap L - V
    gaps = []
    for c in [x for x in ["T1", "T2", "T3"] if x in conds]:
        for L in LENGTHS:
            for t in TASKS:
                ss = [s for s in samples if s[0] == L and s[1] == t]
                if ss:
                    gaps.append((100 * (S(by, ss, c, "L") - S(by, ss, c, "V")).mean(), c, L, t))
    res["max_typo_gap"] = max(gaps) if gaps else None
    # validity (bar 0)
    c0 = {a: float(np.mean(S(by, samples, "C0", a) == S(by, samples, "C0", "E"))) for a in ARMS}
    evq = {c: float(np.mean(S(by, samples, c, "Ev") == S(by, samples, c, "E"))) for c in conds}
    res["c0_agree"], res["ev_agree"] = c0, evq
    P("")
    P("Bar 0 checks: share of samples whose score equals E's in C0: " + ", ".join("%s %.4f" % (a, c0[a]) for a in ARMS))
    P("              share of samples where Ev equals E, by condition: " + ", ".join("%s %.4f" % (c, evq[c]) for c in conds))
    # second scorers: key choice and strict answers
    P("")
    P("Second scorer 1, key choice: share of queried keys for which the arm chose the %s key (keys whose %s key is stored)"
      % (("stored variant", "variant") if write else ("gold", "gold")))
    kc = {}
    stored = {}
    for s in samples:
        r0 = by[s + ("C0",)]
        stored[s] = [ch is not None for ch in r0["arms"]["E"]["chosen"]]
    for a in ["E", "En", "L", "W", "N", "V"]:
        line, kc[a] = a.ljust(6), {}
        for c in conds:
            ok = tot = 0
            for s in samples:
                r = by[s + (c,)]
                if write:
                    tgt = [v.lower() for v in r["variants"]]
                    st = r["variant_stored"]
                else:
                    tgt, st = r["keys"], stored[s]
                for ch, g, q in zip(r["arms"][a]["chosen"], tgt, st):
                    if q:
                        tot += 1
                        ok += ch == g
            kc[a][c] = ok / max(tot, 1)
            line += "%8.4f" % kc[a][c]
        P(line + "   (n keys %d per condition)" % tot)
    res["key_choice"] = kc
    P("Second scorer 2, strict: the predicted values equal the gold values as a set (points)")
    strict = {}
    for a in ARMS:
        strict[a] = {}
        line = a.ljust(6)
        dis = 0
        for c in conds:
            v = []
            for s in samples:
                r = by[s + (c,)]
                ok = set(r["arms"][a]["pred"].split()) == set(x for x in r["outputs"])
                v.append(ok)
                dis += (r["arms"][a]["score"] == 1.0) != ok
            strict[a][c] = 100 * np.mean(v)
            line += "%8.2f" % strict[a][c]
        P(line + "   (samples where primary = 1 disagrees with strict: %d)" % dis)
    res["strict"] = strict
    # hand-read material
    if hand is not None:
        for c in vconds:
            for o in ["L", "W"]:
                dd = [s for s in samples if by[s + (c,)]["arms"]["V"]["score"] != by[s + (c,)]["arms"][o]["score"]]
                rng = np.random.default_rng(1)
                pick = [dd[i] for i in rng.permutation(len(dd))[:10]] if dd else []
                hand.append("### %s side, %s, V vs %s: %d discordant samples; up to 10 drawn at random" % (label, c, o, len(dd)))
                for s in pick:
                    r = by[s + (c,)]
                    hand.append("  %dk %s #%d | keys %s | variants %s | gold %s" % (s[0] // 1024, s[1], s[2], r["keys"],
                                r["variants"], r["outputs"]))
                    for a in ["V", o]:
                        hand.append("     %s chose %s -> %r (score %.2f)" % (a, r["arms"][a]["chosen"], r["arms"][a]["pred"][:120],
                                                                          r["arms"][a]["score"]))
    return res


def bars(res, out, label):
    P = lambda *a: out.append(" ".join(str(x) for x in a))
    c = res["cmp"]
    b = {}
    b["bar0"] = min(res["c0_agree"].values()) >= 0.999 and min(res["ev_agree"].values()) >= 0.999
    m, lo, hi = c["E|pooled"][:3]
    b["bar1"] = m >= 50 and lo > 0
    m2, lo2, hi2 = c["L|pooled"][:3]
    b["bar2"] = m2 >= 1.0 and lo2 > 0
    m3, lo3, hi3 = c["W|pooled"][:3]
    b["bar3"] = m3 >= 1.0 and lo3 > 0
    P("## Bars, %s side" % label)
    P("  Bar 0 (C0: every arm = E on >= 99.9%%; Ev = E in every condition): min agreement %.4f / %.4f -> %s" % (
        min(res["c0_agree"].values()), min(res["ev_agree"].values()), "PASS" if b["bar0"] else "FAIL"))
    P("  Bar 1 (pooled V - E >= 50, interval above 0): %+.2f [%+.2f, %+.2f] -> %s" % (m, lo, hi, "PASS" if b["bar1"] else "FAIL"))
    P("  Bar 2 (pooled V - L >= +1.0, interval excludes 0): %+.2f [%+.2f, %+.2f] -> %s" % (m2, lo2, hi2, "PASS" if b["bar2"] else "FAIL"))
    P("  Bar 3 (pooled V - W >= +1.0, interval excludes 0): %+.2f [%+.2f, %+.2f] -> %s" % (m3, lo3, hi3, "PASS" if b["bar3"] else "FAIL"))
    return b


def predictions(q, bq, w, bw, choice, out):
    P = lambda *a: out.append(" ".join(str(x) for x in a))
    H = lambda ok: "HELD" if ok else "NOT HELD"
    acc = q["acc"]
    P("## Predictions (part a)")
    P("  Bar 0 passes; every arm = E on all C0 samples; Ev = E in every condition: bar0 %s, min C0 agreement %.4f, min Ev %.4f -> %s"
      % (bq["bar0"], min(q["c0_agree"].values()), min(q["ev_agree"].values()),
         H(bq["bar0"] and min(q["c0_agree"].values()) == 1.0 and min(q["ev_agree"].values()) == 1.0)))
    ev = max(acc[a][c] for a in ("E", "Ev") for c in VAR)
    P("  E and Ev = 0.0 in every variant condition: max %.2f -> %s" % (ev, H(ev == 0.0)))
    P("  En 65-80 in P and <= 0.5 in T1-T3, A: P %.2f, others max %.2f -> %s" % (
        acc["En"]["P"], max(acc["En"][c] for c in ("T1", "T2", "T3", "A")),
        H(65 <= acc["En"]["P"] <= 80 and max(acc["En"][c] for c in ("T1", "T2", "T3", "A")) <= 0.5)))
    lims = {"T1": (99.0, 100), "T2": (98.5, 100), "T3": (96.0, 100), "P": (99.3, 100), "A": (85, 98)}
    for c, (lo, hi) in lims.items():
        P("  V %s in [%g, %g]: %.2f -> %s" % (c, lo, hi, acc["V"][c], H(lo <= acc["V"][c] <= hi)))
    lv = np.mean([acc["L"][c] - acc["V"][c] for c in ("T1", "T2", "T3")])
    P("  Typos: 0 <= L - V <= 1.0 pooled over T1-T3: %+.2f -> %s" % (lv, H(0 <= lv <= 1.0)))
    g = q["max_typo_gap"]
    P("  Largest per-cell typo gap L - V on niah_multikey_2 at 64k/128k: %+.2f at %s %dk %s -> %s" % (
        g[0], g[1], g[2] // 1024, g[3], H(g[3] == "niah_multikey_2" and g[2] in (65536, 131072))))
    pw = all(abs(acc[a]["P"] - acc[a]["C0"]) <= 0.3 for a in ("V", "L", "W", "N"))
    P("  P: V, L, W, N within 0.3 of their C0: " + ", ".join("%s %+.2f" % (a, acc[a]["P"] - acc[a]["C0"]) for a in ("V", "L", "W", "N"))
      + " -> " + H(pw))
    P("  A: V - L >= 10: %+.2f -> %s; |V - W| <= 3: %+.2f -> %s" % (
        q["cmp"]["L|A"][0], H(q["cmp"]["L|A"][0] >= 10), q["cmp"]["W|A"][0], H(abs(q["cmp"]["W|A"][0]) <= 3)))
    P("  Bar 1 pass, bar 2 pass, bar 3 fail: %s / %s / %s -> %s" % (bq["bar1"], bq["bar2"], bq["bar3"],
                                                                   H(bq["bar1"] and bq["bar2"] and not bq["bar3"])))
    vn = q["cmp"]["N|pooled"][0]
    P("  |V - N| <= 0.3 pooled: %+.2f -> %s" % (vn, H(abs(vn) <= 0.3)))
    a4, a128 = q["mk2"].get("A|4096", {}).get("V"), q["mk2"].get("A|131072", {}).get("V")
    if a4 is None or a128 is None:
        P("  V alias on niah_multikey_2 >= 95 at 4k and <= 80 at 128k: n/a (lengths missing)")
    else:
        P("  V alias on niah_multikey_2 >= 95 at 4k and <= 80 at 128k: %.1f / %.1f -> %s" % (a4, a128, H(a4 >= 95 and a128 <= 80)))
    if choice:
        P("  Dev choice V {2,3} or {3} with ng; L OSA: V %s %s, L %s -> %s / %s" % (
            choice["V_nset"], choice["V_bind"], choice["L_dist"],
            H(choice["V_nset"] in ("23", "3") and choice["V_bind"] == "ng"), H(choice["L_dist"] == "osa")))
    if w is not None:
        P("  Write side: see the write-side section for the copy rates; bars as on the query side: bar1 %s bar2 %s bar3 %s -> %s"
          % (bw["bar1"], bw["bar2"], bw["bar3"], H(bw["bar1"] == bq["bar1"] and bw["bar2"] == bq["bar2"] and bw["bar3"] == bq["bar3"])))


def write_extras(path, out):
    P = lambda *a: out.append(" ".join(str(x) for x in a))
    rows = [json.loads(l) for l in open(path)]
    P("## Write side: re-reading and copying")
    for c in ["C0", "T2", "P", "A"]:
        rs = [r for r in rows if r["cond"] == c]
        rr = sum(r["reread"] for r in rs)
        same = sum(r["reread_same"] for r in rs)
        vs = [x for r in rs for x in r["variant_stored"]]
        P("  %-3s re-read chunks %d, identical to B21's banked output %d (%.4f); queried keys whose variant was stored %d / %d (%.4f)"
          % (c, rr, same, same / max(rr, 1), sum(vs), len(vs), sum(vs) / max(len(vs), 1)))
        if c == "C0":
            c0same = same / max(rr, 1)
    return c0same, {c: np.mean([x for r in rows if r["cond"] == c for x in r["variant_stored"]]) for c in ["C0", "T2", "P", "A"]}


def pcm_report(path, out):
    P = lambda *a: out.append(" ".join(str(x) for x in a))
    H = lambda ok: "HELD" if ok else "NOT HELD"
    t = json.load(open(path))["totals"]
    ks = ["1", "2", "4", "8", "16", "32"]
    times = ["60 s", "1 h", "1 d", "30 d", "1 y"]
    P("## Part b: share of the 4,776 probed keys read exactly on the simulated PCM device (16 pairs per vector)")
    P("  arm   k   " + "".join(x.rjust(9) for x in times) + "   bit errors (VSA store, mean over stores)")
    for k in ks:
        for a in ["VSA", "Trep", "T1"]:
            P("  %-5s x%-3s" % (a, k) + "".join("%9.4f" % t["%s | x%s | %s" % (a, k, tl)][2] for tl in times)
              + ("   " + " ".join("%.3f" % t["bit_err | x%s | %s" % (k, tl)] for tl in times) if a == "VSA" else ""))
    g = lambda a, k, tl: t["%s | x%s | %s" % (a, k, tl)][2]
    h0 = all(g(a, "1", tl) >= 0.995 for a in ("VSA", "Trep", "T1") for tl in times)
    h1 = all(g("VSA", k, tl) >= g("Trep", k, tl) for k in ks for tl in times)
    h2 = g("VSA", "8", "30 d") - g("T1", "8", "30 d") >= 0.05
    P("## Bars, part b")
    P("  H0 (k = 1: every arm >= 99.5%% at every time): %s" % ("PASS" if h0 else "FAIL"))
    worst = min((g("VSA", k, tl) - g("Trep", k, tl), k, tl) for k in ks for tl in times)
    P("  H1 (VSA >= Trep at every k and time): worst VSA - Trep %+.4f at x%s %s -> %s" % (worst[0], worst[1], worst[2], "PASS" if h1 else "FAIL"))
    P("  H2 (VSA - T1 >= 5 points at x8, 30 d): %+.4f -> %s" % (g("VSA", "8", "30 d") - g("T1", "8", "30 d"), "PASS" if h2 else "FAIL"))
    P("## Predictions (part b)")
    P("  H0 passes -> %s" % H(h0))
    d = "30 d"
    P("  T1 at 30 d: 60-90%% at x4 (%.1f), 5-35%% at x8 (%.1f), <= 5%% at x16 (%.1f) -> %s" % (
        100 * g("T1", "4", d), 100 * g("T1", "8", d), 100 * g("T1", "16", d),
        H(0.6 <= g("T1", "4", d) <= 0.9 and 0.05 <= g("T1", "8", d) <= 0.35 and g("T1", "16", d) <= 0.05)))
    P("  VSA at 30 d: >= 99.5%% at x4 (%.1f), 95-100%% at x8 (%.1f), < 1%% at x16 (%.1f) -> %s" % (
        100 * g("VSA", "4", d), 100 * g("VSA", "8", d), 100 * g("VSA", "16", d),
        H(g("VSA", "4", d) >= 0.995 and g("VSA", "8", d) >= 0.95 and g("VSA", "16", d) < 0.01)))
    P("  Trep at 30 d: >= 99.5%% up to x8 (%.1f), >= 95%% at x16 (%.1f), 30-80%% at x32 (%.1f) -> %s" % (
        100 * min(g("Trep", k, d) for k in ("1", "2", "4", "8")), 100 * g("Trep", "16", d), 100 * g("Trep", "32", d),
        H(min(g("Trep", k, d) for k in ("1", "2", "4", "8")) >= 0.995 and g("Trep", "16", d) >= 0.95 and 0.3 <= g("Trep", "32", d) <= 0.8)))
    trep_better = all(g("Trep", k, tl) > g("VSA", k, tl) for k in ("16", "32") for tl in times)
    P("  H1 fails with Trep > VSA at x16 and x32 at every time; H2 passes -> %s" % H((not h1) and trep_better and h2))
    return {"H0": h0, "H1": h1, "H2": h2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("write", nargs="?")
    ap.add_argument("--choice")
    ap.add_argument("--pcm")
    ap.add_argument("--gpu")
    ap.add_argument("--out", default="results/b26_verdict.txt")
    ap.add_argument("--hand", default="results/b26_handread.txt")
    ap.add_argument("--json", default="results/b26_scores.json")
    a = ap.parse_args()
    out, hand = [], []
    out.append("# B26 verdict (prereg/B26_vector_form_prereg.md): does the vector form itself beat an equally exact table?")
    out.append("# Part a: RULER needle tasks, entity names varied in surface form; part b: the store and a table on the PCM model")
    out.append("")
    choice = json.load(open(a.choice)) if a.choice else None
    if choice:
        out.append("## Dev choice (results/b26_dev_choice.json, committed before any test run)")
        out.append("  V: n-gram set %s, binding %s; L: %s; N uses V's n-gram set" % (choice["V_nset"], choice["V_bind"], choice["L_dist"]))
        out.append("  dev scores: " + ", ".join("%s %.2f" % kv for kv in sorted(choice["dev_scores"].items())))
        out.append("")
    by, samples, conds = load(a.query)
    q = side_report(by, samples, conds, "Query", out, hand)
    out.append("")
    bq = bars(q, out, "query")
    out.append("")
    w = bw = None
    if a.write:
        byw, sw, cw = load(a.write)
        w = side_report(byw, sw, cw, "Write", out, hand, write=True)
        out.append("")
        bw = bars(w, out, "write")
        out.append("")
        write_extras(a.write, out)
        out.append("")
    predictions(q, bq, w, bw, choice, out)
    out.append("")
    bp = None
    if a.pcm:
        bp = pcm_report(a.pcm, out)
        out.append("")
    if a.gpu:
        led = json.load(open(a.gpu))
        out.append("## Compute")
        tot = 0.0
        for e in led["jobs"]:
            out.append("  %-34s %-10s %-10s %6.2f GPU-h" % (e["what"], e["host"], e["gpu"], e["gpu_hours"]))
            tot += e["gpu_hours"]
        out.append("  Total %.2f GPU-hours (cap 40). %s" % (tot, led.get("note", "")))
    open(a.out, "w").write("\n".join(out) + "\n")
    open(a.hand, "w").write("\n".join(hand) + "\n")
    json.dump({"query": q, "bars_query": bq, "write": w, "bars_write": bw, "bars_pcm": bp}, open(a.json, "w"), indent=1,
              default=lambda x: x if not isinstance(x, (np.floating, np.integer, np.bool_)) else x.item())
    print("\n".join(out))


if __name__ == "__main__":
    main()
