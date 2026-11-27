"""B30b verdict generator (prereg/B30b_referee_baselines_prereg.md). Written before any B30b test output exists; it
reads only saved outputs and writes results/b30b_verdict.txt, results/b30b_scores.json and the per-question /
per-sample answer files b30b_out/logs/ans_<set>_<arm>.jsonl and b30b_out/bab/scored_<L>_<arm>.jsonl.

python3 -B b30b_verdict.py
"""
import collections
import glob
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import b30b_logs as B  # noqa: E402
import b30b_bab as BB  # noqa: E402

b18 = BB.b18
import babilong_metrics_official as OFF  # noqa: E402  (b30b_frozen copy)

OUTR = os.environ.get("B30B_TEST_ROOT", os.path.join(HERE, "b30b_out"))  # test harness only: fake outputs elsewhere
RESR = os.environ.get("B30B_TEST_RESULTS", os.path.join(HERE, "results"))
LO = os.path.join(OUTR, "logs")
BA = os.path.join(OUTR, "bab")
LABELS = os.path.join(HERE, "b23_data", "labels.json")
BANKED_B18 = {"128k": os.path.join(HERE, "b23_ref", "b18_s0"), "512k": os.path.join(HERE, "b30b_ref", "b18_s0"),
              "1M": os.path.join(HERE, "b30b_ref", "b18_s0")}
LENGTHS = ["128k", "512k", "1M"]
NB = 10000
M_A = 0.05  # part (a) margin
ARMS_B = [("q8T", "Qwen3-8B, thinking, recommended sampling"), ("q17T", "Qwen3-1.7B (reader's base), thinking"),
          ("rdT", "reader (1.7B + LoRA), thinking"), ("q8N", "Qwen3-8B, no thinking, greedy"),
          ("q17N", "Qwen3-1.7B, no thinking, greedy"), ("rdN", "reader, no thinking, greedy")]
SIZE = {"8B": ["q8T", "q8N"], "1.7B": ["q17T", "q17N", "rdT", "rdN"]}

out_lines = []


def P(*a):
    s = " ".join(str(x) for x in a)
    out_lines.append(s)
    print(s)


def jl(p):
    return [json.loads(l) for l in open(p)]


# ------------------------------------------------------------------------------------------------ statistics
def cl_boot(A, Bm, stratify=False, seed=0):
    """cluster_bootstrap.cl_boot, verbatim: whole logs resampled with replacement, pooled ratio, 10,000 resamples."""
    ks = sorted(A)
    cl = sorted({A[k][0] for k in ks})
    dsum = collections.Counter()
    nsum = collections.Counter()
    scen = {}
    for k in ks:
        c = A[k][0]
        dsum[c] += A[k][2] - Bm[k][2]
        nsum[c] += 1
        scen[c] = A[k][1]
    D = np.array([dsum[c] for c in cl], float)
    N = np.array([nsum[c] for c in cl], float)
    rng = np.random.default_rng(seed)
    if not stratify:
        idx = rng.integers(0, len(cl), size=(NB, len(cl)))
        stat = D[idx].sum(1) / N[idx].sum(1)
    else:
        groups = [np.array([i for i, c in enumerate(cl) if scen[c] == s]) for s in sorted(set(scen.values()))]
        num = np.zeros(NB)
        den = np.zeros(NB)
        for g in groups:
            idx = g[rng.integers(0, len(g), size=(NB, len(g)))]
            num += D[idx].sum(1)
            den += N[idx].sum(1)
        stat = num / den
    lo, hi = np.percentile(stat, [2.5, 97.5])
    return float(lo), float(hi), len(cl)


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, j) for j in range(k + 1)) / 2 ** n)


# ------------------------------------------------------------------------------------------------ part (a)
def tri(pq, field=3):
    """per_question (cluster, scenario, strict, forced, extra) -> {k: (cluster, scenario, ok)}"""
    return {k: (v[0], v[1], v[field]) for k, v in pq.items()}


def mean(d):
    return sum(v[2] for v in d.values()) / len(d)


def sub(d, keys):
    return {k: d[k] for k in keys}


def reading_a1(d, lo, hi):
    if d >= M_A and lo > 0:
        return "handicap material"
    if hi < M_A:
        return "no material handicap"
    return "inconclusive"


def reading_a2(lo, hi):
    if lo >= 0.20:
        return "system ahead by >= 0.20"
    if lo > 0:
        return "system ahead"
    if hi < 0:
        return "Qwen3-32B ahead"
    return "tie (interval includes 0)"


def run_stats(path, s):
    rs = [r for r in jl(path) if r["id"].split("|")[0] == s]
    if not rs:
        return None
    th = sorted(r["think_tokens"] for r in rs)
    return {"n": len(rs), "forced": sum(r["phase"] == "force" for r in rs), "continued": sum(r["phase"] == "cont" for r in rs),
            "no_close_stop": sum(r["phase"] == "done" and not r["closed_think"] for r in rs),
            "think_mean": round(sum(th) / len(th)), "think_median": th[len(th) // 2], "think_max": th[-1],
            "gen_tokens": sum(r["gen_tokens"] for r in rs), "past_40960": sum(r["max_position"] > 40960 for r in rs)}


def part_a(J):
    P("## Part (a): Qwen3-32B over the whole log, Qwen3's recommended thinking-mode sampling, 32,768-token reasoning budget")
    P("Sets: location 414 (test2), impact 230 (nsr1), window 460 (nsr2); the paper's prompts (banked problem files) and")
    P("scorers (verbatim copies). Accuracy = the 'forced' score as published (a run at the reasoning limit is scored on its")
    P("budget-forced answer); strict in brackets. Intervals: 95% cluster bootstrap over the 40 logs (paired by question,")
    P("10,000 resamples, numpy seed 0); stratified by scenario in braces.\n")
    J["a"] = {}
    r1p, r2p = os.path.join(LO, "s32_r1.jsonl"), os.path.join(LO, "s32_r2.jsonl")
    for s in B.SETS:
        pub = B.per_question(s, *B.banked(s))
        sysd = B.system(s)
        arms = {"published (greedy, 8,192)": pub}
        ged = os.path.join(LO, "ge", s)
        if glob.glob(os.path.join(ged, f"qwen3-32b_{B.TASK[s]}_n*.jsonl")):
            fn = "forced_qwen3-32b.jsonl" if s == "where" else "forced512_qwen3-32b.jsonl"
            arms["G-E re-run (greedy, 8,192, vLLM)"] = B.per_question(s, *B.b6_dir(s, ged, forced_name=fn))
        for tag, p in (("S32 R1", r1p), ("S32 R2", r2p)):
            if os.path.exists(p):
                runs, forced = B.gen_file(s, p)
                if runs:
                    arms[tag] = B.per_question(s, runs, forced)
                    with open(os.path.join(LO, f"ans_{s}_{tag.replace(' ', '_').lower()}.jsonl"), "w") as f:
                        for k, v in B.answers(s, runs, forced).items():
                            f.write(json.dumps(v) + "\n")
        js = J["a"][s] = {"n": len(pub), "system": round(mean(sysd), 4), "arms": {}}
        P(f"### {s} (n = {len(pub)}); system {mean(sysd):.3f}")
        for name, pq in arms.items():
            fo, st = mean(tri(pq, 3)), mean(tri(pq, 2))
            js["arms"][name] = {"forced": round(fo, 4), "strict": round(st, 4),
                                "missing": sum(1 for v in pq.values() if False)}
            P(f"  {name:36s} {fo:.3f} ({st:.3f})")
        for tag, p in (("S32 R1", r1p), ("S32 R2", r2p)):
            if tag in arms:
                rs = run_stats(p, s)
                js["arms"][tag]["runs"] = rs
                P(f"    {tag} runs: {rs}")
        pubt = tri(pub)
        comps = []
        if "S32 R1" in arms:
            r1 = tri(arms["S32 R1"])
            comps += [("D1  S32 R1 - published", r1, pubt), ("D2  system - S32 R1", sysd, r1)]
            if "G-E re-run (greedy, 8,192, vLLM)" in arms:
                comps.append(("D1e S32 R1 - G-E re-run (same engine)", r1, tri(arms["G-E re-run (greedy, 8,192, vLLM)"])))
            if "S32 R2" in arms:
                r2 = tri(arms["S32 R2"])
                both = {k: (r1[k][0], r1[k][1], (r1[k][2] + r2[k][2]) / 2) for k in r1}
                comps += [("    S32 R2 - S32 R1 (seed spread)", r2, r1), ("    system - S32 R2", sysd, r2),
                          ("    system - mean(R1, R2)", sysd, both), ("    mean(R1, R2) - published", both, pubt)]
        if "G-E re-run (greedy, 8,192, vLLM)" in arms:
            comps.append(("G-E re-run - published (engine)", tri(arms["G-E re-run (greedy, 8,192, vLLM)"]), pubt))
        comps.append(("    system - published (as in the paper)", sysd, pubt))
        js["comparisons"] = {}
        for name, A, Bm in comps:
            d = mean(A) - mean(Bm)
            lo, hi, ncl = cl_boot(A, Bm)
            slo, shi, _ = cl_boot(A, Bm, stratify=True)
            js["comparisons"][name.strip()] = {"d": round(d, 4), "ci": [round(lo, 4), round(hi, 4)], "ci_strat": [round(slo, 4), round(shi, 4)]}
            P(f"  {name:44s} {d:+.3f} [{lo:+.3f}, {hi:+.3f}] {{{slo:+.3f}, {shi:+.3f}}}")
        if "S32 R1" in arms:
            c1, c2 = js["comparisons"]["D1  S32 R1 - published"], js["comparisons"]["D2  system - S32 R1"]
            js["A1"] = reading_a1(c1["d"], *c1["ci"])
            js["A2"] = reading_a2(*c2["ci"])
            P(f"  A1 (was the baseline handicapped?): {js['A1']}")
            P(f"  A2 (does the system still lead?):   {js['A2']}")
        if "G-E re-run (greedy, 8,192, vLLM)" in arms and s in ("impact", "window"):
            ge = arms["G-E re-run (greedy, 8,192, vLLM)"]
            d = mean(tri(ge)) - mean(pubt)
            gans = B.answers(s, *B.b6_dir(s, ged, forced_name="forced512_qwen3-32b.jsonl"))
            pans = B.answers(s, *B.banked(s))
            same = sum(gans[k]["pred"] == pans[k]["pred"] for k in pans)
            js["G-E"] = {"d": round(d, 4), "pass": abs(d) <= 0.03, "same_answer": same}
            P(f"  G-E: re-run greedy {mean(tri(ge)):.3f} vs published {mean(pubt):.3f} (|d| <= 0.03: "
              f"{'PASS' if abs(d) <= 0.03 else 'FAIL'}); same parsed answer {same}/{len(pans)}")
        if s == "impact":
            keep = [k for k, v in pub.items() if B.AMBIG not in v[4]["q"]]
            P(f"  excluding the {len(pub) - len(keep)} ambiguous home questions ('{B.AMBIG}...'), n = {len(keep)}:")
            js["excl_ambiguous"] = {"n": len(keep), "system": round(mean(sub(sysd, keep)), 4)}
            for name, pq in arms.items():
                v = mean(sub(tri(pq), keep))
                js["excl_ambiguous"][name] = round(v, 4)
                P(f"    {name:34s} {v:.3f}")
            P(f"    system                             {mean(sub(sysd, keep)):.3f}")
            if "S32 R1" in arms:
                r1 = sub(tri(arms["S32 R1"]), keep)
                for name, A, Bm in (("D1x S32 R1 - published", r1, sub(pubt, keep)), ("D2x system - S32 R1", sub(sysd, keep), r1)):
                    d = mean(A) - mean(Bm)
                    lo, hi, _ = cl_boot(A, Bm)
                    js["excl_ambiguous"][name] = {"d": round(d, 4), "ci": [round(lo, 4), round(hi, 4)]}
                    P(f"    {name:32s} {d:+.3f} [{lo:+.3f}, {hi:+.3f}]")
                js["A3"] = reading_a2(*js["excl_ambiguous"]["D2x system - S32 R1"]["ci"])
                P(f"    A3 (system lead on the 150 unambiguous impact questions): {js['A3']}")
        P("")


# ------------------------------------------------------------------------------------------------ part (b)
def answer_text(r):
    if r.get("mode") == "think32k":
        return r["forced_answer"] if r["hit_limit"] else r["answer_text"]
    return r["text"]


def part_b(J):
    for t, v in json.load(open(LABELS)).items():
        b18._LABELS[t] = set(v)
    P("## Part (b): BABILong, the same reader's events as a notepad (analogue of RULER's facts-in-prompt arm)")
    P("QA1-5, official test release; memory = b18.World over the re-run reader's grounded events; notepad = the same")
    P("events as text lines, answered by a language model. Scorer: b18.correct_official on the whole answer (official")
    P("compare_answers in brackets where it differs). Paired by sample: difference in points (memory - arm), discordant")
    P("pairs (memory only / arm only), exact two-sided McNemar p.\n")
    J["b"] = {}
    for L in LENGTHS:
        ap = os.path.join(BA, f"assembled_{L}.jsonl")
        if not os.path.exists(ap):
            continue
        rows = jl(ap)
        jb = J["b"][L] = {"n": len(rows)}
        # gate G-B
        same = okb = 0
        for t in BB.TASKS:
            old = jl(glob.glob(f"{BANKED_B18[L]}/{L}_{t}.jsonl")[0])
            new = [r for r in rows if r["task"] == t]
            assert len(old) == len(new)
            for o, r in zip(old, new):
                assert o["question"] == r["question"] and str(o["target"]) == str(r["target"])
                same += o["pred"] == r["pred"]
                okb += bool(o["ok_official"])
        mem = {(r["task"], r["i"]): bool(b18.correct_official(r["pred"], r["target"], r["question"], r["task"])) for r in rows}
        n = len(rows)
        am, ab = 100 * sum(mem.values()) / n, 100 * okb / n
        gate = same >= 0.99 * n and abs(am - ab) <= 1.0
        jb["G-B"] = {"same_pred": same, "memory": round(am, 2), "banked": round(ab, 2), "pass": gate}
        nl = sorted(len(r["events"]) for r in rows)
        jb["notepad_lines"] = {"median": nl[n // 2], "p90": nl[int(0.9 * n)], "max": nl[-1]}
        P(f"### {L} (n = {n}): G-B memory answers equal to banked B18 {same}/{n}; memory {am:.1f} vs banked {ab:.1f} -> "
          f"{'PASS' if gate else 'FAIL'}; notepad lines median {nl[n // 2]}, 90th pct {nl[int(0.9 * n)]}, max {nl[-1]}")
        lo, hi = wilson(sum(mem.values()), n)
        P(f"  {'memory (B18 rules over the events)':44s} {am:5.1f} [{100 * lo:.1f}, {100 * hi:.1f}]  " +
          " ".join(f"{t} {100 * sum(v for (tt, _), v in mem.items() if tt == t) / sum(1 for (tt, _) in mem if tt == t):.0f}" for t in BB.TASKS))
        jb["memory"] = round(am, 2)
        jb["arms"] = {}
        for tag, name in ARMS_B:
            fp = os.path.join(BA, f"ans_{L}_{tag}.jsonl")
            if not os.path.exists(fp):
                continue
            got = {}
            for r in jl(fp):
                _, t, i = r["id"].split("|")
                got[(t, int(i))] = r
            if len(got) != n:
                P(f"  {name}: {len(got)}/{n} answered, skipped")
                continue
            ok, ok_off, sc = {}, {}, []
            for r in rows:
                k = (r["task"], r["i"])
                txt = answer_text(got[k])
                ok[k] = bool(b18.correct_official(txt, r["target"], r["question"], r["task"]))
                ok_off[k] = bool(OFF.compare_answers(r["target"], txt, r["question"], OFF.TASK_LABELS[r["task"]]))
                sc.append({"id": got[k]["id"], "task": r["task"], "i": r["i"], "target": r["target"], "answer": txt,
                           "ok": ok[k], "ok_official": ok_off[k], "memory_pred": r["pred"], "memory_ok": mem[k],
                           "hit_limit": got[k].get("hit_limit"), "think_tokens": got[k].get("think_tokens"),
                           "n_lines": len(r["events"])})
            with open(os.path.join(BA, f"scored_{L}_{tag}.jsonl"), "w") as f:
                for x in sc:
                    f.write(json.dumps(x) + "\n")
            a = 100 * sum(ok.values()) / n
            ao = 100 * sum(ok_off.values()) / n
            bm = sum(mem[k] and not ok[k] for k in mem)
            ca = sum(ok[k] and not mem[k] for k in mem)
            p = mcnemar_exact(bm, ca)
            d = am - a
            lo, hi = wilson(sum(ok.values()), n)
            if d >= 1.0 and p < 0.05:
                rd = "memory adds"
            elif d < 0 and p < 0.05:
                rd = "notepad better"
            else:
                rd = "not distinguishable"
            hits = sum(bool(x["hit_limit"]) for x in sc)
            jb["arms"][tag] = {"acc": round(a, 2), "acc_official": round(ao, 2), "wilson95": [round(100 * lo, 1), round(100 * hi, 1)],
                               "d": round(d, 2), "memory_only": bm, "arm_only": ca, "p": p, "reading": rd,
                               "per_task": {t: round(100 * sum(v for (tt, _), v in ok.items() if tt == t) / sum(1 for (tt, _) in ok if tt == t), 1) for t in BB.TASKS},
                               "hit_limit": hits}
            P(f"  {name:44s} {a:5.1f} [{100 * lo:.1f}, {100 * hi:.1f}]" + (f" (official {ao:.1f})" if ao != a else "") +
              f"  d {d:+.1f}, discordant {bm}/{ca}, p {p:.2g} -> {rd}; " +
              " ".join(f"{t} {v:.0f}" for t, v in jb["arms"][tag]["per_task"].items()) + (f"; forced {hits}" if hits else ""))
        for size, tags in SIZE.items():
            have = [t for t in tags if t in jb["arms"]]
            if not have:
                continue
            allr = [jb["arms"][t]["reading"] for t in have]
            verdict = "memory adds" if all(r == "memory adds" for r in allr) else (
                "notepad better" if any(r == "notepad better" for r in allr) else "not shown")
            best = max(have, key=lambda t: jb["arms"][t]["acc"])
            jb[f"conservative_{size}"] = {"arms": have, "complete": len(have) == len(tags), "verdict": verdict, "best": best}
            P(f"  conservative rule, {size} answerers ({', '.join(have)}{'' if len(have) == len(tags) else '; INCOMPLETE'}): "
              f"{verdict} (best notepad arm {best} {jb['arms'][best]['acc']:.1f})")
        P("")
    pr = J["b"].get("128k", {}).get("arms", {})
    if "q8T" in pr:
        J["NB1"] = pr["q8T"]["reading"]
    if "q17T" in pr:
        J["NB2"] = pr["q17T"]["reading"]
    P(f"NB1 (128k, Qwen3-8B thinking): {J.get('NB1', 'not run')};  NB2 (128k, Qwen3-1.7B thinking): {J.get('NB2', 'not run')}\n")


# ------------------------------------------------------------------------------------------------ ledger
def ledger(J):
    p = os.path.join(OUTR, "gpu_ledger.txt")
    if not os.path.exists(p):
        return
    import datetime as dt
    st, tot, rows = {}, 0.0, []
    for l in open(p):
        w = l.split()
        name, gpus, kind, t = w[0], w[1].split("=")[1], w[2], w[3]
        ts = dt.datetime.strptime(t, "%Y-%m-%dT%H:%M:%SZ")
        if kind == "start":
            st[(name, gpus)] = ts
        elif (name, gpus) in st:
            h = (ts - st.pop((name, gpus))).total_seconds() / 3600 * len(gpus.split(","))
            tot += h
            rows.append((name, gpus, round(h, 2), w[4] if len(w) > 4 else ""))
    J["gpu_hours"] = round(tot, 2)
    J["gpu_jobs"] = rows
    P(f"## Compute: {tot:.2f} A800 GPU-hours over {len(rows)} finished jobs (cap 80); still running: {len(st)}")
    for r in rows:
        P(f"  {r[0]:28s} gpus {r[1]:8s} {r[2]:6.2f} h  {r[3]}")


def predictions(J):
    """The registered predictions (prereg, 'Predictions'), checked mechanically where the outputs allow."""
    P("## Registered predictions vs outcome")
    out = []

    def chk(text, ok):
        out.append((text, ok))
        P(f"  [{'held' if ok else 'FAILED' if ok is not None else 'not run'}] {text}")
    a = J.get("a", {})
    ge = [a.get(s, {}).get("G-E") for s in ("impact", "window")]
    if all(ge):
        chk(f"G-E passes on impact and window with >= 80% identical parsed answers (pass: {[g['pass'] for g in ge]}; "
            f"identical {ge[0]['same_answer']}/230, {ge[1]['same_answer']}/460)",
            all(g["pass"] for g in ge) and ge[0]["same_answer"] >= 0.8 * 230 and ge[1]["same_answer"] >= 0.8 * 460)
    w = a.get("where", {}).get("comparisons", {}).get("G-E re-run - published (engine)")
    chk(f"location under vLLM greedy within 0.05 of the published HF number (d = {w['d'] if w else None})",
        abs(w["d"]) <= 0.05 if w else None)
    rng = {"where": (0.55, 0.68), "impact": (0.28, 0.40), "window": (0.20, 0.32)}
    for s, (lo, hi) in rng.items():
        v = a.get(s, {}).get("arms", {}).get("S32 R1", {}).get("forced")
        chk(f"S32 R1 {s} in [{lo}, {hi}] (got {v})", lo <= v <= hi if v is not None else None)
    ex = a.get("impact", {}).get("excl_ambiguous", {}).get("S32 R1")
    chk(f"S32 R1 impact on the 150 unambiguous questions about 0.32 (+-0.05; got {ex})", abs(ex - 0.32) <= 0.05 if ex is not None else None)
    d1 = {s: a.get(s, {}).get("comparisons", {}).get("D1  S32 R1 - published", {}).get("d") for s in rng}
    chk(f"D1 in [0, +0.10] on every set (got {d1})", all(0 <= v <= 0.10 for v in d1.values()) if None not in d1.values() else None)
    a1 = [a.get(s, {}).get("A1") for s in rng]
    chk(f"A1 'handicap material' on at most one set (got {a1})", sum(x == "handicap material" for x in a1) <= 1 if None not in a1 else None)
    a2 = [a.get(s, {}).get("A2") for s in rng] + [a.get("impact", {}).get("A3")]
    chk(f"A2 'system ahead by >= 0.20' on all three sets and A3 too (got {a2})",
        all(x == "system ahead by >= 0.20" for x in a2) if None not in a2 else None)
    rs = {s: a.get(s, {}).get("arms", {}).get("S32 R1", {}).get("runs") for s in rng}
    if None not in rs.values():
        chk("runs forced at the 32,768 budget <= 5% per set (got " + ", ".join(f"{s} {r['forced']}/{r['n']}" for s, r in rs.items()) + ")",
            all(r["forced"] <= 0.05 * r["n"] for r in rs.values()))
        chk("runs past position 40,960 <= 2% per set (got " + ", ".join(f"{s} {r['past_40960']}/{r['n']}" for s, r in rs.items()) + ")",
            all(r["past_40960"] <= 0.02 * r["n"] for r in rs.values()))
    r2 = {s: a.get(s, {}).get("comparisons", {}).get("S32 R2 - S32 R1 (seed spread)", {}).get("d") for s in rng}
    chk(f"R2 within 0.03 of R1 on every set (got {r2})", all(abs(v) <= 0.03 for v in r2.values()) if None not in r2.values() else None)
    b = J.get("b", {})
    chk("G-B passes at every length run (" + ", ".join(f"{L} {v['G-B']['pass']}" for L, v in b.items()) + ")",
        all(v["G-B"]["pass"] for v in b.values()) if b else None)
    b128 = b.get("128k", {})
    if b128:
        chk(f"notepads at 128k: median <= 100 lines (got {b128['notepad_lines']['median']})", b128["notepad_lines"]["median"] <= 100)
        ar = b128.get("arms", {})
        if "q8T" in ar:
            pt = ar["q8T"]["per_task"]
            chk(f"q8T at 128k in [85, 98] (got {ar['q8T']['acc']})", 85 <= ar["q8T"]["acc"] <= 98)
            chk(f"qa3 is q8T's weakest task at 128k (per task {pt})", min(pt, key=pt.get) == "qa3")
            chk(f"NB1 'memory adds' (got {ar['q8T']['reading']})", ar["q8T"]["reading"] == "memory adds")
        if "q17T" in ar:
            chk(f"q17T at 128k in [60, 92] (got {ar['q17T']['acc']})", 60 <= ar["q17T"]["acc"] <= 92)
            chk(f"NB2 'memory adds' (got {ar['q17T']['reading']})", ar["q17T"]["reading"] == "memory adds")
        pairs = [("q8N", "q8T"), ("q17N", "q17T"), ("rdN", "rdT")]
        if all(x in ar and y in ar for x, y in pairs):
            chk("every N arm >= 5 points below the T arm of the same model (" +
                ", ".join(f"{y} {ar[y]['acc']} vs {x} {ar[x]['acc']}" for x, y in pairs) + ")",
                all(ar[y]["acc"] - ar[x]["acc"] >= 5 for x, y in pairs))
        cr = [b128.get(f"conservative_{k}", {}) for k in ("8B", "1.7B")]
        if all(c.get("complete") for c in cr):
            chk(f"conservative rule 'memory adds' for both sizes at 128k (got {[c['verdict'] for c in cr]})",
                all(c["verdict"] == "memory adds" for c in cr))
    for L in ("512k", "1M"):
        if "q8T" in b.get(L, {}).get("arms", {}) and "q8T" in b128.get("arms", {}):
            v, v0 = b[L]["arms"]["q8T"]["acc"], b128["arms"]["q8T"]["acc"]
            chk(f"q8T at {L} at least 2 points below its 128k score (got {v} vs {v0})", v <= v0 - 2)
    J["predictions"] = [{"text": t, "held": ok} for t, ok in out]
    P(f"  {sum(ok is True for _, ok in out)} held, {sum(ok is False for _, ok in out)} failed, "
      f"{sum(ok is None for _, ok in out)} not run\n")


def main():
    B.ungz(OUTR)
    J = {}
    P("# B30b verdict (prereg/B30b_referee_baselines_prereg.md): two referee baselines that may favour the system\n")
    part_a(J)
    part_b(J)
    predictions(J)
    ledger(J)
    os.makedirs(RESR, exist_ok=True)
    json.dump(J, open(os.path.join(RESR, "b30b_scores.json"), "w"), indent=1)
    open(os.path.join(RESR, "b30b_verdict.txt"), "w").write("\n".join(out_lines) + "\n")


if __name__ == "__main__":
    main()
