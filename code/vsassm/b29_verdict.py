"""B29 verdict generator (prereg/B29_same_tooling_prereg.md). Written before any B29 test output existed.

python3 -I -B b29_verdict.py [--out results/b29_verdict.txt] [--json results/b29_scores.json]

Reads only saved per-sample / per-question outputs (b29_out/), B21's banked outputs (b21_dl/b21_test) and, read-only,
the banked whole-log comparators through cluster_bootstrap.py's loaders. Prints every arm that exists, the registered
bars and predictions, the gates, and the GPU-hour and spend ledgers. A missing arm is printed as "not run".
"""
import argparse
import collections
import glob
import json
import os
import site
import sys

if os.path.isdir(site.getusersitepackages()) and site.getusersitepackages() not in sys.path:
    sys.path.append(site.getusersitepackages())
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import ruler_rm as R  # noqa: E402
import b29_ruler as BR  # noqa: E402
import b29_logs as BL  # noqa: E402

RO = os.path.join(ROOT, "b29_out", "ruler")
LO = os.path.join(ROOT, "b29_out", "logs")
LEN = BR.LENGTHS
TASKS = BR.TASKS
B = 10000
MARGIN_R, MARGIN_L = 1.0, 0.05          # registered margins: RULER points, log accuracy
LOG_SETS = [("where", "test2"), ("impact", "nsr1"), ("window", "nsr2")]
PAPER = {"where": 0.959, "impact": 0.930, "window": 0.874}   # banked 1.7B-reader accuracies (paper Table 1)


def jl(p):
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else None


# ------------------------------------------------------------------------------------------------------ RULER
def sample_scores(rows, field):
    """(length, task) -> {i: score in [0, 1]} (official string_match_all)."""
    out = collections.defaultdict(dict)
    for r in rows:
        if r.get("missing") or field not in r:
            continue
        out[(int(r["length"]), r["task"])][r["i"]] = R.string_match_all(r[field], r["outputs"])
    return out


def b21_scores():
    b = BR.b21_rows(os.path.join(ROOT, "b21_dl", "b21_test"))
    d, s = collections.defaultdict(dict), collections.defaultdict(dict)
    for (L, t, i), r in b.items():
        d[(L, t)][i] = R.string_match_all(r["pred_table"], r["outputs"])
        s[(L, t)][i] = R.string_match_all(r["pred_vsa"], r["outputs"])
    return d, s, b


def mean9(sc, L, tasks=TASKS):
    if not all(len(sc.get((L, t), {})) == 100 for t in tasks):
        return None
    return 100 * float(np.mean([v for t in tasks for v in sc[(L, t)].values()]))


def paired9(a, b, L, rng, tasks=TASKS):
    """mean(a) - mean(b) over the tasks at L with a 95% bootstrap interval resampling the 100 samples within each task
    (stratified; the same draw for both arms), as b24_verdict.paired."""
    if not all(len(a.get((L, t), {})) == 100 and len(b.get((L, t), {})) == 100 for t in tasks):
        return None
    da = np.stack([np.array([a[(L, t)][i] for i in range(100)]) - np.array([b[(L, t)][i] for i in range(100)])
                   for t in tasks])
    n = da.shape[1]
    idx = rng.integers(0, n, size=(B, len(tasks), n))
    bs = np.take_along_axis(np.broadcast_to(da, (B, len(tasks), n)), idx, axis=2).mean(axis=(1, 2))
    return 100 * float(da.mean()), 100 * float(np.percentile(bs, 2.5)), 100 * float(np.percentile(bs, 97.5))


def reading(diff, margin):
    """The registered three-way reading of a paired difference (larger reader - small reader)."""
    if diff is None:
        return "not run"
    m, lo, hi = diff
    if m >= margin and lo > 0:
        return "SIZE MATTERS (gain >= margin, interval above 0)"
    if hi < 0:
        return "LARGER READER WORSE (interval below 0)"
    if hi < margin:
        return "SMALL READER SUFFICES (interval below the margin)"
    return "INCONCLUSIVE"


def ruler_part(P, J):
    d8, s8, b21 = b21_scores()
    arms = {"8B (B21, banked)": (d8, s8)}
    files = {"8B replay (gate G1)": "test_rel_q8.jsonl", "32B": "test_rel_q32.jsonl", "1.7B": "test_rel_q17.jsonl",
             "32B full read (vLLM)": "test_full_q32.jsonl"}
    rows = {}
    for name, f in files.items():
        rws = jl(os.path.join(RO, f))
        rows[name] = rws
        if rws is not None:
            arms[name] = (sample_scores(rws, "pred_dict"), sample_scores(rws, "pred_store"))
    J["ruler"] = {"means": {}, "diffs": {}, "gates": {}, "bars": {}}
    P("## RULER memory tasks (9 tasks: niah_single_1-3, niah_multikey_1-3, niah_multivalue, niah_multiquery, vt)")
    P("B19 seed-42 test sets, 100 samples per task and length, official string_match_all. Same extraction prompt, chunking,")
    P("grounding, dict (B21's table) and sharded Fourier HRR store for every reader; only the reader model differs.\n")
    P("9-task mean, dict (plain table) answers")
    P("reader".ljust(26) + "".join(f"{L // 1024}k".rjust(8) for L in LEN))
    for name, (dsc, _) in arms.items():
        cells = [mean9(dsc, L) for L in LEN]
        J["ruler"]["means"][name] = {str(L): (None if c is None else round(c, 2)) for L, c in zip(LEN, cells)}
        P(name.ljust(26) + "".join((f"{c:8.1f}" if c is not None else "       -") for c in cells))
    P("\n128k per task, dict answers")
    P("reader".ljust(26) + "".join(t.replace("niah_", "")[:10].rjust(11) for t in TASKS))
    for name, (dsc, _) in arms.items():
        cells = [100 * float(np.mean(list(dsc[(131072, t)].values()))) if len(dsc.get((131072, t), {})) == 100 else None
                 for t in TASKS]
        P(name.ljust(26) + "".join((f"{c:11.1f}" if c is not None else "          -") for c in cells))
    rng = np.random.default_rng(0)
    P("\nPaired differences, 9-task mean, dict answers (points; 95% stratified paired bootstrap, 10,000 resamples, seed 0)")
    P("comparison".ljust(34) + "".join(f"{L // 1024}k".rjust(22) for L in LEN))
    for a, b in (("32B", "8B (B21, banked)"), ("1.7B", "8B (B21, banked)"), ("8B replay (gate G1)", "8B (B21, banked)"),
                 ("32B full read (vLLM)", "32B")):
        if a not in arms:
            P(f"{a} - {b}".ljust(34) + "  not run")
            continue
        cells = []
        for L in LEN:
            d = paired9(arms[a][0], arms[b][0], L, rng)
            J["ruler"]["diffs"].setdefault(f"{a} - {b}", {})[str(L)] = d
            cells.append(f"{d[0]:+5.2f} [{d[1]:+5.2f},{d[2]:+5.2f}]".rjust(22) if d else "-".rjust(22))
        P(f"{a} - {b}".ljust(34) + "".join(cells))
    # store vs dict on the same reader output
    P("\nStore vs dict on the same reader output (identical predictions / samples)")
    for name in ("8B (B21, banked)", "32B full read (vLLM)", "32B", "1.7B"):
        if name not in arms:
            P(f"  {name}: not run")
            continue
        if name == "8B (B21, banked)":
            same = sum(r["pred_table"] == r["pred_vsa"] for r in b21.values())
            P(f"  {name}: {same}/{len(b21)} (all lengths; B21)")
            continue
        rs = [r for r in rows[name] if "pred_store" in r]
        by = collections.Counter()
        tot = collections.Counter()
        for r in rs:
            by[r["length"]] += r["pred_store"] == r["pred_dict"]
            tot[r["length"]] += 1
        P(f"  {name}: " + ", ".join(f"{L // 1024}k {by[L]}/{tot[L]}" for L in sorted(tot)) +
          ("  (VT only: NIAH store needs every chunk read)" if "full" not in name else ""))
        J["ruler"].setdefault("store_eq_dict", {})[name] = {str(L): [by[L], tot[L]] for L in sorted(tot)}
    if "32B full read (vLLM)" in arms:
        dsc, ssc = arms["32B full read (vLLM)"]
        m_s, m_d = mean9(ssc, 131072), mean9(dsc, 131072)
        P(f"  32B full read, 128k 9-task mean: store {m_s:.2f}, dict {m_d:.2f}" if m_s is not None else
          "  32B full read: incomplete")
    # gates
    P("\nGates")
    g1 = None
    if rows["8B replay (gate G1)"] is not None:
        per, diffs = gate_g1(rows["8B replay (gate G1)"], b21)
        ok_all = True
        for L in LEN:
            c = per.get(L)
            if not c:
                ok_all = False
                continue
            chunk_ok = c[0] >= 0.99 * c[1]
            samp_ok = c[2] >= 0.99 * c[3]
            d = J["ruler"]["diffs"].get("8B replay (gate G1) - 8B (B21, banked)", {}).get(str(L))
            mean_ok = d is not None and abs(d[0]) <= 0.25
            ok_all &= chunk_ok and samp_ok and mean_ok
            P(f"  G1 {L // 1024:>3}k: chunks equal {c[0]}/{c[1]}, samples equal {c[2]}/{c[3]}, mean diff "
              f"{d[0]:+.2f}" if d else f"  G1 {L // 1024}k: incomplete")
        g1 = ok_all
        P(f"  G1 (8B replay reproduces B21: >= 99% chunks, >= 99% samples, |mean diff| <= 0.25 at every length): "
          f"{'PASS' if g1 else 'FAIL'}")
        J["ruler"]["gates"]["G1"] = g1
        for dd in diffs[:12]:
            P("    diff: " + json.dumps(dd)[:300])
    else:
        P("  G1: not run")
    g0 = os.path.join(RO, "g0_b21_replay.txt")
    if os.path.exists(g0):
        P("  G0 (CPU, before registration): " + " | ".join(l.strip() for l in open(g0) if l.startswith("G0")))
    # bars
    P("\nBars (registered)")
    d32 = J["ruler"]["diffs"].get("32B - 8B (B21, banked)", {}).get("131072")
    P(f"  RS1 (32B - 8B at 128k, dict; margin {MARGIN_R} point): " +
      (f"{d32[0]:+.2f} [{d32[1]:+.2f}, {d32[2]:+.2f}] -> {reading(d32, MARGIN_R)}" if d32 else "not run") +
      ("" if g1 else "   [G1 has not passed: does not count]"))
    J["ruler"]["bars"]["RS1"] = reading(d32, MARGIN_R) if d32 else None
    eq = J["ruler"].get("store_eq_dict", {}).get("32B full read (vLLM)", {}).get("131072")
    rs2 = None if not eq or eq[1] < 900 else eq[0] >= 0.99 * eq[1]
    P(f"  RS2 (32B store = dict on >= 99% of the 900 samples at 128k): " +
      (f"{eq[0]}/{eq[1]} -> {'PASS' if rs2 else 'FAIL'}" if rs2 is not None else "not run / incomplete"))
    J["ruler"]["bars"]["RS2"] = rs2
    if rows["32B full read (vLLM)"] is not None and rows["32B"] is not None:
        a = {(r["length"], r["task"], r["i"]): r["pred_dict"] for r in rows["32B full read (vLLM)"] if "pred_dict" in r}
        b = {(r["length"], r["task"], r["i"]): r["pred_dict"] for r in rows["32B"] if "pred_dict" in r}
        ks = sorted(set(a) & set(b))
        same = sum(a[k] == b[k] for k in ks)
        rs3 = same >= 0.98 * len(ks) if ks else None
        P(f"  RS3 (cross-engine: 32B dict answers, vLLM full read = HF needed-chunk read, >= 98%): {same}/{len(ks)} -> "
          f"{'PASS' if rs3 else 'FAIL'}")
        J["ruler"]["bars"]["RS3"] = [same, len(ks)]
    else:
        P("  RS3: not run")
    d17 = J["ruler"]["diffs"].get("1.7B - 8B (B21, banked)", {}).get("131072")
    P(f"  RS4 (secondary, 1.7B - 8B at 128k): " + (f"{d17[0]:+.2f} [{d17[1]:+.2f}, {d17[2]:+.2f}] -> "
                                                    f"{'smaller reader worse' if d17[2] < 0 else 'not worse'}"
                                                    if d17 else "not run"))
    return g1


def gate_g1(rows, b21):
    per = collections.defaultdict(lambda: [0, 0, 0, 0])
    diffs = []
    for r in rows:
        b = b21[(r["length"], r["task"], r["i"])]
        assert b["outputs"] == r["outputs"]
        c = per[r["length"]]
        for j, txt in (r.get("raw") or {}).items():
            c[0] += txt == b["raw"][int(j)]
            c[1] += 1
            if txt != b["raw"][int(j)]:
                diffs.append(["chunk", r["length"], r["task"], r["i"], int(j), b["raw"][int(j)][:80], txt[:80]])
        if r["task"] == "vt":   # VT reads every chunk: compare the grounded pairs instead of 2,000 raw strings
            c[0] += [list(p) for p in r["pairs_q"]] == [list(p) for p in b["pairs"]]
            c[1] += 1
        c[2] += r["pred_dict"] == b["pred_table"]
        c[3] += 1
        if r["pred_dict"] != b["pred_table"]:
            diffs.append(["sample", r["length"], r["task"], r["i"], b["pred_table"][:80], r["pred_dict"][:80]])
    return per, diffs


# ------------------------------------------------------------------------------------------------------- logs
def log_correct(split, tag, field="pred"):
    recs = jl(os.path.join(LO, f"ans_{split}_{tag}.jsonl"))
    if recs is None:
        return None
    c = BL.correct(split, recs, field)
    return {r["id"]: (r["log"], r["scenario"], bool(c[r["id"]])) for r in recs}


def banked_comparators():
    """Whole-log comparators, per question, through cluster_bootstrap.py's loaders (read-only)."""
    import cluster_bootstrap as CB  # noqa: E402
    out = {}
    loc = CB.location("qwen3-32b")
    out[("where", "Qwen3-32B whole log (forced)")] = {f"{k[0]}|{k[1]}": v for k, v in loc.items()}
    imp = CB.impact_1day("arm3_qwen3-32b_forced")
    out[("impact", "Qwen3-32B whole log (forced)")] = {f"{k[0]}|{k[1]}|{k[2]}": v for k, v in imp.items()}
    imp = CB.impact_1day("arm3_gptD_strict")
    out[("impact", "GPT-6.1 Sol whole log (xhigh, OpenAI direct)")] = {f"{k[0]}|{k[1]}|{k[2]}": v for k, v in imp.items()}
    win = CB.window_1day("llm_qwen3-32b_forced")
    out[("window", "Qwen3-32B whole log (forced)")] = dict(win)
    win = CB.window_1day("gptD_strict")
    out[("window", "GPT-6.1 Sol whole log (xhigh, OpenAI direct)")] = dict(win)
    return out, CB


def acc(A):
    return sum(v[2] for v in A.values()) / len(A)


def cl_diff(A, Bm, CB, stratify=False):
    ks = sorted(set(A) & set(Bm))
    A2, B2 = {k: A[k] for k in ks}, {k: Bm[k] for k in ks}
    lo, hi, n = CB.cl_boot(A2, B2, stratify=stratify)
    m = sum(A2[k][2] - B2[k][2] for k in ks) / len(ks)
    return m, lo, hi, len(ks), n


def failed_logs(split, tag):
    """Frontier calls that failed after the client's retries (logged as text "" with 0 tokens) and the logs they belong
    to: a failed prompt contains its message exactly as the pipeline shows it ("Sentence: <msg>" / "Message: <msg>")."""
    import re as _re
    p = os.path.join(LO, f"calls_{split}_{tag}.jsonl")
    if not os.path.exists(p):
        return None
    calls = jl(p)
    failed = [c["p"] for c in calls if c["t"] == "" and c["n"] == 0]
    logs = BL.logs_of(split)
    msgs = {}
    for lid, l in logs.items():
        for i, line in enumerate(l["lines"]):
            if line["kind"] == "human":
                m = _re.sub(r"^\[\d\d:\d\d\]\s*|^\d\d:\d\d\s+SPEECH\s+", "", l["text_lines"][i])
                msgs.setdefault(m, set()).add(lid)
    aff = set()
    for fp in failed:
        hit = [m for m in msgs if f"Sentence: {m}\n" in fp or f"Message: {m}\n" in fp]
        for m in hit:
            aff |= msgs[m]
    return len(calls), len(failed), sorted(aff)


def log_part(P, J, g1):
    comps, CB = banked_comparators()
    J["logs"] = {"acc": {}, "diffs": {}, "gates": {}, "bars": {}, "dict": {}}
    P("\n## Operational logs: the same pipeline (prompt C with its worked examples, forced choice, yes/no check, registry,")
    P("connectors, 256 x 256 matrix memory, engines) with the reader model varied. Set-exact accuracy; paired differences")
    P("with a 95% cluster bootstrap over logs (10,000 resamples, numpy seed 0; cluster_bootstrap.cl_boot), larger - small.\n")
    readers = [("1.7B fine-tuned (banked)", "banked"), ("1.7B fine-tuned (re-run, gate G2)", "ft17"),
               ("Qwen3-8B prompted", "q8"), ("Qwen3-32B prompted", "q32"), ("GPT-6.1 Sol (low) prompted", "gpt61low"),
               ("Claude Opus 5.5 (low) prompted", "opus_low")]
    gates = {}
    for kind, split in LOG_SETS:
        P(f"### {kind} questions ({split})")
        base = log_correct(split, "banked")
        J["logs"]["acc"][kind] = {}
        for name, tag in readers:
            A = log_correct(split, tag)
            if A is None:
                P(f"  {name:44s} not run")
                continue
            line = f"  {name:44s} {acc(A):.3f} ({sum(v[2] for v in A.values())}/{len(A)})"
            J["logs"]["acc"][kind][name] = round(acc(A), 4)
            if tag != "banked":
                D = log_correct(split, tag + "_dict")
                if D is not None:
                    recs = {r["id"]: r for r in jl(os.path.join(LO, f"ans_{split}_{tag}.jsonl"))}
                    recd = {r["id"]: r for r in jl(os.path.join(LO, f"ans_{split}_{tag}_dict.jsonl"))}
                    same = sum(json.dumps(recs[k]["pred"]) == json.dumps(recd.get(k, {}).get("pred")) for k in recs)
                    line += f"   dict {acc(D):.3f}, same answers {same}/{len(recs)}"
                    J["logs"]["dict"].setdefault(kind, {})[name] = [same, len(recs), round(acc(D), 4)]
            if kind == "impact":
                A2 = log_correct(split, tag, "pred_arm2a")
                line += f"   count-table rules {acc(A2):.3f}"
            P(line)
            if tag != "banked":
                for ref_name, ref in [("1.7B fine-tuned (banked)", base)] + (
                        [("1.7B fine-tuned (re-run)", log_correct(split, "ft17"))] if tag not in ("ft17",) else []):
                    if ref is None:
                        continue
                    m, lo, hi, nq, nc = cl_diff(A, ref, CB)
                    ms, los, his, _, _ = cl_diff(A, ref, CB, stratify=True)
                    J["logs"]["diffs"][f"{kind}: {name} - {ref_name}"] = [m, lo, hi, nq, nc, los, his]
                    P(f"      - {ref_name:30s} {m:+.3f} [{lo:+.3f}, {hi:+.3f}] (stratified [{los:+.3f}, {his:+.3f}]; "
                      f"{nq} q, {nc} logs) -> {reading((m, lo, hi), MARGIN_L) if tag not in ('ft17',) else ''}")
        # frontier failed calls (registered: re-run if > 0.5% fail; else report a sensitivity analysis without the
        # logs they touched)
        for name, tag in readers[4:]:
            fl = failed_logs(split, tag)
            A = log_correct(split, tag)
            if fl is None or A is None:
                continue
            n, nf, aff = fl
            J["logs"].setdefault("failed", {})[f"{kind}: {name}"] = [nf, n, aff]
            line = f"  {name}: {nf} of {n} calls failed after retries ({100 * nf / n:.2f}%), touching {len(aff)} logs"
            if nf:
                keep = {k: v for k, v in A.items() if v[0] not in aff}
                refk = {k: v for k, v in base.items() if v[0] not in aff}
                if keep:
                    m, lo, hi, nq, nc = cl_diff(keep, refk, CB)
                    line += (f"; without those logs: {acc(keep):.3f} vs 1.7B {acc(refk):.3f}, {m:+.3f} [{lo:+.3f}, {hi:+.3f}] "
                             f"({nq} q, {nc} logs)")
                    J["logs"]["diffs"][f"{kind}: {name} - 1.7B, logs without failed calls"] = [m, lo, hi, nq, nc]
            P(line)
        # gate G2
        A = log_correct(split, "ft17")
        if A is not None and base is not None:
            recs = {r["id"]: r for r in jl(os.path.join(LO, f"ans_{split}_ft17.jsonl"))}
            recb = {r["id"]: r for r in jl(os.path.join(LO, f"ans_{split}_banked.jsonl"))}
            same = sum(json.dumps(recs[k]["pred"]) == json.dumps(recb[k]["pred"]) for k in recb)
            g = same >= 0.97 * len(recb) and abs(acc(A) - acc(base)) <= 0.015
            gates[kind] = g
            P(f"  G2 {kind}: re-run answers equal the banked ones on {same}/{len(recb)}, accuracy {acc(A):.3f} vs "
              f"{acc(base):.3f} -> {'PASS' if g else 'FAIL'}")
        else:
            gates[kind] = None
            P(f"  G2 {kind}: not run")
        # tooling effect: the same model with and without the pipeline
        for (k2, cname), C in comps.items():
            if k2 != kind:
                continue
            mine = "q32" if cname.startswith("Qwen3-32B") else "gpt61low"
            A = log_correct(split, mine)
            P(f"  {cname:44s} {acc(C):.3f} ({len(C)} q)" +
              ("" if A is None else f"   same model in the pipeline - this: {cl_diff(A, C, CB)[0]:+.3f} "
                                    f"[{cl_diff(A, C, CB)[1]:+.3f}, {cl_diff(A, C, CB)[2]:+.3f}]"))
            if A is not None:
                J["logs"]["diffs"][f"{kind}: tooling effect {cname}"] = list(cl_diff(A, C, CB))
            if A is not None:
                m, lo, hi, nq, nc = cl_diff(log_correct(split, "banked"), C, CB)
                P(f"      (1.7B fine-tuned pipeline - this: {m:+.3f} [{lo:+.3f}, {hi:+.3f}])")
        P("")
    J["logs"]["gates"]["G2"] = gates
    # bars
    P("Bars (registered; margin 0.05; a set counts only if its gate G2 passed, else the re-run is the baseline)")
    for kind, split in LOG_SETS:
        for name, tag in readers[2:]:
            ref = "1.7B fine-tuned (banked)" if gates.get(kind) else "1.7B fine-tuned (re-run)"
            d = J["logs"]["diffs"].get(f"{kind}: {name} - {ref}")
            if d is None:
                continue
            r = reading(d[:3], MARGIN_L)
            J["logs"]["bars"][f"{kind}: {name}"] = r
            P(f"  L-{kind:6s} {name:30s} {d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}] vs {ref} -> {r}")
    # exploratory, not registered: GPT runs with every call answered / the impact run before its registered re-run
    P("\nExploratory (not registered; reported because the registered rule left failed calls in place)")
    for split, kind, tag, what in (("nsr2", "window", "gpt61low_wr", "GPT-6.1 Sol window set with the 29 refused calls "
                                    "re-asked (all other answers replayed from the cache; 44 new calls)"),
                                   ("nsr1", "impact", "run1_gpt61low", "GPT-6.1 Sol impact set, run 1 (103 refused "
                                    "calls left as no events; replaced by the registered re-run)")):
        if tag.startswith("run1_"):
            recs = jl(os.path.join(LO, "run1_ans_nsr1_gpt61low.jsonl"))
            if recs is None:
                continue
            c = BL.correct(split, recs)
            A = {r["id"]: (r["log"], r["scenario"], bool(c[r["id"]])) for r in recs}
        else:
            A = log_correct(split, tag)
        if A is None:
            continue
        base = log_correct(split, "banked")
        m, lo, hi, nq, nc = cl_diff(A, base, CB)
        J["logs"]["diffs"][f"exploratory {kind}: {tag} - 1.7B"] = [m, lo, hi, nq, nc]
        P(f"  {what}: {acc(A):.3f}; - 1.7B {m:+.3f} [{lo:+.3f}, {hi:+.3f}] -> would read {reading((m, lo, hi), MARGIN_L)}")
        if tag == "gpt61low_wr":
            for (k2, cname), C in comps.items():
                if k2 == "window" and cname.startswith("GPT"):
                    m, lo, hi, nq, nc = cl_diff(A, C, CB)
                    P(f"      the same model reading the whole log (xhigh, OpenAI direct): pipeline - whole log {m:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    P("  LD (dict = matrix memory on >= 99% of questions, every reader and set): " + "; ".join(
        f"{k} {n}: {v[0]}/{v[1]}" for k, d in J["logs"]["dict"].items() for n, v in d.items()))


# ------------------------------------------------------------------------------------------------- predictions
def predictions(P, J):
    """Mechanical check of the registered predictions (prereg 'Predictions')."""
    P("\n## Registered predictions")
    r, L, D = J["ruler"], J["logs"], J["logs"]["diffs"]
    out = []
    out.append(("G1 passes", r["gates"].get("G1")))
    d = r["diffs"].get("32B - 8B (B21, banked)", {}).get("131072")
    out.append((f"RS1: d32 at 128k in [-0.5, +0.4] (got {d[0]:+.2f})", -0.5 <= d[0] <= 0.4))
    out.append(("RS1 reads small reader suffices", (r["bars"].get("RS1") or "").startswith("SMALL")))
    out.append(("RS2 holds", r["bars"].get("RS2")))
    rs3 = r["bars"].get("RS3")
    out.append(("RS3 holds", rs3 is not None and rs3[0] >= 0.98 * rs3[1]))
    d = r["diffs"].get("1.7B - 8B (B21, banked)", {}).get("131072")
    out.append((f"RS4: 1.7B at least 1 point below 8B at 128k, interval below 0 (got {d[0]:+.2f})", d[0] <= -1 and d[2] < 0))
    k32 = [D[f"{k}: Qwen3-32B prompted - 1.7B fine-tuned (banked)"][0] for k in ("where", "impact", "window")]
    out.append((f"32B |d| <= 0.03 on each log set (got {', '.join(f'{x:+.3f}' for x in k32)})", all(abs(x) <= 0.03 for x in k32)))
    suff = sum((L["bars"].get(f"{k}: Qwen3-32B prompted") or "").startswith("SMALL") for k in ("where", "impact", "window"))
    out.append((f"32B: small reader suffices on >= 2 of 3 log sets (got {suff})", suff >= 2))
    fr = [(k, n) for k in ("where", "impact", "window") for n in ("GPT-6.1 Sol (low) prompted", "Claude Opus 5.5 (low) prompted")
          if f"{k}: {n} - 1.7B fine-tuned (banked)" in D]
    fd = [D[f"{k}: {n} - 1.7B fine-tuned (banked)"][0] for k, n in fr]
    out.append((f"frontier d in [-0.02, +0.04] on every set run (got {', '.join(f'{x:+.3f}' for x in fd)})",
                all(-0.02 <= x <= 0.04 for x in fd)))
    fs = [(L["bars"].get(f"{k}: {n}") or "").startswith("SMALL") for k, n in fr]
    out.append((f"frontier: small reader suffices on each set run ({sum(fs)} of {len(fs)})", all(fs)))
    k8 = [D[f"{k}: Qwen3-8B prompted - 1.7B fine-tuned (banked)"][0] for k in ("where", "impact", "window")]
    out.append((f"8B prompted within 0.03 of the 1.7B (got {', '.join(f'{x:+.3f}' for x in k8)})", all(abs(x) <= 0.03 for x in k8)))
    ld = [v[0] >= 0.99 * v[1] for d_ in L["dict"].values() for v in d_.values()]
    out.append((f"LD holds for every reader and set ({sum(ld)} of {len(ld)})", all(ld)))
    t32 = [D[f"{k}: tooling effect Qwen3-32B whole log (forced)"][0] for k in ("where", "impact", "window")]
    out.append((f"tooling effect of Qwen3-32B >= +0.30 on all three sets (got {', '.join(f'{x:+.3f}' for x in t32)})",
                all(x >= 0.30 for x in t32)))
    tg = [D[f"{k}: tooling effect GPT-6.1 Sol whole log (xhigh, OpenAI direct)"][0] for k in ("impact", "window")]
    out.append((f"tooling effect of GPT-6.1 Sol within +-0.05 (got {', '.join(f'{x:+.3f}' for x in tg)})",
                all(abs(x) <= 0.05 for x in tg)))
    for what, ok in out:
        P(f"  {'held' if ok else 'FAILED'}: {what}")
    J["predictions"] = [[w, bool(o)] for w, o in out]


# ------------------------------------------------------------------------------------------------- ledgers
def ledgers(P, J):
    P("\n## Compute and spend")
    gh = 0.0
    lp = os.path.join(ROOT, "b29_out", "gpu_ledger.txt")
    rows = []
    if os.path.exists(lp):
        import datetime
        st = {}
        for l in open(lp):
            p = l.split()
            name, gpus, ev, t = p[0], p[1].split("=")[1], p[2], p[3]
            ts = datetime.datetime.strptime(t, "%Y-%m-%dT%H:%M:%SZ")
            if ev == "start":
                st[(name, gpus)] = ts
            elif (name, gpus) in st:
                h = (ts - st.pop((name, gpus))).total_seconds() / 3600 * len(gpus.split(","))
                gh += h
                rows.append((name, gpus, round(h, 3)))
        for name, gpus in st:
            rows.append((name, gpus, "still running"))
    P(f"A800 GPU-hours (claimed GPUs x wall time, b29_out/gpu_ledger.txt): {gh:.2f} (cap 60)")
    for r in rows:
        P(f"  {r[0]:28s} gpus {r[1]:8s} {r[2]}")
    usd, tok, ncalls = 0.0, [0, 0], 0
    for mp in sorted(glob.glob(os.path.join(LO, "*reader_*_gpt61low.jsonl.meta.json"))):   # incl. run1_ and dev
        m = json.load(open(mp))
        mt = m.get("meter", {})
        tok[0] += mt.get("prompt_tokens", 0)
        tok[1] += mt.get("completion_tokens", 0)
        ncalls += mt.get("calls", 0)
    cache = os.path.join(LO, "zheng_cache_gpt-6.1-sol_low.jsonl")
    ncache = sum(1 for _ in open(cache)) if os.path.exists(cache) else 0
    extra = max(0, ncache - ncalls)   # answered calls of runs stopped before writing their meter (location runs)
    if ncalls:
        tok[0] += round(extra * tok[0] / ncalls)
        tok[1] += round(extra * tok[1] / ncalls)
    usd = (tok[0] * 2.0 + tok[1] * 10.0) / 1e6
    P(f"Frontier (GPT-6.1 Sol via api_zheng): {tok[0]:,} prompt + {tok[1]:,} completion tokens = USD {usd:.2f} at the "
      f"list-price estimate of apps/real/api_reader.PRICE (2 / 10 per million); includes {extra} answered calls of "
      f"stopped runs at the mean tokens per call of the metered runs")
    ousd, otok = 0.0, [0, 0]
    for mp in sorted(glob.glob(os.path.join(LO, "reader_*_opus_low*.jsonl.meta.json"))):
        mt = json.load(open(mp)).get("meter", {})
        ousd += mt.get("usd", 0.0)
        otok[0] += mt.get("prompt_tokens", 0)
        otok[1] += mt.get("completion_tokens", 0)
    P(f"Frontier (Claude Opus 5.5 via the user's endpoint): {otok[0]:,} prompt + {otok[1]:,} completion tokens = USD "
      f"{ousd:.2f} (list price incl. cache writes, claude_api.PRICE)")
    P(f"Frontier total: USD {usd + ousd:.2f} (cap 150; dev pilots included where their meta files exist)")
    J["compute"] = {"gpu_hours": round(gh, 3), "jobs": rows, "frontier_tokens": tok, "frontier_usd_est": round(usd, 2),
                    "opus_tokens": otok, "opus_usd": round(ousd, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "b29_verdict.txt"))
    ap.add_argument("--json", default=os.path.join(ROOT, "results", "b29_scores.json"))
    a = ap.parse_args()
    lines = []
    P = lambda s="": lines.append(s)  # noqa: E731
    J = {}
    P("# B29 verdict (prereg/B29_same_tooling_prereg.md): does model size matter given the same tooling?\n")
    g1 = ruler_part(P, J)
    log_part(P, J, g1)
    predictions(P, J)
    ledgers(P, J)
    for extra in ("b29_notes.txt", "b29_literature.txt"):   # hand-written records, printed verbatim
        fp = os.path.join(ROOT, "results", extra)
        if os.path.exists(fp):
            P("")
            P(open(fp).read().rstrip())
    txt = "\n".join(lines) + "\n"
    print(txt)
    with open(a.out, "w") as f:
        f.write(txt)
    with open(a.json, "w") as f:
        json.dump(J, f, indent=1, default=str)


if __name__ == "__main__":
    main()
