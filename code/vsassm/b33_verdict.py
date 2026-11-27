"""B33 verdict (prereg/B33_fresh_task2_prereg.md). Written before any evaluation-set output exists.

Every accuracy is recomputed here from the saved per-question outputs (memory: saved predicted location against the
dataset label; fine-tune: argmax of the saved logits; prompting: saved prediction), independently of the numbers the
runs logged, and compared with them. Gates, bars, predictions and readings are applied exactly as registered.
Intervals: Wilson 95% per seed (questions); for means and differences a bootstrap over STORIES (the five questions
of a story share its text), 10,000 resamples, paired across arms and seeds.

  python b33_verdict.py --runs b33_out --b28 b28_out --gencheck results/b33_gencheck.json \
      --fresh_check results/b33_fresh_check.json --ledger results/b33_gpu_ledger.txt \
      --out results/b33_verdict.txt --scores results/b33_scores.json
"""
import argparse, gzip, json, os, sys
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from b28_common import wilson, ENCODINGS, SEEDS                       # noqa: E402

SETS = ["fresh", "hfrest", "orig500"]
SET_NAME = {"fresh": "FRESH (b33_gen, registered seed; primary)", "hfrest": "HF test 500-999 (secondary)",
            "orig500": "HF test 0-499 (the paper's 500; gates only)"}
FT = {"r16_e6_lr0.0001": "best of twelve (r16, 6 ep, 1e-4)", "r8_e2_lr0.0001": "first configuration (r8, 2 ep, 1e-4)"}
PROMPT_BANK = 0.3860                                                    # v49_logs/v49.json task2/zeroshot/all
B28_FT_ALL500 = {"r16_e6_lr0.0001": [0.968, 0.986, 0.958], "r8_e2_lr0.0001": [0.840, 0.832, 0.846]}  # B28 reruns (expr)
L = []


def P(s=""):
    L.append(s); print(s)


def load(p):
    if os.path.exists(p):
        return json.load(open(p))
    if os.path.exists(p + ".gz"):
        return json.load(gzip.open(p + ".gz", "rt"))
    return None


def mem_matrix(dm, name, arm, enc):
    S = dm["sets"][name]
    lab = np.array(S["labels_all"]); n = len(lab); ii = np.array(S["row_index"], int)
    rows = []
    for s in SEEDS:
        k = "%s/%s/%d" % (arm, enc, s)
        if k not in S["runs"]:
            return None
        c = np.zeros(n, int)
        if len(ii):
            pr = np.array([o["pred"] for o in S["runs"][k]["per_q"]])
            c[ii] = (pr == lab[ii]).astype(int)
        rows.append(c)
        assert abs(c.mean() - S["runs"][k]["acc_all"]) < 1e-9, "recomputed memory accuracy differs from logged"
    return np.array(rows)


def ft_matrix(dl, name):
    kept = dl["kept"]; lab = np.array(dl["sets"][name]["labels_all"])
    rows = []
    for r in sorted(dl["runs"], key=lambda r: SEEDS.index(r["seed"])):
        pr = np.array([kept[int(np.argmax(x))] for x in r["sets"][name]["logits"]])
        c = (pr == lab).astype(int)
        assert abs(c.mean() - r["sets"][name]["acc"]) < 1e-9, "recomputed fine-tune accuracy differs from logged"
        rows.append(c)
    return np.array(rows) if len(rows) == 3 else None


def prompt_vec(dp, name):
    S = dp["sets"][name]
    c = (np.array(S["pred"]) == np.array(S["labels"])).astype(int)
    assert abs(c.mean() - S["acc"]) < 1e-9
    return c


def boot_idx(sid, B=10000, seed=0):
    """Resample stories: returns a list of B index arrays over questions."""
    rng = np.random.default_rng(seed)
    sid = np.asarray(sid); stories = np.unique(sid)
    members = [np.where(sid == s)[0] for s in stories]
    out = []
    for _ in range(B):
        pick = rng.integers(0, len(stories), len(stories))
        out.append(np.concatenate([members[j] for j in pick]))
    return out


def ci_mean(X, idx):
    X = np.atleast_2d(X)
    m = np.array([X[:, ix].mean() for ix in idx])
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def ci_diff(A, Bm, idx):
    A = np.atleast_2d(A); Bm = np.atleast_2d(Bm)
    d = np.array([A[:, ix].mean() - Bm[:, ix].mean() for ix in idx])
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def seedline(X):
    return ", ".join("%.3f [%.3f-%.3f]" % (x.mean(), *wilson(int(x.sum()), len(x))) for x in np.atleast_2d(X))


def reading_ft(per, lo, hi):
    if all(per >= 0.10):
        return "MEMORY LEADS"
    if all(-per >= 0.02) and hi < 0:
        return "FINE-TUNE LEADS"
    if all(per >= 0.02) and lo > 0:
        return "MEMORY AHEAD, BELOW THE TEN-POINT BAR"
    return "NO CLEAR DIFFERENCE"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="b33_out")
    ap.add_argument("--b28", default="b28_out")
    ap.add_argument("--gencheck", default="results/b33_gencheck.json")
    ap.add_argument("--fresh_check", default="results/b33_fresh_check.json")
    ap.add_argument("--ledger", default="results/b33_gpu_ledger.txt")
    ap.add_argument("--out", default="results/b33_verdict.txt")
    ap.add_argument("--scores", default="results/b33_scores.json")
    ap.add_argument("--gates_only", action="store_true", help="print the gates and stop (read before any result)")
    a = ap.parse_args()
    S = {"gates": {}, "sets": {}, "bars": {}, "predictions": {}}
    P("# B33 verdict (prereg/B33_fresh_task2_prereg.md): the corrected parse on bAbI task-2 questions no part of the")
    P("# paper has used. Generated by b33_verdict.py on %s from the per-question outputs in %s."
      % (datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), a.runs))
    P()
    mc = load(os.path.join(a.runs, "b33_mem_completed.json"))
    mp = load(os.path.join(a.runs, "b33_mem_published.json"))
    pr = load(os.path.join(a.runs, "b33_prompt.json"))
    ft = {}
    for tag in FT:
        for trn in ("expr", "all"):
            d_ = load(os.path.join(a.runs, "b33_lora_%s_%s.json" % (tag, trn)))
            if d_ is None:                       # I/O only: runs split by seed (one file per seed) are merged
                parts = [load(os.path.join(a.runs, "b33_lora_%s_%s_s%d.json" % (tag, trn, s))) for s in SEEDS]
                if all(parts):
                    d_ = dict(parts[0]); d_["runs"] = [r for p_ in parts for r in p_["runs"]]
            ft[(tag, trn)] = d_
    gc = load(a.gencheck); fc = load(a.fresh_check)

    # ------------------------------------------------------------------ gates
    P("## Gates")
    if gc:
        P("  G-DIST  generator vs the other 100 official training stories (before registration): %d statistics, min "
          "story-bootstrap p = %.4f (threshold %.4f); tracker %d/%d official; question rule %d/%d -> %s" % (
              len(gc["stats"]), gc["min_p_check_half"], gc["alpha"],
              gc["official_tracker"][0], gc["official_tracker"][1], gc["rule_check_generated"]["explained"],
              gc["rule_check_generated"]["slots"], "PASS" if gc["gate"] else "FAIL"))
        S["gates"]["G-DIST"] = gc["gate"]
    if fc:
        ok = fc["gate"]
        P("  G-FRESH fresh set: %d questions in %d stories; second scorer (b33_track) agrees with every label: %s; "
          "identical (story, question) rows shared with the official task-2 splits: %d; longest input %d tokens (%d over 768) "
          "-> %s" % (fc["n_questions"], fc["n_stories"], fc["labels_agree"], fc["shared_with_official"], fc["max_tokens"],
                     fc["n_over_768"], "PASS" if ok else "FAIL"))
        S["gates"]["G-FRESH"] = ok
    b28c = load(os.path.join(a.b28, "b28_task2_completed.json"))
    b28p = load(os.path.join(a.b28, "b28_task2.json"))
    for tag, dm, db in (("G0 corrected parse", mc, b28c), ("G1 published parse", mp, b28p)):
        if dm is None or db is None or "orig500" not in dm["sets"]:
            P("  %s: missing" % tag); continue
        So = dm["sets"]["orig500"]
        same_rows = So["row_index"] == db["row_index"]
        agree = []
        for k, v in db["runs"].items():
            if k in So["runs"]:
                a_ = [x["pred"] for x in So["runs"][k]["per_q"]]; b_ = [x["pred"] for x in v["per_q"]]
                agree.append(float(np.mean([x == y for x, y in zip(a_, b_)])) if len(a_) == len(b_) else 0.0)
        X = mem_matrix(dm, "orig500", "probe_argmax", "comm")
        ok = same_rows and agree and min(agree) == 1.0
        S["gates"][tag.split()[0]] = bool(ok)
        P("  %s on the paper's 500: same parsed rows as B28 %s; per-question agreement with B28 min %.4f over %d "
          "cells; naming head comm per seed %s -> %s" % (tag, same_rows, min(agree) if agree else float("nan"),
                                                       len(agree), "/".join("%.3f" % x for x in X.mean(axis=1)),
                                                       "PASS" if ok else "FAIL"))
    if pr is not None and "orig500" in pr["sets"]:
        v = prompt_vec(pr, "orig500").mean()
        ok = abs(v - PROMPT_BANK) <= 0.004
        S["gates"]["G2"] = bool(ok)
        P("  G2 prompting (V-49 zero-shot) on the paper's 500: %.4f vs banked %.4f (tolerance 0.004) -> %s"
          % (v, PROMPT_BANK, "PASS" if ok else "FAIL"))
    for tag in FT:
        d_ = ft.get((tag, "expr"))
        if d_ is None:
            continue
        X = ft_matrix(d_, "orig500")
        per = X.mean(axis=1)
        ok = all(abs(x - y) <= 0.03 for x, y in zip(per, B28_FT_ALL500[tag]))
        b28l = load(os.path.join(a.b28, "b28_lora2_%s_expr.json" % tag))
        agr = []
        if b28l:
            kept = b28l["kept"]
            for r, r2 in zip(sorted(b28l["runs"], key=lambda r: SEEDS.index(r["seed"])),
                             sorted(d_["runs"], key=lambda r: SEEDS.index(r["seed"]))):
                p1 = [kept[int(np.argmax(x))] for x in r["logitsB"]]
                p2 = [kept[int(np.argmax(x))] for x in r2["sets"]["orig500"]["logits"]]
                agr.append(float(np.mean([x == y for x, y in zip(p1, p2)])))
        S["gates"]["G3/" + tag] = bool(ok)
        P("  G3 fine-tune rerun %s on the paper's 500: %s vs B28 %s (tolerance 0.03; reported, not blocking); "
          "per-question agreement with B28 %s -> %s" % (FT[tag], "/".join("%.3f" % x for x in per),
                                                        "/".join("%.3f" % x for x in B28_FT_ALL500[tag]),
                                                        "/".join("%.3f" % x for x in agr), "PASS" if ok else "MISS"))
    blocking = [S["gates"].get(k) for k in ("G-FRESH", "G0", "G1", "G2")]
    S["gates"]["blocking_all_pass"] = all(x is True for x in blocking)
    P("  blocking gates (G-FRESH, G0, G1, G2) all pass: %s   (G-DIST is reported, not blocking: it failed before "
      "registration)" % S["gates"]["blocking_all_pass"])
    P()
    if a.gates_only:
        return

    # ------------------------------------------------------------------ results per set
    res = {}
    for name in SETS:
        if mc is None or name not in mc["sets"]:
            continue
        sid = mc["sets"][name]["story_id"]
        idx = boot_idx(sid)
        n = len(sid)
        P("## %s: %d questions, %d stories" % (SET_NAME[name], n, len(set(sid))))
        arms = {}
        for dm, lab in ((mc, "corrected parse"), (mp, "published parse")):
            if dm is None or name not in dm["sets"]:
                continue
            Sx = dm["sets"][name]
            P("  memory, %s: parse reads %d of %d, %d expressible under the ordering rule; %d inputs > 768 tokens"
              % (lab, Sx["n_parsed"], n, sum(Sx["expressible"]), Sx["n_longer_than_768_tokens"]))
            for arm in ("probe_argmax", "parser"):
                for enc in ENCODINGS:
                    X = mem_matrix(dm, name, arm, enc)
                    if X is not None:
                        arms["memory %s, %s, %s" % ("naming head" if arm == "probe_argmax" else "exact rows",
                                                    enc, lab)] = X
        for tag in FT:
            for trn in ("expr", "all"):
                d_ = ft.get((tag, trn))
                if d_ is not None and name in d_["sets"]:
                    X = ft_matrix(d_, name)
                    if X is not None:
                        arms["fine-tune %s, trained on %s" % (FT[tag], "775 (as published)" if trn == "expr"
                                                              else "all 900")] = X
        if pr is not None and name in pr["sets"]:
            arms["prompting (V-49 zero-shot)"] = prompt_vec(pr, name)[None]
        out = {}
        for k, X in arms.items():
            lo, hi = ci_mean(X, idx)
            out[k] = {"per_seed": [float(x) for x in np.atleast_2d(X).mean(axis=1)], "mean": float(X.mean()),
                      "ci_story_boot": [lo, hi]}
            P("    %-78s mean %.3f [%.3f, %.3f]  seeds %s" % (k, X.mean(), lo, hi, seedline(X)))
        MEM = "memory naming head, comm, corrected parse"
        con = {}
        if MEM in arms:
            Mm = arms[MEM]
            for k, X in arms.items():
                if k == MEM or k.startswith("memory"):
                    continue
                Xs = np.repeat(X, 3, axis=0) if X.shape[0] == 1 else X
                per = Mm.mean(axis=1) - Xs.mean(axis=1)
                lo, hi = ci_diff(Mm, Xs, idx)
                rd = reading_ft(per, lo, hi) if k.startswith("fine-tune") else ""
                con[k] = {"per_seed": [float(x) for x in per], "mean": float(per.mean()), "ci": [lo, hi],
                          "reading": rd}
                P("    memory - %-62s per seed %s  mean %+.3f [%+.3f, %+.3f] %s" % (
                    k, "/".join("%+.3f" % x for x in per), per.mean(), lo, hi, ("-> " + rd) if rd else ""))
            # second angle inside the set: every memory error listed for hand-reading
            S_ = mc["sets"][name]
            errs = sorted({int(S_["row_index"][j]) for kk, v in S_["runs"].items() if kk.startswith("probe_argmax/comm")
                           for j, o in enumerate(v["per_q"]) if not o["correct"]} |
                          (set(range(n)) - set(S_["row_index"])))
            P("    memory (naming head, comm, corrected parse) errors or unanswered questions: %d %s" % (
                len(errs), errs[:40]))
            out["_memory_errors"] = errs
        res[name] = {"arms": out, "contrasts": con}
        P()
    S["sets"] = res

    # ------------------------------------------------------------------ bars on the fresh set
    P("## Bars (memory = naming head, comm, corrected parse). Each bar is read on the fresh set and on hfrest; a bar")
    P("## is CONFIRMED only if it holds on both (fresh: never used, but the generator's G-DIST failed on two statistics;")
    P("## hfrest: the official distribution, but featurized by the August pooled probes)")
    MEM = "memory naming head, comm, corrected parse"
    FT12 = "fine-tune %s, trained on 775 (as published)" % FT["r16_e6_lr0.0001"]
    FT1 = "fine-tune %s, trained on 775 (as published)" % FT["r8_e2_lr0.0001"]
    PR = "prompting (V-49 zero-shot)"
    for nm in ("fresh", "hfrest"):
        F_ = res.get(nm, {})
        if MEM not in F_.get("arms", {}):
            continue
        bs = S["bars"].setdefault(nm, {})
        m = F_["arms"][MEM]
        bs["B1"] = all(x >= 0.98 for x in m["per_seed"])
        P("  [%s] B1 memory >= 0.98 in every seed: %s -> %s" % (nm, "/".join("%.3f" % x for x in m["per_seed"]),
                                                               "HELD" if bs["B1"] else "FAILED"))
        c = F_["contrasts"].get(PR)
        if c:
            bs["B2"] = all(x >= 0.40 for x in c["per_seed"]) and c["ci"][0] > 0.30
            P("  [%s] B2 memory - prompting >= 0.40 in every seed and 95%% CI lower bound > 0.30: %s, CI [%+.3f, %+.3f]"
              " -> %s" % (nm, "/".join("%+.3f" % x for x in c["per_seed"]), c["ci"][0], c["ci"][1],
                          "HELD" if bs["B2"] else "FAILED"))
        for key, lab in ((FT12, "B3"), (FT1, "B4")):
            c = F_["contrasts"].get(key)
            if c:
                bs[lab] = c["reading"]
                P("  [%s] %s memory vs %s: %s" % (nm, lab, "best of twelve" if lab == "B3" else "first configuration",
                                                c["reading"]))
    both = {}
    for b in ("B1", "B2", "B3", "B4"):
        v = [S["bars"].get(nm, {}).get(b) for nm in ("fresh", "hfrest")]
        if b in ("B1", "B2"):
            both[b] = "CONFIRMED" if all(x is True for x in v) else ("FAILED ON BOTH" if all(x is False for x in v)
                                                                    else "SPLIT (%s fresh / %s hfrest)" % tuple(v))
        else:
            both[b] = v[0] if v[0] == v[1] else "SPLIT (%s on fresh / %s on hfrest)" % tuple(v)
        P("  %s combined: %s" % (b, both[b]))
    S["bars"]["combined"] = both
    F_ = res.get("fresh", {})
    off = [res.get(nm, {}).get("arms", {}).get(FT12) for nm in ("orig500", "hfrest")]
    if all(off) and FT12 in F_.get("arms", {}):
        off_mean = float(np.mean([o["mean"] for o in off]))
        gap = F_["arms"][FT12]["mean"] - off_mean
        S["bars"]["shift_flag"] = bool(gap < -0.03)
        P("  shift safeguard: best of twelve on fresh %.3f vs on the 1,000 official test questions %.3f (difference "
          "%+.3f; flag if < -0.03) -> %s" % (F_["arms"][FT12]["mean"], off_mean, gap,
                                            "FLAG: the hfrest contrasts are reported first" if gap < -0.03
                                            else "no flag"))
    P()
    # ------------------------------------------------------------------ predictions
    P("## Predictions (registered)")

    def H(label, val):
        S["predictions"][label] = bool(val)
        P("  %-104s %s" % (label, "HELD" if val else "NOT HELD"))

    g = S["gates"]
    H("P1 blocking gates G-FRESH, G0, G1, G2 pass (G-DIST failed before registration, disclosed)", g.get("blocking_all_pass"))
    H("P2 G3: both fine-tune reruns within 0.03 of B28 per seed", all(g.get("G3/" + t) for t in FT))
    if MEM in F_.get("arms", {}):
        A = F_["arms"]
        H("P3 memory on fresh >= 0.995 in every seed (%s)" % "/".join("%.3f" % x for x in A[MEM]["per_seed"]),
          all(x >= 0.995 for x in A[MEM]["per_seed"]))
        ex = "memory exact rows, comm, corrected parse"
        if ex in A:
            H("P4 exact rows equal the naming head on fresh within 0.002 per seed",
              all(abs(x - y) <= 0.002 for x, y in zip(A[ex]["per_seed"], A[MEM]["per_seed"])))
        pub = "memory naming head, comm, published parse"
        if pub in A:
            H("P5 published parse on fresh: mean in [0.78, 0.90] (%.3f)" % A[pub]["mean"], 0.78 <= A[pub]["mean"] <= 0.90)
        if FT12 in A:
            H("P6 best of twelve on fresh: mean in [0.93, 0.99] (%.3f)" % A[FT12]["mean"], 0.93 <= A[FT12]["mean"] <= 0.99)
        if FT1 in A:
            H("P7 first configuration on fresh: mean in [0.72, 0.90] (%.3f)" % A[FT1]["mean"], 0.72 <= A[FT1]["mean"] <= 0.90)
        if PR in A:
            H("P8 prompting on fresh in [0.30, 0.46] (%.3f)" % A[PR]["mean"], 0.30 <= A[PR]["mean"] <= 0.46)
        cb = S["bars"]["combined"]
        H("P9 B1 and B2 CONFIRMED (hold on fresh and on hfrest)", cb.get("B1") == "CONFIRMED" and cb.get("B2") == "CONFIRMED")
        H("P10 B3 reading = MEMORY AHEAD, BELOW THE TEN-POINT BAR on both sets",
          cb.get("B3") == "MEMORY AHEAD, BELOW THE TEN-POINT BAR")
        H("P11 B4 reading = MEMORY LEADS on both sets", cb.get("B4") == "MEMORY LEADS")
        H("P12 no distribution-shift flag", S["bars"].get("shift_flag") is False)
        hr = res.get("hfrest", {})
        if MEM in hr.get("arms", {}) and PR in hr.get("arms", {}):
            c3 = hr["contrasts"].get(FT12, {}).get("reading")
            H("P13 hfrest: memory >= 0.98 every seed, memory - prompting >= 0.40 every seed, B3 reading as on fresh",
              all(x >= 0.98 for x in hr["arms"][MEM]["per_seed"]) and
              all(x >= 0.40 for x in hr["contrasts"][PR]["per_seed"]) and c3 == S["bars"].get("fresh", {}).get("B3"))
        fa = "fine-tune %s, trained on all 900" % FT["r16_e6_lr0.0001"]
        if fa in A and FT12 in A:
            H("P14 best of twelve trained on all 900 >= trained on the 775 on fresh (mean)", A[fa]["mean"] >= A[FT12]["mean"])
    if os.path.exists(a.ledger):
        tot = 0.0
        starts = {}
        for line in open(a.ledger):
            p_ = line.split()
            if len(p_) >= 4 and p_[2] in ("start", "end"):
                t = datetime.strptime(p_[3], "%Y-%m-%dT%H:%M:%SZ")
                if p_[2] == "start":
                    starts[(p_[0], p_[1])] = t
                elif (p_[0], p_[1]) in starts:
                    tot += (t - starts.pop((p_[0], p_[1]))).total_seconds() / 3600
        S["gpu_hours"] = tot
        H("P15 total GPU-hours <= 4 (%.2f)" % tot, tot <= 4)
    open(a.out, "w").write("\n".join(L) + "\n")
    json.dump(S, open(a.scores, "w"), indent=1)


if __name__ == "__main__":
    main()
