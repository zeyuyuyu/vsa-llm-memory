"""Cluster bootstrap of the paired operational-log comparisons (referee: questions are clustered within logs / sites).

Read-only over nsr/ and apps/real/: imports their scorers (run with `python3 -I -B`, so nothing is written there) and
writes only results/cluster_bootstrap.txt (+ results/cluster_bootstrap.json).

For each comparison: per-question correctness of both arms with the original scoring functions (accuracies must
reproduce the paper's), then
  (a) question-level paired bootstrap, exactly as nsr/nsr3_compare.py and nsr/nsr5_compare.py do it
      (sorted question ids, d = okA - okB, random.seed(0), 10,000 x random.choice, bs[250] / bs[9750]);
  (b) cluster bootstrap: resample whole logs (one-day sets) or whole multi-day sites with replacement,
      numpy default_rng(0), 10,000 resamples, statistic = sum(d) / sum(n) over the resampled clusters
      (the ratio estimator, i.e. the pooled accuracy difference of the resampled set), 2.5 / 97.5 percentiles;
  (c) sensitivity: the same cluster bootstrap stratified by scenario (logs/sites resampled within each of the four
      scenarios, matching the fixed 10-logs-per-scenario design).
Difference is always ours - other (negative = the other arm is ahead).

python3 -I -B /path/to/vsassm/cluster_bootstrap.py
"""
import collections
import json
import os
import random
import site
import sys

if site.getusersitepackages() not in sys.path:  # -I drops the user site, where numpy is installed
    sys.path.append(site.getusersitepackages())
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
NSR = os.path.join(ROOT, "nsr")
REAL = os.path.join(ROOT, "apps", "real")
OUTN = os.path.join(NSR, "out")
for p in (REAL, ROOT, NSR):
    sys.path.insert(0, p)

import nsr1_score  # noqa: E402  gold(), the NSR1 independent scorer
import nsr3_compare  # noqa: E402  gold_all(), NSR3 window replay (nsr3_score.states / affected)
import nsr5_sites  # noqa: E402  sites(K), the multi-day sites
import run_real  # noqa: E402  ok(), resolve_answer(), load(), gid()
import build_real  # noqa: E402  parse_forced()
from product import Registry  # noqa: E402

B = 10000
USED = collections.OrderedDict()  # comparison set -> description of files and functions


def jl(path):
    return [json.loads(l) for l in open(path)]


# ------------------------------------------------------------------------------------------- per-question scoring
def impact_1day(arm):
    """NSR1 impact questions (split nsr1, nsr/data): nsr1_score.gold + the scorer's set-exact rule."""
    logs = {}
    for sc in ("warehouse", "biobank", "home", "ops"):
        for d in jl(os.path.join(NSR, "data", f"{sc}_nsr1.jsonl")):
            logs[d["id"]] = d
    out = {}
    for r in jl(os.path.join(OUTN, f"nsr1_nsr1_{arm}.jsonl")):
        g = nsr1_score.gold(logs[r["log"]], r["qtype"], r["target"])
        p = r["pred"]
        out[(r["log"], r["qtype"], r["target"])] = (r["log"], r["scenario"], p is not None and sorted(p) == g)
    return out


_G3 = {}


def window_1day(arm):
    """NSR3 window questions (split nsr2): nsr3_compare.gold_all (window union of nsr3_score replay) + set equality."""
    if not _G3:
        _G3["gold"], _G3["logs"] = nsr3_compare.gold_all("nsr2", os.path.join(NSR, "data"))
    gold, logs = _G3["gold"], _G3["logs"]
    out = {}
    for r in jl(os.path.join(OUTN, f"nsr3_nsr2_{arm}.jsonl")):
        lid = r["id"].split("|")[0]
        out[r["id"]] = (lid, logs[lid]["scenario"], r["pred"] is not None and set(r["pred"]) == gold(r["id"]))
    return out


_S5 = {}


def multiday(K, arm):
    """NSR5 multi-day impact questions: nsr5_sites.sites(K) + nsr1_score.gold, as nsr5_compare.py."""
    if K not in _S5:
        _S5[K] = {s["id"]: s for s, _ in nsr5_sites.sites(K)}
    sites = _S5[K]
    out = {}
    for r in jl(os.path.join(OUTN, f"nsr5_K{K}_{arm}.jsonl")):
        sid, qt, tg = r["id"].split("|")
        g = nsr1_score.gold(sites[sid], qt, tg)
        out[r["id"]] = (sid, sites[sid]["scenario"], r["pred"] is not None and sorted(r["pred"]) == g)
    return out


_L = {}


def location(arm):
    """apps/real test2 'where is it now' questions: run_real.ok / resolve_answer, forced answers via
    build_real.parse_forced, exactly as build_real.main computes the README's forced score."""
    if not _L:
        _L["logs"] = run_real.load("test2")
        _L["rd"] = {r["id"]: r for r in jl(os.path.join(REAL, "out", "reader_test2_verify_v4.jsonl"))}
    out = {}
    if arm == "rm":
        for l in _L["logs"]:
            reg, r = Registry(l), _L["rd"][l["id"]]
            for qi, q in enumerate(l["questions"]):
                out[(l["id"], qi)] = (l["id"], l["scenario"], run_real.ok(q, r["answers"][qi]["pred"], reg))
        return out
    import glob
    g = {}
    for p in glob.glob(os.path.join(REAL, "out", "test2a", f"{arm}_REAL*.jsonl")):
        for r in jl(p):
            g[(r["n"], r["idx"])] = r
    fp = os.path.join(REAL, "out", "test2a", f"forced_{arm}.jsonl")
    forced = {(r["n"], r["idx"]): r["forced_answer"] for r in jl(fp)}
    for l in _L["logs"]:
        n, reg = run_real.gid(l), Registry(l)
        for qi, q in enumerate(l["questions"]):
            x = g.get((n, qi))
            if x is None:
                continue
            strict = (not x["hit_limit"]) and run_real.ok(q, run_real.resolve_answer(q, x["answer_text"], reg), reg)
            fa = forced.get((n, qi)) if x["hit_limit"] else None
            okl = strict or (fa is not None and run_real.ok(q, build_real.parse_forced(q, fa, reg), reg))
            out[(l["id"], qi)] = (l["id"], l["scenario"], okl)
    return out


# ------------------------------------------------------------------------------------------------- bootstraps
def q_boot(A, Bm):
    """nsr3_compare.py / nsr5_compare.py verbatim: sorted ids, random.seed(0), random.choice, bs[250], bs[9750]."""
    ks = sorted(A)
    d = [A[k][2] - Bm[k][2] for k in ks]
    random.seed(0)
    bs = sorted(sum(random.choice(d) for _ in d) / len(d) for _ in range(B))
    return bs[250], bs[9750]


def cl_boot(A, Bm, stratify=False, seed=0):
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
        idx = rng.integers(0, len(cl), size=(B, len(cl)))
        stat = D[idx].sum(1) / N[idx].sum(1)
    else:
        groups = [np.array([i for i, c in enumerate(cl) if scen[c] == s]) for s in sorted(set(scen.values()))]
        num = np.zeros(B)
        den = np.zeros(B)
        for g in groups:
            idx = g[rng.integers(0, len(g), size=(B, len(g)))]
            num += D[idx].sum(1)
            den += N[idx].sum(1)
        stat = num / den
    lo, hi = np.percentile(stat, [2.5, 97.5])
    return float(lo), float(hi), len(cl)


# ------------------------------------------------------------------------------------------------ comparisons
# (label, set, loader, armA (ours), armB, paper accA, paper accB, paper question-level CI of ours - B or None)
COMPS = [
    ("1  impact 1-day: ours vs GPT-6.1 Sol direct", "impact1", impact_1day, "arm1", "arm3_gptD_strict",
     0.930, 0.952, (-0.061, 0.022)),
    ("1x impact 1-day: ours vs Opus 5.5 normalised", "impact1", impact_1day, "arm1", "arm3n_opusr_strict",
     0.930, 0.935, None),
    ("2  window 1-day: ours vs GPT-6.1 Sol direct", "window1", window_1day, "history", "gptD_strict",
     0.874, 0.904, (-0.067, 0.007)),
    ("2b window 1-day: ours vs Opus 5.5 normalised", "window1", window_1day, "history", "opusrnn_strict",
     0.874, 0.878, (-0.046, 0.035)),
    ("3a impact 4-day: ours vs GPT-6.1 Sol direct", "K4", lambda a: multiday(4, a), "pipeline", "direct",
     0.801, 0.925, (-0.186, -0.062)),
    ("3b impact 8-day: ours vs GPT-6.1 Sol direct", "K8", lambda a: multiday(8, a), "pipeline", "direct",
     0.710, 0.855, (-0.261, -0.043)),
    ("3x impact 4-day: ours vs Opus 5.5 normalised", "K4", lambda a: multiday(4, a), "pipeline", "opusn",
     0.801, 0.845, None),
    ("3y impact 8-day: ours vs Opus 5.5 normalised", "K8", lambda a: multiday(8, a), "pipeline", "opusn",
     0.710, 0.754, None),
    ("4a location 1-day: ours vs Qwen3-32B (forced)", "loc", location, "rm", "qwen3-32b", 0.959, 0.543, None),
    ("4b impact 1-day: ours vs Qwen3-32B (forced)", "impact1", impact_1day, "arm1", "arm3_qwen3-32b_forced",
     0.930, 0.278, None),
    ("4c window 1-day: ours vs Qwen3-32B (forced)", "window1", window_1day, "history", "llm_qwen3-32b_forced",
     0.874, 0.202, None),
]

FILES = {
    "impact1": "logs nsr/data/{warehouse,biobank,home,ops}_nsr1.jsonl; answers nsr/out/nsr1_nsr1_<arm>.jsonl; "
               "gold nsr1_score.gold(log, qtype, target); ok = pred is not None and sorted(pred) == gold "
               "(nsr1_score.main); cluster = record['log']",
    "window1": "logs nsr/data/*_nsr2.jsonl; answers nsr/out/nsr3_nsr2_<arm>.jsonl; gold nsr3_compare.gold_all('nsr2') "
               "(union of nsr3_score.states/affected over the window); ok = pred is not None and set(pred) == gold; "
               "cluster = id.split('|')[0] (the log)",
    "K4": "sites nsr5_sites.sites(4) (logs of splits test/nsr1/nsr2, 4 consecutive days per site); answers "
          "nsr/out/nsr5_K4_<arm>.jsonl; gold nsr1_score.gold(site, qtype, target); ok as nsr5_compare.py; "
          "cluster = id.split('|')[0] (the site)",
    "K8": "as K4 with K = 8: nsr5_sites.sites(8), nsr/out/nsr5_K8_<arm>.jsonl",
    "loc": "logs apps/real/data/*_test2.jsonl (run_real.load('test2')); ours apps/real/out/reader_test2_verify_v4.jsonl; "
           "Qwen3-32B apps/real/out/test2a/qwen3-32b_REAL_n*.jsonl + forced_qwen3-32b.jsonl; ok = run_real.ok with "
           "run_real.resolve_answer, forced answers via build_real.parse_forced (as build_real.main); cluster = log id",
}


def main():
    rows, cache = [], {}
    for label, cset, fn, a, b, pa, pb, pci in COMPS:
        A = cache.setdefault((cset, a), fn(a))
        Bm = cache.setdefault((cset, b), fn(b))
        assert sorted(A) == sorted(Bm), label
        assert all(A[k][0] == Bm[k][0] for k in A), label
        n = len(A)
        acc_a = sum(v[2] for v in A.values()) / n
        acc_b = sum(v[2] for v in Bm.values()) / n
        repro = round(acc_a, 3) == pa and round(acc_b, 3) == pb
        qlo, qhi = q_boot(A, Bm)
        clo, chi, ncl = cl_boot(A, Bm)
        slo, shi, _ = cl_boot(A, Bm, stratify=True)
        sizes = collections.Counter(v[0] for v in A.values())
        rows.append({"label": label, "set": cset, "arm_a": a, "arm_b": b, "n": n, "clusters": ncl,
                     "cluster_size_min_max": [min(sizes.values()), max(sizes.values())],
                     "acc_a": round(acc_a, 4), "acc_b": round(acc_b, 4), "paper_acc": [pa, pb], "reproduced": repro,
                     "diff": round(acc_a - acc_b, 4), "q_ci": [round(qlo, 3), round(qhi, 3)], "paper_q_ci": pci,
                     "cluster_ci": [round(clo, 3), round(chi, 3)], "strat_cluster_ci": [round(slo, 3), round(shi, 3)],
                     "q_excl0": qlo > 0 or qhi < 0, "cl_excl0": clo > 0 or chi < 0, "st_excl0": slo > 0 or shi < 0})
        print(json.dumps(rows[-1]), flush=True)
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    json.dump(rows, open(os.path.join(ROOT, "results", "cluster_bootstrap.json"), "w"), indent=1)
    f = lambda x: f"{x:+.3f}"  # noqa: E731
    L = ["# Cluster bootstrap of the paired operational-log comparisons (2026-11-17)",
         "# Difference = ours - other (negative: the other arm is ahead). 10,000 resamples, percentile 95% intervals.",
         "# q-level: nsr3_compare.py / nsr5_compare.py procedure verbatim (random.seed(0), random.choice over sorted ids).",
         "# cluster: whole logs (1-day sets) or whole multi-day sites resampled with replacement, numpy default_rng(0),",
         "#          statistic sum(d)/sum(n) over the resampled clusters (pooled accuracy difference).",
         "# strat:   the same cluster bootstrap with logs/sites resampled within each of the 4 scenarios (sensitivity).",
         "# Script: /path/to/vsassm/cluster_bootstrap.py (python3 -I -B; imports the nsr/apps-real scorers read-only).",
         "",
         "| comparison | n q | n clusters (size min-max) | acc ours / other (repro) | diff | q-level CI (paper) | cluster CI | strat. cluster CI | conclusion |",
         "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        pci = f" (paper {f(r['paper_q_ci'][0])} to {f(r['paper_q_ci'][1])})" if r["paper_q_ci"] else ""
        same = "unchanged" if r["q_excl0"] == r["cl_excl0"] else "CHANGES"
        concl = f"{same}: {'excludes' if r['cl_excl0'] else 'includes'} 0"
        L.append(f"| {r['label']} | {r['n']} | {r['clusters']} ({r['cluster_size_min_max'][0]}-{r['cluster_size_min_max'][1]}) | "
                 f"{r['acc_a']:.3f} / {r['acc_b']:.3f} ({'yes' if r['reproduced'] else 'NO'}) | {f(r['diff'])} | "
                 f"{f(r['q_ci'][0])} to {f(r['q_ci'][1])}{pci} | {f(r['cluster_ci'][0])} to {f(r['cluster_ci'][1])} | "
                 f"{f(r['strat_cluster_ci'][0])} to {f(r['strat_cluster_ci'][1])} | {concl} |")
    L += ["", "## Files and scoring functions (all under /path/to/vsassm/)"]
    for k, v in FILES.items():
        L.append(f"- {k}: {v}")
    L += ["", "## Arms (answer files)"]
    for r in rows:
        L.append(f"- {r['label']}: ours = {r['arm_a']}, other = {r['arm_b']}")
    open(os.path.join(ROOT, "results", "cluster_bootstrap.txt"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
