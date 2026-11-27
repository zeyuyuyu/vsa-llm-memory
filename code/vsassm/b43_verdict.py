"""B43 verdict (prereg/B43_natural_text_and_chunked_baseline_prereg.md), part B: the paper's store system against a
Chain-of-Agents (CoA) baseline with the same reader model, on RULER's 9 memory tasks (64k, 128k) and BABILong QA1-5
(128k). Part A (ProPara) is scored by b43_propara.py score; its numbers are copied in here from OUTDIR/scores.json.

python3 b43_verdict.py COA_DIR [--propara PP_OUTDIR] [--cost b43_out/cost.json]   -> stdout (results/b43_verdict.txt)
COA_DIR holds coa_ruler_<L>_*.jsonl and coa_bab_128k*.jsonl (one record per sample, b43_coa.py's output).
Samples are aligned by (task, position in RULER's file); index and gold are asserted equal to B21's.
"""
import argparse
import collections
import glob
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ruler_rm import string_match_all  # noqa: E402

NV = ["niah_single_1", "niah_single_2", "niah_single_3", "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
      "niah_multivalue", "niah_multiquery", "vt"]
LEN = [65536, 131072]
BTASKS = ["qa1", "qa2", "qa3", "qa4", "qa5"]
# CoA camera-ready App. F.4 ("catastrophic collapse"): rejection cues propagate along the chain
REJECT = __import__("re").compile(r"(not mentioned|no mention|does not (contain|mention|provide|include)|"
                                  r"no (information|evidence|relevant)|I don't know|cannot be (determined|found)|"
                                  r"not (provided|found|present|specified))", __import__("re").I)


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def paired(a, b, tasks, rng_seed=0, B=10000):
    """mean(a) - mean(b) over tasks (equal weight), 95% bootstrap resampling samples within each task, same draw."""
    rng = np.random.default_rng(rng_seed)
    d = np.stack([np.asarray(a[t], float) - np.asarray(b[t], float) for t in tasks])
    n = d.shape[1]
    idx = rng.integers(0, n, size=(B, len(tasks), n))
    bs = np.take_along_axis(np.broadcast_to(d, (B, len(tasks), n)), idx, axis=2).mean(axis=(1, 2))
    return 100 * float(d.mean()), 100 * float(np.percentile(bs, 2.5)), 100 * float(np.percentile(bs, 97.5))


def reading(d, lo, hi, m=3.0):
    if d >= m and lo > 0:
        return "store ahead"
    if lo > 0:
        return "store ahead, small"
    if hi < 0:
        return "CoA ahead"
    return "no detectable difference"


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def load_b21(L):
    out = {}
    for f in glob.glob(f"{HERE}/b21_dl/b21_test/{L}_*.jsonl"):
        for l in open(f):
            r = json.loads(l)
            out.setdefault(r["task"], []).append(r)
    return out


def ruler(coa_dir, res):
    print("## RULER, 9 memory tasks, seed-42 test sets, 100 samples per task (n = 900 per length)")
    print("store = B21 (Qwen3-8B reader, 250-character chunks, query-independent extraction, sharded FHRR store);")
    print("dict = the same reader output in a Python dict; CoA = Chain-of-Agents with Qwen3-8B over the same chunks")
    print("(8,192-token windows, verbatim CoA prompts, thinking off, greedy). Official string_match_all.\n")
    for L in LEN:
        recs, dup = {}, []
        for f in sorted(glob.glob(f"{coa_dir}/coa_ruler_{L}_*.jsonl")):
            for l in open(f):
                r = json.loads(l)
                if (r["task"], r["pos"]) in recs:
                    dup.append((r["task"], r["pos"]))
                recs[(r["task"], r["pos"])] = r
        if dup:
            print(f"  DUPLICATE samples at {L}: {len(dup)} {dup[:5]}")
        if not recs:
            print(f"### {L // 1024}k: no CoA output\n")
            continue
        b21 = load_b21(L)
        S = {"store": {}, "dict": {}, "coa": {}}
        diag = collections.Counter()
        tok = {"p": [], "g": [], "win": []}
        missing = []
        for t in NV:
            rows = b21[t]
            assert len(rows) == 100, (t, L, len(rows))
            for k in S:
                S[k][t] = []
            for pos, b in enumerate(rows):
                c = recs.get((t, pos))
                if c is None:
                    missing.append((t, pos))
                    continue
                assert c["index"] == b["index"] and c["outputs"] == b["outputs"], (t, pos)
                S["store"][t].append(string_match_all(b["pred_vsa"], b["outputs"]))
                S["dict"][t].append(string_match_all(b["pred_table"], b["outputs"]))
                sc = string_match_all(c["pred"], c["outputs"])
                S["coa"][t].append(sc)
                gold = [o.lower() for o in c["outputs"]]
                final = c["workers"][-1]["cu"].lower()
                kept = all(g in final for g in gold)
                ever = set(g for w in c["workers"] for g in gold if g in w["cu"].lower())
                lost = any(g in ever and g not in final for g in gold)
                diag["kept"] += kept
                diag["lost_along_chain"] += lost
                diag["never_seen_all"] += len(ever) < len(gold)
                diag["manager_failed_with_all_in_final"] += kept and sc < 1
                diag["final_cu_rejection_phrase"] += bool(REJECT.search(c["workers"][-1]["cu"]))
                diag["capped_worker_calls"] += sum(w["capped"] for w in c["workers"])
                diag["worker_calls"] += len(c["workers"])
                tok["p"].append(c["prompt_tokens"])
                tok["g"].append(c["gen_tokens"])
                tok["win"].append(c["n_windows"])
        n = sum(len(v) for v in S["coa"].values())
        print(f"### {L // 1024}k: CoA samples {n} / 900" + (f" (MISSING {len(missing)}: {missing[:5]})" if missing else ""))
        # registered bar needs all 900; if a task is incomplete, the full-set bar is not computed and the complete
        # tasks are reported (labelled), plus per-task paired differences on the samples that finished
        full = [t for t in NV if len(S["coa"][t]) == 100]
        part = [t for t in NV if 0 < len(S["coa"][t]) < 100]
        if missing:
            print(f"  INCOMPLETE: registered 9-task bar not computable; complete tasks {len(full)} / 9; "
                  f"partial {[(t, len(S['coa'][t])) for t in part]}; absent {[t for t in NV if not S['coa'][t]]}")
        tasks = full if missing else NV
        cell = {}
        lab = "9-task" if not missing else f"{len(tasks)}-complete-task"
        for k in ("store", "dict", "coa"):
            m = 100 * np.mean([np.mean(S[k][t]) for t in tasks])
            kk = sum(sum(S[k][t]) for t in tasks)
            ci = wilson(kk, 100 * len(tasks))
            cell[k] = m
            print(f"  {k:6s} {lab} mean {m:6.2f}  [{ci[0]:.1f}, {ci[1]:.1f}]")
        print("  per task (store / dict / CoA; n):")
        for t in NV:
            if S["coa"][t]:
                print(f"    {t:16s} {100 * np.mean(S['store'][t]):6.1f} {100 * np.mean(S['dict'][t]):6.1f} "
                      f"{100 * np.mean(S['coa'][t]):6.1f}   n={len(S['coa'][t])}")
        out = {"n": n, "complete_tasks": tasks, "incomplete": bool(missing), "mean": cell,
               "per_task": {k: {t: 100 * float(np.mean(S[k][t])) for t in NV if S[k][t]} for k in S},
               "per_task_n": {t: len(S["coa"][t]) for t in NV}}
        for a, b in (("store", "coa"), ("dict", "coa"), ("store", "dict")):
            d, lo, hi = paired(S[a], S[b], tasks)
            rd = (reading(d, lo, hi) if b == "coa" else "") + (" (complete tasks only; not the registered bar)"
                                                               if missing and b == "coa" else "")
            print(f"  {a} - {b} ({lab}): {d:+6.2f} [{lo:+.2f}, {hi:+.2f}]  {rd}")
            out[f"{a}-{b}"] = {"d": d, "ci": [lo, hi], "reading": rd}
        if missing:
            for t in part:
                d, lo, hi = paired({t: S["store"][t]}, {t: S["coa"][t]}, [t])
                print(f"  store - coa, {t} on its {len(S['coa'][t])} finished samples: {d:+6.2f} [{lo:+.2f}, {hi:+.2f}]")
                out[f"store-coa_{t}_partial"] = {"d": d, "ci": [lo, hi], "n": len(S["coa"][t])}
        print(f"  chain diagnostics over {n} samples: final CU holds every gold value {diag['kept']}; "
              f"a gold value seen in some CU but lost by the final one {diag['lost_along_chain']}; "
              f"some gold value never in any CU {diag['never_seen_all']}; "
              f"all gold in the final CU but the manager's answer failed {diag['manager_failed_with_all_in_final']}; "
              f"final CU with a rejection phrase {diag['final_cu_rejection_phrase']}; "
              f"worker calls at the 1,024-token cap {diag['capped_worker_calls']} / {diag['worker_calls']}")
        print(f"  CoA tokens per sample: prompt mean {np.mean(tok['p']):,.0f}, generated mean {np.mean(tok['g']):,.0f}; "
              f"windows mean {np.mean(tok['win']):.1f} (min {min(tok['win'])}, max {max(tok['win'])})\n")
        out.update(diag=dict(diag), coa_tokens={"prompt_mean": float(np.mean(tok["p"])), "gen_mean": float(np.mean(tok["g"])),
                                                "windows_mean": float(np.mean(tok["win"]))})
        res[f"ruler_{L}"] = out
        res[f"_S_ruler_{L}"] = {k: {t: list(map(float, v)) for t, v in S[k].items()} for k in S}


def babilong(coa_dir, res):
    sys.path.insert(0, os.path.join(HERE, "b30b_frozen"))
    import b18
    recs = {}
    for f in sorted(glob.glob(f"{coa_dir}/coa_bab_128k*.jsonl")):
        for l in open(f):
            r = json.loads(l)
            recs[(r["task"], r["index"])] = r
    print("## BABILong QA1-5, 128k, official test release, 100 samples per task (n = 500)")
    if not recs:
        print("  no CoA output (not run)\n")
        return
    mem = {(r["task"], r["i"]): r for r in map(json.loads, open(f"{HERE}/b30b_out/bab/assembled_128k.jsonl"))}
    A, Bc = {t: [] for t in BTASKS}, {t: [] for t in BTASKS}
    disc = [0, 0]
    missing = 0
    for t in BTASKS:
        for i in range(100):
            m, c = mem[(t, i)], recs.get((t, i))
            if c is None:
                missing += 1
                continue
            assert str(c["target"]) == str(m["target"]), (t, i)
            ok_c = b18.correct_official(c["pred"].strip(), c["target"], c["q"], t)
            A[t].append(float(m["ok_official"]))
            Bc[t].append(float(ok_c))
            disc[0] += m["ok_official"] and not ok_c
            disc[1] += ok_c and not m["ok_official"]
    n = sum(len(v) for v in Bc.values())
    print(f"  CoA samples {n} / 500" + (f" (MISSING {missing})" if missing else ""))
    if missing:
        return
    ma, mb = 100 * np.mean([np.mean(A[t]) for t in BTASKS]), 100 * np.mean([np.mean(Bc[t]) for t in BTASKS])
    d, lo, hi = paired(A, Bc, BTASKS)
    print(f"  memory {ma:.1f}   CoA (Qwen3-8B) {mb:.1f}   per task CoA: " +
          " ".join(f"{t} {100 * np.mean(Bc[t]):.0f}" for t in BTASKS))
    print(f"  memory - CoA {d:+.1f} [{lo:+.1f}, {hi:+.1f}]  discordant {disc[0]}/{disc[1]}, McNemar p {mcnemar(*disc):.2g}"
          f"  -> {reading(d, lo, hi)}\n")
    res["babilong_128k"] = {"memory": ma, "coa": mb, "d": d, "ci": [lo, hi], "reading": reading(d, lo, hi),
                            "discordant": disc, "per_task_coa": {t: 100 * float(np.mean(Bc[t])) for t in BTASKS}}


def predictions(res):
    """The registered predictions, checked mechanically where the data exist."""
    print("## Registered predictions")
    r = res.get("ruler_131072")
    if r:
        inc = " (on the complete tasks only)" if r["incomplete"] else ""
        m = r["mean"]["coa"]
        print(f"  PB1 CoA 9-task mean at 128k in [80, 97]: {m:.1f}{inc} -> {'held' if 80 <= m <= 97 else 'FAILED'}")
        rd = r["store-coa"]["reading"]
        print(f"  PB2 C1 reads 'store ahead': {rd} -> {'held' if rd.startswith('store ahead') and 'small' not in rd else 'FAILED'}"
              + (" (not the registered bar: incomplete)" if r["incomplete"] else ""))
        print("  PB3 CoA 64k >= CoA 128k: not tested (64k not run)")
        cost = res.get("store_reader_cost", {}).get("131072")
        if cost:
            coa_t = r["coa_tokens"]["prompt_mean"] + r["coa_tokens"]["gen_mean"]
            st = cost["nocache_prompt"] + cost["nocache_gen"]
            print(f"  PB4 CoA tokens per sample < store reader without cache: {coa_t:,.0f} vs {st:,.0f} -> "
                  f"{'held' if coa_t < st else 'FAILED'}")
        pt = r["per_task"]["coa"]
        low3 = sorted(pt, key=pt.get)[:3]
        print(f"  PB5 weakest CoA tasks are multivalue, multiquery, vt: lowest three {low3} -> "
              f"{'held' if set(low3) == {'niah_multivalue', 'niah_multiquery', 'vt'} else 'FAILED'}")
        d = r["diag"]
        n = r["n"]
        print(f"  PB7 final CU with a rejection phrase < 20%: {d['final_cu_rejection_phrase']}/{n} = "
              f"{100 * d['final_cu_rejection_phrase'] / n:.0f}% -> {'held' if d['final_cu_rejection_phrase'] < 0.2 * n else 'FAILED'};"
              f" failures from values never written or lost (not manager errors): manager errors "
              f"{d['manager_failed_with_all_in_final']} -> {'held' if d['manager_failed_with_all_in_final'] == 0 else 'check'}")
    print("  PB6 BABILong: not tested (not run)")
    pp = res.get("propara")
    if pp:
        c = pp["comparisons"]["store_vsa - direct_think"]
        beats = c["f1_d"] >= 10 and c["f1_ci"][0] > 0
        dt, st = pp["direct_think"]["official_f1"], pp["store_vsa"]["official_f1"]
        print(f"  PA1 A1 does not reach 'beats': {'held' if not beats else 'FAILED'}; direct_think {dt:.1f} in [45, 62]: "
              f"{'held' if 45 <= dt <= 62 else 'FAILED'}; store {st:.1f} in [40, 58]: {'held' if 40 <= st <= 58 else 'FAILED'}")
        print(f"  PA2 VSA = dict on 100% of cells: {'held' if pp['vsa_eq_dict'][0] == pp['vsa_eq_dict'][1] else 'FAILED'}")
        e = pp["extraction"]
        print(f"  PA3 extraction recall < 0.70: {'held' if e['recall'] < 0.70 else 'FAILED'} ({e['recall']:.2f}); "
              f"precision > recall: {'held' if e['precision'] > e['recall'] else 'FAILED'} ({e['precision']:.2f} vs {e['recall']:.2f})")
        c2 = pp["comparisons"]["direct_think - direct_nothink"]
        print(f"  PA4 direct_think >= direct_nothink: {'held' if c2['f1_d'] >= 0 else 'FAILED'} ({c2['f1_d']:+.1f})")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("coa_dir")
    ap.add_argument("--propara")
    ap.add_argument("--cost")
    ap.add_argument("--json", default=f"{HERE}/results/b43_scores.json")
    a = ap.parse_args()
    res = {}
    print("# B43 verdict (prereg/B43_natural_text_and_chunked_baseline_prereg.md)\n")
    ruler(a.coa_dir, res)
    babilong(a.coa_dir, res)
    if a.cost and os.path.exists(a.cost):
        c = json.load(open(a.cost))
        res["store_reader_cost"] = c
        print("## Store reader tokens per sample (B21's chunks and raw outputs recounted; Qwen3 tokenizer, chat template)")
        for L, v in c.items():
            print(f"  {int(L) // 1024}k: " + ", ".join(f"{k} {x:,.0f}" for k, x in v.items()))
        print()
    if a.propara and os.path.exists(f"{a.propara}/scores.json"):
        res["propara"] = pp = json.load(open(f"{a.propara}/scores.json"))
        print("## Part A: ProPara test (54 paragraphs, 236 participants, 1,910 state cells), Qwen3-8B")
        print("official document-level F1 (aristo-leaderboard evaluator; CLI agrees) / precision / recall; state accuracy")
        for arm in ("direct_think", "direct_nothink", "store_vsa", "store_dict"):
            x = pp[arm]
            print(f"  {arm:15s} F1 {x['official_f1']:5.1f}  P {x['official_p']:5.1f}  R {x['official_r']:5.1f}   "
                  f"state acc {x['state_acc']:5.1f}   implied actions P {x['actions_implied']['precision']:.2f} "
                  f"R {x['actions_implied']['recall']:.2f}")
        e = pp["extraction"]
        print(f"  extraction (the extractor's own events vs gold actions): precision {e['precision']:.3f}, recall "
              f"{e['recall']:.3f}, F1 {e['f1']:.3f} ({e['pred']} predicted, {e['gold']} gold, {e['tp']} correct); "
              f"destination accuracy {e['dest_acc']:.3f} on {e['dest_n']} matched create/move")
        print(f"  store_vsa == store_dict on {pp['vsa_eq_dict'][0]} / {pp['vsa_eq_dict'][1]} cells")
        print("  paired differences (95% bootstrap over paragraphs, 10,000 resamples, seed 0):")
        for k, v in pp["comparisons"].items():
            print(f"    {k:32s} F1 {v['f1_d']:+6.1f} [{v['f1_ci'][0]:+.1f}, {v['f1_ci'][1]:+.1f}]   state acc "
                  f"{v['state_d']:+6.1f} [{v['state_ci'][0]:+.1f}, {v['state_ci'][1]:+.1f}]")
        c = pp["comparisons"]["store_vsa - direct_think"]
        d, lo, hi = c["f1_d"], c["f1_ci"][0], c["f1_ci"][1]
        rd = ("store route beats the direct LM" if d >= 10 and lo > 0 else "store ahead, < 10" if lo > 0 else
              "direct ahead" if hi < 0 else "no detectable difference")
        print(f"  A1 (store_vsa - direct_think >= 10 with CI > 0): {d:+.1f} [{lo:+.1f}, {hi:+.1f}] -> {rd}")
        print(f"  A2 (store_vsa = store_dict on 100% of cells): "
              f"{'PASS' if pp['vsa_eq_dict'][0] == pp['vsa_eq_dict'][1] else 'FAIL'}\n")
    predictions(res)
    json.dump({k: v for k, v in res.items() if not k.startswith("_")}, open(a.json, "w"), indent=1)
    json.dump({k: v for k, v in res.items() if k.startswith("_S")}, open(a.json.replace(".json", "_persample.json"), "w"))


if __name__ == "__main__":
    main()
