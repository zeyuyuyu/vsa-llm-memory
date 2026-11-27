"""B23 scoring (prereg/B23_babilong_finetune_baseline_prereg.md).

Per run (arm, seed) x split x length x task: accuracy under our scorer (b18.correct_official on the whole output) and
under the verbatim official compare_answers, Wilson 95% intervals, outputs with no label and outputs at the token
limit; the dev gate and the dev choice of the primary arm; the paired comparison with the banked B18 per-question
results (exact McNemar); the frontier subset (first 25 per task); bars 1 and 2.

python3 b23_score.py --outs b23_out --b18 b23_ref/b18_s0 --labels b23_data/labels.json --json results/b23_scores.json
"""
import argparse
import glob
import json
import os
import re

import b23

TASKS = b23.TASKS
OURS = {"0k": 99.6, "4k": 97.8, "16k": 99.2, "64k": 99.6, "128k": 99.8}  # b18_verdict.txt row QA1-5 (checked below)
FRONTIER = {  # b18_api_verdict.txt, first 25 per task (n = 125): QA1-5 under the official rule on the whole output
    "GPT-6.1 Sol": {"0k": 100.0, "4k": 99.2, "16k": 100.0, "64k": 99.2, "128k": 96.8},
    "Claude Opus 5.5": {"0k": 100.0, "4k": 97.6, "16k": 95.2, "64k": 96.0, "128k": 97.6},
}


def no_label(r, labs):
    out = str(r["output"]).lower().split(".")[0]
    q = r["question"].lower()
    cand = {l for l in labs[r["task"]] if not re.search(r"\b" + re.escape(l) + r"\b", q)}
    return not any(re.search(r"\b" + re.escape(l) + r"\b", out) for l in cand)


def cell(rows):
    k = sum(r["ok"] for r in rows)
    n = len(rows)
    lo, hi = b23.wilson(k, n)
    return {"k": k, "n": n, "acc": round(100 * k / n, 1) if n else None, "wilson95": [round(100 * lo, 1), round(100 * hi, 1)],
            "k_official": sum(r["ok_off"] for r in rows), "scorer_disagree": sum(r["ok"] != r["ok_off"] for r in rows),
            "no_label": sum(r["nolab"] for r in rows), "hit_limit": sum(bool(r.get("hit_limit")) for r in rows)}


def table(rows, lengths):
    out = {}
    for L in lengths:
        rl = [r for r in rows if r["length"] == L]
        if not rl:
            continue
        out[L] = {"QA1-5": cell(rl), **{t: cell([r for r in rl if r["task"] == t]) for t in TASKS}}
    return out


def load_b18(d, L, task):
    fs = sorted(glob.glob(f"{d}/{L}_{task}.jsonl"))
    if not fs:
        return None
    return [json.loads(l) for l in open(fs[0])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outs", default="b23_out")
    ap.add_argument("--b18", default="b23_ref/b18_s0")
    ap.add_argument("--labels", default="b23_data/labels.json")
    ap.add_argument("--json", default="results/b23_scores.json")
    a = ap.parse_args()
    labs = json.load(open(a.labels))
    res = {"runs": {}}
    for f in sorted(glob.glob(f"{a.outs}/*_s*.jsonl")):
        split, arm, seed = re.match(r"(dev|test)_([NY])_s(\d+)", os.path.basename(f)).groups()
        rows = {}
        for r in map(json.loads, open(f)):
            rows[(r["task"], r["length"], r["i"])] = r  # a resumed run never repeats a key; keep the last if it did
        rows = sorted(rows.values(), key=lambda r: (TASKS.index(r["task"]), r["length"], r["i"]))
        for r in rows:
            r["ok"] = b23.ok_ours(r, a.labels)
            r["ok_off"] = b23.ok_official(r)
            r["nolab"] = no_label(r, labs)
        lengths = b23.DEV_LENGTHS if split == "dev" else b23.TEST_LENGTHS
        res["runs"].setdefault(f"{arm}_s{seed}", {})[split] = table(rows, lengths)
        res["runs"][f"{arm}_s{seed}"][split + "_rows"] = rows
    # dev: gate and the primary arm (seed 0)
    sel = {}
    for run, v in res["runs"].items():
        if "dev" in v:
            d = v["dev"]
            five = [d[L]["QA1-5"]["acc"] for L in b23.DEV_LENGTHS if L in d]
            v["dev_five_length_mean"] = round(sum(five) / len(five), 2) if len(five) == 5 else None
            v["bar0"] = d.get("0k", {}).get("QA1-5", {}).get("acc", 0) >= 90.0
            if run.endswith("_s0"):
                sel[run[0]] = v["dev_five_length_mean"]
    if len(sel) == 2 and None not in sel.values():
        res["primary"] = "N" if sel["N"] > sel["Y"] else "Y"
        res["primary_rule"] = f"dev five-length mean: N {sel['N']} vs Y {sel['Y']} (tie -> Y)"
    # paired comparison with B18 (ours) on identical test questions
    for run, v in res["runs"].items():
        if "test_rows" not in v:
            continue
        cmp = {}
        for L in OURS:
            ours_all, base_all, bad = [], [], 0
            for t in TASKS:
                o = load_b18(a.b18, L, t)
                b = [r for r in v["test_rows"] if r["length"] == L and r["task"] == t]
                if o is None or len(b) != 100:
                    break
                o = o[:100]
                bad += sum(x["question"] != y["question"] or str(x["target"]).strip().lower() != y["target"].lower()
                           for x, y in zip(o, sorted(b, key=lambda r: r["i"])))
                ours_all += [bool(x["ok_official"]) for x in o]
                base_all += [r["ok"] for r in sorted(b, key=lambda r: r["i"])]
            else:
                ko, kb = sum(ours_all), sum(base_all)
                bb = sum(x and not y for x, y in zip(ours_all, base_all))
                cc = sum(y and not x for x, y in zip(ours_all, base_all))
                cmp[L] = {"ours": round(100 * ko / 500, 1), "ours_verdict": OURS[L], "baseline": round(100 * kb / 500, 1),
                          "diff_points": round(100 * (ko - kb) / 500, 1), "ours_only": bb, "baseline_only": cc,
                          "mcnemar_p": b23.mcnemar_exact(bb, cc), "mismatched_rows": bad}
        v["vs_ours"] = cmp
        sub = {}
        for L in OURS:
            rl = [r for r in v["test_rows"] if r["length"] == L and r["i"] < 25]
            if len(rl) == 125:
                ours_sub = []
                for t in TASKS:
                    ours_sub += [bool(x["ok_official"]) for x in (load_b18(a.b18, L, t) or [])[:25]]
                sub[L] = {"baseline": cell(rl), "ours": round(100 * sum(ours_sub) / max(1, len(ours_sub)), 1),
                          **{m: FRONTIER[m][L] for m in FRONTIER}}
        v["frontier_subset"] = sub
    prim = res.get("primary")
    if prim and f"{prim}_s0" in res["runs"] and "vs_ours" in res["runs"][f"{prim}_s0"]:
        c = res["runs"][f"{prim}_s0"]["vs_ours"]
        for name, L in (("bar1", "128k"), ("bar2", "64k")):
            if L in c:
                res[name] = {"length": L, "diff": c[L]["diff_points"], "p": c[L]["mcnemar_p"],
                             "pass": c[L]["diff_points"] >= 5.0 and c[L]["mcnemar_p"] < 0.05}
    os.makedirs(os.path.dirname(a.json) or ".", exist_ok=True)
    slim = {k: ({r: {kk: vv for kk, vv in rv.items() if not kk.endswith("_rows")} for r, rv in v.items()}
                if k == "runs" else v) for k, v in res.items()}
    json.dump(slim, open(a.json, "w"), indent=1)
    # printout
    for run, v in slim["runs"].items():
        for split in ("dev", "test"):
            if split not in v:
                continue
            print(f"\n== {run} {split} (our scorer; official scorer in brackets where it differs)")
            print("length  " + "  ".join(f"{t:>6s}" for t in TASKS) + "   QA1-5 [Wilson 95%]   n   no-label  limit")
            for L, row in v[split].items():
                q = row["QA1-5"]
                off = f" (off {100 * q['k_official'] / q['n']:.1f})" if q["scorer_disagree"] else ""
                print(f"{L:6s}  " + "  ".join(f"{row[t]['acc']:6.1f}" for t in TASKS) +
                      f"   {q['acc']:5.1f} [{q['wilson95'][0]:.1f}-{q['wilson95'][1]:.1f}]{off}  {q['n']}   {q['no_label']}   {q['hit_limit']}")
        if "dev_five_length_mean" in v:
            print(f"dev five-length mean {v['dev_five_length_mean']}; bar 0 {'PASS' if v['bar0'] else 'FAIL'}")
        for L, c in v.get("vs_ours", {}).items():
            print(f"vs ours {L}: ours {c['ours']} (verdict {c['ours_verdict']}) baseline {c['baseline']} diff {c['diff_points']:+.1f}"
                  f"  ours-only {c['ours_only']} baseline-only {c['baseline_only']}  McNemar p {c['mcnemar_p']:.2g}"
                  f"  mismatched rows {c['mismatched_rows']}")
        for L, s in v.get("frontier_subset", {}).items():
            print(f"subset {L} (n=125): baseline {s['baseline']['acc']} ours {s['ours']} GPT-6.1 Sol {s['GPT-6.1 Sol']} "
                  f"Opus 5.5 {s['Claude Opus 5.5']}")
    for k in ("primary", "primary_rule", "bar1", "bar2"):
        if k in slim:
            print(k, slim[k])


if __name__ == "__main__":
    main()
