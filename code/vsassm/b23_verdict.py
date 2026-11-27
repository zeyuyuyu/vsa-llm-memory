"""B23 verdict text from results/b23_scores.json (b23_score.py) and the GPU-hour ledger.

python3 b23_verdict.py --scores results/b23_scores.json --gpu results/b23_gpu_hours.json --out results/b23_verdict.txt
The prose notes are added by hand below the generated part.
"""
import argparse
import json

LENS = ["0k", "4k", "16k", "32k", "64k", "128k"]
OURS = {"0k": 99.6, "4k": 97.8, "16k": 99.2, "64k": 99.6, "128k": 99.8}
PRED = {"0k": (97, 90, 100), "4k": (95, None, None), "16k": (93, None, None), "32k": (90, None, None),
        "64k": (85, 65, 97), "128k": (72, 45, 92)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="results/b23_scores.json")
    ap.add_argument("--gpu", default="results/b23_gpu_hours.json")
    ap.add_argument("--out", default="results/b23_verdict.txt")
    a = ap.parse_args()
    S = json.load(open(a.scores))
    G = json.load(open(a.gpu))
    R = S["runs"]
    prim = S.get("primary")
    out = []
    w = out.append
    w("# B23 verdict (prereg/B23_babilong_finetune_baseline_prereg.md): Qwen3-1.7B fine-tuned end to end on BABILong,")
    w("# no memory -- the referee baseline for the B18 reader + memory")
    w("")
    w("Metric: our implementation of the official BABILong rule (b18.correct_official) on the whole output; the verbatim")
    w("official compare_answers is the second scorer. Test = RMT-team/babilong, all 100 samples per task (n = 500 per")
    w("length). Cells: accuracy in per cent; QA1-5 with its 95% Wilson interval.")
    w("")
    w("## Dev (training split, indices 4900-4919, n = 100 per length) -- gate and choice of the primary arm")
    for run in sorted(R):
        v = R[run]
        if "dev" not in v:
            continue
        d = v["dev"]
        row = "  ".join(f"{L} {d[L]['QA1-5']['acc']:.1f}" for L in d)
        w(f"  {run}: {row}   five-length mean {v.get('dev_five_length_mean')}   bar 0 {'PASS' if v.get('bar0') else 'FAIL'}")
    w(f"  Primary arm: {prim} ({S.get('primary_rule')})")
    w("")
    for run in sorted(R, key=lambda r: (r != f"{prim}_s0", r)):
        v = R[run]
        if "test" not in v:
            continue
        tag = " (PRIMARY)" if run == f"{prim}_s0" else (" (seed replicate)" if run.endswith("_s1") else "")
        w(f"## Test, run {run}{tag}")
        w("length     qa1    qa2    qa3    qa4    qa5    QA1-5 [Wilson 95%]   official  no-label  ours (B18)  diff   McNemar p")
        for L in LENS:
            if L not in v["test"]:
                continue
            c = v["test"][L]
            q = c["QA1-5"]
            cmp = v.get("vs_ours", {}).get(L)
            tail = (f"{cmp['ours']:6.1f}   {cmp['diff_points']:+5.1f}   {cmp['mcnemar_p']:.2g} "
                    f"({cmp['ours_only']}/{cmp['baseline_only']})" if cmp else "     -        -")
            w(f"{L:6s} " + " ".join(f"{c[t]['acc']:6.1f}" for t in ["qa1", "qa2", "qa3", "qa4", "qa5"]) +
              f"   {q['acc']:5.1f} [{q['wilson95'][0]:.1f}-{q['wilson95'][1]:.1f}]   {100 * q['k_official'] / q['n']:5.1f}"
              f"     {q['no_label']:3d}     {tail}")
        w("  (diff = ours minus baseline, points; McNemar exact two-sided on the paired questions, discordant pairs"
          " ours-only/baseline-only; n = 500 per length)")
        sub = v.get("frontier_subset", {})
        if sub:
            w("  Frontier subset (first 25 per task, n = 125): " + "; ".join(
                f"{L}: baseline {s['baseline']['acc']:.1f}, ours {s['ours']:.1f}, GPT-6.1 Sol {s['GPT-6.1 Sol']:.1f}, "
                f"Opus 5.5 {s['Claude Opus 5.5']:.1f}" for L, s in sub.items()))
        w("")
    w("## Bars (seed 0, primary arm)")
    for name, txt in (("bar1", "Bar 1, 128k: ours - baseline >= 5.0 points and McNemar p < 0.05"),
                      ("bar2", "Bar 2, 64k: ours - baseline >= 5.0 points and McNemar p < 0.05")):
        b = S.get(name)
        if b:
            w(f"  {txt}: diff {b['diff']:+.1f}, p = {b['p']:.2g} -> {'PASS' if b['pass'] else 'FAIL'}")
    w("  Bar 0 (dev QA1-5 at 0k >= 90): " + ", ".join(
        f"{r} {'PASS' if R[r].get('bar0') else 'FAIL'} ({R[r]['dev']['0k']['QA1-5']['acc']:.1f})" for r in sorted(R) if "dev" in R[r]))
    w("")
    w("## Predictions")
    P = R.get(f"{prim}_s0", {}).get("test", {})
    for L in LENS:
        if L in P:
            p, lo, hi = PRED[L]
            got = P[L]["QA1-5"]["acc"]
            rng = f" (range {lo}-{hi}: {'HELD' if lo <= got <= hi else 'NOT HELD'})" if lo is not None else " (point prediction)"
            w(f"  {L}: predicted {p}, measured {got:.1f}, off by {got - p:+.1f}{rng}")
    ok0 = all(R[r]["dev"]["0k"]["QA1-5"]["acc"] >= 95 for r in R if "dev" in R[r])
    w(f"  Bar 0 passes for every run with dev 0k >= 95: {'HELD' if ok0 else 'NOT HELD'}")
    w(f"  Arm Y is primary (dev Y >= N): {'HELD' if prim == 'Y' else 'NOT HELD'}")
    b1, b2 = S.get("bar1", {}).get("pass"), S.get("bar2", {}).get("pass")
    w(f"  Bars 1 and 2 both pass: {'HELD' if b1 and b2 else 'NOT HELD'}")
    weakest = []
    for L in LENS[1:]:
        if L in P:
            accs = {t: P[L][t]["acc"] for t in ["qa1", "qa2", "qa3", "qa4", "qa5"]}
            weakest.append((L, min(accs, key=accs.get), accs["qa3"] == min(accs.values())))
    w(f"  qa3 weakest at every length from 4k: {('HELD' if all(x[2] for x in weakest) else 'NOT HELD') if weakest else 'n/a'} "
      f"(weakest per length: {', '.join(f'{L} {t}' for L, t, _ in weakest)})")
    if "N_s0" in R and "Y_s0" in R and "test" in R["N_s0"] and "test" in R["Y_s0"] and "128k" in R["N_s0"]["test"]:
        n128, y128 = R["N_s0"]["test"]["128k"]["QA1-5"]["acc"], R["Y_s0"]["test"]["128k"]["QA1-5"]["acc"]
        w(f"  Arm N at 128k >= 5 points below arm Y: N {n128:.1f} vs Y {y128:.1f} -> {'HELD' if y128 - n128 >= 5 else 'NOT HELD'}")
    near = [(L, P[L]["QA1-5"]["acc"]) for L in ("0k", "4k", "16k") if L in P]
    w(f"  Baseline within 3 points of ours at 0k / 4k / 16k: " + ", ".join(
        f"{L} {OURS[L] - x:+.1f}" for L, x in near) + f" -> {('HELD' if all(abs(OURS[L] - x) <= 3 for L, x in near) else 'NOT HELD') if near else 'n/a'}")
    sub = R.get(f"{prim}_s0", {}).get("frontier_subset", {}).get("128k")
    if sub:
        bl = sub["baseline"]["acc"]
        w(f"  At 128k (subset) baseline below GPT-6.1 Sol 96.8 and Opus 5.5 97.6: baseline {bl:.1f} -> "
          f"{'HELD' if bl < 96.8 and bl < 97.6 else 'NOT HELD'}")
    w("")
    w("## Compute (sacct elapsed x 1 GPU, every B23 GPU job including failed and cancelled ones)")
    tot = 0.0
    for j in G["jobs"]:
        w(f"  {j['id']} {j['name']:10s} {j['state']:10s} {j['node']:22s} {j['gpu']:8s} {j['hours']:.2f} h")
        tot += j["hours"]
    w(f"  Total {tot:.1f} GPU-hours; at $1.17 per GPU-hour about ${1.17 * tot:.0f} (actual PACE rates are lower on A100;"
      f" CPU jobs for download / prep / token counts not included, < 0.3 CPU-node-hours).")
    open(a.out, "w").write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
