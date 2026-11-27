"""B24 verdict (prereg/B24_ruler_store_ablation_prereg.md): VSA store (B21) vs arm F (facts in the prompt) vs arm R
(BM25 RAG) vs Qwen3-8B direct (B19), on the 9 RULER memory tasks, per length.

python3 b24_verdict.py B21_DIR B19_DIRECT_DIR B24_DIR [LEDGER.json [SCORES.json]] > results/b24_verdict.txt
Samples are aligned by (task, length, position in RULER's file), and index and gold are asserted equal across systems.
LEDGER.json (optional): [{"job": id, "gpu_hours": h, "gpu": name, "what": text}, ...] from sacct.
Written before any B24 test output exists.
"""
import collections
import glob
import json
import math
import os
import sys

import numpy as np

import ruler_rm as R

NV = R.NIAH + ["vt"]
LEN = [4096, 8192, 16384, 32768, 65536, 131072]
SYS = ["store", "F", "R", "direct"]
NAME = {"store": "VSA store (B21)", "F": "arm F, facts in prompt", "R": "arm R, BM25 top-20", "direct": "direct (B19)"}
USD_PER_GPU_HOUR = 1.17


def rows_by_cell(files, pred_key):
    cells = collections.defaultdict(list)
    for f in files:
        for l in open(f):
            r = json.loads(l)
            cells[(r["task"], int(r["length"]))].append(r)
    return cells, pred_key


def wilson(p, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def main(b21, b19d, b24, ledger=None, scores_path=None):
    b21c = collections.defaultdict(list)
    for f in sorted(glob.glob(os.path.join(b21, "*.jsonl"))):
        for l in open(f):
            r = json.loads(l)
            b21c[(r["task"], int(r["length"]))].append(r)
    dirc = collections.defaultdict(list)
    for f in sorted(glob.glob(os.path.join(b19d, "*.jsonl"))):
        for l in open(f):
            r = json.loads(l)
            dirc[(r["task"], int(r["length"]))].append(r)
    arm = {"F": collections.defaultdict(dict), "R": collections.defaultdict(dict)}
    for a, pat in (("F", "armf_*.jsonl"), ("R", "armr_*.jsonl")):
        for f in sorted(glob.glob(os.path.join(b24, pat))):
            for l in open(f):
                r = json.loads(l)
                arm[a][(r["task"], int(r["length"]))][r["pos"]] = r   # a resumed run may repeat a pos: last wins
    # score[sys][(task, L)] = list of 100 sample scores in file order
    score = {s: {} for s in SYS}
    meta = {"F": collections.defaultdict(list), "R": collections.defaultdict(list)}
    missing = []
    for L in LEN:
        for t in NV:
            ref = b21c[(t, L)]
            score["store"][(t, L)] = [R.string_match_all(r["pred_vsa"], r["outputs"]) for r in ref]
            d = dirc[(t, L)]
            assert len(d) == len(ref) and all(x["index"] == y["index"] and x["outputs"] == y["outputs"]
                                              for x, y in zip(d, ref)), ("direct misaligned", t, L)
            score["direct"][(t, L)] = [R.string_match_all(r["pred"], r["outputs"]) for r in d]
            for a in ("F", "R"):
                got = arm[a].get((t, L), {})
                if len(got) != len(ref):
                    missing.append((a, t, L, len(got)))
                    continue
                rs = [got[i] for i in range(len(ref))]
                assert all(x["index"] == y["index"] and x["outputs"] == y["outputs"] for x, y in zip(rs, ref)), (a, t, L)
                score[a][(t, L)] = [R.string_match_all(r["pred"], r["outputs"]) for r in rs]
                meta[a][L].extend(rs)
    if missing:
        print("INCOMPLETE:", missing)
    out = {"per_task": {}, "per_length": {}, "diff": {}, "bars": {}, "predictions": {}}

    def mean9(s, L):
        if not all((t, L) in score[s] for t in NV):
            return None, None, None
        xs = [x for t in NV for x in score[s][(t, L)]]
        m = float(np.mean(xs))
        return 100 * m, wilson(m, len(xs)), len(xs)

    def task(s, t, L):
        v = score[s].get((t, L))
        return 100 * float(np.mean(v)) if v else None

    rng = np.random.default_rng(0)

    def paired(a, b, L, B=10000):
        """mean(a) - mean(b) over the 9 tasks at L, and a 95% bootstrap interval, resampling the 100 samples within
        each task (stratified), the same draw for both systems."""
        if not all((t, L) in score[a] and (t, L) in score[b] for t in NV):
            return None
        da = np.stack([np.array(score[a][(t, L)]) - np.array(score[b][(t, L)]) for t in NV])   # 9 x 100
        n = da.shape[1]
        idx = rng.integers(0, n, size=(B, len(NV), n))
        bs = np.take_along_axis(np.broadcast_to(da, (B, len(NV), n)), idx, axis=2).mean(axis=(1, 2))
        return 100 * float(da.mean()), 100 * float(np.percentile(bs, 2.5)), 100 * float(np.percentile(bs, 97.5))

    print("# B24 verdict (prereg/B24_ruler_store_ablation_prereg.md): does the VSA store matter beyond the reader's extraction?\n")
    print("9 RULER memory tasks (niah_single_1-3, niah_multikey_1-3, niah_multivalue, niah_multiquery, vt), seed-42 test")
    print("sets, 100 samples per task and length; official string_match_all; same Qwen3-8B everywhere, nothing trained.\n")
    print("9-task mean (95% Wilson interval, n = 900 sample scores per length)")
    print("system".ljust(26) + "".join(f"{L // 1024}k".rjust(20) for L in LEN))
    for s in SYS:
        cells = []
        for L in LEN:
            m, ci, n = mean9(s, L)
            cells.append(f"{m:5.1f} [{ci[0]:4.1f},{ci[1]:5.1f}]".rjust(20) if m is not None else "-".rjust(20))
            if m is not None:
                out["per_length"].setdefault(str(L), {})[s] = {"mean": round(m, 2), "wilson95": [round(ci[0], 2), round(ci[1], 2)], "n": n}
        print(NAME[s].ljust(26) + "".join(cells))
    print("\nPaired differences, 9-task mean (points; 95% stratified paired bootstrap, 10,000 resamples, seed 0)")
    for a, b in (("store", "F"), ("store", "R"), ("F", "direct"), ("store", "direct"), ("F", "R")):
        cells = []
        for L in LEN:
            p = paired(a, b, L)
            cells.append(f"{p[0]:+5.1f} [{p[1]:+5.1f},{p[2]:+5.1f}]".rjust(20) if p else "-".rjust(20))
            if p:
                out["diff"].setdefault(f"{a}-{b}", {})[str(L)] = [round(x, 2) for x in p]
        print(f"{a} - {b}".ljust(26) + "".join(cells))
    print("\nPer task (store / F / R / direct)")
    print("task".ljust(17) + "".join(f"{L // 1024}k".rjust(24) for L in LEN))
    for t in NV:
        cells = []
        for L in LEN:
            v = [task(s, t, L) for s in SYS]
            cells.append(" ".join(f"{x:5.1f}" if x is not None else "    -" for x in v).rjust(24))
            out["per_task"][f"{L}/{t}"] = {s: (round(x, 2) if x is not None else None) for s, x in zip(SYS, v)}
        print(t.ljust(17) + "".join(cells))

    # prompts and YaRN
    print("\nPrompts (template and answer prefix included): max tokens / samples run with YaRN, per length")
    for a in ("F", "R"):
        cells = []
        for L in LEN:
            rs = meta[a].get(L, [])
            cells.append((f"{max(r['prompt_tokens'] for r in rs)} / {sum(r['yarn'] for r in rs)}" if rs else "-").rjust(16))
        print(f"arm {a}".ljust(10) + "".join(cells))
    for L in LEN:
        rs = meta["F"].get(L, [])
        if rs:
            by = collections.defaultdict(list)
            for r in rs:
                by[r["task"]].append(r)
            big = {t: (max(r["prompt_tokens"] for r in v), sum(r["yarn"] for r in v)) for t, v in by.items()
                   if max(r["prompt_tokens"] for r in v) > 4000}
            if big:
                print(f"  arm F {L // 1024}k long fact lists (max tokens, YaRN samples):", big)
    gpus = collections.Counter(r["gpu"] for a in ("F", "R") for L in LEN for r in meta[a].get(L, []))
    print("GPUs (samples):", dict(gpus))

    # bars
    def bar(L):
        p = paired("store", "F", L)
        return None if p is None else (p[0] >= 3.0 and p[1] > 0, p)
    s1, s2 = bar(131072), bar(65536)
    r1 = [paired("store", "R", L) for L in LEN]
    r1ok = None if any(p is None for p in r1) else all(p[0] >= 3.0 and p[1] > 0 for p in r1)
    fmt = lambda b: "n/a" if b is None else ("PASS" if b else "FAIL")
    print("\nBars:")
    print(f"  S1 (128k: store - arm F >= 3.0 points, paired interval above 0): {fmt(s1 and s1[0])}"
          + (f", {s1[1][0]:+.2f} [{s1[1][1]:+.2f}, {s1[1][2]:+.2f}]" if s1 else ""))
    print(f"  S2 (64k: the same): {fmt(s2 and s2[0])}" + (f", {s2[1][0]:+.2f} [{s2[1][1]:+.2f}, {s2[1][2]:+.2f}]" if s2 else ""))
    print(f"  R1 (every length: store - arm R >= 3.0, paired interval above 0): {fmt(r1ok)}"
          + ("" if r1ok is None else ", " + ", ".join(f"{L // 1024}k {p[0]:+.1f} [{p[1]:+.1f},{p[2]:+.1f}]" for L, p in zip(LEN, r1))))
    out["bars"] = {"S1": None if s1 is None else bool(s1[0]), "S2": None if s2 is None else bool(s2[0]), "R1": r1ok}
    rev = [(L, p) for L in LEN if (p := paired("F", "store", L)) and p[0] >= 1.0 and p[1] > 0]
    print("  Reverse check (arm F above the store by >= 1.0 with interval above 0):", rev or "none")

    # predictions
    m = lambda s, L: mean9(s, L)[0]
    T7 = ["niah_single_1", "niah_single_2", "niah_single_3", "niah_multikey_1", "niah_multivalue", "niah_multiquery", "vt"]
    preds = [
        ("PF1 arm F 9-task mean >= 98.5 at 4k-32k", lambda: all(m("F", L) >= 98.5 for L in LEN[:4])),
        ("PF2 arm F below the store at 64k and 128k, S1 and S2 PASS",
         lambda: m("F", 65536) < m("store", 65536) and m("F", 131072) < m("store", 131072) and s1[0] and s2[0]),
        ("PF3 arm F 9-task mean at 128k in [80, 95]", lambda: 80 <= m("F", 131072) <= 95),
        ("PF4 arm F multikey_3 <= 80 at 64k and <= 60 at 128k; multikey_2 <= 90 at 128k",
         lambda: task("F", "niah_multikey_3", 65536) <= 80 and task("F", "niah_multikey_3", 131072) <= 60
         and task("F", "niah_multikey_2", 131072) <= 90),
        ("PF5 arm F >= 97 on each of the other 7 tasks at every length",
         lambda: all(task("F", t, L) >= 97 for t in T7 for L in LEN)),
        ("PF6 arm F 9-task mean >= direct at 64k and 128k", lambda: all(m("F", L) >= m("direct", L) for L in LEN[4:])),
        ("PR1 arm R >= 97 on each NIAH task at every length", lambda: all(task("R", t, L) >= 97 for t in R.NIAH for L in LEN)),
        ("PR2 arm R vt <= 40 at every length", lambda: all(task("R", "vt", L) <= 40 for L in LEN)),
        ("PR3 arm R 9-task mean in [88, 93] at every length", lambda: all(88 <= m("R", L) <= 93 for L in LEN)),
        ("PR4 R1 PASS", lambda: bool(r1ok)),
    ]
    print("\nPredictions:")
    for name, fn in preds:
        try:
            ok = bool(fn())
        except (TypeError, KeyError):
            ok = None
        out["predictions"][name] = ok
        print(f"  {name}: {'held' if ok else ('n/a' if ok is None else 'did not hold')}")
    # failures of PF5 / PR1, listed
    for a, ts, thr in (("F", T7, 97), ("R", R.NIAH, 97)):
        low = [(t, L // 1024, round(task(a, t, L), 1)) for t in ts for L in LEN if task(a, t, L) is not None and task(a, t, L) < thr]
        if low:
            print(f"  arm {a} cells below {thr}: {low}")

    if ledger and os.path.exists(ledger):
        led = json.load(open(ledger))
        h = sum(x["gpu_hours"] for x in led)
        print(f"\nCompute: {h:.2f} GPU-hours, about ${h * USD_PER_GPU_HOUR:.2f} at ${USD_PER_GPU_HOUR}/GPU-hour")
        by = collections.defaultdict(float)
        for x in led:
            by[(x["what"], x["gpu"])] += x["gpu_hours"]
        for (w, g), v in sorted(by.items()):
            print(f"  {w} on {g}: {v:.2f} GPU-hours")
        out["gpu_hours"] = round(h, 3)
    scores_path = scores_path or os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "b24_scores.json")
    json.dump(out, open(scores_path, "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:])
