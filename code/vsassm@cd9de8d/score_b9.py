"""Independent scorer for B9 (prereg/B9_prereg.md): gold names recomputed from the stored events and name lists.
Usage: python3 score_b9.py b9_out
"""
import glob
import json
import statistics
import sys
from collections import Counter, defaultdict

NS = (10, 20, 40, 80, 160)
SPLITS = ("B6-new", "C-new", "F-new", "C-old", "B6-new2", "C-new2", "F-new2")


def gold(task, r):
    if task == "T2":
        loc = {}
        for op, a, b in r["acts"]:
            if op == "put":
                loc[a] = b
            else:
                loc = {o: (b if c == a else a if c == b else c) for o, c in loc.items()}
        return r["places"][loc[r["query"]]]
    par = {a: b for a, b in r["acts"]}
    i = r["query"]
    while i in par:
        i = par[i]
    return r["ents"][i]


def global_root(r):
    par = {a: b for a, b in r["acts"]}

    def top(i):
        while i in par:
            i = par[i]
        return i
    return r["ents"][Counter(top(i) for i in range(len(r["ents"]))).most_common(1)[0][0]]


def main():
    out = sys.argv[1]
    suf = sys.argv[2] if len(sys.argv) > 2 else ""  # B9b: "2" = claims on the fresh pool
    claim = (f"B6-new{suf}", f"C-new{suf}", f"F-new{suf}")
    acc, read = defaultdict(list), defaultdict(lambda: [0, 0])
    for path in glob.glob(f"{out}/T?_*.jsonl"):
        task = path.split("/")[-1][:2]
        for r in map(json.loads, open(path)):
            g = gold(task, r)
            acc[("MEMS-OPEN", task, r["split"], r["n"])].append(r["mem"] == g)
            acc[("DICT-OPEN", task, r["split"], r["n"])].append(r["dict"] == g)
            if task == "T3":
                acc[("GLOBAL-ROOT", task, r["split"], r["n"])].append(global_root(r) == g)
            if "read_total" in r:
                read[(r["split"], r["n"])][0] += r["read_ok"]
                read[(r["split"], r["n"])][1] += r["read_total"]

    def a(arm, t, s, n):
        v = acc.get((arm, t, s, n))
        return sum(v) / len(v) if v else None

    for t in ("T2", "T3"):
        for s in SPLITS:
            if not any(("MEMS-OPEN", t, s, n) in acc for n in NS):
                continue
            print(f"\n=== {t}{'′' if t == 'T3' else ''} {s}: accuracy at N = {NS}")
            for arm in ("MEMS-OPEN", "DICT-OPEN", "GLOBAL-ROOT"):
                vals = [a(arm, t, s, n) for n in NS]
                if any(v is not None for v in vals):
                    print(f"  {arm:12s} " + " ".join("  -  " if v is None else f"{v:.3f}" for v in vals))
            wr = f"{out}/{t}_{s}_writer.json"
            try:
                print("  writer:", json.load(open(wr)))
            except FileNotFoundError:
                pass
    print("\n=== claims")
    cells = {(t, s): [a("MEMS-OPEN", t, s, n) for n in NS] for t in ("T2", "T3") for s in claim}
    if any(None in v for v in cells.values()):
        print("P1: INCOMPLETE")
    else:
        fails = [f"{t}/{s} min {min(v):.3f}" for (t, s), v in cells.items() if min(v) < 0.95]
        print("P1 MEMS-OPEN >= 0.95 at every N, T2 and T3', test names, B6/C/F wording:",
              "PASS" if not fails else "FAIL " + "; ".join(fails))
        gaps = [a("MEMS-OPEN", "T3", s, n) - a("GLOBAL-ROOT", "T3", s, n) for s in claim for n in NS]
        print(f"P2 beats global-root by >= 0.30 at every N on T3': min gap {min(gaps):+.3f} ->", "PASS" if min(gaps) >= 0.30 else "FAIL")
        for t in ("T2", "T3"):
            for s in claim:
                d = statistics.mean(cells[(t, s)]) - statistics.mean([a("DICT-OPEN", t, s, n) for n in NS])
                print(f"P3 {t}/{s} |MEMS - DICT| <= 0.03: {d:+.3f} ->", "PASS" if abs(d) <= 0.03 else "FAIL")
    for s in SPLITS:
        v = [read[(s, n)][0] / read[(s, n)][1] for n in NS if read[(s, n)][1]]
        if v:
            print(f"reported: T2 {s} every placed object read correctly:", [round(x, 4) for x in v])


if __name__ == "__main__":
    main()
