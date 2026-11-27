"""B26 checks from other angles (independent of b26_fuzzy.py's arm code and b26_verdict.py's scorer).

python3 b26_check.py b26_out/test_query.jsonl [N_SAMPLE]

1. Re-scores every row with a scorer written here (each gold string must occur in the prediction, case-insensitive),
   and recomputes the per-condition means and the pooled V - L / V - W differences without numpy's bootstrap code.
2. On a random sample of rows, rebuilds the key vocabulary from the banked B21 pairs and recomputes, in pure Python,
   the key that L (Levenshtein / longer length) and N (cosine of character trigram counts) choose, and compares it with
   the logged choice. Ties are counted separately (the arms break ties by first-read order).
3. E's C0 accuracy per length recomputed from the rows, for comparison with B21's verdict.
"""
import collections
import json
import math
import random
import re
import sys

import b26_fuzzy as B   # only for load_split (data) -- not for any arm


def norm(s):
    return re.sub(r"[^0-9a-z]+", " ", s.lower()).strip()


def lev(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def tri(s):
    t = " " + norm(s) + " "
    return collections.Counter(t[i:i + 3] for i in range(len(t) - 2))


def cos(a, b):
    dot = sum(v * b.get(g, 0) for g, v in a.items())
    return dot / math.sqrt(sum(v * v for v in a.values()) * sum(v * v for v in b.values()))


def score(pred, gold):
    p = pred.lower()
    return sum(g.lower() in p for g in gold) / len(gold)


def main(path, nsample=400):
    rows = [json.loads(l) for l in open(path)]
    # 1. independent scorer
    bad = 0
    acc = collections.defaultdict(list)
    per = collections.defaultdict(dict)
    for r in rows:
        for a, x in r["arms"].items():
            s = score(x["pred"], r["outputs"])
            bad += abs(s - x["score"]) > 1e-9
            acc[(r["cond"], a)].append(s)
            per[(r["cell"][0], r["cell"][1], r["pos"])][(r["cond"], a)] = s
    print("rows %d; logged scores that differ from the independent scorer: %d" % (len(rows), bad))
    conds = ["C0", "T1", "T2", "T3", "P", "A"]
    for a in ["E", "En", "L", "W", "N", "V"]:
        print("  %-3s" % a + "".join(" %s %6.2f" % (c, 100 * sum(acc[(c, a)]) / len(acc[(c, a)])) for c in conds if (c, a) in acc))
    for o in ["L", "W", "N"]:
        d = [sum(v[(c, "V")] - v[(c, o)] for c in conds[1:]) / 5 for v in per.values()]
        print("  pooled V - %s over T1-A: %+.3f points (n = %d samples)" % (o, 100 * sum(d) / len(d), len(d)))
    # 2. independent key choice for L and N on a sample
    split = "dev" if "dev_" in path else "test"
    data = {(c[0], c[1], p): row for c, p, row, s in B.load_split(split)}
    rng = random.Random(7)
    pick = rng.sample([r for r in rows if r["cond"] != "C0"], min(nsample, len(rows)))
    agree = collections.Counter()
    for r in pick:
        keys = list(dict.fromkeys(k for k, _, _ in data[(r["cell"][0], r["cell"][1], r["pos"])]["pairs"]))
        nk = [norm(k) for k in keys]
        tk = [tri(k) for k in keys]
        for qi, q in enumerate(r["variants"]):
            d = [lev(norm(q), x) / max(len(norm(q)), len(x), 1) for x in nk]
            best = min(d)
            cands = {keys[i] for i, v in enumerate(d) if abs(v - best) < 1e-12}
            ch = r["arms"]["L"]["chosen"][qi]
            agree["L", "same" if ch == keys[d.index(best)] else ("tie" if ch in cands else "DIFF")] += 1
            qt = tri(q)
            c = [cos(qt, t) for t in tk]
            best = max(c)
            cands = {keys[i] for i, v in enumerate(c) if abs(v - best) < 1e-9}
            ch = r["arms"]["N"]["chosen"][qi]
            agree["N", "same" if ch == keys[c.index(best)] else ("tie" if ch in cands else "DIFF")] += 1
            chv = r["arms"]["V"]["chosen"][qi]
            agree["V vs exact trigram cosine", "same" if chv == keys[c.index(best)] else ("tie" if chv in cands else "differs")] += 1
    print("independent key choice on %d sampled rows:" % len(pick), dict(agree))
    # 3. E in C0 per length
    print("E, C0, per length:", {L: round(100 * sum(v[("C0", "E")] for k, v in per.items() if k[0] == L) /
                                         sum(1 for k in per if k[0] == L), 2) for L in sorted({k[0] for k in per})})


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 400)
