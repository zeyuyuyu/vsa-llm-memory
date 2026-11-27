"""B26 post-hoc sensitivity (NOT registered; reported as such): the fuzzy arms with every stored key shorter than
3 characters (after normalization) removed from the candidate vocabulary, e.g. the reader's junk key "m." from the
essay haystack. L and W are recomputed; V and N keep their logged choice (asserted never to be a removed key)."""
import collections, json, sys
import numpy as np
from rapidfuzz import fuzz, process, utils
from rapidfuzz.distance import Levenshtein
import b26_fuzzy as B
import b26_verdict as Vd

rows = [json.loads(l) for l in open(sys.argv[1])]
data = {(c[0], c[1], p): row for c, p, row, s in B.load_split("test")}
out, nredo = {}, [0]
for r in rows:
    pairs = data[(r["cell"][0], r["cell"][1], r["pos"])]["pairs"]
    table = collections.OrderedDict()
    for k, v, j in pairs:
        table.setdefault(k, []).append(v)
    keys = [k for k in table if len(B.norm(k)) >= 3]
    nk = [B.norm(k) for k in keys]
    sc = {}
    redo = [a for a in ("V", "N") if any(ch is not None and len(B.norm(ch)) < 3 for ch in r["arms"][a]["chosen"])]
    if redo:                         # rebuild V / N over the filtered pairs (junk pairs never stored)
        B.clear_cache()
        fp = [tuple(p) for p in pairs if len(B.norm(p[0])) >= 3]
        mem = B.Memories(fp, nsets=("3",), binds=("ng",))
        for a in redo:
            sc[a] = B.score(mem.answer(a, r["question"], ns="3", bind="ng")[0], r["outputs"])
        nredo[0] += 1
    for a in ("L", "W"):
        vals = []
        for q in B.qkeys(r["question"]):
            q = q.strip()
            if not keys:
                continue
            if a == "L":
                i = int(np.argmin(process.cdist([B.norm(q)], nk, scorer=Levenshtein.normalized_distance)[0]))
            else:
                i = int(np.argmax(process.cdist([q.lower()], keys, scorer=fuzz.WRatio, processor=utils.default_process)[0]))
            vals += table[keys[i]]
        sc[a] = B.score(" ".join(vals), r["outputs"])
    for a in ("V", "N"):
        sc.setdefault(a, r["arms"][a]["score"])
    out[(r["cell"][0], r["cell"][1], r["pos"], r["cond"])] = sc
samples = sorted({k[:3] for k in out})
conds = ["C0", "T1", "T2", "T3", "P", "A"]
print("rows where V or N had chosen a removed key and were recomputed: %d" % nredo[0])
print("post hoc, keys shorter than 3 characters removed for every arm (points):")
for a in ("L", "W", "N", "V"):
    print("  %s " % a + " ".join("%s %6.2f" % (c, 100 * np.mean([out[s + (c,)][a] for s in samples])) for c in conds))
for o in ("L", "W"):
    for c in ["A", "pooled"]:
        if c == "pooled":
            d = np.stack([[out[s + (cc,)]["V"] - out[s + (cc,)][o] for s in samples] for cc in conds[1:]], 1)
        else:
            d = np.array([out[s + (c,)]["V"] - out[s + (c,)][o] for s in samples])
        print("  V-%s %-6s %s" % (o, c, Vd.fmt_cmp(d)[0]))
