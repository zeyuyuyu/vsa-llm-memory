"""Post-hoc paired analysis of NSR5 answer files (not a registered measure): per-question item errors, where the item
loss comes from, both/only/neither counts and a paired bootstrap interval. python3 nsr/nsr5_compare.py K armA armB"""
import collections, json, os, random, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from nsr1_score import gold  # noqa: E402
import nsr5_sites as S  # noqa: E402


def load(K, arm):
    return {r["id"]: r for r in map(json.loads, open(os.path.join(HERE, "out", f"nsr5_K{K}_{arm}.jsonl")))}


def main(K, a, b, ids=None):
    K = int(K)
    sites = {s["id"]: s for s, _ in S.sites(K)}
    A, B = load(K, a), load(K, b)
    ks = sorted(A) if ids is None else sorted(ids)
    G = {k: gold(sites[k.split("|")[0]], *k.split("|")[1:]) for k in ks}
    ok = lambda P, k: P[k]["pred"] is not None and sorted(P[k]["pred"]) == G[k]  # noqa: E731
    oa = {k: ok(A, k) for k in ks}; ob = {k: ok(B, k) for k in ks}
    c = collections.Counter((oa[k], ob[k]) for k in ks)
    def errs(P):
        miss = extra = none = 0; big = []
        for k in ks:
            p, g = P[k]["pred"], set(G[k])
            if p is None:
                none += 1; miss += len(g); big.append((len(g), 0, k)); continue
            m, e = len(g - set(p)), len(set(p) - g); miss += m; extra += e
            if m + e: big.append((m, e, k))
        big.sort(reverse=True)
        return {"items_gold": sum(len(G[k]) for k in ks), "missed": miss, "extra": extra, "no_answer": none,
                "worst": [(m, e, k.split("|", 1)[1], len(G[k])) for m, e, k in big[:6]]}
    d = [oa[k] - ob[k] for k in ks]; random.seed(0)
    bs = sorted(sum(random.choice(d) for _ in d) / len(d) for _ in range(10000))
    print(json.dumps({"K": K, "n": len(ks), "acc": {a: round(sum(oa.values()) / len(ks), 4), b: round(sum(ob.values()) / len(ks), 4)},
                      "diff_a_minus_b": round(sum(d) / len(d), 4), "ci95": [round(bs[250], 3), round(bs[9750], 3)],
                      "both": c[(True, True)], "only_" + a: c[(True, False)], "only_" + b: c[(False, True)],
                      "neither": c[(False, False)], "errors": {a: errs(A), b: errs(B)}}, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:4])
