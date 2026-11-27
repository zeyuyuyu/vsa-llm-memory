"""Paired comparison of two NSR3 answer files (post-hoc analysis, not a registered measure): set-exact for each arm,
micro-F1, both/only/neither counts and a paired bootstrap interval for the difference. Uses nsr3_score's replay."""
import collections, json, os, random, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import nsr3_score as S3  # noqa: E402


def gold_all(split, logs_dir):
    logs = {}
    for sc in ("warehouse", "biobank", "home", "ops"):
        for line in open(os.path.join(logs_dir, f"{sc}_{split}.jsonl")):
            d = json.loads(line); logs[d["id"]] = d
    S = {lid: S3.states(log) for lid, log in logs.items()}
    def gold(qid):
        lid, qt, tg, i, j = qid.split("|"); i, j = int(i), int(j)
        g = set()
        for st in S[lid][max(i - 1, 0):j + 1]:
            g |= S3.affected(logs[lid], st, qt, tg)
        return g
    return gold, logs


def main(split, a, b):
    gold, logs = gold_all(split, os.path.join(HERE, "data"))
    A = {r["id"]: r["pred"] for r in map(json.loads, open(os.path.join(HERE, "out", f"nsr3_{split}_{a}.jsonl")))}
    B = {r["id"]: r["pred"] for r in map(json.loads, open(os.path.join(HERE, "out", f"nsr3_{split}_{b}.jsonl")))}
    ks = sorted(A); assert ks == sorted(B)
    ok = lambda P, k: P[k] is not None and set(P[k]) == gold(k)  # noqa: E731
    oa = {k: ok(A, k) for k in ks}; ob = {k: ok(B, k) for k in ks}
    c = collections.Counter((oa[k], ob[k]) for k in ks)
    def f1(P):
        tp = np_ = ng = 0
        for k in ks:
            p, g = set(P[k] or []), gold(k); tp += len(p & g); np_ += len(p); ng += len(g)
        pr, rc = tp / max(np_, 1), tp / max(ng, 1); return round(2 * pr * rc / max(pr + rc, 1e-9), 4)
    d = [oa[k] - ob[k] for k in ks]; random.seed(0)
    bs = sorted(sum(random.choice(d) for _ in d) / len(d) for _ in range(10000))
    print(json.dumps({"a": a, "b": b, "acc_a": round(sum(oa.values()) / len(ks), 4), "acc_b": round(sum(ob.values()) / len(ks), 4),
                      "f1_a": f1(A), "f1_b": f1(B), "diff": round(sum(d) / len(d), 4), "ci95": [round(bs[250], 3), round(bs[9750], 3)],
                      "both": c[(True, True)], "only_a": c[(True, False)], "only_b": c[(False, True)], "neither": c[(False, False)],
                      "either": round(1 - c[(False, False)] / len(ks), 4)}))


if __name__ == "__main__":
    main(*sys.argv[1:4])
