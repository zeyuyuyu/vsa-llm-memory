import json, sys, numpy as np
d = sys.argv[1]; tag = sys.argv[2] if len(sys.argv) > 2 else ""; split = sys.argv[3] if len(sys.argv) > 3 else "test"
V = [json.loads(l) for l in open(f"{d}/preds_{split}_vec_s0{tag}.jsonl")]
D = [json.loads(l) for l in open(f"{d}/preds_{split}_disc_s0{tag}.jsonl")]
assert len(V) == len(D) and all((a["op"], a["x"], a["y"]) == (b["op"], b["x"], b["y"]) for a, b in zip(V, D))
gold = [str(r["gold"]) for r in V]
acc = {
    "base": np.array([r["base"] == g for r, g in zip(V, gold)]),
    "vec_published": np.array([r["vec_pub"] == g for r, g in zip(V, gold)]),
    "vec_best_match": np.array([r["vec_best"] == g for r, g in zip(V, gold)]),
    "discrete": np.array([r["disc"] == g for r, g in zip(D, gold)]),
}
mod = {
    "vec_published": np.array([r["vec_pub_modsol"] == r["gold"] and r["vec_pub_gate"] for r in V]),
    "vec_best_match": np.array([r["vec_best_modsol"] == r["gold"] and r["vec_best_gate"] for r in V]),
    "discrete": np.array([r["disc_modsol"] == r["gold"] and r["disc_gate"] for r in D]),
}
rng = np.random.default_rng(0)
n = len(gold); idx = rng.integers(0, n, size=(20000, n))
def diff(a, b):
    x = acc[a].astype(float) - acc[b].astype(float)
    bs = x[idx].mean(1)
    return {"diff_points": round(100 * x.mean(), 2), "ci95_points": [round(100 * np.percentile(bs, 2.5), 2), round(100 * np.percentile(bs, 97.5), 2)]}
def wilson(p, n, z=1.96):
    den = 1 + z * z / n; c = (p + z * z / (2 * n)) / den; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(c - h, 4), round(c + h, 4)]
out = {
    "split": split, "n_items": n, "seed": 0,
    "accuracy": {k: round(float(v.mean()), 4) for k, v in acc.items()},
    "accuracy_wilson95": {k: wilson(float(v.mean()), n) for k, v in acc.items()},
    "module_answer_correct_and_gate_open": {k: round(float(v.mean()), 4) for k, v in mod.items()},
    "followed_given_module_right": {k: round(float(acc[k][mod[k]].mean()), 4) for k in mod},
    "vector_published_minus_discrete": diff("vec_published", "discrete"),
    "vector_best_match_minus_discrete": diff("vec_best_match", "discrete"),
    "per_op": {op: {k: round(float(v[[r["op"] == op for r in V]].mean()), 3) for k, v in acc.items()}
               for op in sorted(set(r["op"] for r in V))},
}
print(json.dumps(out, indent=1))
