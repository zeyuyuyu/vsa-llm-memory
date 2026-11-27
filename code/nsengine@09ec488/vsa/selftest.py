"""Self-test: the gold choice reproduces the target exactly; a perturbed choice does not."""
import json, sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import resonator as R

rng = np.random.default_rng(0)
# subset sum
S = [18, 4, 5, 10, 2, 23]; T = 21
p = R.subset_sum_problem(S, T, 2000, rng)
ch = [1, 0, 0, 0, 0, 1]  # 18 + ... no: find a true one
import itertools
for c in itertools.product([0, 1], repeat=6):
    if sum(s for s, b in zip(S, c) if b) == T:
        ch = list(c); break
prod = np.ones(2000, complex)
for f, C in enumerate(p.C):
    prod *= C[:, ch[f]]
print("subset-sum gold reproduces target:", np.allclose(prod, p.t), "verify", p.verify(ch))

items = [json.loads(l) for l in open("data/nsr8_items.jsonl") if '"family": "vans"' in l]
ok = 0
for it in items:
    d = it["data"]; k = it["optimum"]
    prob = R.vans_problem(d, k, 2000, rng, symmetry=False)
    groups, apart, gid = R.vans_groups(d)
    gold = {key.split(":")[1]: v for key, v in it["gold"].items()}
    choice = [gold[g["members"][0]] - 1 for g in groups]
    # slacks
    Wu = d["W"] // 10
    for v in range(k):
        lw = sum(g["w"] for gi, g in enumerate(groups) if choice[gi] == v) // 10
        ls = sum(g["s"] for gi, g in enumerate(groups) if choice[gi] == v)
        choice += [Wu - lw, d["S"] - ls]
    for p_, (a, b) in enumerate(apart):
        for v in range(k):
            cnt = (choice[a] == v) + (choice[b] == v)
            choice.append(1 - cnt)
    prod = np.ones(2000, complex)
    for f, C in enumerate(prob.C):
        prod *= C[:, choice[f]]
    good = np.allclose(prod, prob.t) and prob.verify(choice[:prob.n_groups])
    ok += good
print("vans gold reproduces target on %d of %d NSR8 items" % (ok, len(items)))
it = items[-1]; prob = R.vans_problem(it["data"], it["optimum"], 2000, rng)
print("largest NSR8 vans: factors", prob.n_factors, "groups", prob.n_groups, "log10 search space %.1f" % prob.space_log10)
