# Independent plain-loop re-derivation of the SDM (data addresses) arm on test stores of two conditions:
# budget check, prediction agreement, and split disjointness of hard-location addresses and stored/queried names.
import json, math, sys
import numpy as np
import cr

cr.setup('qwen8')
tr = set(cr.G['train_rows'].tolist())
for (M, V, b) in [(256, 16, 8), (256, 2, 16)]:
    res = json.load(open(f'{cr.WD}/out_qwen8/cell_M{M}_V{V}_b{b}.json'))
    cfg = res['sdm']['cfg']; prep, addr, a, ctr, sh = cfg[1], cfg[2], int(cfg[3]), cfg[4], float(cfg[5])
    B = b * M; cs = M * 1000 + V * 10 + b
    test = cr.make_stores('test', M, V, 20 if M == 256 else 5, 2000 + cs)
    agree = []; ok_budget = True; correct = 0; nq = 0
    for si, st in enumerate(test[:3]):
        assert not (set(st['keys'].tolist()) & tr) and not (set(st['q'].tolist()) & tr)
        Sims = cr.sdm_sims(st, prep, addr)
        p1 = cr.sdm_eval(st, Sims, B, V, a, ctr, sh, np.random.default_rng(0))
        # plain loop version: find the N the vectorised code used by re-running its size rule, then loop
        nk = len(st['keys']); ns = max(1, int(round(sh * nk)))
        Ns = sorted(set(int(x) for x in np.unique(np.round(np.geomspace(max(a, 8), Sims.shape[1], 64)))))
        best = None
        for N in Ns:
            C = {}
            for i in range(ns):
                act = np.argsort(-Sims[i, :N], kind='stable')[:a]
                for l in act:
                    if ctr == 'onehot':
                        C[(l, int(st['vals'][i]))] = C.get((l, int(st['vals'][i])), 0) + 1
                    else:
                        nb = max(1, math.ceil(math.log2(V)))
                        for j in range(nb):
                            C[(l, j)] = C.get((l, j), 0) + (1 if (st['vals'][i] >> j) & 1 else -1)
            ncells = N * (V if ctr == 'onehot' else max(1, math.ceil(math.log2(V))))
            vals = list(C.values()); nz = ncells - len(vals)
            hist = {}
            for v in vals: hist[v] = hist.get(v, 0) + 1
            hist[0] = hist.get(0, 0) + nz
            p = np.array(list(hist.values()), float) / ncells
            cost = ncells * float(-(p[p > 0] * np.log2(p[p > 0])).sum())
            if cost <= B: best = (N, C, cost)
        N, C, cost = best
        ok_budget &= cost <= B
        p2 = []
        for qi in range(len(st['q'])):
            act = np.argsort(-Sims[nk + qi, :N], kind='stable')[:a]
            if ctr == 'onehot':
                s = np.array([sum(C.get((l, v), 0) for l in act) for v in range(V)], float)
            else:
                nb = max(1, math.ceil(math.log2(V)))
                r = np.array([sum(C.get((l, j), 0) for l in act) for j in range(nb)], float)
                s = np.array([sum(r[j] * (1 if (v >> j) & 1 else -1) for j in range(nb)) for v in range(V)])
            p2.append(int(np.argmax(s)) if (s == s.max()).sum() == 1 else -1)
        p2 = np.array(p2)
        decided = p2 >= 0
        agree.append(float((p1[decided] == p2[decided]).mean()))
        correct += int((p2[decided] == st['qv'][decided]).sum()); nq += int(decided.sum())
        print(f'M{M} V{V} b{b} store {si}: N={N} cost={cost:.0f} <= B={B}: {cost <= B}; agreement on non-tied queries {agree[-1]:.4f} ({decided.sum()}/{len(p2)})', flush=True)
    print(f'M{M} V{V} b{b} cfg={cfg} budget ok={ok_budget} mean agreement={np.mean(agree):.4f}', flush=True)
