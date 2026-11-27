"""Re-derive per-query test correctness for my development-chosen configs and cluster-bootstrap over entities."""
import sys, json
import numpy as np
ENC_ARG = sys.argv[1]
sys.argv = ['x', ENC_ARG, '/dev/null', 'centred']
src = open('cr_equalbit_fast2.py').read()
exec(src[:src.index('for white in')])          # data, prep, stores, quantizer, codes (my own code)
mine = json.load(open('my_cells.json'))[ENC_ARG]['cells']
ents_test = [x for x in ents if x['split'] == 'test']
per = {}   # cond -> dict(mem -> per-query correct array), entity id per query
cache = {}
for ck, c in mine.items():
    _, _, M, V, b = ck.split('|'); M, V, b = int(M), int(V), int(b)
    stores = STORES[('test', M, V)]
    pn, mp, k, sh = c['store_cfg'].split('|'); k = int(k); sh = float(sh)
    white = pn == 'white'
    Pt = prep(SPL['test']['rows'], white)
    d = store_dim(b * M, k)
    key = (pn, mp)
    if key not in cache:
        G1 = gaussian([ENC == 'qwen8', 1], DMAX, n)
        Z = Pt @ G1.T
        if mp == 'lin':
            F = Z
        else:
            F = np.where(Z >= 0, 1, -1).astype(np.int8)
            if mp == 'prod3':
                for t in [2, 3]:
                    G = gaussian([ENC == 'qwen8', t], DMAX, n)
                    F = F * np.where(Pt @ G.T >= 0, 1, -1).astype(np.int8)
        cache = {key: F}
    F = cache[key]
    CV = value_codes(V, d)
    sc_store, ent_ids = [], []
    for si, st in enumerate(stores):
        Fl = F[st['lab'], :d].astype(np.float32); Fq = F[st['qrows'], :d].astype(np.float32)
        m = int(round(sh * M))
        bq = quant_vec((Fl[:m] * CV[st['vals'][:m]]).sum(0), k)
        pred = ((Fq * bq[None, :]) @ CV.T).argmax(1)
        sc_store.append(pred == st['qvals'])
        # entity of each query: label local index of owner
        owner = []
        for j, ej in enumerate(st['lab']):
            owner += [int(ej)] * len(SPL['test']['al'].get(int(ej), []))
        ent_ids.append(np.array(owner))
    # LS
    pn2, feat, k2, C, scale, a = c['lamc_cfg'].split('|'); k2 = int(k2); C = int(C); a = float(a)
    assert feat == 'principal', 'only principal features re-derived here'
    P2 = prep(SPL['test']['rows'], pn2 == 'white')
    p = ls_dim(b * M, k2, C, scale, n)
    sc_ls = []
    for st in stores:
        Fl = P2[st['lab'], :p].astype(np.float64); Fq = P2[st['qrows'], :p].astype(np.float64)
        T = np.eye(V)[st['vals']] - 1.0 / V
        primal = p < M
        Gm = (Fl.T @ Fl) if primal else (Fl @ Fl.T)
        md = np.trace(Gm) / M
        RHS = Fl.T @ T if primal else T
        X = np.linalg.solve(Gm + a * md * np.eye(Gm.shape[0]), RHS)
        W = (X if primal else Fl.T @ X)[:, :C]
        if scale == 'shared':
            Wq = quant_vec(W.ravel().astype(np.float32), k2).reshape(W.shape)
        else:
            Wq = np.stack([quant_vec(W[:, cc].astype(np.float32), k2) for cc in range(C)], 1)
        S = Fq @ Wq.astype(np.float64)
        if C == V - 1:
            S = np.hstack([S, -S.sum(1, keepdims=True)])
        sc_ls.append(S.argmax(1) == st['qvals'])
    per[ck] = dict(store=np.concatenate(sc_store).astype(float), ls=np.concatenate(sc_ls).astype(float),
                   ent=np.concatenate(ent_ids))
    print(ck, 'rederived store %.2f (my_cells %.2f)  ls %.2f (my_cells %.2f)' % (
        100 * per[ck]['store'].mean(), c['STORE'], 100 * per[ck]['ls'].mean(), c['LAMC']), flush=True)

NE = len(ents_test)
rng = np.random.default_rng(2026)
R = 2000
agg = {}
for ck, x in per.items():
    agg[ck] = (np.bincount(x['ent'], x['store'] - x['ls'], NE), np.bincount(x['ent'], None, NE))
out = {}
for name, f in [('V2', lambda c: c.split('|')[3] == '2'), ('V16', lambda c: c.split('|')[3] == '16'), ('R1', lambda c: True)]:
    ks = [c for c in agg if f(c)]
    point = np.mean([agg[c][0].sum() / agg[c][1].sum() for c in ks]) * 100
    bs = np.empty(R)
    for r in range(R):
        w = rng.multinomial(NE, np.full(NE, 1.0 / NE))
        bs[r] = np.mean([(w * agg[c][0]).sum() / (w * agg[c][1]).sum() for c in ks]) * 100
    lo, hi = np.percentile(bs, [2.5, 97.5])
    lab = 'store leads' if lo > 0 else ('least squares leads' if hi < 0 else 'level')
    out[name] = dict(point=point, lo=lo, hi=hi, label=lab)
    print(ENC_ARG, name, '%.2f [%.2f, %.2f] %s' % (point, lo, hi, lab))
json.dump(out, open(f'boot_{ENC_ARG}.json', 'w'), indent=1)
