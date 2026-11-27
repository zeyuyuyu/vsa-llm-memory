"""B70 checks (prereg/B70_useful_budgets_prereg.md, 'Checks'): a second angle on the verdict.

python3 b70_check.py scorer      # plain-loop cell accuracies and point statistics vs results/b70_verdict.json
python3 b70_check.py bits        # B70 job rows and the existing arms' logged sizes at the R2 cells within budget
python3 b70_check.py rederive    # independent float64 NLMSc (genuine one-fact-at-a-time loop) and SDMt on a sample
python3 b70_check.py sample      # sample-reading material (Qwen3-8B, e5): R2 cell with the largest |STORE - LAMc|
All append a summary line to results/b70_check.txt.
"""
import gzip
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
RES = f'{ROOT}/results'
J70 = f'{ROOT}/b70_out/jobs'
PANEL = ['qwen8', 'b46', 'llama', 'mistral', 'e5', 'bge', 'minilm']
NF = {'qwen8': 4096, 'b46': 4096, 'llama': 4096, 'mistral': 4096, 'e5': 768, 'bge': 384, 'minilm': 384}
HQ = {1: 1.0, 2: 1.9037, 3: 2.7606, 4: 3.6024}


def log(line):
    print(line, flush=True)
    with open(f'{RES}/b70_check.txt', 'a') as fo:
        fo.write(line + '\n')


def acc_loop(recs, k):
    hit = tot = 0
    for r in recs:
        for p, g in zip(r['pred'][k], r['qgold']):
            hit += int(p == g)
            tot += 1
    return hit / tot


def new_pred_loop(enc, M, V, b, arm, rep, qgold):
    """The new arms' test predictions read straight from B70's job files (own lookup of the committed choice)."""
    ch = json.load(open(f'{RES}/b70_choice_{enc}.json'))['choice']
    cfg = ch[f'{M}|{V}|{b}|{arm}']['cfg']
    fn = f'{J70}/{arm}_{enc}_test_M{M}_V{V}_r{rep}' + (f'_{cfg[2]}' if arm == 'NLMSC' else '') + '.json.gz'
    j = json.load(gzip.open(fn, 'rt'))
    assert j['qgold'] == qgold
    for x in j['rows']:
        if x['cfg'] == cfg and x['bpf'] == b:
            return x['pred']
    raise KeyError((fn, cfg, b))


def scorer():
    import b70_verdict as VV
    J = json.load(open(f'{RES}/b70_verdict.json'))
    worst, n = 0.0, 0
    allacc = {}
    for enc in PANEL:
        data, cells = VV.load(enc)
        P = J['parts'][enc]
        arms = P['arms']
        for (M, V, b) in cells:
            recs = data[M, V]
            for arm in arms:
                k = VV.key(arm, b)
                a = acc_loop(recs, k)
                if arm in ('NLMSC', 'SDMT') and b in VV.NEW:  # second path from the raw job files
                    hit = tot = 0
                    for r in recs:
                        pr = new_pred_loop(enc, M, V, b, arm, r['rep'], r['qgold'])
                        for p, g in zip(pr, r['qgold']):
                            hit += int(p == g)
                            tot += 1
                    worst = max(worst, abs(hit / tot - a))
                    n += 1
                allacc[enc, M, V, b, arm] = a
                worst = max(worst, abs(P['cells'][f'{enc}|{M}|{V}|{b}'][arm] - a))
                n += 1
        for x, y, _ in VV.PRIM + VV.SEC:
            if x not in arms or y not in arms:
                continue
            s = 0.0
            for (M, V, b) in cells:
                s += allacc[enc, M, V, b, x] - allacc[enc, M, V, b, y]
            s /= len(cells)
            worst = max(worst, abs(s - P['contrasts'][f'{x}-{y}']['diff']))
            n += 1
            s = 0.0
            for (M, V, b) in cells:
                s += (allacc[enc, M, V, b, x] - allacc[enc, M, V, b, y]) / (1 - 1 / V)
            if f'{x}-{y}' in P['cn_contrasts']:
                worst = max(worst, abs(s / len(cells) - P['cn_contrasts'][f'{x}-{y}']['diff']))
                n += 1
        for arm in arms:
            raw = resc = fr = frr = 0.0
            for (M, V, b) in cells:
                a, e, c = allacc[enc, M, V, b, arm], allacc[enc, M, V, b, 'EXACT'], 1.0 / V
                raw += a
                resc += (a - c) / (1 - c)
                fr += a / e
                frr += (a - c) / (e - c)
            for nm, v in (('raw', raw), ('rescaled', resc), ('frac_exact', fr), ('frac_exact_rescaled', frr)):
                worst = max(worst, abs(v / len(cells) - P['means'][arm][nm]))
                n += 1
    for gn, grp in (('LLM', PANEL[:4]), ('sentence', PANEL[4:])):
        G = J['groups'][gn]
        cells = [(e, M, V, b) for (e, M, V, b, a) in allacc if e in grp and a == 'EXACT']
        for x, y, _ in VV.PRIM + VV.SEC:
            if f'{x}-{y}' not in G['contrasts']:
                continue
            s = sum(allacc[c + (x,)] - allacc[c + (y,)] for c in cells) / len(cells)
            worst = max(worst, abs(s - G['contrasts'][f'{x}-{y}']['diff']))
            n += 1
    json.dump({'max_abs_diff': worst, 'n_compared': n}, open(f'{RES}/b70_check_scorer.json', 'w'))
    log(f'scorer: {n} cell accuracies and point statistics recomputed by plain loops (new arms also from the raw job '
        f'files); max |difference| {worst:.3e} ({"PASS" if worst < 1e-9 else "FAIL"} at 1e-9)')


def own_bits(p, c, q, scale):
    if p == 0:
        return 0.0
    groups = c * ((p + 8191) // 8192) if scale == 'col' else (c * p + 8191) // 8192
    return c * p * HQ[q] + 32 * groups


def bits():
    import b32_common as C
    import b70_verdict as VV
    over = notmax = n = 0
    bad = []
    for fn in sorted(os.listdir(J70)):
        r = json.load(gzip.open(os.path.join(J70, fn), 'rt'))
        s = r['spec']
        for x in r['rows']:
            B = x['bpf'] * s['M']
            n += 1
            if x['arm'] == 'NLMSC':
                scale, t, prep, feat, _, _, q = x['cfg']
                c = s['V'] - 1 if t == 'V-1' else s['V']
                p = x['p']
                cap = 65536 if feat == 'rp' else NF[s['enc']]
                b1 = own_bits(p, c, q, scale)
                if b1 > B or abs(b1 - x['bits']) > 1e-6:
                    over += 1
                    bad.append([fn, x['cfg'], x['bpf'], p, b1, x['bits']])
                if p < cap and own_bits(p + 1, c, q, scale) <= B:
                    notmax += 1
                    bad.append([fn, x['cfg'], x['bpf'], p, 'not maximal'])
            elif x['bits'] > B:
                over += 1
                bad.append([fn, x['cfg'], x['bpf'], x['bits']])
    # existing arms at the R2 cells: logged sizes within budget
    nx = ox = 0
    for enc in PANEL:
        data, cells = VV.load(enc)
        for (M, V, b) in cells:
            for r in data[M, V]:
                B = b * M
                d, q, ns = r['vsa_d'][f'VSA|{b}|ideal']
                nx += 1
                ox += C.vsa_bits(d, q, 'ideal') > B if hasattr(C, 'vsa_bits') else d > C.vsa_dim(B, q, 'ideal')
                cfg, p, bt = r['b60'][f'LAMC|{b}|ideal']
                nx += 1
                ox += bt > B
                for a in ('NLMSC',) if enc in VV.CUT_SDMT else ('NLMSC', 'SDMT'):
                    bt = r['b70'][f'{a}|{b}|ideal'][2]
                    nx += 1
                    ox += bt > B
    json.dump({'n': n, 'over_or_mismatch': over, 'not_maximal': notmax, 'examples': bad[:20], 'existing_n': nx,
               'existing_over': int(ox)}, open(f'{RES}/b70_check_bits.json', 'w'))
    ok = over == 0 and notmax == 0 and ox == 0
    log(f'bits: {n} B70 job rows (dev + test): over budget or logged bits mismatching {over}, NLMSc not maximal {notmax}; '
        f'{nx} logged sizes of the store / LAMc / NLMSc / SDMt at the R2 test records: over budget {int(ox)} '
        f'({"PASS" if ok else "FAIL"})')


def rederive():
    """NLMSc: own float64 preps / eigenbasis / rp blocks from the registered seeds, genuine one-fact-at-a-time loop,
    own quantizer and read (b62_check's functions); SDMt: b64_check.rederive_one. Rep 0 of every R2 setting with a new
    budget, at its largest R2 budget."""
    import b32_common as C
    import b62_check as K62
    import b64_check as K64
    reg = json.load(open(f'{RES}/b70_region.json'))
    sample = sorted({(e, M, V, max(b for e2, M2, V2, b in reg['cells'] if (e2, M2, V2) == (e, M, V)))
                     for e, M, V, b in reg['cells'] if b in (32, 64)}, key=lambda x: PANEL.index(x[0]))
    out = {}
    cwd = os.getcwd()
    for enc in PANEL:
        sm = [x for x in sample if x[0] == enc]
        if not sm:
            continue
        ch = json.load(open(f'{RES}/b70_choice_{enc}.json'))['choice']
        os.chdir(f'{ROOT}/b49_out')
        pool = C.Pool(K62.TAG[enc])
        os.chdir(cwd)
        Tr = pool.train_matrix().astype(np.float64)
        mu = Tr.mean(0)
        w, U = np.linalg.eigh(np.cov((Tr - mu).T))
        w = np.maximum(w, 0)
        U0 = U[:, np.argsort(-w)]
        Wz = (U / np.sqrt(w + 0.1 * w.mean())) @ U.T
        n = Tr.shape[1]

        def prep(X, kind):
            Y = np.asarray(X, np.float64) - mu
            if kind == 'whiten':
                Y = Y @ Wz
            return Y / np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)

        def fmap(pr, feat, p):
            if feat == 'pca':
                return U0[:, :p]
            return np.hstack([np.random.default_rng(K62.seed_of('LAMG', enc, pr, j0)).standard_normal(
                (n, 8192), dtype=np.float32)[:, :min(8192, p - j0)].astype(np.float64) / math.sqrt(n)
                for j0 in range(0, p, 8192)])
        D = K64.Data(enc)
        preps = {p: D.prep(p) for p in ('center', 'whiten')}
        locs = {l: D.locs(l) for l in ('LA', 'L', 'A')}
        for (_, M, V, b) in sm:
            st = C.make_store(pool, 'test', M, V, 0)
            order = np.random.default_rng(K62.seed_of('subset', 'test', M, V, 0)).permutation(M)
            # NLMSc
            cfg = ch[f'{M}|{V}|{b}|NLMSC']['cfg']
            scale, t, pr, feat, x, y, q = cfg
            j = json.load(gzip.open(f'{J70}/NLMSC_{enc}_test_M{M}_V{V}_r0_{pr}.json.gz', 'rt'))
            row = [r for r in j['rows'] if r['cfg'] == cfg and r['bpf'] == b][0]
            p = row['p']
            G = fmap(pr, feat, p)
            Fk, Fq = prep(st['keys'], pr) @ G, prep(st['queries'], pr) @ G
            Tm = np.eye(V)[st['vals'][order]] - 1.0 / V
            if t == 'V-1':
                Tm = Tm[:, 1:]
            Wl, Wa = K62.own_nlms(Fk[order], Tm, x)
            Wq = K62.own_quantW(Wl if y == 'last' else Wa, q, scale, np.random.default_rng(1))
            Sc = Fq @ Wq
            if t == 'V-1':
                Sc = np.concatenate([-Sc.sum(1, keepdims=True), Sc], 1)
            pred = Sc.argmax(1)
            lg = np.array(row['pred'])
            out[f'{enc}|{M}|{V}|{b}|NLMSC'] = {'cfg': cfg, 'p': p, 'agree': float((pred == lg).mean()),
                                               'acc_logged': float((lg == st['qgold']).mean()),
                                               'acc_rederived': float((pred == st['qgold']).mean())}
            print(f'{enc}|{M}|{V}|{b}|NLMSC', out[f'{enc}|{M}|{V}|{b}|NLMSC'], flush=True)
            del G, Fk, Fq
            # SDMt
            if f'{M}|{V}|{b}|SDMT' not in ch:
                continue
            cfg = ch[f'{M}|{V}|{b}|SDMT']['cfg']
            j = json.load(gzip.open(f'{J70}/SDMT_{enc}_test_M{M}_V{V}_r0.json.gz', 'rt'))
            row = [r for r in j['rows'] if r['cfg'] == cfg and r['bpf'] == b][0]
            pr2, N, bt = K64.rederive_one(D, preps, locs, enc, M, V, 0, tuple(cfg), b, 'ideal')
            out[f'{enc}|{M}|{V}|{b}|SDMT'] = {'cfg': cfg, 'agree': float(np.mean(np.array(pr2) == np.array(row['pred']))),
                                              'N_log': row['N'], 'N_own': N, 'bits_log': row['bits'], 'bits_own': bt,
                                              'within_budget_own': bt <= b * M}
            print(f'{enc}|{M}|{V}|{b}|SDMT', out[f'{enc}|{M}|{V}|{b}|SDMT'], flush=True)
    json.dump(out, open(f'{RES}/b70_check_rederive.json', 'w'), indent=0)
    an = [v['agree'] for k, v in out.items() if k.endswith('NLMSC')]
    asd = [v['agree'] for k, v in out.items() if k.endswith('SDMT')]
    nN = sum(v['N_log'] == v['N_own'] for k, v in out.items() if k.endswith('SDMT'))
    log(f'rederive: float64 independent code on rep 0 of {len(an)} R2 settings at their largest R2 budget: NLMSc '
        f'(genuine one-fact-at-a-time loop) agreement min {min(an):.4f} mean {np.mean(an):.4f}; SDMt (b64_check) '
        f'agreement min {min(asd):.4f} mean {np.mean(asd):.4f}, N identical {nN} of {len(asd)} '
        f'({"PASS" if min(an + asd) >= 0.99 else "FAIL"} at 0.99)')


def sample():
    import b70_verdict as VV
    J = json.load(open(f'{RES}/b70_verdict.json'))
    name = {e['uri']: e['name'] for e in json.load(open(f'{ROOT}/b49_out/b32_data/entities.json'))['entities']}
    L = []
    for enc in ('qwen8', 'e5'):
        data, cells = VV.load(enc)
        P = J['parts'][enc]['cells']
        d, M, V, b = max((abs(P[f'{enc}|{M}|{V}|{b}']['STORE'] - P[f'{enc}|{M}|{V}|{b}']['LAMC']), M, V, b)
                         for (M, V, b) in cells)
        recs = data[M, V]
        groups = {'only store right': [], 'only LAMc right': [], 'only BESTc right (store and LAMc wrong)': []}
        for r in recs:
            for i, g in enumerate(r['qgold']):
                s_ok = r['pred'][f'STORE|{b}|ideal'][i] == g
                l_ok = r['pred'][f'LAMC|{b}|ideal'][i] == g
                c_ok = r['pred'][f'BESTC|{b}|ideal'][i] == g
                item = (r['rep'], r['qalias'][i], name.get(r['uris'][r['qent'][i]], r['uris'][r['qent'][i]]), g,
                        r['pred'][f'STORE|{b}|ideal'][i], r['pred'][f'LAMC|{b}|ideal'][i],
                        r['pred'][f'BESTC|{b}|ideal'][i], r['pred']['EXACT'][i])
                if s_ok and not l_ok:
                    groups['only store right'].append(item)
                elif l_ok and not s_ok:
                    groups['only LAMc right'].append(item)
                if c_ok and not s_ok and not l_ok:
                    groups['only BESTc right (store and LAMc wrong)'].append(item)
        fam = J['parts'][enc]['devbest'][f'{M}|{V}|{b}']
        L.append(f'## {enc} M={M} V={V} bpf={b}: STORE {100 * P[f"{enc}|{M}|{V}|{b}"]["STORE"]:.1f}, LAMc '
                 f'{100 * P[f"{enc}|{M}|{V}|{b}"]["LAMC"]:.1f}, BESTc {100 * P[f"{enc}|{M}|{V}|{b}"]["BESTC"]:.1f} '
                 f'(dev-best family {fam}), EXACT {100 * P[f"{enc}|{M}|{V}|{b}"]["EXACT"]:.1f}; counts ' +
                 ', '.join(f'{k} {len(v)}' for k, v in groups.items()))
        rng = np.random.default_rng(70)
        for nm, lst in groups.items():
            L.append(f'  {nm}: (rep, alias, stored label, gold, store, LAMc, BESTc, EXACT answers)')
            for j in rng.choice(len(lst), size=min(10, len(lst)), replace=False):
                L.append(f'    {lst[j]}')
    open(f'{RES}/b70_sample_material.txt', 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    {'scorer': scorer, 'bits': bits, 'rederive': rederive, 'sample': sample}[sys.argv[1]]()
