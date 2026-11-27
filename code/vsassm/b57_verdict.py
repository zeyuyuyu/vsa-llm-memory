"""B57 verdict (prereg/B57_pca_store_maps_online_prereg.md) on b57_out/fam/{enc}/all57_* (B56's merged records + the
B57 arms). Statistics as B49 / B53 / B56 (b36_verdict.per_query / boot_region); maps: B52's two-level bootstrap.

python3 b57_verdict.py part ENC
python3 b57_verdict.py panel
python3 b57_verdict.py preds
"""
import collections
import gzip
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import b36_verdict as V36  # noqa: E402
import b32_common as C  # noqa: E402
import b49_common as B  # noqa: E402

FAM = f'{ROOT}/b57_out/fam'
RES = f'{ROOT}/results'
PANEL = ['qwen8', 'b46', 'llama', 'mistral', 'e5', 'bge', 'minilm']
LLM = PANEL[:4]
MAP_ENCS = ['qwen8', 'e5', 'minilm']
NAMES = {'qwen8': 'Qwen3-8B', 'b46': 'Granite-3.3-8B', 'llama': 'Llama-3.1-8B', 'mistral': 'Mistral-7B',
         'e5': 'e5-base-v2', 'bge': 'bge-small', 'minilm': 'MiniLM-L6'}
NDIM = {'qwen8': 4096, 'b46': 4096, 'llama': 4096, 'mistral': 4096, 'e5': 768, 'bge': 384, 'minilm': 384}
M_GRID, VN, BPF, ACCTS = [256, 1024], [2, 16], [1, 2, 4, 8, 16, 32, 64], ['ideal', 'real']
R1B = (4, 8, 16)
REG = {}
for nm, bs in (('R1', R1B), ('R0', (1, 2)), ('RH', (32, 64))):
    REG[nm] = [(M, V, b) for M in M_GRID for V in VN for b in bs]
    for V in VN:
        REG[f'{nm}-V{V}'] = [(M, V, b) for M in M_GRID for b in bs]
PAIRS = [('VSAP', 'LAM'), ('VSAP', 'BEST'), ('VSAP', 'VSAS'), ('VSAPX', 'LAM'), ('VSAPX', 'BEST'), ('NLMS', 'LAM'),
         ('NLMS', 'VSAS'), ('RLS', 'LAM'), ('RLS', 'VSAS'), ('VSAS', 'LAM'), ('VSAS', 'BEST'), ('VSAP', 'BEST36'),
         ('VSAS', 'BEST36'), ('VSAPX', 'VSAS')]
ARMS = ['EXACT', 'VSA', 'VSAS', 'VSAP', 'VSAPX', 'NLMS', 'RLS', 'LAMRF', 'LAM', 'BEST', 'BEST36']
MARGIN = 0.02


def load(enc):
    return {(M, V): [json.loads(l) for l in gzip.open(f'{FAM}/{enc}/all57_{enc}_M{M}_V{V}.jsonl.gz', 'rt')]
            for M in M_GRID for V in VN}


def summ(s, dr):
    return {'diff': s, 'ci95': [float(np.quantile(dr, 0.025)), float(np.quantile(dr, 0.975))],
            'ci90': [float(np.quantile(dr, 0.05)), float(np.quantile(dr, 0.95))]}


def add_vsapx(data, enc):
    ch56 = json.load(open(f'{RES}/b56_choice_{enc}.json'))['choice']
    ch57 = json.load(open(f'{RES}/b57_choice_{enc}.json'))['choice']
    src = {}
    for M in M_GRID:
        for V in VN:
            for b in BPF:
                for a, a3 in (('ideal', 'ideal'), ('real', 'raw')):
                    s = ch56[f'{M}|{V}|{b}|{a3}|STAR']
                    p = ch57[f'{M}|{V}|{b}|{a3}|PSTAR']
                    ds = s['dev'][s['fam']]
                    dp = p['dev'][p['fam']]
                    src[M, V, b, a] = 'VSAP' if dp > ds + 0.001 else 'VSAS'
                    for r in data[M, V]:
                        r['pred'][f'VSAPX|{b}|{a}'] = r['pred'][f'{src[M, V, b, a]}|{b}|{a}']
    return src


def write_cost(r0, enc, M, V, b, ch57):
    """Multiply-adds and working memory (numbers) to write the M facts of one cell, from the chosen shapes (ideal)."""
    n = NDIM[enc]
    out = {}
    cfg = r0['cfg']
    # store (B56 VSAS): Hebbian bundle, plus retraining if the dev rule is RT
    fam = cfg[f'VSAS|{b}|ideal'][0]
    d, q, ns = r0['b56'][f'{"VSAT" if fam == "RT" else "VSAL"}|{b}|ideal']
    fmap = cfg[f'VSAT|{b}|ideal'][3] if fam == 'RT' else cfg[f'VSAL|{b}|ideal'][3]
    K = 3 if fmap == 'sgn3' else 1
    out['feat_store'] = ns * n * d * K
    out['hebb'] = (ns * d, d)
    if fam == 'RT':
        e = int(str(cfg[f'VSAT|{b}|ideal'][1]).split(':')[0])
        out['store'] = (ns * d + e * ns * d * (V + 2) + e * math.ceil(ns / 32) * V * d, ns * d + V * d + d)
    else:
        out['store'] = out['hebb']
    # LAM (B49 choice)
    lc = cfg[f'LAM|{b}|ideal']
    ft, q = lc[1], lc[3]
    c = V - 1
    p = B.lam_p(b * M, c, q, 'ideal', 65536 if ft == 'rp' else n)
    out['feat_lam'] = M * n * p
    out['lam'] = (M * M * p + M ** 3 / 3 + M * M * c + p * M * c, M * p + M * M + p * c)
    nc = ch57[f'{M}|{V}|{b}|ideal|NLMS']['cfg']
    pn = B.lam_p(b * M, c, nc[3], 'ideal', 65536 if nc[1] == 'rp' else n)
    out['nlms'] = (2 * M * pn * c + M * pn, pn * c + pn)
    rc = ch57[f'{M}|{V}|{b}|ideal|RLS']['cfg']
    pr = B.lam_p(b * M, c, rc[3], 'ideal', n)
    out['rls'] = (M * (2 * pr * pr + 2 * pr * c), pr * pr + pr * c)
    out['bits'] = b * M
    return out


def twolevel(per_map_cells, B_=10000, seed=57):
    """per_map_cells: list over cells of (list over maps of per-query differences, cluster ids). Returns per-map point
    estimates, the two-level bootstrap draws of the map mean, and the entity-only draws of the map mean."""
    rng = np.random.default_rng(seed)
    nm = len(per_map_cells[0][0])
    pts = np.zeros(nm)
    draws = np.zeros((nm, B_))
    for diffs, cl in per_map_cells:
        k, inv = V36.clusters(cl)
        n_ = np.bincount(inv, minlength=k).astype(np.float64)
        sums = [np.bincount(inv, weights=d, minlength=k) for d in diffs]
        for m in range(nm):
            pts[m] += diffs[m].mean()
        for c0 in range(0, B_, 2000):
            cb = min(2000, B_ - c0)
            idx = rng.integers(0, k, size=(cb, k)).astype(np.int32)
            nn = n_[idx].sum(1)
            for m in range(nm):
                draws[m, c0:c0 + cb] += sums[m][idx].sum(1) / nn
    pts /= len(per_map_cells)
    draws /= len(per_map_cells)
    mi = rng.integers(0, nm, size=(B_, nm))
    two = draws[mi, np.arange(B_)[:, None]].mean(1)
    ent = draws.mean(0)
    return pts, two, ent


def part(enc):
    ei = PANEL.index(enc)
    data = load(enc)
    src = add_vsapx(data, enc)
    ch57 = json.load(open(f'{RES}/b57_choice_{enc}.json'))['choice']
    J = {'cells': {}, 'regions': {}, 'cn_regions': {}, 'vsapx_src': {f'{k[0]}|{k[1]}|{k[2]}|{k[3]}': v for k, v in src.items()}}
    pqc = {}

    def pq(M, V, arm):
        if (M, V, arm) not in pqc:
            pqc[M, V, arm] = V36.per_query(data[M, V], arm)
        return pqc[M, V, arm]

    def key(arm, b, a):
        return 'EXACT' if arm == 'EXACT' else f'{arm}|{b}|{a}'
    for ai, acct in enumerate(ACCTS):
        for M in M_GRID:
            for V in VN:
                for b in BPF:
                    J['cells'][f'{enc}|{acct}|{M}|{V}|{b}'] = {x: float(pq(M, V, key(x, b, acct))[0].mean()) for x in ARMS}
        for ri, (rn, R) in enumerate(REG.items()):
            for k, (x, y) in enumerate(PAIRS):
                diffs, cdiffs = [], []
                for (M, V, b) in R:
                    v, cl = pq(M, V, key(x, b, acct))
                    w = pq(M, V, key(y, b, acct))[0]
                    diffs.append((v - w, cl))
                    cdiffs.append(((v - w) / (1 - 1 / V), cl))
                nb = 10000 if (x, y) in (('VSAP', 'LAM'), ('VSAP', 'BEST')) and acct == 'ideal' and rn == 'R1' else 2000
                s, _, _, dr = V36.boot_region(diffs, nb, 5700 + ri * 100 + k + ei * 1000 + ai * 7)
                J['regions'][f'{enc}|{acct}|{rn}|{x}-{y}'] = summ(s, dr)
                if acct == 'ideal' and rn.startswith('R1') and (x, y) in (('VSAP', 'LAM'), ('VSAP', 'BEST'),
                                                                        ('VSAPX', 'LAM')):
                    s, _, _, dr = V36.boot_region(cdiffs, 2000, 9700 + ri * 100 + k + ei * 1000)
                    J['cn_regions'][f'{enc}|{acct}|{rn}|{x}-{y}'] = summ(s, dr)
    # maps
    if enc in MAP_ENCS:
        J['maps'] = {}
        for acct in ACCTS:
            for y in ('LAM', 'BEST', 'BEST36', 'LAMRF'):
                cells = []
                for (M, V, b) in REG['R1']:
                    _, cl = pq(M, V, key('VSAS', b, acct))
                    ds = []
                    for m in range(4):
                        st = 'VSAS' if m == 0 else f'VSASm{m}'
                        yy = y if (m == 0 or y == 'BEST36') else f'{y}m{m}'
                        ds.append(pq(M, V, key(st, b, acct))[0] - pq(M, V, key(yy, b, acct))[0])
                    cells.append((ds, cl))
                pts, two, ent = twolevel(cells, 10000 if acct == 'ideal' else 2000, 57 + ei)
                J['maps'][f'{enc}|{acct}|R1|VSAS-{y}'] = {
                    'per_map': pts.tolist(), 'mean': float(pts.mean()), 'sd': float(pts.std(ddof=1)),
                    'two_level95': [float(np.quantile(two, 0.025)), float(np.quantile(two, 0.975))],
                    'entity95': [float(np.quantile(ent, 0.025)), float(np.quantile(ent, 0.975))],
                    'registered_rank': int(np.argsort(np.argsort(pts))[0]) + 1}
            for m in range(1, 4):
                for y in ('LAM', 'BEST', 'BEST36'):
                    diffs = []
                    for (M, V, b) in REG['R1']:
                        v, cl = pq(M, V, key(f'VSASm{m}', b, acct))
                        yy = y if y == 'BEST36' else f'{y}m{m}'
                        diffs.append((v - pq(M, V, key(yy, b, acct))[0], cl))
                    s, _, _, dr = V36.boot_region(diffs, 2000, 7700 + m * 10 + ei * 1000)
                    J['maps'][f'{enc}|{acct}|R1|m{m}|VSAS-{y}'] = summ(s, dr)
    # write cost (ideal, R1 cells) and the Methods table
    wc = collections.defaultdict(list)
    meth = collections.defaultdict(list)
    for (M, V, b) in REG['R1']:
        r0 = data[M, V][0]
        w = write_cost(r0, enc, M, V, b, ch57)
        for k_, v in w.items():
            wc[k_].append(v)
        c = r0['cfg']
        meth['VSA'].append(c[f'VSA|{b}|ideal'] + [r0['vsa_d'][f'VSA|{b}|ideal'][0]])
        fam = c[f'VSAS|{b}|ideal'][0]
        nm = 'VSAT' if fam == 'RT' else 'VSAL'
        meth['VSAS'].append([fam] + c[f'{nm}|{b}|ideal'] + [r0['b56'][f'{nm}|{b}|ideal'][0]])
        meth['VSAP'].append(r0['b57'][f'VSAP|{b}|ideal'])
    J['write_cost'] = {k_: v for k_, v in wc.items()}
    J['methods'] = {k_: v for k_, v in meth.items()}
    json.dump(J, open(f'{RES}/b57_part_{enc}.json', 'w'))
    print('part', enc, flush=True)


def f(r, k='ci95'):
    return f'{100 * r["diff"]:+6.2f} [{100 * r[k][0]:+6.2f}, {100 * r[k][1]:+6.2f}]'


def labels(r):
    lo, hi = r['ci95']
    return {'not_behind': lo >= -MARGIN, 'tost': r['ci90'][0] > -MARGIN and r['ci90'][1] < MARGIN, 'ahead': lo > 0,
            'behind': hi < 0}


def g(x):
    return f'{x:.2g}'


def panel():
    J = {'cells': {}, 'regions': {}, 'cn_regions': {}, 'maps': {}, 'write_cost': {}, 'methods': {}}
    for enc in PANEL:
        P = json.load(open(f'{RES}/b57_part_{enc}.json'))
        for k in ('cells', 'regions', 'cn_regions', 'maps'):
            J[k].update(P.get(k, {}))
        J['write_cost'][enc] = P['write_cost']
        J['methods'][enc] = P['methods']
    R = J['regions']
    J49 = json.load(open(f'{RES}/b49_verdict.json'))
    J53 = json.load(open(f'{RES}/b53_verdict.json'))
    J56 = json.load(open(f'{RES}/b56_verdict.json'))
    L = ['# B57 verdict (prereg/B57_pca_store_maps_online_prereg.md). Points; 95% cluster bootstrap (90% for TOST).',
         '# VSAP = store on LAM\'s principal-component features (B56 read rules, dev-selected); VSAPX = dev-best of VSAS',
         '# (B56, random features) and VSAP; NLMS = one-pass normalised delta rule; RLS = recursive least squares (pca);',
         '# LAM, BEST = B49; BEST36 = best of the six key-free memories (B36).', '']
    for arm in ('VSAP', 'VSAPX'):
        L.append(f'## {"Primary" if arm == "VSAP" else "Secondary"}: R1 {arm} - LAM (ideal)')
        bars = {}
        for enc in PANEL:
            r = R[f'{enc}|ideal|R1|{arm}-LAM']
            lb = labels(r)
            bars[enc] = lb
            L.append(f'  {NAMES[enc]:15s} {f(r)}  90% [{100 * r["ci90"][0]:+.2f}, {100 * r["ci90"][1]:+.2f}]  '
                     f'not_behind={lb["not_behind"]} TOST={lb["tost"]} ahead={lb["ahead"]} LAM_ahead={lb["behind"]}')
        for nm, grp in (('panel', PANEL), ('LLM', LLM), ('sentence', PANEL[4:])):
            L.append(f'  {nm}: not behind {sum(bars[e]["not_behind"] for e in grp)} of {len(grp)}; TOST '
                     f'{sum(bars[e]["tost"] for e in grp)}; ahead {sum(bars[e]["ahead"] for e in grp)}; LAM ahead '
                     f'{sum(bars[e]["behind"] for e in grp)}')
        J.setdefault('bars', {})[arm] = bars
        L.append('')
    for x in ('VSAP', 'VSAPX'):
        for y in ('LAM', 'BEST'):
            L.append(f'## {x} - {y} by region (ideal); R2 = B49 store, unchanged')
            L.append('| encoder | R1 | R1 V=2 | R1 V=16 | R0 | R0 V=2 | R0 V=16 | RH | RH V=2 | RH V=16 | R2 |')
            L.append('| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |')
            for enc in PANEL:
                row = [f(R[f'{enc}|ideal|{rn}|{x}-{y}']) for rn in ('R1', 'R1-V2', 'R1-V16', 'R0', 'R0-V2', 'R0-V16',
                                                                  'RH', 'RH-V2', 'RH-V16')]
                row.append(f(J49['regions'][f'{enc}|ideal|R2|{y}']))
                L.append(f'| {NAMES[enc]} | ' + ' | '.join(row) + ' |')
            L.append('')
    L.append('## R1 component and online-write comparisons (ideal)')
    hdr = ['VSAP-VSAS', 'VSAPX-VSAS', 'VSAP-BEST36', 'VSAS-BEST36', 'NLMS-LAM', 'NLMS-VSAS', 'RLS-LAM', 'RLS-VSAS']
    L.append('| encoder | ' + ' | '.join(hdr) + ' |')
    L.append('|' + ' --- |' * (len(hdr) + 1))
    for enc in PANEL:
        L.append(f'| {NAMES[enc]} | ' + ' | '.join(f(R[f"{enc}|ideal|R1|{h}"]) for h in hdr) + ' |')
    L.append('')
    L.append('## Per V (ideal): R1-V2 / R1-V16')
    for h in ('VSAP-VSAS', 'NLMS-LAM', 'NLMS-VSAS', 'RLS-LAM'):
        L.append(f'  {h}: ' + ' | '.join(f'{NAMES[e]} {f(R[f"{e}|ideal|R1-V2|{h}"])} / {f(R[f"{e}|ideal|R1-V16|{h}"])}'
                                         for e in PANEL))
    L.append('')
    L.append('## Chance-normalised R1 (per-cell difference / (1 - 1/V)), ideal')
    for enc in PANEL:
        L.append(f'  {NAMES[enc]:15s} ' + ' | '.join(f'{rn} {h} {f(J["cn_regions"][f"{enc}|ideal|{rn}|{h}"])}'
                                                    for rn in ('R1', 'R1-V2', 'R1-V16')
                                                    for h in ('VSAP-LAM', 'VSAP-BEST', 'VSAPX-LAM')))
    L.append('')
    L.append('## R1 mean accuracy by V (ideal), raw | chance-normalised')
    for enc in PANEL:
        for V in VN:
            cells = [J['cells'][f'{enc}|ideal|{M}|{V}|{b}'] for M in M_GRID for b in R1B]
            m = {x: float(np.mean([c[x] for c in cells])) for x in ARMS}
            L.append(f'  {NAMES[enc]:15s} V={V:2d} ' + ' '.join(f'{x} {100 * m[x]:.1f}' for x in ARMS) + ' | cn ' +
                     ' '.join(f'{x} {100 * (m[x] - 1 / V) / (1 - 1 / V):.1f}' for x in ARMS))
    L.append('')
    L.append('## Map variance (R1, store = VSAS; maps 0 = registered, 1-3 new; two-level bootstrap maps x entities)')
    for enc in MAP_ENCS:
        for acct in ACCTS:
            for y in ('LAM', 'BEST', 'BEST36', 'LAMRF'):
                x = J['maps'][f'{enc}|{acct}|R1|VSAS-{y}']
                L.append(f'  {NAMES[enc]:11s} {acct:5s} VSAS-{y:6s} per map ' + ' '.join(f'{100 * v:+.2f}' for v in x['per_map'])
                         + f' | mean {100 * x["mean"]:+.2f} two-level [{100 * x["two_level95"][0]:+.2f}, '
                         f'{100 * x["two_level95"][1]:+.2f}] entity-only [{100 * x["entity95"][0]:+.2f}, '
                         f'{100 * x["entity95"][1]:+.2f}] between-map SD {100 * x["sd"]:.2f}; registered map rank '
                         f'{x["registered_rank"]} of 4 (1 = lowest)')
        for m in (1, 2, 3):
            L.append(f'  {NAMES[enc]:11s} map {m}: ' + ' | '.join(f'VSAS-{y} {f(J["maps"][f"{enc}|ideal|R1|m{m}|VSAS-{y}"])}'
                                                              for y in ('LAM', 'BEST', 'BEST36')))
    L.append('')
    L.append('## Write cost to store M facts (ideal; median over the 12 R1 cells [min, max]; multiply-adds | working '
             'memory in numbers; feature maps separately)')
    for enc in PANEL:
        W = J['write_cost'][enc]
        parts = []
        for k_ in ('hebb', 'store', 'lam', 'nlms', 'rls'):
            fl = [v[0] for v in W[k_]]
            mem = [v[1] for v in W[k_]]
            parts.append(f'{k_} {g(np.median(fl))} [{g(min(fl))}, {g(max(fl))}] | {g(np.median(mem))}')
        parts.append(f'features: store {g(np.median(W["feat_store"]))}, LAM {g(np.median(W["feat_lam"]))}; bits '
                     f'{g(np.median(W["bits"]))}')
        L.append(f'  {NAMES[enc]:15s} ' + '; '.join(parts))
    L.append('')
    L.append('## Accounting: R1 store - LAM and store - BEST, ideal vs real')
    L.append('| encoder | VSA(B53) - LAM ideal / real | VSAS(B56) - LAM | VSAP - LAM | VSAPX - LAM | VSA(B53) - BEST | '
             'VSAS - BEST | VSAP - BEST | VSAPX - BEST |')
    L.append('|' + ' --- |' * 9)
    for enc in PANEL:
        row = [f'{f(J53["regions"][f"{enc}|ideal|R1|LAM"])} / {f(J53["regions"][f"{enc}|real|R1|LAM"])}',
               f'{f(J56["regions"][f"{enc}|ideal|R1|VSAS-LAM"])} / {f(J56["regions"][f"{enc}|real|R1|VSAS-LAM"])}',
               f'{f(R[f"{enc}|ideal|R1|VSAP-LAM"])} / {f(R[f"{enc}|real|R1|VSAP-LAM"])}',
               f'{f(R[f"{enc}|ideal|R1|VSAPX-LAM"])} / {f(R[f"{enc}|real|R1|VSAPX-LAM"])}',
               f'{f(J53["regions"][f"{enc}|ideal|R1|BEST"])} / {f(J53["regions"][f"{enc}|real|R1|BEST"])}',
               f'{f(J56["regions"][f"{enc}|ideal|R1|VSAS-BEST"])} / {f(J56["regions"][f"{enc}|real|R1|VSAS-BEST"])}',
               f'{f(R[f"{enc}|ideal|R1|VSAP-BEST"])} / {f(R[f"{enc}|real|R1|VSAP-BEST"])}',
               f'{f(R[f"{enc}|ideal|R1|VSAPX-BEST"])} / {f(R[f"{enc}|real|R1|VSAPX-BEST"])}']
        L.append(f'| {NAMES[enc]} | ' + ' | '.join(row) + ' |')
    L.append('')
    L.append('## Methods: the store\'s dev-chosen configuration over the 12 R1 cells (ideal)')
    for enc in PANEL:
        Mt = J['methods'][enc]
        va = Mt['VSA']
        L.append(f'  {NAMES[enc]} (n = {NDIM[enc]}): B53 VSA prep {dict(collections.Counter(c[0] for c in va))}, map '
                 f'{dict(collections.Counter(c[1] for c in va))}, q {dict(collections.Counter(c[2] for c in va))}, '
                 f'share {dict(collections.Counter(c[3] for c in va))}, d median {int(np.median([c[4] for c in va]))} '
                 f'[{min(c[4] for c in va)}, {max(c[4] for c in va)}]')
        vs = Mt['VSAS']
        L.append(f'      B56 VSAS rule {dict(collections.Counter(c[0] for c in vs))}, alpha / E:eta '
                 f'{dict(collections.Counter(str(c[2]) for c in vs))}, q {dict(collections.Counter(c[1] for c in vs))}, '
                 f'd median {int(np.median([c[-1] for c in vs]))} [{min(c[-1] for c in vs)}, {max(c[-1] for c in vs)}]')
        vp = Mt['VSAP']
        L.append(f'      B57 VSAP prep/map {dict(collections.Counter(c[3] + ":" + c[4] for c in vp))}, q '
                 f'{dict(collections.Counter(c[1] for c in vp))}, d median {int(np.median([c[0] for c in vp]))} '
                 f'[{min(c[0] for c in vp)}, {max(c[0] for c in vp)}], rule {dict(collections.Counter(c[-1] for c in vp))}')
    json.dump(J, open(f'{RES}/b57_verdict.json', 'w'), indent=1)
    open(f'{RES}/b57_verdict.txt', 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L[:22]))


def preds():
    J = json.load(open(f'{RES}/b57_verdict.json'))
    R = J['regions']

    def r(e, rn, h):
        return R[f'{e}|ideal|{rn}|{h}']
    out = []
    v = [r(e, 'R1-V2', 'VSAP-VSAS')['ci95'][0] > 0 for e in PANEL]
    out.append(('Q1', sum(v) >= 4, f'R1-V2 VSAP - VSAS above 0 on {sum(v)} of 7: ' + ', '.join(
        f'{e} {100 * r(e, "R1-V2", "VSAP-VSAS")["diff"]:+.2f}' for e in PANEL)))
    v = [r(e, 'R1-V16', 'VSAP-VSAS')['ci95'][1] < 0 for e in PANEL]
    out.append(('Q2', sum(v) >= 5, f'R1-V16 VSAP - VSAS below 0 on {sum(v)} of 7: ' + ', '.join(
        f'{e} {100 * r(e, "R1-V16", "VSAP-VSAS")["diff"]:+.2f}' for e in PANEL)))
    b = J['bars']['VSAP']
    nb, ah = sum(b[e]['not_behind'] for e in PANEL), sum(b[e]['ahead'] for e in PANEL)
    out.append(('Q3', nb <= 3 and ah == 0, f'VSAP not behind {nb}, ahead {ah}'))
    b = J['bars']['VSAPX']
    nb = sum(b[e]['not_behind'] for e in PANEL)
    out.append(('Q4', nb >= 5, f'VSAPX not behind {nb}'))
    ok = True
    txt = []
    for e in MAP_ENCS:
        x = J['maps'][f'{e}|ideal|R1|VSAS-LAM']
        c1 = x['sd'] <= 0.006 and x['two_level95'][0] <= x['per_map'][0] <= x['two_level95'][1]
        ok &= c1
        txt.append(f'{e} SD {100 * x["sd"]:.2f} registered {100 * x["per_map"][0]:+.2f} in [{100 * x["two_level95"][0]:+.2f}, '
                   f'{100 * x["two_level95"][1]:+.2f}]')
    out.append(('Q5', ok, '; '.join(txt)))
    ok = True
    txt = []
    for e in MAP_ENCS:
        x = J['maps'][f'{e}|ideal|R1|VSAS-BEST36']
        pm = x['per_map']
        low = pm[0] == min(pm) and (min(pm[1:]) - pm[0]) > x['sd']
        ok &= not low
        txt.append(f'{e} per map ' + ' '.join(f'{100 * v:+.2f}' for v in pm))
    out.append(('Q6', ok, '; '.join(txt)))
    v = [r(e, 'R1', 'NLMS-LAM')['ci95'][1] < 0 for e in PANEL]
    v2 = [r(e, 'R1-V2', 'NLMS-VSAS')['diff'] > 0 for e in PANEL]
    out.append(('Q7', sum(v) >= 5 and sum(v2) >= 4, f'NLMS - LAM below 0 on {sum(v)} of 7; R1-V2 NLMS - VSAS > 0 on '
                f'{sum(v2)} of 7: ' + ', '.join(f'{e} {100 * r(e, "R1", "NLMS-LAM")["diff"]:+.2f}' for e in PANEL)))
    v = [abs(r(e, 'R1', 'RLS-LAM')['diff']) <= 0.01 for e in PANEL]
    out.append(('Q8', sum(v) >= 5, f'|RLS - LAM| <= 1 on {sum(v)} of 7: ' + ', '.join(
        f'{e} {100 * r(e, "R1", "RLS-LAM")["diff"]:+.2f}' for e in PANEL)))
    L = [f'- {p}: {"held" if ok else "FAILED"} ({t})' for p, ok, t in out]
    open(f'{RES}/b57_predictions.txt', 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    {'part': lambda: part(sys.argv[2]), 'panel': panel, 'preds': preds}[sys.argv[1]]()
