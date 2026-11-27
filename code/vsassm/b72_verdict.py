"""B72 verdict (prereg/B72_format_side_info_prereg.md): (A) the optimal write in the store's own format, (B) symmetric
side-information accounting, (C) store-level inference (t interval, exact sign-flip permutation test), (D) LAMc+ and
the heterogeneity of STORE - LAMc.

Arms: existing test predictions as b71_verdict.load reads them (+ SDMT as b66_storeboot.load), B72's test jobs
(b72_out/jobs, b72_out/lamp/jobs) at the committed choices (results/b72_choice_{enc}.json). Ideal accounting, R1.

python3 b72_verdict.py part ENC        # -> results/b72_part_{enc}.json
python3 b72_verdict.py panel           # -> results/b72_verdict.{json,txt}
"""
import gzip
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import b66_storeboot as SB  # noqa: E402
import b71_verdict as V71  # noqa: E402

RES = os.environ.get('B72_RES', f'{ROOT}/results')
OUT = os.environ.get('B72_OUT', f'{ROOT}/b72_out')
PANEL = SB.PANEL
LLM, SENT = PANEL[:4], PANEL[4:]
NAMES = SB.NAMES
M_GRID, VN, BPF = [256, 1024], [2, 16], [4, 8, 16]
TEST_REPS = {256: 20, 1024: 5}
REG = SB.REG
RN = ('R1-V2', 'R1-V16', 'R1')
A = 'ideal'
KS = [1e3, 1e4, 1e5]
KT = {1e3: 'k3', 1e4: 'k4', 1e5: 'k5'}
PRIMARY_A = [('OPTR', 'STORE'), ('OPTR', 'LAMC'), ('OPTR', 'NLMSC'), ('OPTX', 'VSAPCX'), ('OPTX', 'LAMC'),
             ('OPTX', 'NLMSC')]
SECONDARY_A = [('OPTP', 'VSAPC'), ('OPTQ', 'OPTR'), ('OPTR', 'LSFR'), ('STORE', 'OPTR'), ('LSFR', 'LAMC'),
               ('STORE', 'LAMC')]
PRIMARY_D = [('LAMCP', 'LAMC'), ('STORE', 'LAMCP')]
HEAD_C = [('STORE', 'LAMC'), ('VSAPCX', 'LAMC'), ('STORE', 'NLMSC'), ('VSAPCX', 'NLMSC')]
B_PAIRS = [('STORE', 'LAMC'), ('VSAR', 'SDMT'), ('STORE', 'NLMSC')]


def choice(enc):
    return json.load(open(f'{RES}/b72_choice_{enc}.json'))['choice']


def kstr(K):
    return f'{K:.0e}'.replace('+0', '').replace('+', '')


def test_index(enc):
    """{key: pred} from B72's test jobs; '_q' keys hold qgold per job group."""
    out = {}
    for jd, lamp in ((os.path.join(OUT, 'jobs'), False), (os.path.join(OUT, 'lamp', 'jobs'), True)):
        if not os.path.isdir(jd):
            continue
        for fn in sorted(os.listdir(jd)):
            if not fn.startswith(f'{enc}_') or '_test_' not in fn:
                continue
            r = json.load(gzip.open(os.path.join(jd, fn), 'rt'))
            s = r['spec']
            if s['enc'] != enc:
                continue
            if lamp:
                gk = ('LAMP', s['M'], s['V'], s['rep'], s['prep'], s['fmap'])
                out[('_q',) + gk] = r['qgold']
                for x in r['rows']:
                    out[gk + (x['bpf'], tuple(x['cfg']))] = x['pred']
            elif s['arm'] == 'OPT':
                gk = ('OPT' + s['kind'], s['M'], s['V'], s['rep'], s['prep'], s['fmap'])
                out[('_q',) + gk] = r['qgold']
                for x in r['rows']:
                    out[gk + (x['bpf'], x['q'], tuple(x['cfg']))] = x['pred']
            else:
                gk = (s['mem'], s['K'], s['M'], s['V'], s['rep'])
                out.setdefault(('_q',) + gk, r['qgold'])
                assert out[('_q',) + gk] == r['qgold']
                for x in r['rows']:
                    out[gk + (x['bpf'], tuple(x['cfg']))] = x['pred']
    return out


def load(enc, with_b=True):
    import b64_verdict as V64
    import b32_common as C
    data = V71.load(enc)
    ch = choice(enc)
    ch64 = V64.choice(enc)
    ti = test_index(enc)
    pool = None
    for (M, V), recs in data.items():
        rows64 = V64.job_rows(enc, M, V)
        for r in recs:
            rep = r['rep']
            jr = rows64.get(rep)
            assert jr is not None and jr['_qgold'] == r['qgold'], ('SDMT test job', enc, M, V, rep)
            for b in BPF:
                cell = f'{M}|{V}|{b}'
                r['pred'][f'SDMT|{b}|{A}'] = jr[tuple(ch64[f'{cell}|{A}|SDMT']['cfg']), b, A][0]
                # (A)
                for kd in ('R', 'S', 'P', 'Q'):
                    c = ch[f'{cell}|OPT{kd}']
                    prep, fmap, q = c['fmt']
                    jk = 'OPTP' if kd == 'P' else 'OPTR'
                    gk = (jk, M, V, rep, prep, fmap)
                    assert ti[('_q',) + gk] == r['qgold'], ('OPT qgold', enc, gk)
                    r['pred'][f'OPT{kd}|{b}|{A}'] = ti[gk + (b, q, tuple(c['cfg']))]
                r['pred'][f'OPTX|{b}|{A}'] = r['pred'][f'{ch[cell + "|OPTX"]["src"]}|{b}|{A}']
                # (D)
                c = ch[f'{cell}|LAMP']
                if c['src'] == 'LAMP':
                    gk = ('LAMP', M, V, rep, c['feat'][0], c['feat'][1])
                    assert ti[('_q',) + gk] == r['qgold']
                    r['pred'][f'LAMCP|{b}|{A}'] = ti[gk + (b, tuple(c['cfg4']))]
                else:
                    r['pred'][f'LAMCP|{b}|{A}'] = r['pred'][f'LAMC|{b}|{A}']
                # (B)
                if not with_b or enc not in LLM:
                    continue
                for K in KS:
                    for mem in ('STORE', 'VSAR', 'LAMC', 'NLMSC', 'SDMT'):
                        c = ch.get(f'{cell}|{mem}|K{kstr(K)}')
                        if c is None:
                            continue
                        if c['cfg'] is None:
                            if pool is None:
                                import b66_lsfc as LS
                                pool = LS.setup(enc)['pool']
                            st = C.make_store(pool, 'test', M, V, rep)
                            dflt = int(np.bincount(st['vals'], minlength=V).argmax())
                            r['pred'][f'{mem}{KT[K]}|{b}|{A}'] = [dflt] * len(r['qgold'])
                            continue
                        gk = (mem, K, M, V, rep)
                        assert ti[('_q',) + gk] == r['qgold']
                        r['pred'][f'{mem}{KT[K]}|{b}|{A}'] = ti[gk + (b, tuple(c['cfg']))]
    return data


# ----------------------------------------------------------------------------------------------- (C) store level

def store_units(data, x, y, R):
    """[(stratum, D_u, Q_u)] with D_u = mean over the stratum's region budgets of the per-query difference in store u."""
    strata = sorted({(M, V) for (M, V, b) in R})
    units = []
    for (M, V) in strata:
        bs = [b for (M_, V_, b) in R if (M_, V_) == (M, V)]
        for r in data[M, V]:
            g = np.asarray(r['qgold'])
            ds = [((np.asarray(r['pred'][f'{x}|{b}|{A}']) == g).astype(float)
                   - (np.asarray(r['pred'][f'{y}|{b}|{A}']) == g).astype(float)).mean() for b in bs]
            units.append(((M, V), float(np.mean(ds)), len(g)))
    return units, strata


def t_interval(units, strata):
    from scipy import stats
    G = len(strata)
    T, var, a_g, n_g = 0.0, 0.0, [], []
    for g in strata:
        u = [(d, q) for (s, d, q) in units if s == g]
        n = len(u)
        Q = sum(q for _, q in u)
        Tg = sum(q * d for d, q in u) / Q
        qb = Q / n
        s2 = sum((q / qb) ** 2 * (d - Tg) ** 2 for d, q in u) / (n * (n - 1))
        T += Tg / G
        a_g.append(s2 / G ** 2)
        n_g.append(n)
    var = sum(a_g)
    df = var ** 2 / sum(a ** 2 / (n - 1) for a, n in zip(a_g, n_g)) if var > 0 else float('inf')
    h = stats.t.ppf(0.975, df) * math.sqrt(var)
    return T, T - h, T + h, df


def sign_flip_exact(a, T):
    """Exact two-sided p of the sign-flip test: #{|sum s_u a_u| >= |T|} / 2^S by meet-in-the-middle."""
    a = np.asarray(a, np.float64)
    S = len(a)
    t = abs(T) - 1e-12 * max(1.0, abs(T))

    def sums(v):
        s = np.zeros(1)
        for x in v:
            s = np.concatenate([s + x, s - x])
        return s
    h = S // 2
    A1 = sums(a[:h])
    B1 = np.sort(sums(a[h:]))
    # count pairs with A1 + B1 >= t (the distribution is symmetric: two-sided = 2 x one tail, |T| > 0)
    cnt = 0
    for c0 in range(0, len(A1), 1 << 22):
        blk = A1[c0:c0 + (1 << 22)]
        cnt += int((len(B1) - np.searchsorted(B1, t - blk, side='left')).sum())
    if abs(T) <= 1e-15:
        return 1.0
    return min(1.0, 2.0 * cnt / float(2 ** S))


def c_analysis(data, pairs, regions):
    out = {}
    for (x, y) in pairs:
        for rn in regions:
            R = REG[rn]
            try:
                units, strata = store_units(data, x, y, R)
            except KeyError:
                continue
            G = len(strata)
            Qg = {g: sum(q for s, _, q in units if s == g) for g in strata}
            w = [q / (G * Qg[s]) for s, _, q in units]
            a = [wi * d for wi, (_, d, _) in zip(w, units)]
            T = float(sum(a))
            T2, lo, hi, df = t_interval(units, strata)
            assert abs(T - T2) < 1e-12
            p = sign_flip_exact(a, T)
            out[f'{rn}|{x}-{y}'] = {'diff': T, 't_ci95': [lo, hi], 'df': df, 't_label': SB.lab(x, y, lo, hi),
                                    'perm_p': p, 'n_stores': len(units), 'a': a}
    return out


# ----------------------------------------------------------------------------------------------- (D) heterogeneity

def hetero(data, enc):
    import b36_verdict as V36
    b53 = json.load(open(f'{ROOT}/results/b53_choice_{enc}.json'))['choice']
    ch60 = json.load(open(f'{ROOT}/results/b60_choice_{enc}.json'))['choice']
    out = {}
    for (M, V), recs in data.items():
        lay, cl = SB.cell_layout(recs)
        for b in BPF:
            d = SB.correct(recs, f'STORE|{b}|{A}') - SB.correct(recs, f'LAMC|{b}|{A}')
            s, lo, hi, _ = V36.boot_region([(d, cl)], 2000, 72000 + PANEL.index(enc) * 100 + M // 256 + V + b)
            ps = [r['b60'][f'LAMC|{b}|{A}'] for r in recs if 'b60' in r and f'LAMC|{b}|{A}' in r['b60']]
            out[f'{M}|{V}|{b}'] = {'diff': s, 'ci95': [lo, hi], 'label': SB.lab('STORE', 'LAMC', lo, hi),
                                   'acc_store': float(SB.correct(recs, f'STORE|{b}|{A}').mean()),
                                   'acc_lamc': float(SB.correct(recs, f'LAMC|{b}|{A}').mean()),
                                   'store_cfg': b53[f'{M}|{V}|{b}|ideal|VSA']['cfg'],
                                   'lamc_cfg': ch60[f'{M}|{V}|{b}|ideal|LAMC']['cfg'],
                                   'lamc_p': ps[0][1] if ps else None}
    return out


# ----------------------------------------------------------------------------------------------- part / panel

def part(enc):
    data = load(enc, with_b=False)
    ei = PANEL.index(enc)
    J1 = SB.analyse(data, ei + 7200, [c + ('P',) for c in PRIMARY_A + PRIMARY_D + HEAD_C], accts=[A],
                    nb={A: 10000}, regions=REG)
    J2 = SB.analyse(data, ei + 7300, [c + ('S',) for c in SECONDARY_A], accts=[A], nb={A: 2000}, regions=REG)
    bp = []
    if enc in LLM:  # K = infinity here; finite K on the box (b72_bverdict.py, compute note 2)
        bp += [(x, y, 'Binf') for (x, y) in B_PAIRS]
    J3 = SB.analyse(data, ei + 7400, bp, accts=[A], nb={A: 2000}, regions=REG) if bp else {'regions': {},
                                                                                          'missing': []}
    pairsA = [(x, y) for x, y in PRIMARY_A + PRIMARY_D]
    fc = f'{RES}/b72_cpart_{enc}.json'
    JC = json.load(open(fc))['C'] if os.path.exists(fc) else c_analysis(data, HEAD_C, RN)
    JC.update(c_analysis(data, pairsA, ('R1',)))
    acc = {}
    arms = sorted({a for c in PRIMARY_A + SECONDARY_A + PRIMARY_D + HEAD_C for a in c[:2]} |
                  {a for c in bp for a in c[:2]} | {'VSAR', 'SDMT', 'STORE', 'LAMC', 'NLMSC'})
    for (M, V), recs in data.items():
        for b in BPF:
            for arm in arms:
                c = SB.correct(recs, f'{arm}|{b}|{A}')
                if c is not None:
                    acc[f'{M}|{V}|{b}|{arm}'] = float(c.mean())
    J = {'enc': enc, 'primary': J1['regions'], 'secondary': J2['regions'], 'B': J3['regions'],
         'missing': J1['missing'] + J2['missing'] + J3['missing'], 'C': JC, 'hetero': hetero(data, enc),
         'cell_acc': acc}
    json.dump(J, open(f'{RES}/b72_part_{enc}.json', 'w'))
    print('part', enc, 'missing', len(J['missing']), flush=True)


def fmt(e, k='one'):
    lo, hi = e[k]['ci95']
    return f'{100 * e["diff"]:+.2f} [{100 * lo:+.2f}, {100 * hi:+.2f}]'


def short(e, x, y, k='one'):
    lb = e[k]['label']
    return lb.replace(f'{x} leads', f'{x}').replace(f'{y} leads', f'{y}')


def panel():
    P = {enc: json.load(open(f'{RES}/b72_part_{enc}.json')) for enc in PANEL
         if os.path.exists(f'{RES}/b72_part_{enc}.json')}
    for enc in P:
        fb = f'{RES}/b72_bpart_{enc}.json'
        if os.path.exists(fb):
            jb = json.load(open(fb))
            P[enc]['B'].update(jb['B'])
            P[enc]['bcell_acc'] = jb['cell_acc']
            P[enc]['missing'] += jb['missing']
    L = ['# B72 verdict (prereg/B72_format_side_info_prereg.md). Ideal accounting, R1 (V 2 / 16 x 4, 8, 16 bits per fact',
         '# x M 256 / 1,024). Points; one-level 95% interval (entity within store) = registered label; two-level label',
         '# (stores, then entities) in brackets where it differs. Labels by B60\'s rule.', '']

    def table(title, pairs, grp='primary', regions=RN, encs=PANEL):
        L.append(f'## {title}')
        for (x, y) in pairs:
            nm = f'{x}-{y}'
            L.append(f'### {nm}')
            L.append('| encoder | ' + ' | '.join(regions) + ' |')
            L.append('| --- |' + ' --- |' * len(regions))
            cnt = {rn: {} for rn in regions}
            for enc in encs:
                if enc not in P:
                    continue
                cells = []
                for rn in regions:
                    e = P[enc][grp].get(f'{A}|{rn}|{nm}')
                    if e is None:
                        cells.append('n/a')
                        continue
                    lb1, lb2 = e['one']['label'], e['two']['label']
                    cells.append(f'{fmt(e)} {lb1}' + (f' (two-level: {lb2})' if lb2 != lb1 else ''))
                    cnt[rn][lb1] = cnt[rn].get(lb1, 0) + 1
                L.append(f'| {NAMES[enc]} | ' + ' | '.join(cells) + ' |')
            for rn in regions:
                L.append(f'  {rn}: ' + ', '.join(f'{k} {v}' for k, v in sorted(cnt[rn].items())))
            L.append('')
    table('(A) optimal write in the store\'s own format', PRIMARY_A)
    table('(A) secondary', SECONDARY_A, grp='secondary')
    # decomposition STORE - LAMC = (STORE - OPTR) + (OPTR - LSFR) + (LSFR - LAMC)
    L.append('## (A) decomposition over R1: STORE - LAMC = [STORE - OPTR] write + [OPTR - LSFR] format + [LSFR - LAMC] '
             'features')
    L.append('| encoder | total | write | format | features | largest |')
    L.append('| --- | --- | --- | --- | --- | --- |')
    for enc in PANEL:
        if enc not in P:
            continue
        s = P[enc]['secondary']
        t = [s[f'{A}|R1|{nm}']['diff'] for nm in ('STORE-LAMC', 'STORE-OPTR', 'OPTR-LSFR', 'LSFR-LAMC')]
        big = ['write', 'format', 'features'][int(np.argmax(np.abs(t[1:])))]
        L.append(f'| {NAMES[enc]} | {100 * t[0]:+.2f} | {100 * t[1]:+.2f} | {100 * t[2]:+.2f} | {100 * t[3]:+.2f} | '
                 f'{big} |')
    L.append('')
    table('(D) LAMc+ (sign and sgn3 random features added to LAMc)', PRIMARY_D)
    # (C)
    L.append('## (C) store-level inference: t interval over per-store differences (stratified, Welch df) and exact '
             'sign-flip permutation test over test stores')
    L.append('| contrast | encoder | region | diff | one-level | two-level | t interval (df) | t label | perm p |')
    L.append('| --- | --- | --- | --- | --- | --- | --- | --- | --- |')
    for (x, y) in HEAD_C + PRIMARY_A + PRIMARY_D:
        nm = f'{x}-{y}'
        for enc in PANEL:
            if enc not in P:
                continue
            for rn in (RN if (x, y) in HEAD_C else ('R1',)):
                c = P[enc]['C'].get(f'{rn}|{nm}')
                if c is None:
                    continue
                e = (P[enc]['primary'].get(f'{A}|{rn}|{nm}') or P[enc]['secondary'].get(f'{A}|{rn}|{nm}'))
                one = f'{fmt(e)} {e["one"]["label"]}' if e else 'n/a'
                two = e['two']['label'] if e else 'n/a'
                lo, hi = c['t_ci95']
                L.append(f'| {nm} | {NAMES[enc]} | {rn} | {100 * c["diff"]:+.2f} | {one} | {two} | '
                         f'[{100 * lo:+.2f}, {100 * hi:+.2f}] ({c["df"]:.1f}) | {c["t_label"]} | {c["perm_p"]:.2g} |')
    L.append('')
    # (D) heterogeneity
    L.append('## (D) heterogeneity of STORE - LAMC: cells (of 12 per encoder) with the store ahead')
    for enc in PANEL:
        if enc not in P:
            continue
        H = P[enc]['hetero']
        pos = [(k, v) for k, v in sorted(H.items()) if v['diff'] > 0]
        lead = [k for k, v in pos if v['label'] == 'STORE leads']
        L.append(f'- {NAMES[enc]}: point > 0 in {len(pos)} of 12; store leads in {len(lead)}: ' + '; '.join(
            f'M={k.split("|")[0]} V={k.split("|")[1]} {k.split("|")[2]} bpf {100 * v["diff"]:+.2f} '
            f'[{100 * v["ci95"][0]:+.2f}, {100 * v["ci95"][1]:+.2f}] store {v["store_cfg"]} LAMc {v["lamc_cfg"]} '
            f'p={v["lamc_p"]}' for k, v in pos))
    L.append('')
    # (B)
    L.append('## (B) symmetric side information (8 bits per number amortised over K stores; language models)')
    for (x, y) in B_PAIRS:
        L.append(f'### {x} - {y}')
        L.append('| encoder | K | R1-V2 | R1-V16 | R1 |')
        L.append('| --- | --- | --- | --- | --- |')
        for enc in LLM:
            if enc not in P:
                continue
            for K, tg in [(1e3, 'k3'), (1e4, 'k4'), (1e5, 'k5'), (None, '')]:
                nm = f'{x}{tg}-{y}{tg}'
                cells = []
                for rn in RN:
                    e = P[enc]['B'].get(f'{A}|{rn}|{nm}')
                    if e is None:
                        cells.append('n/a')
                        continue
                    lb1, lb2 = e['one']['label'], e['two']['label']
                    cells.append(f'{fmt(e)} {lb1}' + (f' (two-level: {lb2})' if lb2 != lb1 else ''))
                L.append(f'| {NAMES[enc]} | {kstr(K) if K else "inf"} | ' + ' | '.join(cells) + ' |')
        L.append('')
    L.append('### memory(K) - memory(inf): cost of the charge to each memory itself, R1 point difference of region means')
    L.append('(no interval: the K = infinity predictions are not on the box; compute note 2)')
    for enc in LLM:
        if enc not in P or 'bcell_acc' not in P[enc]:
            continue
        ca, cb = P[enc]['cell_acc'], P[enc]['bcell_acc']
        cells = [(M, V, b) for M in M_GRID for V in VN for b in BPF]
        parts = []
        for mem in ('STORE', 'VSAR', 'LAMC', 'NLMSC', 'SDMT'):
            if f'256|2|4|{mem}' not in ca:
                continue
            parts.append(mem + ' ' + ', '.join(
                f'{kstr(K)}: {100 * np.mean([cb[f"{M}|{V}|{b}|{mem}{KT[K]}"] - ca[f"{M}|{V}|{b}|{mem}"] for (M, V, b) in cells]):+.2f}'
                for K in KS))
        L.append(f'- {NAMES[enc]}: ' + '; '.join(parts))
    L.append('')
    miss = {enc: P[enc]['missing'] for enc in P if P[enc]['missing']}
    L.append(f'missing: {json.dumps(miss)[:2000]}')
    open(f'{RES}/b72_verdict.txt', 'w').write('\n'.join(L) + '\n')
    json.dump({enc: {k: v for k, v in P[enc].items() if k != 'cell_acc'} for enc in P},
              open(f'{RES}/b72_verdict.json', 'w'))
    print('\n'.join(L))


def cpart(enc):
    """(C) on the four headline contrasts from the saved outputs of B60 / B62 / B71 only (no B72 arm needed)."""
    data = V71.load(enc)
    JC = c_analysis(data, HEAD_C, RN)
    json.dump({'enc': enc, 'C': JC}, open(f'{RES}/b72_cpart_{enc}.json', 'w'))
    print('cpart', enc, {k: (round(100 * v['diff'], 2), v['t_label'], v['perm_p']) for k, v in JC.items()}, flush=True)


def main():
    a = sys.argv[1:]
    if a[0] == 'part':
        part(a[1])
    elif a[0] == 'cpart':
        cpart(a[1])
    elif a[0] == 'panel':
        panel()
    else:
        raise SystemExit(__doc__)


if __name__ == '__main__':
    main()
