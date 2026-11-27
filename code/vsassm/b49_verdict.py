"""B49 verdict (prereg/B49_panel_mechanism_competitors_prereg.md, sections 4-6): scores the merged per-query test
predictions of every panel encoder (b49_out/fam/{enc}/test/all49_{enc}_M{M}_V{V}.jsonl.gz) with b36_verdict's own
functions (per_query, boot_region, sign_test, rate) and applies the registered bars mechanically; then the mechanism
interventions (Delta of the R1 margin, paired on the same queries) and the geometry association.

python3 b49_verdict.py panel [--boot 10000] [--encs qwen8,b46,...]   # -> results/b49_verdict.{json,txt}
python3 b49_verdict.py mech  [--boot 10000]                          # -> results/b49_mech_verdict.{json,txt}
"""
import gzip
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import b36_verdict as V36  # noqa: E402

PANEL = ['qwen8', 'b46', 'llama', 'mistral', 'e5', 'bge', 'minilm']
LLM = ['qwen8', 'b46', 'llama', 'mistral']
NAMES = {'qwen8': 'Qwen3-8B L9 mean', 'b46': 'Granite-3.3-8B L30 mean', 'llama': 'Llama-3.1-8B L24 mean',
         'mistral': 'Mistral-7B L24 PromptEOL', 'e5': 'e5-base-v2', 'bge': 'bge-small-en-v1.5',
         'minilm': 'all-MiniLM-L6-v2'}
M_GRID, V_GRID, BPF = [256, 1024], [2, 16, 128, 1024], [1, 2, 4, 8, 16, 32, 64]
ACCTS = ['ideal', 'real']
OLD = ['IDX', 'SF', 'CM', 'BF', 'FBF', 'SDM']
NEW = ['LAM', 'LHI', 'LHC', 'LHS']
ALL10 = OLD + NEW
REGIONS = {'R1': [(M, V, b) for M in M_GRID for V in (2, 16) for b in (4, 8, 16)],
           'R0': [(M, V, b) for M in M_GRID for V in (2, 16) for b in (1, 2)],
           'RH': [(M, V, b) for M in M_GRID for V in (2, 16) for b in (32, 64)],
           'R2': [(M, 1024, b) for M in M_GRID for b in BPF]}
MARGIN = 0.02
NCELL = 56


def load(enc, Vs=V_GRID, d=None):
    d = d or f'{ROOT}/b49_out/fam/{enc}/test'
    return {(M, V): [json.loads(l) for l in gzip.open(f'{d}/all49_{enc}_M{M}_V{V}.jsonl.gz', 'rt')]
            for M in M_GRID for V in Vs}


def pq(recs, arm):
    return V36.per_query(recs, arm)


def cell(recs, a, b, B, seed):
    x, cl = pq(recs, a)
    y, _ = pq(recs, b)
    stat, lo, hi, draws = V36.boot_region([(x - y, cl)], B, seed)
    al = 0.05 / NCELL
    return {'diff': stat, 'ci95': [lo, hi], 'ci_bonf': [float(np.quantile(draws, al / 2)),
                                                       float(np.quantile(draws, 1 - al / 2))],
            'a': float(x.mean()), 'b': float(y.mean())}


def panel_one(B, enc):
    """One encoder's cells, regions and rates (seeds depend on the encoder's position in PANEL only), written to
    results/b49_verdict_part_{enc}.{json,txt}; panel() combines the parts."""
    J = {'cells': {}, 'regions': {}, 'rates': {}}
    L = []
    for ei, enc in [(PANEL.index(enc), enc)]:
        data = load(enc)
        for ai, acct in enumerate(ACCTS):
            L.append(f'## {enc} ({NAMES[enc]}), {acct}: accuracy (points); VSA - BEST [95% CI] [Bonferroni]')
            L.append(f'{"M":>5} {"V":>5} {"bpf":>3} {"EXACT":>6} ' + ' '.join(f'{c:>5}' for c in ALL10) +
                     f' {"VSA":>5} {"VSAL":>5} {"BEST":>9} {"diff":>7} {"95% CI":>16} {"Bonf":>16} label')
            for M in M_GRID:
                for V in V_GRID:
                    recs = data[M, V]
                    ex, cl = pq(recs, 'EXACT')
                    for b in BPF:
                        seed = 49 + M * 7 + V * 13 + b * 17 + ei * 100000 + ai * 50000
                        v, _ = pq(recs, f'VSA|{b}|{acct}')
                        comp = {c: float(pq(recs, f'{c}|{b}|{acct}')[0].mean()) for c in ALL10}
                        c0 = cell(recs, f'VSA|{b}|{acct}', f'BEST|{b}|{acct}', B, seed)
                        lab = 'VSA WIN' if c0['diff'] >= MARGIN and c0['ci_bonf'][0] > 0 else (
                            'BEST WIN' if c0['diff'] <= -MARGIN and c0['ci_bonf'][1] < 0 else 'tie/inconclusive')
                        fam = recs[0]['cfg'][f'BEST|{b}|{acct}'][0]
                        per = {}
                        for c in ALL10 + ['BEST36', 'BESTNL']:
                            x = cell(recs, f'VSA|{b}|{acct}', f'{c}|{b}|{acct}', 2000, seed + 7 + len(per))
                            x['lead'] = x['diff'] <= -MARGIN and x['ci_bonf'][1] < 0  # family leads VSA (Bonferroni)
                            per[c] = x
                        oracle = max(ALL10, key=lambda c: comp[c])
                        vl = float(pq(recs, f'VSAL|{b}|{acct}')[0].mean())
                        J['cells'][f'{enc}|{acct}|{M}|{V}|{b}'] = {
                            'nq': int(len(v)), 'EXACT': float(ex.mean()), 'VSA': float(v.mean()), 'VSAL': vl,
                            'BEST': c0['b'], 'BEST_fam': fam, 'diff': c0['diff'], 'ci95': c0['ci95'],
                            'ci_bonf': c0['ci_bonf'], 'label': lab, 'per_fam': per, 'acc': comp, 'oracle_fam': oracle,
                            'sign': list(V36.sign_test(v, pq(recs, f'BEST|{b}|{acct}')[0])[:2])}
                        L.append(f'{M:5d} {V:5d} {b:3d} {100 * ex.mean():6.2f} ' +
                                 ' '.join(f'{100 * comp[c]:5.1f}' for c in ALL10) +
                                 f' {100 * v.mean():5.1f} {100 * vl:5.1f} {fam:>4}{100 * c0["b"]:5.1f} '
                                 f'{100 * c0["diff"]:+7.2f} [{100 * c0["ci95"][0]:+6.2f},{100 * c0["ci95"][1]:+6.2f}] '
                                 f'[{100 * c0["ci_bonf"][0]:+6.2f},{100 * c0["ci_bonf"][1]:+6.2f}] {lab}')
            L.append('')
            for rn, R in REGIONS.items():
                for vs in ['BEST', 'BEST36', 'BESTNL', 'ORACLE'] + ALL10 + ['VSAL']:
                    diffs = []
                    for (M, V, b) in R:
                        recs = data[M, V]
                        v, cl = pq(recs, f'VSA|{b}|{acct}')
                        if vs == 'ORACLE':
                            o = J['cells'][f'{enc}|{acct}|{M}|{V}|{b}']['oracle_fam']
                            x, _ = pq(recs, f'{o}|{b}|{acct}')
                            diffs.append((v - x, cl))
                        elif vs == 'VSAL':
                            x, _ = pq(recs, f'VSAL|{b}|{acct}')
                            diffs.append((x - v, cl))  # VSAL - VSA
                        else:
                            x, _ = pq(recs, f'{vs}|{b}|{acct}')
                            diffs.append((v - x, cl))
                    nb = B if vs in ('BEST', 'BEST36') else 2000
                    stat, lo, hi, _ = V36.boot_region(diffs, nb, 1 + list(REGIONS).index(rn) * 100 + ei * 1000 + ai * 7
                                                      + (['BEST', 'BEST36', 'BESTNL', 'ORACLE'] + ALL10 + ['VSAL'])
                                                      .index(vs))
                    J['regions'][f'{enc}|{acct}|{rn}|{vs}'] = {'diff': stat, 'ci95': [lo, hi]}
                    lab = 'VSAL - VSA' if vs == 'VSAL' else f'VSA - {vs}'
                    L.append(f'  region {rn} {lab:13s}: {100 * stat:+6.2f} [{100 * lo:+6.2f}, {100 * hi:+6.2f}]')
                cs = [J['cells'][f'{enc}|{acct}|{M}|{V}|{b}'] for (M, V, b) in R]
                L.append(f'  region {rn} mean accuracy: VSA {100 * np.mean([c["VSA"] for c in cs]):.2f}  BEST '
                         f'{100 * np.mean([c["BEST"] for c in cs]):.2f}  EXACT {100 * np.mean([c["EXACT"] for c in cs]):.2f}'
                         f'  BEST families: ' + str({f: sum(1 for c in cs if c['BEST_fam'] == f) for f in ALL10
                                                     if any(c['BEST_fam'] == f for c in cs)}))
            L.append('')
            # rates (B36's statistic) and useful accuracy
            L.append(f'## {enc}, {acct}: bits per fact to reach a target (running max; ">" = not within 64); '
                     f'U = 0.9 x EXACT')
            for M in M_GRID:
                for V in V_GRID:
                    recs = data[M, V]
                    ex, _ = pq(recs, 'EXACT')
                    arms = ['VSA', 'BEST', 'BEST36'] + ALL10 + ['VSAL']
                    curves = {a: [float(pq(recs, f'{a}|{b}|{acct}')[0].mean()) for b in BPF] for a in arms}
                    tg = {'lift0.25': 1 / V + 0.25 * (ex.mean() - 1 / V), 'lift0.5': 1 / V + 0.5 * (ex.mean() - 1 / V),
                          'U90': 0.9 * ex.mean()}
                    rr = {t: {a: V36.rate(curves[a], x) for a in arms} for t, x in tg.items()}
                    refs = {r: float(pq(recs, r)[0].mean()) for r in ('VSAINF', 'LAMINF')}
                    J['rates'][f'{enc}|{acct}|{M}|{V}'] = {'targets': tg, 'rates': rr, 'refs': refs,
                                                          'EXACT': float(ex.mean()), 'curves': curves}
                    for t in tg:
                        fewer = [a for a in ALL10 if (rr[t][a][1] != '>' and (rr[t]['VSA'][1] == '>' or
                                                                              rr[t][a][0] < rr[t]['VSA'][0]))]
                        J['rates'][f'{enc}|{acct}|{M}|{V}']['fewer_bits_than_VSA_' + t] = fewer
                        L.append(f'  M={M:5d} V={V:5d} {t:8s} (acc {100 * tg[t]:5.1f}): ' +
                                 ' '.join(f'{a} {rr[t][a][1]}{rr[t][a][0]:.1f}' for a in arms) +
                                 f' | fewer bits than VSA: {",".join(fewer) or "-"}')
                    L.append(f'  M={M:5d} V={V:5d} references: EXACT {100 * ex.mean():.1f}  U {100 * tg["U90"]:.1f}  '
                             f'VSA-inf {100 * refs["VSAINF"]:.1f}  LAM-inf {100 * refs["LAMINF"]:.1f}  VSA at 64 '
                             f'{100 * curves["VSA"][-1]:.1f}  BEST at 64 {100 * curves["BEST"][-1]:.1f}')
            L.append('')
    json.dump(J, open(f'{ROOT}/results/b49_verdict_part_{enc}.json', 'w'))
    open(f'{ROOT}/results/b49_verdict_part_{enc}.txt', 'w').write('\n'.join(L) + '\n')
    print(f'part {enc} written', flush=True)


def panel(B, encs):
    J = {'cells': {}, 'regions': {}, 'rates': {}, 'useful': {}, 'bars': {}, 'item5': {}}
    L = ['# B49 verdict, panel (prereg/B49_panel_mechanism_competitors_prereg.md)',
         '# VSA store vs every dev-chosen competitor at equal bits; BEST = dev-chosen family among IDX SF CM BF FBF SDM',
         '# LAM LHI LHC LHS; BEST36 = B36\'s six; BESTNL = BEST without LAM; ORACLE = test-best of the ten', '']
    for enc in encs:
        if not os.path.exists(f'{ROOT}/results/b49_verdict_part_{enc}.json'):
            panel_one(B, enc)
        P = json.load(open(f'{ROOT}/results/b49_verdict_part_{enc}.json'))
        for k in ('cells', 'regions', 'rates'):
            J[k].update(P[k])
        L += open(f'{ROOT}/results/b49_verdict_part_{enc}.txt').read().splitlines()
    # bars
    L.append('## Registered bars (ideal accounting)')
    passes = {}
    for enc in encs:
        r = J['regions'][f'{enc}|ideal|R1|BEST']
        r36 = J['regions'][f'{enc}|ideal|R1|BEST36']
        rnl = J['regions'][f'{enc}|ideal|R1|BESTNL']
        rr_ = J['regions'][f'{enc}|real|R1|BEST']
        ro = J['regions'][f'{enc}|ideal|R1|ORACLE']
        p = r['diff'] >= MARGIN and r['ci95'][0] > 0
        lo = r['diff'] <= -MARGIN and r['ci95'][1] < 0
        passes[enc] = p
        J['bars'][enc] = {'H1_B49': p, 'loss': lo, 'R1': r, 'R1_BEST36_pass': r36['diff'] >= MARGIN and r36['ci95'][0] > 0,
                          'R1_BESTNL_pass': rnl['diff'] >= MARGIN and rnl['ci95'][0] > 0,
                          'R1_real_pass': rr_['diff'] >= MARGIN and rr_['ci95'][0] > 0,
                          'R1_ORACLE_pass': ro['diff'] >= MARGIN and ro['ci95'][0] > 0}
        f = lambda x: f'{100 * x["diff"]:+.2f} [{100 * x["ci95"][0]:+.2f}, {100 * x["ci95"][1]:+.2f}]'
        L.append(f'  {enc:8s} ({NAMES[enc]}): H1-B49 {"PASS" if p else ("LOSS" if lo else "tie")}  R1 VSA - BEST {f(r)}'
                 f' | vs BEST36 {f(r36)} | vs BESTNL {f(rnl)} | real {f(rr_)} | ORACLE {f(ro)}')
    nl = sum(passes[e] for e in encs if e in LLM)
    ns = sum(passes[e] for e in encs if e not in LLM)
    J['bars']['panel'] = {'pass_total': sum(passes.values()), 'pass_llm': nl, 'pass_sentence': ns,
                          'n_llm': sum(1 for e in encs if e in LLM), 'n_sentence': sum(1 for e in encs if e not in LLM)}
    L.append(f'  PANEL: {sum(passes.values())} of {len(encs)} encoders pass; LLM families {nl} of '
             f'{J["bars"]["panel"]["n_llm"]}; sentence encoders {ns} of {J["bars"]["panel"]["n_sentence"]}')
    # item 5: cells where any competitor family leads the VSA (Bonferroni, per family)
    L.append('')
    L.append('## Item 5: cells (ideal) where a competitor family leads the VSA by >= 2 points, Bonferroni interval < 0')
    for enc in encs:
        lst = []
        for k, c in J['cells'].items():
            e, acct, M, V, b = k.split('|')
            if e != enc or acct != 'ideal':
                continue
            fams = [f for f in ALL10 if c['per_fam'][f]['lead']]
            if fams:
                lst.append(f'M{M} V{V} b{b}: {",".join(fams)}')
        J['item5'][enc] = lst
        r1 = [x for x in lst if x.split()[1] in ('V2', 'V16') and x.split()[2].rstrip(':') in ('b4', 'b8', 'b16')]
        L.append(f'  {enc}: {len(lst)} cells; in R1: {"; ".join(r1) or "none"}')
        L.append(f'      all: {"; ".join(lst) or "none"}')
    # useful accuracy summary
    L.append('')
    L.append('## Item 4: useful accuracy U = 0.9 x EXACT (ideal); bits per fact to reach U (">64" = not within 64)')
    for enc in encs:
        for M in M_GRID:
            for V in (2, 16):
                x = J['rates'][f'{enc}|ideal|{M}|{V}']
                ru = x['rates']['U90']
                J['useful'][f'{enc}|{M}|{V}'] = {'VSA': ru['VSA'], 'BEST': ru['BEST'], 'U': x['targets']['U90'],
                                                'VSAINF': x['refs']['VSAINF'], 'LAMINF': x['refs']['LAMINF'],
                                                'reached': {a: ru[a][1] != '>' for a in ru}}
                L.append(f'  {enc:8s} M={M:5d} V={V:3d}: U {100 * x["targets"]["U90"]:5.1f}  VSA {ru["VSA"][1]}'
                         f'{ru["VSA"][0]:.1f}  BEST {ru["BEST"][1]}{ru["BEST"][0]:.1f}  VSA-inf '
                         f'{100 * x["refs"]["VSAINF"]:5.1f} ({"below U" if x["refs"]["VSAINF"] < x["targets"]["U90"] else "reaches U"})'
                         f'  LAM-inf {100 * x["refs"]["LAMINF"]:5.1f}  families reaching U: '
                         f'{",".join(a for a in ALL10 if ru[a][1] != ">") or "none"}')
    # geometry association
    try:
        from scipy.stats import spearmanr
        g = {**json.load(open(f'{ROOT}/results/b49_geom_existing.json')),
             **json.load(open(f'{ROOT}/results/b49_geom_candidates.json'))}
        g['b46'] = g.get('granite', g.get('b46'))
        m = [J['regions'][f'{e}|ideal|R1|BEST']['diff'] for e in encs]
        rows = {'whitened PR': [g[e]['whiten']['pr'] for e in encs],
                'whitened d\'': [g[e]['whiten']['dprime'] for e in encs],
                'centred PR / n': [g[e]['center']['pr_frac'] for e in encs]}
        J['geom_assoc'] = {}
        L.append('')
        L.append('## Geometry association over the panel (Spearman, R1 VSA - BEST)')
        for k, xs in rows.items():
            rho, pv = spearmanr(xs, m)
            J['geom_assoc'][k] = [float(rho), float(pv)]
            L.append(f'  {k:16s}: rho {rho:+.3f} (p {pv:.3f})')
    except Exception as ex:  # noqa: BLE001
        L.append(f'  geometry association not computed: {ex}')
    json.dump(J, open(f'{ROOT}/results/b49_verdict.json', 'w'), indent=1)
    open(f'{ROOT}/results/b49_verdict.txt', 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L[-60:]))


def mech(B):
    J, L = {}, ['# B49 mechanism verdict (prereg/B49 section 6): R1 margin on transformed codes vs the original codes',
                '# Delta = (VSA - BEST)_T - (VSA - BEST)_original, paired on the same queries; BEST = all ten families', '']
    R1 = REGIONS['R1']
    for base, T in (('qwen8', 'T1'), ('e5', 'T2'), ('qwen8', 'T3'), ('b46', 'T1'), ('bge', 'T2'), ('b46', 'T3')):
        d = f'{ROOT}/b49_out/fam/{base}_{T}/test'
        if not os.path.exists(f'{d}/all49_{base}_{T}_M1024_V16.jsonl.gz'):
            L.append(f'{base} {T}: not run')
            continue
        dT = load(f'{base}_{T}', (2, 16), d)
        d0 = load(base, (2, 16))
        for vs in ('BEST', 'BEST36'):
            dl, dt, d00 = [], [], []
            for (M, V, b) in R1:
                a, b_ = dT[M, V], d0[M, V]
                assert [r['qgold'] for r in a] == [r['qgold'] for r in b_] and \
                    [r['qalias'] for r in a] == [r['qalias'] for r in b_]
                vt, cl = pq(a, f'VSA|{b}|ideal')
                vt_b = pq(a, f'{vs}|{b}|ideal')[0]
                v0, _ = pq(b_, f'VSA|{b}|ideal')
                v0_b = pq(b_, f'{vs}|{b}|ideal')[0]
                dl.append(((vt - vt_b) - (v0 - v0_b), cl))
                dt.append((vt - vt_b, cl))
                d00.append((v0 - v0_b, cl))
            s, lo, hi, _ = V36.boot_region(dl, B, 4949)
            st_, lt, ht, _ = V36.boot_region(dt, 2000, 4950)
            s0, l0, h0, _ = V36.boot_region(d00, 2000, 4951)
            changed = abs(s) >= MARGIN and (lo > 0 or hi < 0)
            J[f'{base}|{T}|{vs}'] = {'delta': s, 'ci95': [lo, hi], 'margin_T': [st_, lt, ht], 'margin_0': [s0, l0, h0],
                                     'changed': changed}
            L.append(f'{base} {T} vs {vs}: margin original {100 * s0:+.2f} [{100 * l0:+.2f}, {100 * h0:+.2f}] -> '
                     f'transformed {100 * st_:+.2f} [{100 * lt:+.2f}, {100 * ht:+.2f}]; Delta {100 * s:+.2f} '
                     f'[{100 * lo:+.2f}, {100 * hi:+.2f}] ({"changed" if changed else "not changed by the bar"})')
        # accuracies
        for lab, dd in (('original', d0), ('transformed', dT)):
            accs = {a: 100 * np.mean([pq(dd[M, V], f'{a}|{b}|ideal')[0].mean() for (M, V, b) in R1])
                    for a in ['VSA', 'BEST'] + ALL10}
            ex = 100 * np.mean([pq(dd[M, V], 'EXACT')[0].mean() for (M, V, b) in R1])
            L.append(f'   {lab:11s} R1 mean accuracy: EXACT {ex:.1f} ' + ' '.join(f'{a} {x:.1f}' for a, x in accs.items()))
        L.append('')
    # readings
    def get(k):
        return J.get(k)
    t1, t2, t3 = get('qwen8|T1|BEST'), get('e5|T2|BEST'), get('qwen8|T3|BEST')
    if t1 and t2 and t3:
        hd = (t1['delta'] > -MARGIN) and (t2['delta'] < MARGIN) and (t3['delta'] <= -MARGIN and t3['ci95'][1] < 0)
        ha = (t1['delta'] <= -MARGIN and t1['ci95'][1] < 0) and (t2['delta'] >= MARGIN and t2['ci95'][0] > 0)
        J['reading'] = 'HD supported' if hd else ('HA supported' if ha else 'neither account explains the interventions')
        L.append(f'READING (primary runs qwen8 T1, e5 T2, qwen8 T3): {J["reading"]}')
    json.dump(J, open(f'{ROOT}/results/b49_mech_verdict.json', 'w'), indent=1)
    open(f'{ROOT}/results/b49_mech_verdict.txt', 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    a = sys.argv[1:]
    B = int(a[a.index('--boot') + 1]) if '--boot' in a else 10000
    encs = a[a.index('--encs') + 1].split(',') if '--encs' in a else PANEL
    if a[0] == 'panel':
        panel(B, encs)
    elif a[0] == 'panel1':
        panel_one(B, a[1])
    else:
        mech(B)
