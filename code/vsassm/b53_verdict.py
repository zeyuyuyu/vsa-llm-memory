"""B53 verdict (prereg/B53_value_codes_prereg.md): the store with antipodal (V = 2) / orthogonal (V = 16) value codes
against LAM, BEST and B49's random-code store (VSAR), on B49's merged test files with the VSA replaced
(b53_out/fam/{enc}/all53_*). Uses b36_verdict's per_query / boot_region (B49's statistics). V in {128, 1024} cells (and
R2) are B49's, unchanged: their numbers are taken from results/b49_verdict.json.

python3 b53_verdict.py part ENC      # -> results/b53_part_{enc}.json
python3 b53_verdict.py panel         # parts -> results/b53_verdict.{json,txt}
"""
import gzip
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import b36_verdict as V36  # noqa: E402

FAM = os.environ.get('B53_FAM', f'{ROOT}/b53_out/fam')
RES = os.environ.get('B53_RES', f'{ROOT}/results')
PANEL = ['qwen8', 'b46', 'llama', 'mistral', 'e5', 'bge', 'minilm']
LLM = PANEL[:4]
NAMES = {'qwen8': 'Qwen3-8B L9 mean', 'b46': 'Granite-3.3-8B L30 mean', 'llama': 'Llama-3.1-8B L24 mean',
         'mistral': 'Mistral-7B L24 PromptEOL', 'e5': 'e5-base-v2', 'bge': 'bge-small-en-v1.5',
         'minilm': 'all-MiniLM-L6-v2'}
M_GRID, VN, BPF, ACCTS = [256, 1024], [2, 16], [1, 2, 4, 8, 16, 32, 64], ['ideal', 'real']
ALL10 = ['IDX', 'SF', 'CM', 'BF', 'FBF', 'SDM', 'LAM', 'LHI', 'LHC', 'LHS']
REG = {}
for nm, bs in (('R1', (4, 8, 16)), ('R0', (1, 2)), ('RH', (32, 64))):
    REG[nm] = [(M, V, b) for M in M_GRID for V in VN for b in bs]
    for V in VN:
        REG[f'{nm}-V{V}'] = [(M, V, b) for M in M_GRID for b in bs]
COMP = ['LAM', 'BEST', 'VSAR', 'BEST36', 'BESTNL', 'ORACLE']
NCELL = 28
MARGIN = 0.02


def load(enc):
    return {(M, V): [json.loads(l) for l in gzip.open(f'{FAM}/{enc}/all53_{enc}_M{M}_V{V}.jsonl.gz', 'rt')]
            for M in M_GRID for V in VN}


def part(enc):
    ei = PANEL.index(enc)
    data = load(enc)
    J = {'cells': {}, 'regions': {}}
    pqc = {}

    def pq(M, V, arm):
        if (M, V, arm) not in pqc:
            pqc[M, V, arm] = V36.per_query(data[M, V], arm)
        return pqc[M, V, arm]
    for ai, acct in enumerate(ACCTS):
        for M in M_GRID:
            for V in VN:
                ex = pq(M, V, 'EXACT')[0].mean()
                for b in BPF:
                    v, cl = pq(M, V, f'VSA|{b}|{acct}')
                    accs = {a: float(pq(M, V, f'{a}|{b}|{acct}')[0].mean()) for a in ALL10 + ['VSAR', 'BEST', 'BEST36',
                                                                                         'BESTNL']}
                    orc = max(ALL10, key=lambda c: accs[c])
                    c = {'VSA': float(v.mean()), 'EXACT': float(ex), 'acc': accs, 'oracle_fam': orc,
                         'BEST_fam': data[M, V][0]['cfg'][f'BEST|{b}|{acct}'][0],
                         'VSA_cfg': data[M, V][0]['cfg'][f'VSA|{b}|{acct}'], 'cmp': {}}
                    for k, x in enumerate(('LAM', 'BEST', 'VSAR')):
                        y, _ = pq(M, V, f'{x}|{b}|{acct}')
                        s, lo, hi, dr = V36.boot_region([(v - y, cl)], 2000, 53 + M * 7 + V * 13 + b * 17 + ei * 100000
                                                        + ai * 50000 + k)
                        al = 0.05 / NCELL
                        c['cmp'][x] = {'diff': s, 'ci95': [lo, hi], 'ci_bonf': [float(np.quantile(dr, al / 2)),
                                                                               float(np.quantile(dr, 1 - al / 2))]}
                    J['cells'][f'{enc}|{acct}|{M}|{V}|{b}'] = c
        for ri, (rn, R) in enumerate(REG.items()):
            for k, x in enumerate(COMP):
                diffs = []
                for (M, V, b) in R:
                    v, cl = pq(M, V, f'VSA|{b}|{acct}')
                    arm = J['cells'][f'{enc}|{acct}|{M}|{V}|{b}']['oracle_fam'] if x == 'ORACLE' else x
                    diffs.append((v - pq(M, V, f'{arm}|{b}|{acct}')[0], cl))
                nb = 10000 if (x in ('LAM', 'BEST') and acct == 'ideal') else 2000
                s, lo, hi, _ = V36.boot_region(diffs, nb, 5300 + ri * 100 + k + ei * 1000 + ai * 7)
                J['regions'][f'{enc}|{acct}|{rn}|{x}'] = {'diff': s, 'ci95': [lo, hi]}
        # half-the-coordinates check (P2) and useful accuracy at V 2 / 16 (point estimates)
        for M in M_GRID:
            for V in VN:
                ex = pq(M, V, 'EXACT')[0].mean()
                cur = [float(pq(M, V, f'VSA|{b}|{acct}')[0].mean()) for b in BPF]
                curR = [float(pq(M, V, f'VSAR|{b}|{acct}')[0].mean()) for b in BPF]
                J.setdefault('curves', {})[f'{enc}|{acct}|{M}|{V}'] = {'VSA': cur, 'VSAR': curR, 'EXACT': float(ex),
                                                                     'U90_VSA': V36.rate(cur, 0.9 * ex),
                                                                     'U90_VSAR': V36.rate(curR, 0.9 * ex)}
    json.dump(J, open(f'{RES}/b53_part_{enc}.json', 'w'))
    print('part', enc, flush=True)


def f(r):
    return f'{100 * r["diff"]:+6.2f} [{100 * r["ci95"][0]:+6.2f}, {100 * r["ci95"][1]:+6.2f}]'


def lab(r):
    return 'VSA ahead' if r['ci95'][0] > 0 else ('LAM ahead' if r['ci95'][1] < 0 else 'level')


def tuning():
    import b36_run as R36
    import b32_run as R32
    g = R32.GRID
    out = {'VSA': len(g['prep']) * len(g['fmap']) * len(g['q']) * len(g['f']),
           'IDX': len(g['prep']) * len(g['family']) * len(g['f'])}
    for fam in R36.FAMS:
        out[fam] = len(R36.PREPS) * len(R36.fam_grid(fam, 16))
    for enc in PANEL:
        ch = json.load(open(f'{ROOT}/results/b49_choice_{enc}.json'))['choice']
        for fam in ('LAM', 'LHI', 'LHC', 'LHS'):
            ns = {ch[f'{M}|{V}|{b}|ideal|{fam}']['n_cfg'] for M in M_GRID for V in VN for b in BPF}
            assert len(ns) == 1, (enc, fam, ns)
            n = ns.pop()
            assert out.setdefault(fam, n) == n, (enc, fam, n)
    out['BEST'] = sum(out[x] for x in ALL10)
    return out


def panel():
    J = {'cells': {}, 'regions': {}, 'curves': {}}
    for enc in PANEL:
        P = json.load(open(f'{RES}/b53_part_{enc}.json'))
        for k in J:
            J[k].update(P[k])
    J49 = json.load(open(f'{ROOT}/results/b49_verdict.json'))
    L = ['# B53 verdict (prereg/B53_value_codes_prereg.md): the store with antipodal (V = 2) / orthogonal (V = 16) value',
         '# codes; competitors, dev choices and test predictions are B49\'s. VSAR = B49\'s random-code store. Points;',
         '# 95% cluster bootstrap (entity within store); cells Bonferroni over the 28 re-run cells per encoder/accounting.',
         '# V in {128, 1024} (incl. R2) unchanged: random codes, B49\'s numbers.', '']
    T = tuning()
    J['tuning'] = T
    L.append('## Tuning-grid size per arm (configurations searched on dev per cell, V 2 / 16)')
    L.append('  ' + ', '.join(f'{k} {v}' for k, v in T.items()) + '  (LHI/LHC/LHS also choose a learned map among 16 '
             'per encoder; BEST = best of the ten families\' chosen configurations)')
    L.append('')
    for enc in PANEL:
        L.append(f'## {enc} ({NAMES[enc]}), ideal: cells (V 2 / 16)')
        L.append(f'{"M":>5} {"V":>3} {"bpf":>3} {"EXACT":>6} {"VSAR":>5} {"VSA":>5} {"LAM":>5} {"BEST":>9} '
                 f'{"VSA-VSAR":>8} {"VSA - LAM [95%] [Bonf]":>40} {"VSA - BEST [95%] [Bonf]":>40}  VSA cfg')
        for M in M_GRID:
            for V in VN:
                for b in BPF:
                    c = J['cells'][f'{enc}|ideal|{M}|{V}|{b}']
                    a = c['acc']
                    g = lambda x: (f'{100 * c["cmp"][x]["diff"]:+6.2f} [{100 * c["cmp"][x]["ci95"][0]:+6.2f},'
                                   f'{100 * c["cmp"][x]["ci95"][1]:+6.2f}] [{100 * c["cmp"][x]["ci_bonf"][0]:+6.2f},'
                                   f'{100 * c["cmp"][x]["ci_bonf"][1]:+6.2f}]')
                    L.append(f'{M:5d} {V:3d} {b:3d} {100 * c["EXACT"]:6.2f} {100 * a["VSAR"]:5.1f} {100 * c["VSA"]:5.1f} '
                             f'{100 * a["LAM"]:5.1f} {c["BEST_fam"]:>4}{100 * a["BEST"]:5.1f} '
                             f'{100 * (c["VSA"] - a["VSAR"]):+8.2f} {g("LAM"):>40} {g("BEST"):>40}  {c["VSA_cfg"]}')
        L.append('')
        for acct in ACCTS:
            for rn in REG:
                L.append(f'  {acct:5s} {rn:7s}: ' + ' | '.join(f'VSA - {x} {f(J["regions"][f"{enc}|{acct}|{rn}|{x}"])}'
                                                              for x in ('LAM', 'BEST', 'VSAR')))
            for x in ('LAM', 'BEST'):
                r = J49['regions'][f'{enc}|{acct}|R2|{x}']
                J['regions'][f'{enc}|{acct}|R2|{x}'] = r
                L.append(f'  {acct:5s} R2     : VSA - {x} {f(r)} (B49, unchanged)')
            L.append(f'  {acct:5s} R1 secondary: VSA - BEST36 {f(J["regions"][f"{enc}|{acct}|R1|BEST36"])} | VSA - '
                     f'BESTNL {f(J["regions"][f"{enc}|{acct}|R1|BESTNL"])} | VSA - ORACLE '
                     f'{f(J["regions"][f"{enc}|{acct}|R1|ORACLE"])}')
        L.append('')
    # bars
    L.append('## Primary bar: R1 VSA - LAM (ideal); label by the 95% interval; B49 for comparison')
    cnt = {k: {'all': 0, 'llm': 0, 'sent': 0} for k in ('ge', 'ahead')}
    J['bars'] = {}
    for enc in PANEL:
        r = J['regions'][f'{enc}|ideal|R1|LAM']
        r49 = J49['regions'][f'{enc}|ideal|R1|LAM']
        lb = lab(r)
        pm = 'VSA WIN (+2)' if r['diff'] >= MARGIN and r['ci95'][0] > 0 else (
            'LAM WIN (-2)' if r['diff'] <= -MARGIN and r['ci95'][1] < 0 else 'tie (+-2 rule)')
        J['bars'][enc] = {'R1_LAM': r, 'label': lb, 'pm2': pm, 'R1_LAM_B49': r49,
                          'V2': lab(J['regions'][f'{enc}|ideal|R1-V2|LAM']),
                          'V16': lab(J['regions'][f'{enc}|ideal|R1-V16|LAM'])}
        for k, ok in (('ge', lb != 'LAM ahead'), ('ahead', lb == 'VSA ahead')):
            if ok:
                cnt[k]['all'] += 1
                cnt[k]['llm' if enc in LLM else 'sent'] += 1
        L.append(f'  {enc:8s}: R1 {f(r)} {lb:9s} ({pm}); B49 {f(r49)} | R1-V2 '
                 f'{f(J["regions"][f"{enc}|ideal|R1-V2|LAM"])} {J["bars"][enc]["V2"]} | R1-V16 '
                 f'{f(J["regions"][f"{enc}|ideal|R1-V16|LAM"])} {J["bars"][enc]["V16"]}')
    J['bars']['count'] = cnt
    L.append(f'  PANEL: VSA >= LAM on {cnt["ge"]["all"]} of 7 (LLM {cnt["ge"]["llm"]} of 4, sentence '
             f'{cnt["ge"]["sent"]} of 3); VSA ahead on {cnt["ahead"]["all"]} of 7 (LLM {cnt["ahead"]["llm"]}, sentence '
             f'{cnt["ahead"]["sent"]})')
    L.append('')
    L.append('## Secondary: R1 VSA - BEST (H1-B49 bar), B49 for comparison')
    npass = 0
    for enc in PANEL:
        r = J['regions'][f'{enc}|ideal|R1|BEST']
        p = r['diff'] >= MARGIN and r['ci95'][0] > 0
        lo = r['diff'] <= -MARGIN and r['ci95'][1] < 0
        npass += p
        J['bars'][enc]['H1_B49'] = 'PASS' if p else ('LOSS' if lo else 'tie')
        L.append(f'  {enc:8s}: R1 {f(r)} H1-B49 {J["bars"][enc]["H1_B49"]}; B49 {f(J49["regions"][f"{enc}|ideal|R1|BEST"])}'
                 f' | R1-V2 {f(J["regions"][f"{enc}|ideal|R1-V2|BEST"])} | R1-V16 '
                 f'{f(J["regions"][f"{enc}|ideal|R1-V16|BEST"])}')
    L.append(f'  PANEL: H1-B49 passes on {npass} of 7')
    L.append('')
    # useful accuracy
    L.append('## Useful accuracy U = 0.9 x EXACT (ideal), bits per fact to reach U on the 1-64 grid (VSA B53 / VSAR B49)')
    for enc in PANEL:
        s = []
        for M in M_GRID:
            for V in VN:
                c = J['curves'][f'{enc}|ideal|{M}|{V}']
                s.append(f'M{M} V{V}: {c["U90_VSA"][1]}{c["U90_VSA"][0]:.1f} / {c["U90_VSAR"][1]}{c["U90_VSAR"][0]:.1f}')
        L.append(f'  {enc:8s}: ' + '; '.join(s))
    L.append('')
    # predictions
    L.append('## Predictions (prereg section "Predictions")')
    R = J['regions']
    g2 = [R[f'{e}|ideal|R1-V2|VSAR'] for e in PANEL]
    p1 = all(x['ci95'][0] > 0 for x in g2) and 0.015 <= np.mean([x['diff'] for x in g2]) <= 0.05
    half = {}
    for e in PANEL:
        dd = []
        for M in M_GRID:
            c = J['curves'][f'{e}|ideal|{M}|2']
            for b in (4, 8, 16):
                dd.append(c['VSA'][BPF.index(b)] - c['VSAR'][BPF.index(2 * b)])
        half[e] = float(np.mean(dd))
    J['half'] = half
    p2 = sum(abs(x) <= 0.02 for x in half.values()) >= 5
    p3 = all(abs(R[f'{e}|ideal|R1-V16|VSAR']['diff']) <= 0.015 for e in PANEL)
    p4a = all(R[f'{e}|ideal|R1|LAM']['diff'] > J49['regions'][f'{e}|ideal|R1|LAM']['diff'] for e in PANEL)
    p4b = sum(R[f'{e}|ideal|R1-V2|LAM']['ci95'][1] < 0 for e in PANEL) >= 5
    p4c = cnt['ge']['all'] <= 3
    p5 = npass == 0 and sum(R[f'{e}|ideal|R1|BEST']['ci95'][1] < 0 for e in PANEL) >= 6
    p6 = all(R[f'{e}|ideal|R1-V16|LAM']['ci95'][0] > 0 for e in LLM) and \
        not any(R[f'{e}|ideal|R1-V16|LAM']['ci95'][0] > 0 for e in PANEL if e not in LLM)
    p7 = all(R[f'{e}|ideal|R0-V2|VSAR']['diff'] > 0 for e in PANEL)
    P = {'P1': (p1, f'R1-V2 VSA - VSAR: ' + ', '.join(f'{e} {100 * x["diff"]:+.2f}' for e, x in zip(PANEL, g2)) +
                f'; mean {100 * np.mean([x["diff"] for x in g2]):+.2f}'),
         'P2': (p2, 'VSA(b) - VSAR(2b) over R1-V2: ' + ', '.join(f'{e} {100 * x:+.2f}' for e, x in half.items())),
         'P3': (p3, 'R1-V16 VSA - VSAR: ' + ', '.join(f'{e} {100 * R[f"{e}|ideal|R1-V16|VSAR"]["diff"]:+.2f}'
                                                      for e in PANEL)),
         'P4a': (p4a, 'R1 VSA - LAM rises on every encoder'), 'P4b': (p4b, 'LAM ahead over R1-V2 on >= 5 of 7'),
         'P4c': (p4c, f'VSA >= LAM over R1 on at most 3 of 7 (observed {cnt["ge"]["all"]})'),
         'P5': (p5, f'H1-B49 0 of 7 and BEST ahead on >= 6 of 7 (passes {npass})'),
         'P6': (p6, 'R1-V16: VSA ahead of LAM on all 4 LLMs, on no sentence encoder'),
         'P7': (p7, 'R0-V2 VSA - VSAR > 0 on all 7')}
    J['predictions'] = {k: [bool(v[0]), v[1]] for k, v in P.items()}
    for k, (ok, s) in P.items():
        L.append(f'  {k}: {"HELD" if ok else "FAILED"} - {s}')
    json.dump(J, open(f'{RES}/b53_verdict.json', 'w'), indent=1)
    open(f'{RES}/b53_verdict.txt', 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L[-40:]))


if __name__ == '__main__':
    a = sys.argv[1:]
    if a[0] == 'part':
        part(a[1])
    else:
        panel()
