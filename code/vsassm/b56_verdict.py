"""B56 verdict (prereg/B56_decoders_retraining_prereg.md) on b56_out/fam/{enc}/all56_* (B53's merged test records +
the B56 arms). Statistics are B49 / B53's: per-query correctness, cluster bootstrap with entity-within-store clusters
resampled within each cell, region statistic = mean over cells of the cell mean (b36_verdict.boot_region).

python3 b56_verdict.py part ENC      # -> results/b56_part_{enc}.json
python3 b56_verdict.py panel         # -> results/b56_verdict.{json,txt}
"""
import gzip
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import b36_verdict as V36  # noqa: E402

FAM = f'{ROOT}/b56_out/fam'
RES = f'{ROOT}/results'
PANEL = ['qwen8', 'b46', 'llama', 'mistral', 'e5', 'bge', 'minilm']
LLM = PANEL[:4]
NAMES = {'qwen8': 'Qwen3-8B', 'b46': 'Granite-3.3-8B', 'llama': 'Llama-3.1-8B', 'mistral': 'Mistral-7B',
         'e5': 'e5-base-v2', 'bge': 'bge-small', 'minilm': 'MiniLM-L6'}
M_GRID, VN, BPF, ACCTS = [256, 1024], [2, 16], [1, 2, 4, 8, 16, 32, 64], ['ideal', 'real']
REG = {}
for nm, bs in (('R1', (4, 8, 16)), ('R0', (1, 2)), ('RH', (32, 64))):
    REG[nm] = [(M, V, b) for M in M_GRID for V in VN for b in bs]
    for V in VN:
        REG[f'{nm}-V{V}'] = [(M, V, b) for M in M_GRID for b in bs]
PAIRS = [('VSAS', 'LAM'), ('VSAS', 'BEST'), ('VSAS', 'LAMRF'), ('VSAS', 'VSA'), ('VSAL', 'VSA'), ('VSAT', 'VSA'),
         ('VSAW', 'VSA'), ('VSAX', 'LAM'), ('VSAX', 'BEST'), ('LAMRF', 'LAM'), ('VSA', 'LAMRF'), ('VSA', 'LAM'),
         ('VSAL', 'LAM'), ('VSAT', 'LAM')]
ARMS = ['EXACT', 'VSA', 'VSAL', 'VSAT', 'VSAS', 'VSAW', 'VSAX', 'LAMRF', 'LAM', 'BEST']
MARGIN = 0.02


def load(enc):
    return {(M, V): [json.loads(l) for l in gzip.open(f'{FAM}/{enc}/all56_{enc}_M{M}_V{V}.jsonl.gz', 'rt')]
            for M in M_GRID for V in VN}


def summ(s, dr):
    return {'diff': s, 'ci95': [float(np.quantile(dr, 0.025)), float(np.quantile(dr, 0.975))],
            'ci90': [float(np.quantile(dr, 0.05)), float(np.quantile(dr, 0.95))]}


def part(enc):
    ei = PANEL.index(enc)
    data = load(enc)
    J = {'cells': {}, 'regions': {}, 'cn_regions': {}}
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
                    J['cells'][f'{enc}|{acct}|{M}|{V}|{b}'] = {x: float(pq(M, V, key(x, b, acct))[0].mean())
                                                               for x in ARMS}
        for ri, (rn, R) in enumerate(REG.items()):
            for k, (x, y) in enumerate(PAIRS):
                diffs, cdiffs = [], []
                for (M, V, b) in R:
                    v, cl = pq(M, V, key(x, b, acct))
                    w = pq(M, V, key(y, b, acct))[0]
                    diffs.append((v - w, cl))
                    cdiffs.append(((v - w) / (1 - 1 / V), cl))
                prim = (x, y) == ('VSAS', 'LAM') and acct == 'ideal'
                nb = 10000 if (prim or (x, y) == ('VSAS', 'BEST')) and acct == 'ideal' else 2000
                s, _, _, dr = V36.boot_region(diffs, nb, 5600 + ri * 100 + k + ei * 1000 + ai * 7)
                J['regions'][f'{enc}|{acct}|{rn}|{x}-{y}'] = summ(s, dr)
                if acct == 'ideal' and (x, y) in (('VSAS', 'LAM'), ('VSAS', 'BEST'), ('VSAS', 'LAMRF'),
                                                  ('VSA', 'LAM')) and rn.startswith('R1'):
                    s, _, _, dr = V36.boot_region(cdiffs, 2000, 9600 + ri * 100 + k + ei * 1000)
                    J['cn_regions'][f'{enc}|{acct}|{rn}|{x}-{y}'] = summ(s, dr)
    # dev choices of the B56 arms (counts over cells, ideal accounting)
    ch = json.load(open(f'{RES}/b56_choice_{enc}.json'))['choice']
    cnt = {}
    for M in M_GRID:
        for V in VN:
            for b in BPF:
                reg = 'R1' if b in (4, 8, 16) else ('R0' if b <= 2 else 'RH')
                c = ch
                for arm in ('LM', 'RT', 'LR'):
                    cnt.setdefault(f'{reg}|{arm}', []).append(c[f'{M}|{V}|{b}|ideal|{arm}']['cfg'])
                cnt.setdefault(f'{reg}|W', []).append(c[f'{M}|{V}|{b}|ideal|W']['cfg'])
                cnt.setdefault(f'{reg}|STAR', []).append(c[f'{M}|{V}|{b}|ideal|STAR']['fam'])
                cnt.setdefault(f'{reg}|ANY', []).append(c[f'{M}|{V}|{b}|ideal|ANY']['fam'])
    J['choices'] = cnt
    json.dump(J, open(f'{RES}/b56_part_{enc}.json', 'w'))
    print('part', enc, flush=True)


def f(r, k='ci95'):
    return f'{100 * r["diff"]:+6.2f} [{100 * r[k][0]:+6.2f}, {100 * r[k][1]:+6.2f}]'


def labels(r):
    lo, hi = r['ci95']
    nb = lo >= -MARGIN
    tost = r['ci90'][0] > -MARGIN and r['ci90'][1] < MARGIN
    ahead = lo > 0
    behind = hi < 0
    return {'not_behind': nb, 'tost': tost, 'ahead': ahead, 'behind': behind}


def panel():
    J = {'cells': {}, 'regions': {}, 'cn_regions': {}, 'choices': {}}
    for enc in PANEL:
        P = json.load(open(f'{RES}/b56_part_{enc}.json'))
        for k in ('cells', 'regions', 'cn_regions'):
            J[k].update(P[k])
        J['choices'][enc] = P['choices']
    R = J['regions']
    L = ['# B56 verdict (prereg/B56_decoders_retraining_prereg.md). Points; 95% cluster bootstrap (90% for TOST);',
         '# VSA = B53 store (matched filter); VSAL = LMMSE read (training-split statistics); VSAT = retrained bundle;',
         '# VSAS = per cell the dev-selected of VSAL / VSAT (primary store arm); VSAW = widened grid (sparse ternary maps);',
         '# VSAX = dev-best of VSAL / VSAT / VSAW; LAMRF = LAM on the store\'s own features; LAM, BEST = B49\'s.', '']
    L.append('## Primary: R1 VSAS - LAM (ideal); not behind = 95% lower bound >= -2; TOST = 90% interval inside +-2')
    bars = {}
    for enc in PANEL:
        r = R[f'{enc}|ideal|R1|VSAS-LAM']
        lb = labels(r)
        bars[enc] = lb
        L.append(f'  {NAMES[enc]:15s} {f(r)}  90% [{100 * r["ci90"][0]:+.2f}, {100 * r["ci90"][1]:+.2f}]  '
                 f'not_behind={lb["not_behind"]} TOST={lb["tost"]} ahead={lb["ahead"]} LAM_ahead={lb["behind"]}')
    for nm, grp in (('panel', PANEL), ('LLM', LLM), ('sentence', PANEL[4:])):
        L.append(f'  {nm}: not behind {sum(bars[e]["not_behind"] for e in grp)} of {len(grp)}; TOST-equivalent '
                 f'{sum(bars[e]["tost"] for e in grp)}; ahead {sum(bars[e]["ahead"] for e in grp)}; LAM ahead (CI below 0) '
                 f'{sum(bars[e]["behind"] for e in grp)}')
    J['bars'] = bars
    L.append('')
    J49 = json.load(open(f'{RES}/b49_verdict.json'))
    for y in ('LAM', 'BEST'):
        L.append(f'## Compact (ideal): VSAS - {y} by region; R2 (V 1,024) = B49 store, unchanged')
        L.append('| encoder | R1 | R1 V=2 | R1 V=16 | R0 | R0 V=2 | R0 V=16 | RH | RH V=2 | RH V=16 | R2 |')
        L.append('| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |')
        for enc in PANEL:
            row = [f(R[f'{enc}|ideal|{rn}|VSAS-{y}']) for rn in ('R1', 'R1-V2', 'R1-V16', 'R0', 'R0-V2', 'R0-V16', 'RH',
                                                               'RH-V2', 'RH-V16')]
            row.append(f(J49['regions'][f'{enc}|ideal|R2|{y}']))
            L.append(f'| {NAMES[enc]} | ' + ' | '.join(row) + ' |')
        L.append('')
    L.append('## Compact (ideal) R1 component effects')
    L.append('| encoder | VSAL - VSA | VSAT - VSA | VSAW - VSA | VSAS - VSA | LAMRF - LAM | VSA - LAMRF | VSAS - LAMRF | VSAX - LAM |'
             ' VSAX - BEST |')
    L.append('| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |')
    for enc in PANEL:
        L.append(f'| {NAMES[enc]} | ' + ' | '.join(f(R[f'{enc}|ideal|R1|{h}']) for h in (
            'VSAL-VSA', 'VSAT-VSA', 'VSAW-VSA', 'VSAS-VSA', 'LAMRF-LAM', 'VSA-LAMRF', 'VSAS-LAMRF', 'VSAX-LAM',
            'VSAX-BEST')) + ' |')
    L.append('')
    for acct in ACCTS:
        for rn in REG:
            L.append(f'## {acct} {rn}')
            hdr = ['VSAS-LAM', 'VSAS-BEST', 'VSAS-LAMRF', 'VSAS-VSA', 'VSAL-VSA', 'VSAT-VSA', 'VSAW-VSA', 'VSAX-LAM',
                   'VSAX-BEST', 'LAMRF-LAM', 'VSA-LAMRF', 'VSA-LAM']
            for enc in PANEL:
                L.append(f'  {NAMES[enc]:15s} ' + ' | '.join(f'{h} {f(R[f"{enc}|{acct}|{rn}|{h}"])}' for h in hdr))
        L.append('')
    L.append('## Chance-normalised R1 differences (per-cell difference / (1 - 1/V)), ideal')
    for enc in PANEL:
        L.append(f'  {NAMES[enc]:15s} ' + ' | '.join(f'{rn} {h} {f(J["cn_regions"][f"{enc}|ideal|{rn}|{h}"])}'
                                                    for rn in ('R1', 'R1-V2', 'R1-V16') for h in ('VSAS-LAM', 'VSAS-BEST',
                                                                                                'VSAS-LAMRF', 'VSA-LAM')))
    L.append('')
    L.append('## R1 mean accuracy by V (ideal), raw and chance-normalised (acc - 1/V) / (1 - 1/V)')
    for enc in PANEL:
        for V in VN:
            cells = [J['cells'][f'{enc}|ideal|{M}|{V}|{b}'] for M in M_GRID for b in (4, 8, 16)]
            m = {x: float(np.mean([c[x] for c in cells])) for x in ARMS}
            L.append(f'  {NAMES[enc]:15s} V={V:2d} ' + ' '.join(f'{x} {100 * m[x]:.1f}' for x in ARMS) + ' || cn ' +
                     ' '.join(f'{x} {100 * (m[x] - 1 / V) / (1 - 1 / V):.1f}' for x in ARMS))
    L.append('')
    L.append('## Dev choices (ideal; counts over the cells of the region)')
    for enc in PANEL:
        C_ = J['choices'][enc]
        for rn in ('R0', 'R1', 'RH'):
            lm = C_[f'{rn}|LM']
            rt = C_[f'{rn}|RT']
            w = C_[f'{rn}|W']
            L.append(f'  {NAMES[enc]:15s} {rn}: LM alpha=inf {sum(1 for c in lm if c[1] == float("inf"))}/{len(lm)} '
                     f'(alphas {sorted(str(c[1]) for c in lm)}); RT epochs {sorted(c[1] for c in rt)}; '
                     f'W sparse {sum(1 for c in w if c[1].startswith("tern"))}/{len(w)}; STAR RT '
                     f'{C_[f"{rn}|STAR"].count("RT")}/{len(lm)}; ANY {sorted(C_[f"{rn}|ANY"])}')
    json.dump(J, open(f'{RES}/b56_verdict.json', 'w'), indent=1)
    open(f'{RES}/b56_verdict.txt', 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L[:20]))




def preds():
    """The registered predictions P1-P7 against results/b56_verdict.json (P8 = the checks, b56_check.py)."""
    J = json.load(open(f'{RES}/b56_verdict.json'))
    R = J['regions']

    def g(e, rn, h):
        return R[f'{e}|ideal|{rn}|{h}']
    out = []
    v = [abs(g(e, 'R1', 'VSAL-VSA')['diff']) <= 0.01 for e in PANEL]
    out.append(('P1', sum(v) >= 6, f'|R1 VSAL - VSA| <= 1 on {sum(v)} of 7: ' + ', '.join(
        f'{e} {100 * g(e, "R1", "VSAL-VSA")["diff"]:+.2f}' for e in PANEL)))
    v = [g(e, 'R1-V2', 'VSAT-VSA')['ci95'][0] > 0 for e in PANEL]
    out.append(('P2', sum(v) >= 5, f'R1-V2 VSAT - VSA interval above 0 on {sum(v)} of 7: ' + ', '.join(
        f'{e} {100 * g(e, "R1-V2", "VSAT-VSA")["diff"]:+.2f}' for e in PANEL)))
    b = J['bars']
    nb, to, ah = (sum(b[e][k] for e in PANEL) for k in ('not_behind', 'tost', 'ahead'))
    out.append(('P3', nb >= 6 and to >= 4 and ah <= 4, f'not behind {nb}, TOST {to}, ahead {ah}'))
    v = [g(e, 'R1', 'LAMRF-LAM')['ci95'][1] < 0 for e in PANEL]
    out.append(('P4', sum(v) >= 4, f'R1 LAMRF - LAM interval below 0 on {sum(v)} of 7: ' + ', '.join(
        f'{e} {100 * g(e, "R1", "LAMRF-LAM")["diff"]:+.2f}' for e in PANEL)))
    sp = [sum(1 for c in J['choices'][e]['R1|W'] if c[1].startswith('tern')) for e in PANEL]
    v = [abs(g(e, 'R1', 'VSAW-VSA')['diff']) <= 0.01 for e in PANEL]
    out.append(('P5', all(x <= 4 for x in sp) and sum(v) >= 6, f'sparse R1 cells per encoder {sp}; |VSAW - VSA| <= 1 on '
                f'{sum(v)} of 7: ' + ', '.join(f'{e} {100 * g(e, "R1", "VSAW-VSA")["diff"]:+.2f}' for e in PANEL)))
    v = [g(e, 'R1', 'VSAS-BEST')['ci95'][1] < 0 for e in PANEL]
    ps = [g(e, 'R1', 'VSAS-BEST')['diff'] >= 0.02 and g(e, 'R1', 'VSAS-BEST')['ci95'][0] > 0 for e in PANEL]
    out.append(('P6', sum(v) >= 6 and sum(ps) == 0, f'BEST ahead on {sum(v)} of 7; H1-B49 passes {sum(ps)}'))
    v2 = [g(e, 'R1-V2', 'VSAS-LAM')['ci95'][1] < 0 for e in PANEL]
    v16 = [g(e, 'R1-V16', 'VSAS-LAM')['ci95'][0] > 0 for e in LLM]
    out.append(('P7', sum(v2) >= 4 and all(v16), f'R1-V2 LAM ahead on {sum(v2)} of 7; R1-V16 VSAS ahead on {sum(v16)} of 4 LLMs'))
    L = [f'- {p}: {"held" if ok else "FAILED"} ({txt})' for p, ok, txt in out]
    open(f'{RES}/b56_predictions.txt', 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    if sys.argv[1] == 'part':
        part(sys.argv[2])
    elif sys.argv[1] == 'preds':
        preds()
    else:
        panel()
