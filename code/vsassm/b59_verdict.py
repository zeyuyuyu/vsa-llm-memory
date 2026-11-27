"""B59 part 1 verdict (prereg/B59_maps_all_encoders_protected_full_prereg.md) on b59_out/fam/{enc}/all59_*: map means
over the registered map and three new maps, B57's two-level bootstrap (maps x entity clusters), per region R1, R1-V2,
R1-V16; primary bar B1 (the paper's final-design verdicts against the +-2 margin on the map means).

python3 b59_verdict.py part ENC
python3 b59_verdict.py panel
"""
import gzip
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import b36_verdict as V36  # noqa: E402

FAM = f'{ROOT}/b59_out/fam'
RES = f'{ROOT}/results'
PANEL = ['qwen8', 'b46', 'llama', 'mistral', 'e5', 'bge', 'minilm']
LLM = PANEL[:4]
SENT = PANEL[4:]
NEW = ['b46', 'llama', 'mistral', 'bge']
NAMES = {'qwen8': 'Qwen3-8B', 'b46': 'Granite-3.3-8B', 'llama': 'Llama-3.1-8B', 'mistral': 'Mistral-7B',
         'e5': 'e5-base-v2', 'bge': 'bge-small', 'minilm': 'MiniLM-L6'}
M_GRID, VN, ACCTS = [256, 1024], [2, 16], ['ideal', 'real']
R1B = (4, 8, 16)
REG = {'R1': [(M, V, b) for M in M_GRID for V in VN for b in R1B]}
for V in VN:
    REG[f'R1-V{V}'] = [(M, V, b) for M in M_GRID for b in R1B]
STORES = {'VSAS': lambda m: 'VSAS' if m == 0 else f'VSASm{m}', 'VSA': lambda m: f'VSAm{m}'}
COMPS = ['LAM', 'BEST', 'LAMRF', 'BEST36']
MARGIN = 0.02
NB = 10000


def load(enc):
    return {(M, V): [json.loads(l) for l in gzip.open(f'{FAM}/{enc}/all59_{enc}_M{M}_V{V}.jsonl.gz', 'rt')]
            for M in M_GRID for V in VN}


def comp_key(y, m):
    return y if (m == 0 or y == 'BEST36') else f'{y}m{m}'


def twolevel(cells, B_, seed):
    """B57's two-level bootstrap (b57_verdict.twolevel's algorithm), returning the draws: cells = list of (list over
    maps of per-query differences, cluster ids). Maps drawn with replacement; entity clusters resampled once per draw
    and shared across maps."""
    rng = np.random.default_rng(seed)
    nm = len(cells[0][0])
    pts = np.zeros(nm)
    draws = np.zeros((nm, B_))
    for diffs, cl in cells:
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
    pts /= len(cells)
    draws /= len(cells)
    mi = rng.integers(0, nm, size=(B_, nm))
    two = draws[mi, np.arange(B_)[:, None]].mean(1)
    ent = draws.mean(0)
    return pts, two, ent


def q(a, p):
    return float(np.quantile(a, p))


def part(enc):
    ei = PANEL.index(enc)
    data = load(enc)
    pqc = {}

    def pq(M, V, arm):
        if (M, V, arm) not in pqc:
            pqc[M, V, arm] = V36.per_query(data[M, V], arm)
        return pqc[M, V, arm]
    J = {'maps': {}, 'cells': {}}
    for ai, acct in enumerate(ACCTS):
        for M in M_GRID:
            for V in VN:
                for b in R1B:
                    J['cells'][f'{enc}|{acct}|{M}|{V}|{b}'] = {
                        f'{s}m{m}': float(pq(M, V, f'{STORES[s](m)}|{b}|{acct}')[0].mean()) for s in STORES for m in range(4)}
                    for y in COMPS:
                        for m in range(4):
                            J['cells'][f'{enc}|{acct}|{M}|{V}|{b}'][f'{y}m{m}'] = float(
                                pq(M, V, f'{comp_key(y, m)}|{b}|{acct}')[0].mean())
        for si, s in enumerate(STORES):
            for yi, y in enumerate(COMPS):
                for ri, (rn, R) in enumerate(REG.items()):
                    cells = []
                    for (M, V, b) in R:
                        _, cl = pq(M, V, f'VSAS|{b}|{acct}')
                        ds = [pq(M, V, f'{STORES[s](m)}|{b}|{acct}')[0] - pq(M, V, f'{comp_key(y, m)}|{b}|{acct}')[0]
                              for m in range(4)]
                        cells.append((ds, cl))
                    nb = NB if acct == 'ideal' else 2000
                    pts, two, ent = twolevel(cells, nb, 59 + ei * 1000 + ai * 100 + si * 30 + yi * 5 + ri)
                    J['maps'][f'{enc}|{acct}|{rn}|{s}-{y}'] = {
                        'per_map': pts.tolist(), 'mean': float(pts.mean()), 'sd': float(pts.std(ddof=1)),
                        'two95': [q(two, 0.025), q(two, 0.975)], 'two90': [q(two, 0.05), q(two, 0.95)],
                        'ent95': [q(ent, 0.025), q(ent, 0.975)],
                        'registered_rank': int(np.argsort(np.argsort(pts))[0]) + 1}
    json.dump(J, open(f'{RES}/b59_part_{enc}.json', 'w'))
    print('part', enc, flush=True)


def lab(lo, hi):
    within = lo > -MARGIN and hi < MARGIN
    behind = hi < 0
    return {'within': within, 'not_behind': lo >= -MARGIN, 'behind': behind, 'ahead': lo > 0,
            'behind_nw': behind and not within}


def name(lb):
    if lb['within']:
        return 'within the margin' + (' (behind)' if lb['behind'] else (' (ahead)' if lb['ahead'] else ''))
    if lb['behind_nw']:
        return 'behind, not within the margin'
    if lb['ahead']:
        return 'ahead, not within the margin'
    return 'not within the margin, interval includes 0'


def f2(x):
    return f'{100 * x:+.2f}'


def panel():
    J = {'maps': {}, 'cells': {}}
    for enc in PANEL:
        P = json.load(open(f'{RES}/b59_part_{enc}.json'))
        J['maps'].update(P['maps'])
        J['cells'].update(P['cells'])
    Mp = J['maps']
    J56 = json.load(open(f'{RES}/b56_verdict.json'))['regions']
    J53 = json.load(open(f'{RES}/b53_verdict.json'))['regions']

    def old(enc, s, y, rn='R1'):
        if s == 'VSAS':
            r = J56[f'{enc}|ideal|{rn}|VSAS-{y}']
        else:
            r = J53[f'{enc}|ideal|{rn}|{y}']
        return r['diff'], r['ci95']
    L = ['# B59 part 1 verdict (prereg/B59_maps_all_encoders_protected_full_prereg.md). Points of accuracy, ideal accounting.',
         '# Map mean over 4 maps (0 = registered, 1-3 new); two-level 95% interval (maps x entity clusters, 10,000 draws);',
         '# entity-only interval of the map mean; between-map SD. VSAS = store with B56 rules (primary); VSA = matched-filter',
         '# read of the final design (secondary). Labels on the two-level 95% interval: within = inside +-2.', '']
    B1 = {}
    for s in ('VSAS', 'VSA'):
        L.append(f'## {s}: R1 map means (ideal), store - LS (LAM) and store - BEST, with V split')
        L.append('| encoder | comparison | region | per map (0,1,2,3) | map mean | two-level 95% | entity-only 95% | SD | '
                 'label (map mean) | registered map, entity-only (B56/B53) | label (registered) |')
        L.append('|' + ' --- |' * 11)
        for enc in PANEL:
            for y in ('LAM', 'BEST'):
                for rn in ('R1', 'R1-V2', 'R1-V16'):
                    x = Mp[f'{enc}|ideal|{rn}|{s}-{y}']
                    lb = lab(*x['two95'])
                    d0, c0 = old(enc, s, y, rn)
                    lb0 = lab(*c0)
                    L.append(f'| {NAMES[enc]} | {s}-{"LS" if y == "LAM" else y} | {rn} | '
                             + ' '.join(f2(v) for v in x['per_map'])
                             + f' | {f2(x["mean"])} | [{f2(x["two95"][0])}, {f2(x["two95"][1])}] | '
                             f'[{f2(x["ent95"][0])}, {f2(x["ent95"][1])}] | {100 * x["sd"]:.2f} | {name(lb)} | '
                             f'{f2(d0)} [{f2(c0[0])}, {f2(c0[1])}] | {name(lb0)} |')
        L.append('')
        # B1
        v1 = {e: lab(*Mp[f'{e}|ideal|R1|{s}-LAM']['two95'])['within'] for e in LLM}
        lbB = {e: lab(*Mp[f'{e}|ideal|R1|{s}-BEST']['two95']) for e in PANEL}
        six = [e for e in PANEL if e != 'llama']
        v2 = all(lbB[e]['behind_nw'] for e in six) and lbB['llama']['within']
        changed = []
        for e in PANEL:
            for y in ('LAM', 'BEST'):
                d0, c0 = old(e, s, y)
                a, b_ = name(lab(*c0)), name(lab(*Mp[f'{e}|ideal|R1|{s}-{y}']['two95']))
                if a != b_:
                    changed.append(f'{NAMES[e]} {s}-{"LS" if y == "LAM" else y}: "{a}" -> "{b_}"')
        B1[s] = {'V1': v1, 'V1_survives': all(v1.values()), 'V2_labels': {e: name(lbB[e]) for e in PANEL},
                 'V2_survives': v2, 'B1': 'SURVIVES' if all(v1.values()) and v2 else 'CHANGES', 'changed_labels': changed}
        L.append(f'### Bar B1 ({s}{", PRIMARY" if s == "VSAS" else ", secondary"})')
        L.append(f'  V1 (within the margin of LS on the four language models): ' +
                 ', '.join(f'{NAMES[e]} {v1[e]}' for e in LLM) + f' -> {"survives" if all(v1.values()) else "CHANGES"}')
        L.append(f'  V2 (behind BEST, not within the margin, on six; within on Llama): ' +
                 ', '.join(f'{NAMES[e]}: {name(lbB[e])}' for e in PANEL) + f' -> {"survives" if v2 else "CHANGES"}')
        L.append(f'  B1 = {B1[s]["B1"]}')
        L.append('  label changes against the registered map (any comparison, R1): ' + ('; '.join(changed) if changed else 'none'))
        # V3, V4
        L.append('  V3 sentence encoders vs LS: ' + ', '.join(
            f'{NAMES[e]}: {name(lab(*Mp[f"{e}|ideal|R1|{s}-LAM"]["two95"]))}' for e in SENT))
        v4 = []
        for e in LLM:
            a2 = Mp[f'{e}|ideal|R1-V2|{s}-LAM']['two95'][1] < 0
            a16 = Mp[f'{e}|ideal|R1-V16|{s}-LAM']['two95'][0] > 0
            v4 += [a2, a16]
        B1[s]['V4'] = sum(v4)
        L.append(f'  V4 (LS ahead at V = 2, store ahead at V = 16, on the four language models): {sum(v4)} of 8')
        L.append('')
    J['B1'] = B1
    # secondary comparisons and raw
    L.append('## Secondary: map means of store - LAMRF and store - BEST36 (R1, ideal); raw accounting of the primary pairs')
    for enc in PANEL:
        parts = []
        for s in ('VSAS', 'VSA'):
            for y in ('LAMRF', 'BEST36'):
                x = Mp[f'{enc}|ideal|R1|{s}-{y}']
                parts.append(f'{s}-{y} {f2(x["mean"])} [{f2(x["two95"][0])}, {f2(x["two95"][1])}]')
            for y in ('LAM', 'BEST'):
                x = Mp[f'{enc}|real|R1|{s}-{y}']
                parts.append(f'raw {s}-{y} {f2(x["mean"])} [{f2(x["two95"][0])}, {f2(x["two95"][1])}] {name(lab(*x["two95"]))}')
        L.append(f'  {NAMES[enc]:15s} ' + ' | '.join(parts))
    L.append('')
    L.append('## Interval widening (R1 VSAS - LAM, ideal): two-level width / entity-only width of the map mean; registered map rank (1 = lowest of 4)')
    J['widen'] = {}
    for enc in PANEL:
        for y in ('LAM', 'BEST'):
            x = Mp[f'{enc}|ideal|R1|VSAS-{y}']
            w = (x['two95'][1] - x['two95'][0]) / (x['ent95'][1] - x['ent95'][0])
            J['widen'][f'{enc}|{y}'] = w
            L.append(f'  {NAMES[enc]:15s} VSAS-{y:4s} widening {100 * (w - 1):+.0f}%  SD {100 * x["sd"]:.2f}  registered rank {x["registered_rank"]}')
    json.dump(J, open(f'{RES}/b59_verdict.json', 'w'), indent=1)
    # predictions (part 1)
    P = []
    P.append(('P1', B1['VSAS']['V1_survives'], 'V1 on VSAS map means: ' + str(B1['VSAS']['V1'])))
    P.append(('P2', B1['VSAS']['V2_survives'], 'V2 on VSAS map means: ' + str(B1['VSAS']['V2_labels'])))
    sds = {e: Mp[f'{e}|ideal|R1|VSAS-LAM']['sd'] for e in NEW}
    P.append(('P3', all(v <= 0.006 for v in sds.values()), 'SD ' + ', '.join(f'{e} {100 * v:.2f}' for e, v in sds.items())))
    ws = {e: J['widen'][f'{e}|LAM'] for e in NEW}
    P.append(('P4', all(1.05 <= v <= 1.60 for v in ws.values()), 'widening ' + ', '.join(f'{e} {100 * (v - 1):+.0f}%' for e, v in ws.items())))
    P.append(('P5', B1['VSAS']['V4'] == 8, f'V4 {B1["VSAS"]["V4"]} of 8'))
    lo = Mp['bge|ideal|R1|VSAS-LAM']['two95'][0]
    P.append(('P6', lo < -MARGIN, f'bge two-level lower bound {f2(lo)}'))
    P.append(('P7', B1['VSA']['B1'] == B1['VSAS']['B1'], f'VSA B1 {B1["VSA"]["B1"]}, VSAS B1 {B1["VSAS"]["B1"]}'))
    L.append('')
    L.append('## Predictions (part 1)')
    L += [f'- {p}: {"held" if ok else "FAILED"} ({t})' for p, ok, t in P]
    open(f'{RES}/b59_verdict.txt', 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    {'part': lambda: part(sys.argv[2]), 'panel': panel}[sys.argv[1]]()
