"""B73 (equal bits) verdict (prereg/B73_confirmatory_prereg.md): the headline contrasts on the fresh entity set.

Input: the assembled records b73_out/asm/{enc}.json.gz (b73_run.py assemble; records per (M, V) with qgold, qent, rep
and pred['ARM|bpf|ideal']). Statistics: b66_storeboot.analyse (B60's region statistic; one-level = b36_verdict.
boot_region, entity clusters within store; two-level = B66's stores-then-entities scheme), 10,000 draws, R1 / R1-V2 /
R1-V16, ideal accounting. Labels: B60's rule. Claims C1-C4 decided on the one-level R1 label (registered criteria).

python3 b73_verdict.py part [--asm DIR] [--encs a,b,..]   # per encoder -> results/b73_part_{enc}.json
python3 b73_verdict.py [--encs a,b,..]                     # panel from the part files -> results/b73_verdict.{txt,json}
"""
import gzip
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import b66_storeboot as SB  # noqa: E402

PANEL = SB.PANEL
NAMES = SB.NAMES
PRIMARY = [('STORE', 'LAMC', 'C1'), ('VSAPCX', 'LAMC', 'C2'), ('STORE', 'NLMSC', 'C3'), ('VSAPCX', 'NLMSC', 'C4')]
SECONDARY = [('VSAS', 'LAMC', 's'), ('VSAPC', 'LAMC', 's'), ('VSAR', 'LAMC', 's'), ('STORE', 'VSAR', 's'),
             ('VSAS', 'NLMSC', 's'), ('NLMSC', 'LAMC', 's'), ('VSAR', 'SDMT', 's')]
# paper (PopQA) R1 values, points, and one-level labels (prereg table)
PAPER = {'C1': ([-4.06, -5.55, -5.56, -5.23, -6.53, -4.18, -4.30], ['LAMC'] * 7),
         'C2': ([-2.01, -3.48, -4.02, -3.91, -5.31, -3.47, -3.66], ['LAMC'] * 7),
         'C3': ([-2.11, -2.87, -1.51, -4.79, -3.48, -2.02, -2.16], ['NLMSC'] * 7),
         'C4': ([-0.06, -0.80, +0.03, -3.47, -2.26, -1.32, -1.52],
                ['level', 'NLMSC', 'level', 'NLMSC', 'NLMSC', 'NLMSC', 'NLMSC']),
         'VSAR-SDMT': ([-1.89, -2.25, -1.20, -3.56, -0.92, -4.73, -3.80], ['SDMT'] * 7)}
ARMS = ['STORE', 'VSAR', 'VSAS', 'VSAPC', 'VSAPCX', 'LAMC', 'NLMSC', 'SDMT']


def short(lab, x, y):
    return x if lab == f'{x} leads' else (y if lab == f'{y} leads' else 'level')


def margin_ok(fresh, paper):
    if abs(fresh - paper) <= 2.0:
        return True
    return bool(paper != 0 and np.sign(fresh) == np.sign(paper) and 0.5 <= fresh / paper <= 2.0)


def load(asm, enc):
    fn = os.path.join(asm, f'{enc}.json.gz')
    if not os.path.exists(fn):
        return None
    d = json.load(gzip.open(fn, 'rt'))
    data = {}
    for k, recs in d.items():
        M, V = map(int, k.split('|'))
        data[M, V] = recs
    return data


def popqa_acc():
    """PopQA R1 cell accuracies of STORE / LAMC / NLMSC (results/b62_verdict.txt's table), for Q7."""
    out = {}
    on = False
    fn = os.path.join(ROOT, 'results', 'b62_verdict.txt')
    if not os.path.exists(fn):
        return {}
    for l in open(fn):
        if l.startswith('## Cell accuracies'):
            on = True
            continue
        if on:
            if not l.strip() or l.startswith('#'):
                if out:
                    break
                continue
            p = l.split()
            enc, M, V, b = p[0].split('|')
            for arm, v in zip(['EXACT', 'STORE', 'LAMC', 'NLMSC'], p[1:5]):
                out.setdefault((enc, arm), []).append(float(v))
    return {k: float(np.mean(v)) for k, v in out.items()}


def mean_acc(data, arm):
    cells = []
    for (M, V), recs in data.items():
        for b in (4, 8, 16):
            c = SB.correct(recs, f'{arm}|{b}|ideal')
            if c is None:
                return None
            cells.append(c.mean())
    return 100 * float(np.mean(cells)) if len(cells) == 12 else None


def part(asm, enc):
    """Per-encoder statistics -> results/b73_part_{enc}.json (so that encoders analysed on different machines can be
    joined by 'panel'; the numbers do not depend on where they are computed)."""
    data = load(asm, enc)
    if data is None or not all(data.get((M, V)) for M in SB.M_GRID for V in SB.VN):
        print('no assembled records', enc)
        return
    R = SB.analyse(data, PANEL.index(enc), PRIMARY + SECONDARY, accts=['ideal'], nb={'ideal': 10000})
    P = {'enc': enc, 'regions': R['regions'], 'missing': R['missing'], 'acc': {arm: mean_acc(data, arm) for arm in ARMS}}
    json.dump(P, open(os.path.join(ROOT, 'results', f'b73_part_{enc}.json'), 'w'), indent=1)
    print('analysed', enc, 'missing', len(R['missing']), flush=True)


def main():
    a = sys.argv[1:]
    asm = a[a.index('--asm') + 1] if '--asm' in a else os.path.join(ROOT, 'b73_out', 'asm')
    encs = a[a.index('--encs') + 1].split(',') if '--encs' in a else PANEL
    if a and a[0] == 'part':
        for enc in encs:
            part(asm, enc)
        return
    J = {'encs': [], 'regions': {}, 'missing': {}, 'acc': {}}
    for enc in encs:
        fn = os.path.join(ROOT, 'results', f'b73_part_{enc}.json')
        if not os.path.exists(fn):
            J['missing'][enc] = 'no part file'
            continue
        P = json.load(open(fn))
        J['encs'].append(enc)
        J['regions'][enc] = P['regions']
        J['missing'][enc] = P['missing']
        J['acc'][enc] = P['acc']
    pq = popqa_acc()
    L = ['# B73 (equal bits) verdict (prereg/B73_confirmatory_prereg.md): fresh Wikidata entities outside PopQA, every '
         'configuration frozen.', '# Accuracy points over R1 (V 2 / 16 x 4, 8, 16 bits per fact x M 256 / 1,024), ideal '
         'accounting. One-level = entity within store (registered label); two-level = stores, then entities (B66).', '']
    J['claims'] = {}
    for (x, y, cid) in PRIMARY + SECONDARY:
        nm = f'{x}-{y}'
        key = cid if cid != 's' else nm
        L += [f'## {cid if cid != "s" else "secondary"}: {x} - {y}', '',
              '| encoder | R1-V2 (one-level) | R1-V16 (one-level) | R1 (one-level) | R1 two-level | paper R1 | margin |',
              '| --- | --- | --- | --- | --- | --- | --- |']
        lab_ok = mar_ok = n = 0
        rows = {}
        for enc in encs:
            if enc not in J['regions'] or f'ideal|R1|{nm}' not in J['regions'][enc]:
                L.append(f'| {NAMES[enc]} | not run | | | | | |')
                continue
            e = {rn: J['regions'][enc][f'ideal|{rn}|{nm}'] for rn in SB.RN}

            def f1(z, k='one'):
                lo, hi = z[k]['ci95']
                return f'{100 * z["diff"]:+6.2f} [{100 * lo:+6.2f}, {100 * hi:+6.2f}] {short(z[k]["label"], x, y)}'
            pv = ''
            mk = ''
            if key in PAPER:
                i = PANEL.index(enc)
                pval, plab = PAPER[key][0][i], PAPER[key][1][i]
                got = short(e['R1']['one']['label'], x, y)
                m = margin_ok(100 * e['R1']['diff'], pval)
                lab_ok += got == plab
                mar_ok += m
                n += 1
                pv = f'{pval:+.2f} {plab}'
                mk = ('label ok' if got == plab else 'LABEL DIFFERS') + (', in margin' if m else ', OUT of margin')
                rows[enc] = {'diff': 100 * e['R1']['diff'], 'ci_one': [100 * v for v in e['R1']['one']['ci95']],
                             'label_one': got, 'ci_two': [100 * v for v in e['R1']['two']['ci95']],
                             'label_two': short(e['R1']['two']['label'], x, y), 'paper': pval, 'paper_label': plab,
                             'label_ok': got == plab, 'margin_ok': bool(m),
                             'V2': 100 * e['R1-V2']['diff'], 'V16': 100 * e['R1-V16']['diff']}
            L.append(f'| {NAMES[enc]} | {f1(e["R1-V2"])} | {f1(e["R1-V16"])} | {f1(e["R1"])} | '
                     f'{f1(e["R1"], "two")} | {pv} | {mk} |')
        if key in PAPER:
            lead_x = sum(r['label_one'] == x for r in rows.values())
            if cid in ('C1', 'C2', 'C3'):
                ok = n == 7 and lab_ok == 7 and mar_ok >= 6
                rule = 'label on 7 of 7 and margin on >= 6 of 7'
            elif cid == 'C4':
                ok = n == 7 and lead_x == 0 and lab_ok >= 6 and mar_ok >= 6
                rule = f'{x} leads on none, label = paper on >= 6 of 7, margin on >= 6 of 7'
            else:
                ok = None
                rule = 'secondary (no claim)'
            verdict = ('CONFIRMED' if ok else ('NOT CONFIRMED' if n == 7 else f'NOT DECIDED ({n} of 7 encoders run)')) \
                if ok is not None else '-'
            J['claims'][key] = {'contrast': nm, 'n': n, 'label_ok': int(lab_ok), 'margin_ok': int(mar_ok),
                                'x_leads': int(lead_x),
                                'two_level_label_ok': int(sum(r['label_two'] == r['paper_label'] for r in rows.values())),
                                'verdict': verdict, 'rule': rule, 'rows': rows}
            L += ['', f'  {key}: label as paper {lab_ok} / {n}, magnitude in margin {mar_ok} / {n}, {x} leads on {lead_x};'
                  f' two-level label as paper {J["claims"][key]["two_level_label_ok"]} / {n} -> **{verdict}** ({rule})']
        L.append('')
    L += ['## Mean R1 accuracy (points; mean over the 12 cells) on the fresh set; PopQA in brackets where available', '',
          '| encoder | ' + ' | '.join(ARMS) + ' |', '| --- |' + ' --- |' * len(ARMS)]
    for enc in J['encs']:
        cells = []
        for arm in ARMS:
            v = J['acc'][enc][arm]
            p = pq.get((enc, arm))
            cells.append('-' if v is None else (f'{v:.2f}' + (f' [{p:.2f}]' if p is not None else '')))
        L.append(f'| {NAMES[enc]} | ' + ' | '.join(cells) + ' |')
    J['popqa_acc'] = {f'{k[0]}|{k[1]}': v for k, v in pq.items()}
    L += ['', '## Missing (arm not run for a cell)', '']
    for enc, m in J['missing'].items():
        L.append(f'  {enc}: {m if isinstance(m, str) else (len(m), sorted({z.split(": ")[1] for z in m}))}')
    open(os.path.join(ROOT, 'results', 'b73_verdict.txt'), 'w').write('\n'.join(L) + '\n')
    json.dump(J, open(os.path.join(ROOT, 'results', 'b73_verdict.json'), 'w'), indent=1)
    print('\n'.join(L))


if __name__ == '__main__':
    main()
