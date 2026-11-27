"""B70 verdict (prereg/B70_useful_budgets_prereg.md): the equal-bit contrasts on R2 (per setting, every grid budget from
the first at which the dev-best memory reaches 0.9 x the exact index's dev accuracy; results/b70_region.json).

Test predictions: b60_verdict.load (STORE = B53's VSA, VSAR, the six families, LH*, LAMc, BESTc, VSAPcX); BEST36 from the
merged records; NLMSc at 8 / 16 from B62's test jobs and choices, at 32 / 64 from B70's; SDMt and BEST7 at 8 / 16 from
B64's, at 32 / 64 from B70's. Statistic: b36_verdict.boot_region (entity within store, resampled within each cell;
region statistic = mean over cells of the cell mean).

python3 b70_verdict.py part ENC     # -> results/b70_part_{enc}.json
python3 b70_verdict.py panel        # -> results/b70_verdict.{json,txt}
"""
import gzip
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import b36_verdict as V36  # noqa: E402
import b60_verdict as V60  # noqa: E402

RES = f'{ROOT}/results'
J70 = f'{ROOT}/b70_out/jobs'
J62 = f'{ROOT}/b62_out/jobs'
J64 = f'{ROOT}/b64_out/jobs'
PANEL = ['qwen8', 'b46', 'llama', 'mistral', 'e5', 'bge', 'minilm']
LLM, SENT = PANEL[:4], PANEL[4:]
NAMES = V60.NAMES
TEST_REPS = {256: 20, 1024: 5}
NEW = (32, 64)
ARMS = ['EXACT', 'STORE', 'VSAPCX', 'VSAR', 'LAMC', 'NLMSC', 'SDMT', 'BEST36', 'BEST7', 'BESTC', 'LHI', 'IDX']
PRIM = [('STORE', 'LAMC', 'rule'), ('STORE', 'NLMSC', 'rule'), ('VSAPCX', 'LAMC', 'rule'),
        ('STORE', 'BEST36', 'bar'), ('STORE', 'BEST7', 'bar')]
SEC = [('VSAR', 'SDMT', 'rule'), ('STORE', 'SDMT', 'rule'), ('VSAR', 'BEST7', 'bar'), ('STORE', 'BESTC', 'rule'),
       ('NLMSC', 'LAMC', 'rule'), ('LAMC', 'BEST36', 'rule'), ('BESTC', 'LAMC', 'rule')]
NB = 10000
CUT_SDMT = {'e5', 'bge', 'minilm'}  # compute note 3: SDMt and BEST7 cut on the sentence encoders
CUT_ARMS = ('SDMT', 'BEST7')


def region_cells():
    return [tuple(x) for x in json.load(open(f'{RES}/b70_region.json'))['cells']]


def cells_of(enc):
    return [(M, V, b) for e, M, V, b in region_cells() if e == enc]


def add_nlms(recs, enc, M, V, bs):
    ch62 = json.load(open(f'{RES}/b62_choice_{enc}.json'))['choice']
    ch70 = json.load(open(f'{RES}/b70_choice_{enc}.json'))['choice'] if any(b in NEW for b in bs) else {}
    for r in recs:
        for b in bs:
            if b in NEW:
                cfg = ch70[f'{M}|{V}|{b}|NLMSC']['cfg']
                fn = f'{J70}/NLMSC_{enc}_test_M{M}_V{V}_r{r["rep"]}_{cfg[2]}.json.gz'
            else:
                cfg = ch62[f'{M}|{V}|{b}|NLMSC']['cfg']
                fn = f'{J62}/{enc}_test_M{M}_V{V}_r{r["rep"]}_{cfg[2]}.json.gz'
            j = json.load(gzip.open(fn, 'rt'))
            assert j['qgold'] == r['qgold'], (enc, M, V, r['rep'])
            x = [x for x in j['rows'] if x['arm'] == 'NLMSC' and x['cfg'] == cfg and x['bpf'] == b]
            assert len(x) == 1, (fn, cfg, b)
            r['pred'][f'NLMSC|{b}|ideal'] = x[0]['pred']
            r.setdefault('b70', {})[f'NLMSC|{b}|ideal'] = [cfg, x[0]['p'], x[0]['bits']]


def add_sdmt(recs, enc, M, V, bs):
    ch64 = json.load(open(f'{RES}/b64_choice_{enc}.json'))['choice']
    ch70 = json.load(open(f'{RES}/b70_choice_{enc}.json'))['choice'] if any(b in NEW for b in bs) else {}
    for r in recs:
        for b in bs:
            if b in NEW:
                c = ch70[f'{M}|{V}|{b}|SDMT']
                fam = ch70[f'{M}|{V}|{b}|BEST7']['fam']
                fn = f'{J70}/SDMT_{enc}_test_M{M}_V{V}_r{r["rep"]}.json.gz'
            else:
                c = ch64[f'{M}|{V}|{b}|ideal|SDMT']
                fam = ch64[f'{M}|{V}|{b}|ideal|BEST7']['fam']
                fn = f'{J64}/{enc}_test_M{M}_V{V}_r{r["rep"]}.json.gz'
            j = json.load(gzip.open(fn, 'rt'))
            assert j['qgold'] == r['qgold'], (enc, M, V, r['rep'])
            x = [x for x in j['rows'] if x['cfg'] == c['cfg'] and x['bpf'] == b and x['acct'] == 'ideal']
            assert len(x) == 1, (fn, c['cfg'], b)
            r['pred'][f'SDMT|{b}|ideal'] = x[0]['pred']
            r.setdefault('b70', {})[f'SDMT|{b}|ideal'] = [c['cfg'], x[0]['N'], x[0]['bits']]
            r['pred'][f'BEST7|{b}|ideal'] = r['pred'][f'{fam}|{b}|ideal']
            r['b70'][f'BEST7|{b}|ideal'] = fam


def load(enc):
    data, status, ch60 = V60.load(enc)
    cells = cells_of(enc)
    ch49 = json.load(open(f'{RES}/b49_choice_{enc}.json'))['choice']
    out = {}
    for (M, V) in sorted({(M, V) for M, V, b in cells}):
        recs = data[M, V]
        bs = [b for m, v, b in cells if (m, v) == (M, V)]
        for r in recs:
            for b in bs:
                f = ch49[f'{M}|{V}|{b}|ideal|BEST36']['fam']
                assert r['pred'][f'BEST36|{b}|ideal'] == r['pred'][f'{f}|{b}|ideal'], (enc, M, V, b)
        add_nlms(recs, enc, M, V, bs)
        if enc not in CUT_SDMT:
            add_sdmt(recs, enc, M, V, bs)
        out[M, V] = recs
    return out, cells


def key(arm, b):
    return 'EXACT' if arm == 'EXACT' else f'{arm}|{b}|ideal'


def summ(s, dr, kind):
    lo, hi = float(np.quantile(dr, 0.025)), float(np.quantile(dr, 0.975))
    d = {'diff': s, 'ci95': [lo, hi], 'rule': 'first leads' if lo > 0 else ('second leads' if hi < 0 else 'level')}
    if kind == 'bar':
        d['bar'] = 'passes' if (s >= 0.02 and lo > 0) else ('loss' if (s <= -0.02 and hi < 0) else 'tie')
    return d


def stats(data, cells, seed0):
    """Per-cell accuracies, region means (raw, chance-rescaled, fraction of EXACT, rescaled fraction), U reach counts,
    and the registered contrasts over the given cells: cells = [(tag, M, V, b)] with data[tag][M, V] records."""
    pq = {}

    def get(t, M, V, arm, b):
        k = (t, M, V, key(arm, b))
        if k not in pq:
            pq[k] = V36.per_query(data[t][M, V], key(arm, b))
        return pq[k]
    J = {'cells': {}, 'means': {}, 'contrasts': {}, 'cn_contrasts': {}, 'reach': {}}
    arms = [a for a in ARMS if not (a in CUT_ARMS and any(t in CUT_SDMT for (t, M, V, b) in cells))]
    J['arms'] = arms
    for (t, M, V, b) in cells:
        ex = get(t, M, V, 'EXACT', b)[0].mean()
        J['cells'][f'{t}|{M}|{V}|{b}'] = {a: float(get(t, M, V, a, b)[0].mean()) for a in arms}
        J['cells'][f'{t}|{M}|{V}|{b}']['U_test'] = 0.9 * float(ex)
    for a in arms:
        raw = [J['cells'][f'{t}|{M}|{V}|{b}'][a] for (t, M, V, b) in cells]
        ex = [J['cells'][f'{t}|{M}|{V}|{b}']['EXACT'] for (t, M, V, b) in cells]
        ch = [1.0 / V for (t, M, V, b) in cells]
        J['means'][a] = {'raw': float(np.mean(raw)),
                         'rescaled': float(np.mean([(x - c) / (1 - c) for x, c in zip(raw, ch)])),
                         'frac_exact': float(np.mean([x / e for x, e in zip(raw, ex)])),
                         'frac_exact_rescaled': float(np.mean([(x - c) / (e - c) for x, e, c in zip(raw, ex, ch)]))}
        J['reach'][a] = int(sum(J['cells'][f'{t}|{M}|{V}|{b}'][a] >= J['cells'][f'{t}|{M}|{V}|{b}']['U_test']
                                for (t, M, V, b) in cells))
    for pi, (x, y, kind) in enumerate(PRIM + SEC):
        if x not in arms or y not in arms:
            continue
        diffs, cdiffs = [], []
        for (t, M, V, b) in cells:
            v, cl = get(t, M, V, x, b)
            w = get(t, M, V, y, b)[0]
            diffs.append((v - w, cl))
            cdiffs.append(((v - w) / (1 - 1 / V), cl))
        nb = NB if pi < len(PRIM) else 2000
        s, _, _, dr = V36.boot_region(diffs, nb, seed0 + pi)
        J['contrasts'][f'{x}-{y}'] = summ(s, dr, kind)
        if pi < len(PRIM):
            s, _, _, dr = V36.boot_region(cdiffs, 2000, seed0 + 100 + pi)
            J['cn_contrasts'][f'{x}-{y}'] = summ(s, dr, kind)
    return J


def part(enc):
    data, cells = load(enc)
    ei = PANEL.index(enc)
    J = stats({enc: data}, [(enc,) + c for c in cells], 70000 + 1000 * ei)
    # dev-best family per R2 cell and dev accuracies of every arm (region file + B62 / B64 / B70 choices)
    reg = json.load(open(f'{RES}/b70_region.json'))['settings']
    J['devbest'] = {f'{M}|{V}|{b}': reg[f'{enc}|{M}|{V}']['best_fam'][str(b)] for (M, V, b) in cells}
    # per-cell configuration records of the new arms (rep of the first record)
    J['cfg'] = {f'{M}|{V}|{b}|{a}': data[M, V][0]['b70'][f'{a}|{b}|ideal'] for (M, V, b) in cells
                for a in (('NLMSC',) if enc in CUT_SDMT else ('NLMSC', 'SDMT', 'BEST7'))}
    json.dump(J, open(f'{RES}/b70_part_{enc}.json', 'w'))
    print('part', enc, len(cells), 'cells', flush=True)
    return data, cells


def sensitivity():
    """Q9: would NLMSc's / SDMt's dev accuracies (B62 / B64 at 4-16, B70 at its budgets) have lowered b*?"""
    reg = json.load(open(f'{RES}/b70_region.json'))['settings']
    out = {}
    for enc in PANEL:
        ch62 = json.load(open(f'{RES}/b62_choice_{enc}.json'))['choice']
        ch64 = json.load(open(f'{RES}/b64_choice_{enc}.json'))['choice']
        f70 = f'{RES}/b70_choice_{enc}.json'
        ch70 = json.load(open(f70))['choice'] if os.path.exists(f70) else {}
        for M in (256, 1024):
            for V in (2, 16):
                s = reg[f'{enc}|{M}|{V}']
                U = s['U_dev']
                hits = []
                for b in (4, 8, 16, 32, 64):
                    if s['bstar'] is not None and b >= s['bstar']:
                        break
                    accs = {}
                    if b in (4, 8, 16):
                        accs['NLMSC'] = ch62[f'{M}|{V}|{b}|NLMSC']['dev_acc']
                        accs['SDMT'] = ch64[f'{M}|{V}|{b}|ideal|SDMT']['dev_acc']
                    else:
                        for a in ('NLMSC', 'SDMT'):
                            if f'{M}|{V}|{b}|{a}' in ch70:
                                accs[a] = ch70[f'{M}|{V}|{b}|{a}']['dev_acc']
                    hits += [(b, a, x) for a, x in accs.items() if x >= U]
                out[f'{enc}|{M}|{V}'] = {'bstar': s['bstar'], 'U_dev': U, 'would_lower': hits}
    return out


def fmt(r, k='rule'):
    return f'{100 * r["diff"]:+6.2f} [{100 * r["ci95"][0]:+6.2f}, {100 * r["ci95"][1]:+6.2f}] {r[k]}'


def num(r):
    return f'{100 * r["diff"]:+6.2f} [{100 * r["ci95"][0]:+6.2f}, {100 * r["ci95"][1]:+6.2f}]'


def lab(x, y, r):
    t = r['rule']
    return f'{x} leads' if t == 'first leads' else (f'{y} leads' if t == 'second leads' else 'level')


def panel():
    P = {enc: json.load(open(f'{RES}/b70_part_{enc}.json')) for enc in PANEL}
    # pooled groups (cells of different encoders are independent)
    G = {}
    for gi, (gn, grp) in enumerate((('LLM', LLM), ('sentence', SENT))):
        data = {}
        cells = []
        for enc in grp:
            d, c = load(enc)
            data[enc] = d
            cells += [(enc,) + x for x in c]
        G[gn] = stats(data, cells, 90000 + 1000 * gi)
    J = {'parts': P, 'groups': G, 'sensitivity': sensitivity()}
    reg = json.load(open(f'{RES}/b70_region.json'))
    L = ['# B70 verdict (prereg/B70_useful_budgets_prereg.md). Ideal accounting; accuracy points; 95% cluster bootstrap',
         '# (entity within store, resampled within each cell). R2 = per setting every grid budget from the first at which',
         '# the dev-best of 13 memories reaches 0.9 x the exact index (dev). STORE = B53 final design; VSAPCX = per cell',
         '# dev-best of VSAS / VSAPc; VSAR = first design; LAMC = corrected least squares; NLMSC = one-pass delta rule;',
         '# SDMT = SDM at training codes; BEST36 = best of B36\'s six; BEST7 = six + SDMt; BESTC = best of ten with LAMc.',
         '# rule: "X leads" iff lower > 0, "Y leads" iff upper < 0, else level. bar (B36): passes >= +2 & lower > 0;',
         '# loss <= -2 & upper < 0; else tie.', '']
    L.append('## R2 cells per encoder (M, V, bits per fact) and the dev-best memory there')
    for enc in PANEL:
        L.append(f'  {NAMES[enc]:15s} ' + '; '.join(f'({k.replace("|", ", ")}) {v}' for k, v in P[enc]['devbest'].items()))
    L.append('')
    L.append('## Primary contrasts over R2 (per encoder; pooled groups below)')
    hdr = [f'{x} - {y}' for x, y, _ in PRIM]
    L.append('| encoder | cells | ' + ' | '.join(hdr) + ' |')
    L.append('| --- | --- |' + ' --- |' * len(PRIM))
    rows = [(NAMES[e], P[e], len(P[e]['devbest'])) for e in PANEL] + \
        [('4 language models', G['LLM'], len(G['LLM']['cells'])), ('3 sentence encoders', G['sentence'],
                                                                     len(G['sentence']['cells']))]
    for nm, X, nc in rows:
        cs = []
        for x, y, kind in PRIM:
            r = X['contrasts'].get(f'{x}-{y}')
            if r is None:
                cs.append('cut')
                continue
            cs.append(num(r) + f' {lab(x, y, r)}' + (f' / bar {r["bar"]}' if kind == 'bar'
                                                                                else ''))
        L.append(f'| {nm} | {nc} | ' + ' | '.join(cs) + ' |')
    L.append('')
    L.append('## Secondary contrasts over R2')
    L.append('| encoder | ' + ' | '.join(f'{x} - {y}' for x, y, _ in SEC) + ' |')
    L.append('| --- |' + ' --- |' * len(SEC))
    for nm, X, nc in rows:
        cs = []
        for x, y, kind in SEC:
            r = X['contrasts'].get(f'{x}-{y}')
            if r is None:
                cs.append('cut')
                continue
            cs.append(num(r) + f' {lab(x, y, r)}' + (f' / bar {r["bar"]}' if kind == 'bar' else ''))
        L.append(f'| {nm} | ' + ' | '.join(cs) + ' |')
    L.append('')
    L.append('## Chance-rescaled primary contrasts ((a - b) / (1 - 1/V) per cell; 2,000 draws)')
    L.append('| encoder | ' + ' | '.join(hdr) + ' |')
    L.append('| --- |' + ' --- |' * len(PRIM))
    for nm, X, nc in rows:
        cs = []
        for x, y, kind in PRIM:
            r = X['cn_contrasts'].get(f'{x}-{y}')
            if r is None:
                cs.append('cut')
                continue
            cs.append(num(r) + f' {lab(x, y, r)}')
        L.append(f'| {nm} | ' + ' | '.join(cs) + ' |')
    L.append('')
    for what, title in (('raw', 'raw accuracy'), ('rescaled', 'rescaled for chance ((a - 1/V) / (1 - 1/V))'),
                        ('frac_exact', 'fraction of the exact index (a / EXACT)'),
                        ('frac_exact_rescaled', 'chance-rescaled fraction of the exact index ((a - 1/V) / (EXACT - 1/V))')):
        L.append(f'## R2 region means: {title}' + (' (points)' if what in ('raw', 'rescaled') else ''))
        L.append('| encoder | ' + ' | '.join(ARMS) + ' |')
        L.append('| --- |' + ' --- |' * len(ARMS))
        for nm, X, nc in rows:
            sc = 100 if what in ('raw', 'rescaled') else 1
            L.append(f'| {nm} | ' + ' | '.join((f'{sc * X["means"][a][what]:.{1 if sc == 100 else 3}f}'
                                                 if a in X['means'] else 'cut') for a in ARMS) + ' |')
        L.append('')
    L.append('## R2 cells in which each arm reaches U_test = 0.9 x EXACT (test)')
    L.append('| encoder | cells | ' + ' | '.join(ARMS[1:]) + ' |')
    L.append('| --- | --- |' + ' --- |' * (len(ARMS) - 1))
    for nm, X, nc in rows:
        L.append(f'| {nm} | {nc} | ' + ' | '.join(str(X['reach'].get(a, 'cut')) for a in ARMS[1:]) + ' |')
    L.append('')
    L.append('## Per-cell test accuracies (points)')
    L.append('| encoder | M | V | bpf | U_test | ' + ' | '.join(ARMS) + ' |')
    L.append('| --- | --- | --- | --- | --- |' + ' --- |' * len(ARMS))
    for enc in PANEL:
        for k, c in P[enc]['cells'].items():
            _, M, V, b = k.split('|')
            L.append(f'| {NAMES[enc]} | {M} | {V} | {b} | {100 * c["U_test"]:.1f} | ' +
                     ' | '.join((f'{100 * c[a]:.1f}' if a in c else 'cut') for a in ARMS) + ' |')
    L.append('')
    L.append('## New arms\' chosen configurations at the R2 cells (first test store)')
    for enc in PANEL:
        L.append(f'  {NAMES[enc]:15s} ' + '; '.join(f'{k}: {v}' for k, v in P[enc]['cfg'].items()))
    L.append('')
    L.append('## Settings without a useful budget within 64 bits per fact (dev), with B49\'s existing TEST references')
    b49 = json.load(open(f'{RES}/b49_verdict.json'))['useful']
    L.append('| encoder | M | V | U_dev | dev best at 64 (family) | U_test (B49) | VSA-inf (B49) | LAM-inf (B49) |')
    L.append('| --- | --- | --- | --- | --- | --- | --- | --- |')
    nolow = 0
    for enc in PANEL:
        for M in (256, 1024):
            for V in (2, 16):
                s = reg['settings'][f'{enc}|{M}|{V}']
                if s['bstar'] is not None:
                    continue
                u = b49[f'{enc}|{M}|{V}']
                nolow += u['VSAINF'] < u['U'] and u['LAMINF'] < u['U']
                L.append(f'| {NAMES[enc]} | {M} | {V} | {100 * s["U_dev"]:.1f} | {100 * s["best_dev"]["64"]:.1f} '
                         f'({s["best_fam"]["64"]}) | {100 * u["U"]:.1f} | {100 * u["VSAINF"]:.1f} | '
                         f'{100 * u["LAMINF"]:.1f} |')
    L.append(f'  both infinite-width reads below U in {nolow} of these settings')
    L.append('')
    L.append('## Sensitivity (Q9): NLMSc / SDMt dev accuracies at budgets below b* that reach U_dev')
    lw = {k: v for k, v in J['sensitivity'].items() if v['would_lower']}
    L.append(f'  settings where b* would be lower: {len(lw)} ' + (str(lw) if lw else ''))
    J['text'] = L
    json.dump(J, open(f'{RES}/b70_verdict.json', 'w'))
    open(f'{RES}/b70_verdict.txt', 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    if sys.argv[1] == 'part':
        part(sys.argv[2])
    elif sys.argv[1] == 'panel':
        panel()
    else:
        raise SystemExit(__doc__)
