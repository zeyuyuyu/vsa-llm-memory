"""B60 verdict (prereg/B60_corrected_lam_prereg.md): LAMc's test predictions (b60_out/jobs) added to the merged test
records (b59_out/fam/{enc}/all59_* for V 2 / 16; b49_out/fam/{enc}/test/all49_*_V1024 for R2); B49 / B53 / B56's
statistics (per-query correctness, cluster bootstrap with entity-within-store clusters resampled within each cell,
region statistic = mean over cells of the cell mean; b36_verdict.boot_region); B59's two-level bootstrap for the maps.

python3 b60_verdict.py part ENC      # -> results/b60_part_{enc}.json
python3 b60_verdict.py panel         # -> results/b60_verdict.{json,txt}
"""
import gzip
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import b36_verdict as V36  # noqa: E402
import b59_verdict as V59  # noqa: E402

RES = f'{ROOT}/results'
JOBS = f'{ROOT}/b60_out/jobs'
VSAPC = f'{ROOT}/b60_out/vsapc'
PANEL = ['qwen8', 'b46', 'llama', 'mistral', 'e5', 'bge', 'minilm']
LLM, SENT = PANEL[:4], PANEL[4:]
NAMES = {'qwen8': 'Qwen3-8B', 'b46': 'Granite-3.3-8B', 'llama': 'Llama-3.1-8B', 'mistral': 'Mistral-7B',
         'e5': 'e5-base-v2', 'bge': 'bge-small', 'minilm': 'MiniLM-L6'}
M_GRID, VN, BPF, ACCTS = [256, 1024], [2, 16], [1, 2, 4, 8, 16, 32, 64], ['ideal', 'real']
A3 = {'ideal': 'ideal', 'real': 'raw'}
REG = {}
for nm, bs in (('R1', (4, 8, 16)), ('R0', (1, 2)), ('RH', (32, 64))):
    REG[nm] = [(M, V, b) for M in M_GRID for V in VN for b in bs]
    for V in VN:
        REG[f'{nm}-V{V}'] = [(M, V, b) for M in M_GRID for b in bs]
REG['R2'] = [(M, 1024, b) for M in M_GRID for b in BPF]
PRIMARY = ('STORE', 'LAMC')
PAIRS = [('STORE', 'LAMC'), ('STORE', 'BESTC'), ('LAMC', 'LAM'), ('STORE', 'LAM'), ('VSAR', 'LAMC'), ('VSAS', 'LAMC'),
         ('VSAP', 'LAMC'), ('VSAPX', 'LAMC'), ('VSAPC', 'LAMC'), ('VSAPCX', 'LAMC'), ('VSAPC', 'VSAP'),
         ('VSAS', 'BESTC'), ('BESTC', 'BEST'), ('VSAR', 'BESTC')]
CN_PAIRS = [('STORE', 'LAMC'), ('STORE', 'BESTC'), ('VSAR', 'LAMC'), ('VSAS', 'LAMC')]
ARMS = ['EXACT', 'STORE', 'VSAR', 'VSAS', 'VSAP', 'VSAPX', 'VSAPC', 'VSAPCX', 'LAM', 'LAMC', 'BEST', 'BESTC']
MAPS = [1, 2, 3]
NB = 10000


def choice(enc):
    return json.load(open(f'{RES}/b60_choice_{enc}.json'))['choice']


def job_rows(enc, m, M, V):
    """{(rep, prep): {(cfg tuple, bpf, acct): pred}} from the test jobs."""
    out = {}
    for rep in range({256: 20, 1024: 5}[M]):
        for prep in ('center', 'whiten'):
            fn = f'{JOBS}/{enc}_m{m}_test_M{M}_V{V}_r{rep}_{prep}.json.gz'
            if not os.path.exists(fn):
                continue
            r = json.load(gzip.open(fn, 'rt'))
            out[rep, prep] = {(tuple(x['cfg']), x['bpf'], x['acct']): (x['pred'], x['p'], x['bits']) for x in r['rows']}
            out[rep, prep]['_qgold'] = r['qgold']
    return out


def add_lamc(recs, enc, M, V, ch, m=0, bpfs=BPF):
    """Adds 'LAMC|b|a' (m = 0) or 'LAMCm{m}|b|a' to every record; returns False if any job is missing.
    Maps: R1 budgets only (compute note 2)."""
    rows = job_rows(enc, m, M, V)
    arm = 'LAMC' if m == 0 else f'LAMCm{m}'
    for r in recs:
        for b in bpfs:
            for a in ACCTS:
                k = f'{M}|{V}|{b}|{a}|{arm}'
                cfg = tuple(ch[k]['cfg'])
                jr = rows.get((r['rep'], cfg[2]))
                if jr is None or (cfg, b, a) not in jr:
                    return False
                assert jr['_qgold'] == r['qgold'], (enc, M, V, r['rep'])
                r['pred'][f'{arm}|{b}|{a}'] = jr[cfg, b, a][0]
                r.setdefault('b60', {})[f'{arm}|{b}|{a}'] = [list(cfg), jr[cfg, b, a][1], jr[cfg, b, a][2]]
    return True


def add_bestc(recs, M, V, ch):
    for r in recs:
        for b in BPF:
            for a in ACCTS:
                f = ch[f'{M}|{V}|{b}|{a}|BESTC']['fam']
                r['pred'][f'BESTC|{b}|{a}'] = r['pred'][f'LAMC|{b}|{a}'] if f == 'LAM' else r['pred'][f'{f}|{b}|{a}']


def add_vsapx(data, enc):
    """B57's VSAPX (dev-best of VSAS and VSAP, ties VSAS) and, if run, VSAPC / VSAPCX (same rule with VSAPc)."""
    ch56 = json.load(open(f'{RES}/b56_choice_{enc}.json'))['choice']
    ch57 = json.load(open(f'{RES}/b57_choice_{enc}.json'))['choice']
    fpc = f'{RES}/b60_choicepc_{enc}.json'
    chpc = json.load(open(fpc))['choice'] if os.path.exists(fpc) else None
    have_pc = chpc is not None and all(os.path.exists(f'{VSAPC}/{enc}/decp_test_{enc}_M{M}_V{V}.jsonl.gz')
                                       for M in M_GRID for V in VN) and any(k.endswith('|PSTAR') for k in chpc)
    for M in M_GRID:
        for V in VN:
            dp = [json.loads(l) for l in gzip.open(f'{VSAPC}/{enc}/decp_test_{enc}_M{M}_V{V}.jsonl.gz', 'rt')] \
                if have_pc else None
            for b in BPF:
                for a in ACCTS:
                    a3 = A3[a]
                    s = ch56[f'{M}|{V}|{b}|{a3}|STAR']
                    p = ch57[f'{M}|{V}|{b}|{a3}|PSTAR']
                    ds, dpv = s['dev'][s['fam']], p['dev'][p['fam']]
                    src = 'VSAP' if dpv > ds + 0.001 else 'VSAS'
                    if have_pc:
                        pc = chpc[f'{M}|{V}|{b}|{a3}|PSTAR']
                        c = chpc[f'{M}|{V}|{b}|{a3}|{pc["fam"]}']['cfg']
                        kk = '|'.join(map(str, (pc['fam'][1:], b, a3, c[0], c[1])))
                        srcc = 'VSAPC' if pc['dev'][pc['fam']] > ds + 0.001 else 'VSAS'
                    for i, r in enumerate(data[M, V]):
                        r['pred'][f'VSAPX|{b}|{a}'] = r['pred'][f'{src}|{b}|{a}']
                        if have_pc:
                            assert dp[i]['rep'] == r['rep'] and dp[i]['qgold'] == r['qgold']
                            r['pred'][f'VSAPC|{b}|{a}'] = dp[i]['pred'][kk]
                            r['pred'][f'VSAPCX|{b}|{a}'] = r['pred'][f'{srcc}|{b}|{a}']
    return have_pc


def load(enc):
    ch = choice(enc)
    data, status = {}, {'maps': {}, 'R2': False}
    for M in M_GRID:
        for V in VN:
            recs = [json.loads(l) for l in gzip.open(f'{ROOT}/b59_out/fam/{enc}/all59_{enc}_M{M}_V{V}.jsonl.gz', 'rt')]
            recs.sort(key=lambda r: r['rep'])
            for r in recs:
                for b in BPF:
                    for a in ACCTS:
                        r['pred'][f'STORE|{b}|{a}'] = r['pred'][f'VSA|{b}|{a}']
                        r['pred'][f'VSASm0|{b}|{a}'] = r['pred'][f'VSAS|{b}|{a}']
            assert add_lamc(recs, enc, M, V, ch), ('LAMC test jobs missing', enc, M, V)
            add_bestc(recs, M, V, ch)
            for m in MAPS:
                ok = f'{M}|{V}|4|ideal|LAMCm{m}' in ch and add_lamc(recs, enc, M, V, ch, m, bpfs=(4, 8, 16))
                status['maps'].setdefault(m, True)
                status['maps'][m] = status['maps'][m] and ok
            data[M, V] = recs
    status['vsapc'] = add_vsapx(data, enc)
    if f'256|1024|4|ideal|LAMC' in ch:
        ok = True
        for M in M_GRID:
            fn = f'{ROOT}/b49_out/fam/{enc}/test/all49_{enc}_M{M}_V1024.jsonl.gz'
            recs = [json.loads(l) for l in gzip.open(fn, 'rt')]
            recs.sort(key=lambda r: r['rep'])
            for r in recs:
                for b in BPF:
                    for a in ACCTS:
                        r['pred'][f'STORE|{b}|{a}'] = r['pred'][f'VSA|{b}|{a}']
            if not add_lamc(recs, enc, M, 1024, ch):
                ok = False
                break
            add_bestc(recs, M, 1024, ch)
            data[M, 1024] = recs
        status['R2'] = ok
    return data, status, ch


def summ(s, dr):
    lo, hi = float(np.quantile(dr, 0.025)), float(np.quantile(dr, 0.975))
    return {'diff': s, 'ci95': [lo, hi], 'ci90': [float(np.quantile(dr, 0.05)), float(np.quantile(dr, 0.95))],
            'label': 'store leads' if lo > 0 else ('LAMc leads' if hi < 0 else 'level')}


def part(enc):
    ei = PANEL.index(enc)
    data, status, ch = load(enc)
    J = {'status': status, 'cells': {}, 'regions': {}, 'cn_regions': {}, 'maps': {}, 'rates': {}, 'choices': {}}
    pqc = {}

    def pq(M, V, key):
        if (M, V, key) not in pqc:
            pqc[M, V, key] = V36.per_query(data[M, V], key)
        return pqc[M, V, key]

    def k_(arm, b, a):
        return 'EXACT' if arm == 'EXACT' else f'{arm}|{b}|{a}'

    def have(arm, M, V):
        return arm == 'EXACT' or f'{arm}|4|ideal' in data[M, V][0]['pred']

    regs = {rn: R for rn, R in REG.items() if all((M, V) in data for (M, V, b) in R)}
    for ai, acct in enumerate(ACCTS):
        for (M, V) in data:
            for b in BPF:
                J['cells'][f'{enc}|{acct}|{M}|{V}|{b}'] = {x: float(pq(M, V, k_(x, b, acct))[0].mean()) for x in ARMS
                                                           if have(x, M, V)}
        for ri, (rn, R) in enumerate(regs.items()):
            for pi, (x, y) in enumerate(PAIRS):
                if not all(have(x, M, V) and have(y, M, V) for (M, V, b) in R):
                    continue
                diffs, cdiffs = [], []
                for (M, V, b) in R:
                    v, cl = pq(M, V, k_(x, b, acct))
                    w = pq(M, V, k_(y, b, acct))[0]
                    diffs.append((v - w, cl))
                    cdiffs.append(((v - w) / (1 - 1 / V), cl))
                big = acct == 'ideal' and (x, y) in (PRIMARY, ('STORE', 'BESTC')) and rn.startswith('R1')
                s, _, _, dr = V36.boot_region(diffs, NB if big else 2000, 6000 + ei * 1000 + ai * 500 + ri * 20 + pi)
                J['regions'][f'{enc}|{acct}|{rn}|{x}-{y}'] = summ(s, dr)
                if (x, y) in CN_PAIRS and rn.startswith('R1'):
                    s, _, _, dr = V36.boot_region(cdiffs, 2000, 16000 + ei * 1000 + ai * 500 + ri * 20 + pi)
                    J['cn_regions'][f'{enc}|{acct}|{rn}|{x}-{y}'] = summ(s, dr)
        # maps: STORE(m) = VSAm{m} (m0 = B53's VSA), VSAS(m); LAMc(m)
        if all(status['maps'].get(m) for m in MAPS):
            for si, (s_name, sk) in enumerate((('STORE', lambda m: f'VSAm{m}'), ('VSAS', lambda m: f'VSASm{m}'))):
                for ri, rn in enumerate(('R1', 'R1-V2', 'R1-V16')):  # compute note 2: maps on R1 budgets only
                    cells = []
                    for (M, V, b) in REG[rn]:
                        _, cl = pq(M, V, k_('LAMC', b, acct))
                        ds = [pq(M, V, k_(sk(m), b, acct))[0] - pq(M, V, k_('LAMC' if m == 0 else f'LAMCm{m}', b, acct))[0]
                              for m in range(4)]
                        cells.append((ds, cl))
                    pts, two, ent = V59.twolevel(cells, NB if acct == 'ideal' else 2000,
                                                 60 + ei * 1000 + ai * 100 + si * 30 + ri)
                    lo, hi = V59.q(two, 0.025), V59.q(two, 0.975)
                    J['maps'][f'{enc}|{acct}|{rn}|{s_name}-LAMC'] = {
                        'per_map': pts.tolist(), 'mean': float(pts.mean()), 'sd': float(pts.std(ddof=1)),
                        'two95': [lo, hi], 'ent95': [V59.q(ent, 0.025), V59.q(ent, 0.975)],
                        'registered_rank': int(np.argsort(np.argsort(pts))[0]) + 1,
                        'label': 'store leads' if lo > 0 else ('LAMc leads' if hi < 0 else 'level')}
        # useful accuracy U = 0.9 x EXACT (B49's rate rule), V 2 / 16
        for M in M_GRID:
            for V in VN:
                ex = pq(M, V, 'EXACT')[0].mean()
                rr = {}
                for arm in ('STORE', 'VSAR', 'LAM', 'LAMC', 'BEST', 'BESTC'):
                    curve = [float(pq(M, V, k_(arm, b, acct))[0].mean()) for b in BPF]
                    rr[arm] = V36.rate(curve, 0.9 * ex)
                J['rates'][f'{enc}|{acct}|{M}|{V}'] = {'U90': 0.9 * float(ex), 'rates': rr}
    # dev choice composition (ideal) and BESTc family counts
    comp = {}
    for (M, V) in data:
        for b in BPF:
            reg = 'R1' if b in (4, 8, 16) else ('R0' if b <= 2 else 'RH')
            if V == 1024:
                reg = 'R2'
            c = ch[f'{M}|{V}|{b}|ideal|LAMC']['cfg']
            comp.setdefault(f'{reg}|V{V}', []).append(c)
            comp.setdefault(f'{reg}|V{V}|BESTC', []).append(ch[f'{M}|{V}|{b}|ideal|BESTC']['fam'])
    J['choices'] = comp
    json.dump(J, open(f'{RES}/b60_part_{enc}.json', 'w'))
    print('part', enc, status, flush=True)


def f(r, k='ci95'):
    return f'{100 * r["diff"]:+6.2f} [{100 * r[k][0]:+6.2f}, {100 * r[k][1]:+6.2f}]'


def panel():
    J = {'cells': {}, 'regions': {}, 'cn_regions': {}, 'maps': {}, 'rates': {}, 'choices': {}, 'status': {}}
    for enc in PANEL:
        P = json.load(open(f'{RES}/b60_part_{enc}.json'))
        for k in ('cells', 'regions', 'cn_regions', 'maps', 'rates'):
            J[k].update(P[k])
        J['choices'][enc] = P['choices']
        J['status'][enc] = P['status']
    R = J['regions']
    L = ['# B60 verdict (prereg/B60_corrected_lam_prereg.md). Accuracy points; 95% cluster bootstrap (entity within store).',
         '# STORE = B53\'s VSA (final design, matched filter); LAMc = corrected least squares (B60); LAM / BEST = B49\'s;',
         '# BESTc = B49\'s best of ten with LAM replaced by LAMc; VSAR = B49\'s store (first design); VSAS = B56; VSAP /',
         '# VSAPX = B57 (whiten basis = B49\'s wrong basis); VSAPC / VSAPCX = VSAP with the corrected basis (B60).',
         '# Decision rule: "store leads" iff 95% lower bound > 0; "LAMc leads" iff upper bound < 0; else "level".', '']
    L.append('## PRIMARY (ideal): STORE - LAMc')
    L.append('| encoder | R1-V2 | label | R1-V16 | label | R1 | label |')
    L.append('| --- | --- | --- | --- | --- | --- | --- |')
    bars = {}
    for enc in PANEL:
        row = []
        for rn in ('R1-V2', 'R1-V16', 'R1'):
            r = R[f'{enc}|ideal|{rn}|STORE-LAMC']
            bars[f'{enc}|{rn}'] = r['label']
            row += [f(r), r['label']]
        L.append(f'| {NAMES[enc]} | ' + ' | '.join(row) + ' |')
    L.append('')
    counts = {}
    for rn in ('R1-V2', 'R1-V16', 'R1'):
        for gn, grp in (('LLM', LLM), ('sentence', SENT)):
            c = {lb: sum(bars[f'{e}|{rn}'] == lb for e in grp) for lb in ('store leads', 'level', 'LAMc leads')}
            counts[f'{rn}|{gn}'] = c
            L.append(f'  {rn:7s} {gn:8s}: store leads {c["store leads"]}, level {c["level"]}, LAMc leads {c["LAMc leads"]} '
                     f'(of {len(grp)})')
    J['bars'] = bars
    J['counts'] = counts
    L.append('')
    L.append('  continuity labels (R1 STORE - LAMc): not behind = lower >= -2; within the margin = 95% interval inside (-2, +2)')
    for enc in PANEL:
        r = R[f'{enc}|ideal|R1|STORE-LAMC']
        L.append(f'    {NAMES[enc]:15s} not_behind={r["ci95"][0] >= -0.02} within={r["ci95"][0] > -0.02 and r["ci95"][1] < 0.02}')
    L.append('')

    def table(title, pair, regs, key='regions', acct='ideal'):
        L.append(f'## {title}')
        L.append('| encoder | ' + ' | '.join(regs) + ' |')
        L.append('| --- |' + ' --- |' * len(regs))
        for enc in PANEL:
            cells = []
            for rn in regs:
                r = J[key].get(f'{enc}|{acct}|{rn}|{pair}')
                cells.append(f(r) + (f' {r["label"]}' if pair.endswith('LAMC') else '') if r else 'n/a')
            L.append(f'| {NAMES[enc]} | ' + ' | '.join(cells) + ' |')
        L.append('')
    R1s = ('R1-V2', 'R1-V16', 'R1')
    table('LAMc - LAM (B49) (ideal): how much the fix matters', 'LAMC-LAM', R1s + ('R0', 'RH', 'R2'))
    table('STORE - BESTc (ideal)', 'STORE-BESTC', R1s + ('R0', 'RH', 'R2'))
    table('STORE - LAM (B49), for reference (ideal)', 'STORE-LAM', R1s)
    for x in ('VSAR', 'VSAS', 'VSAP', 'VSAPX', 'VSAPC', 'VSAPCX'):
        table(f'{x} - LAMc (ideal)', f'{x}-LAMC', R1s + (('R2',) if x == 'VSAR' else ()))
    table('VSAPC - VSAP (ideal): the corrected basis for the store on principal components', 'VSAPC-VSAP', R1s)
    table('BESTc - BEST (ideal)', 'BESTC-BEST', R1s)
    table('STORE - LAMc, R0 and RH (ideal)', 'STORE-LAMC', ('R0-V2', 'R0-V16', 'R0', 'RH-V2', 'RH-V16', 'RH', 'R2'))
    for pr in ('STORE-LAMC', 'STORE-BESTC', 'VSAR-LAMC', 'VSAS-LAMC'):
        table(f'Chance-rescaled {pr} (ideal; (a - b) / (1 - 1/V) per cell)', pr, R1s, key='cn_regions')
    table('REAL accounting: STORE - LAMc', 'STORE-LAMC', R1s + ('R0', 'RH', 'R2'), acct='real')
    table('REAL accounting: STORE - BESTc', 'STORE-BESTC', R1s, acct='real')
    table('REAL accounting: LAMc - LAM', 'LAMC-LAM', R1s, acct='real')
    table('REAL accounting: VSAR - LAMc', 'VSAR-LAMC', R1s, acct='real')
    # maps
    if J['maps']:
        L.append('## Map variance (maps 0-3; two-level 95% interval over maps x entities; ideal)')
        L.append('| encoder | arm | region | per map (0 = registered) | mean | SD | two-level 95% | label | registered label | rank of map 0 |')
        L.append('| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |')
        for enc in PANEL:
            for s_name in ('STORE', 'VSAS'):
                for rn in ('R1-V2', 'R1-V16', 'R1'):
                    x = J['maps'].get(f'{enc}|ideal|{rn}|{s_name}-LAMC')
                    if not x:
                        continue
                    regl = R[f'{enc}|ideal|{rn}|{s_name}-LAMC']['label'] if f'{enc}|ideal|{rn}|{s_name}-LAMC' in R else ''
                    L.append(f'| {NAMES[enc]} | {s_name} | {rn} | ' + ' '.join(f'{100 * v:+.2f}' for v in x['per_map']) +
                             f' | {100 * x["mean"]:+.2f} | {100 * x["sd"]:.2f} | [{100 * x["two95"][0]:+.2f}, '
                             f'{100 * x["two95"][1]:+.2f}] | {x["label"]} | {regl} | {x["registered_rank"]} |')
        L.append('')
    # BESTc family counts and dev choice composition
    L.append('## BESTc: family chosen on dev (ideal), R1 cells (12 per encoder); B49\'s BEST beside')
    n_lam, n_lamc = 0, 0
    for enc in PANEL:
        ch = choice(enc)
        ch49 = json.load(open(f'{RES}/b49_choice_{enc}.json'))['choice']
        fc = [ch[f'{M}|{V}|{b}|ideal|BESTC']['fam'] for M in M_GRID for V in VN for b in (4, 8, 16)]
        fb = [ch49[f'{M}|{V}|{b}|ideal|BEST']['fam'] for M in M_GRID for V in VN for b in (4, 8, 16)]
        n_lamc += fc.count('LAM')
        n_lam += fb.count('LAM')
        L.append(f'  {NAMES[enc]:15s} BESTc: ' + ', '.join(f'{x} {fc.count(x)}' for x in sorted(set(fc))) +
                 ' | BEST (B49): ' + ', '.join(f'{x} {fb.count(x)}' for x in sorted(set(fb))))
    L.append(f'  least squares the dev-chosen best of ten in R1: BEST (B49) {n_lam} of 84; BESTc {n_lamc} of 84')
    J['best_counts'] = {'BEST_LAM': n_lam, 'BESTC_LAMC': n_lamc}
    L.append('')
    L.append('## LAMc dev choices (ideal), counts over cells: scale / target / prep:feat')
    for enc in PANEL:
        for key in sorted(J['choices'][enc]):
            if key.endswith('BESTC'):
                continue
            cs = J['choices'][enc][key]
            cnt = {}
            for c in cs:
                for lab_ in (c[0], c[1], f'{c[2]}:{c[3]}'):
                    cnt[lab_] = cnt.get(lab_, 0) + 1
            L.append(f'  {NAMES[enc]:15s} {key:10s} ({len(cs)} cells): ' + ', '.join(f'{k} {v}' for k, v in sorted(cnt.items())))
    L.append('')
    # useful accuracy
    L.append('## Useful accuracy U = 0.9 x EXACT (ideal), V 2 / 16 x M x 7 encoders = 28 settings: settings reached within 64')
    for arm in ('STORE', 'VSAR', 'LAM', 'LAMC', 'BEST', 'BESTC'):
        hits = [k for k, v in J['rates'].items() if k.split('|')[1] == 'ideal' and v['rates'][arm][1] != '>']
        bpf = [J['rates'][k]['rates'][arm][0] for k in hits]
        L.append(f'  {arm:6s}: {len(hits)} of 28' + (f' (at {min(bpf):.1f} to {max(bpf):.1f} bits per fact)' if hits else ''))
    J['U90_counts'] = {arm: sum(1 for k, v in J['rates'].items() if k.split('|')[1] == 'ideal' and v['rates'][arm][1] != '>')
                       for arm in ('STORE', 'VSAR', 'LAM', 'LAMC', 'BEST', 'BESTC')}
    L.append('')
    L.append('## Run status per encoder: ' + json.dumps(J['status']))
    json.dump(J, open(f'{RES}/b60_verdict.json', 'w'), indent=0)
    open(f'{RES}/b60_verdict.txt', 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    if sys.argv[1] == 'part':
        part(sys.argv[2])
    elif sys.argv[1] == 'panel':
        panel()


def predictions():
    J = json.load(open(f'{RES}/b60_verdict.json'))
    R, L = J['regions'], []

    def g(enc, rn, pair, acct='ideal'):
        return R.get(f'{enc}|{acct}|{rn}|{pair}')
    lab = J['bars']
    n = sum(lab[f'{e}|R1-V16'] == 'LAMc leads' for e in LLM)
    s16 = sum(lab[f'{e}|R1-V16'] == 'store leads' for e in PANEL)
    L.append(f'P1: {"HELD" if n >= 3 and s16 == 0 else "FAILED"} - R1-V16 LAMc leads on {n} of 4 LLMs (>= 3); store leads on {s16} of 7 (0)')
    n = sum(lab[f'{e}|R1-V2'] == 'LAMc leads' for e in LLM)
    L.append(f'P2: {"HELD" if n == 4 else "FAILED"} - R1-V2 LAMc leads on {n} of 4 LLMs (4)')
    n = sum(lab[f'{e}|R1'] == 'LAMc leads' for e in PANEL)
    s = sum(lab[f'{e}|R1'] == 'store leads' for e in PANEL)
    L.append(f'P3: {"HELD" if n >= 6 and s == 0 else "FAILED"} - R1 LAMc leads on {n} of 7 (>= 6); store leads on {s} (0)')
    d16 = {e: g(e, 'R1-V16', 'LAMC-LAM') for e in PANEL}
    d2 = {e: g(e, 'R1-V2', 'LAMC-LAM') for e in PANEL}
    ok1 = all(0.03 <= d16[e]['diff'] <= 0.10 and d16[e]['ci95'][0] > 0 for e in PANEL)
    ok2 = sum(abs(d2[e]['diff']) <= 0.01 for e in PANEL) >= 6
    L.append(f'P4: {"HELD" if ok1 and ok2 else "FAILED"} - LAMc - LAM R1-V16: ' +
             ', '.join(f'{e} {100 * d16[e]["diff"]:+.2f} [{100 * d16[e]["ci95"][0]:+.2f}, {100 * d16[e]["ci95"][1]:+.2f}]' for e in PANEL) +
             ' (each +3..+10, CI > 0); R1-V2 |.| <= 1 on ' + str(sum(abs(d2[e]['diff']) <= 0.01 for e in PANEL)) + ' of 7 (>= 6): ' +
             ', '.join(f'{e} {100 * d2[e]["diff"]:+.2f}' for e in PANEL))
    a = sum(g(e, 'R1', 'STORE-BESTC')['ci95'][1] < 0 for e in PANEL)
    b = sum(g(e, 'R1-V16', 'STORE-BESTC')['ci95'][1] < 0 for e in PANEL)
    L.append(f'P5: {"HELD" if a == 7 and b >= 6 else "FAILED"} - BESTc ahead over R1 on {a} of 7 (7); at R1-V16 on {b} (>= 6)')
    a = sum(g(e, 'R1', 'VSAS-LAMC')['ci95'][1] < 0 for e in PANEL)
    b = sum(g(e, 'R1', 'VSAPX-LAMC')['ci95'][1] < 0 for e in PANEL)
    L.append(f'P6: {"HELD" if a >= 6 and b >= 6 else "FAILED"} - LAMc ahead of VSAS over R1 on {a} of 7, of VSAPX on {b} (>= 6 each)')
    cs = [c for e in PANEL for c in J['choices'][e]['R1|V16']]
    fl = sum(c[0] == 'flat' for c in cs)
    vt = sum(c[1] == 'V' for c in cs)
    wl = [c for e in LLM for k in ('R1|V2', 'R1|V16') for c in J['choices'][e][k]]
    wp = sum(c[2] == 'whiten' and c[3] == 'pca' for c in wl)
    L.append(f'P7: {"HELD" if fl * 3 >= 2 * len(cs) and vt * 2 >= len(cs) and wp * 2 >= len(wl) else "FAILED"} - R1 V=16 cells: '
             f'flat {fl} of {len(cs)} (>= 2/3), V targets {vt} (>= 1/2); LLM R1 cells whiten pca {wp} of {len(wl)} (>= 1/2)')
    if all(f'{e}|ideal|R1|VSAPC-VSAP' in R for e in PANEL):
        n = sum(abs(g(e, 'R1', 'VSAPC-VSAP')['diff']) <= 0.015 for e in PANEL)
        L.append(f'P8: {"HELD" if n >= 5 else "FAILED"} - |R1 VSAPc - VSAP| <= 1.5 on {n} of 7 (>= 5): ' +
                 ', '.join(f'{e} {100 * g(e, "R1", "VSAPC-VSAP")["diff"]:+.2f}' for e in PANEL))
    else:
        L.append('P8: NOT EVALUATED (VSAPc not run)')
    if J['maps']:
        sds = {e: J['maps'][f'{e}|ideal|R1|STORE-LAMC']['sd'] for e in PANEL if f'{e}|ideal|R1|STORE-LAMC' in J['maps']}
        same = sum(J['maps'][f'{e}|ideal|{rn}|STORE-LAMC']['label'] == lab[f'{e}|{rn}'] for e in sds
                   for rn in ('R1-V2', 'R1-V16', 'R1'))
        L.append(f'P9: {"HELD" if all(v <= 0.008 for v in sds.values()) and same >= 19 and len(sds) == 7 else "FAILED"} - '
                 f'between-map SD of R1 STORE - LAMc: ' + ', '.join(f'{e} {100 * v:.2f}' for e, v in sds.items()) +
                 f' (<= 0.8 each); map-mean labels equal registered in {same} of {3 * len(sds)} (>= 19 of 21)')
    else:
        L.append('P9: NOT EVALUATED (maps not run)')
    if all(f'{e}|ideal|R2|LAMC-LAM' in R for e in PANEL):
        n = sum(abs(g(e, 'R2', 'LAMC-LAM')['diff']) <= 0.02 for e in PANEL)
        L.append(f'P10: {"HELD" if n >= 5 else "FAILED"} - |R2 LAMc - LAM| <= 2 on {n} of 7 (>= 5): ' +
                 ', '.join(f'{e} {100 * g(e, "R2", "LAMC-LAM")["diff"]:+.2f}' for e in PANEL))
    else:
        L.append('P10: NOT EVALUATED (R2 not run)')
    L.append('P11: see results/b60_check.txt')
    open(f'{RES}/b60_predictions.txt', 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__' and sys.argv[1] == 'predictions':
    predictions()
