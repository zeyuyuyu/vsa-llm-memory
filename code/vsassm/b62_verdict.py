"""B62 verdict (prereg/B62_online_ls_write_prereg.md): NLMSc / RLSc / RLScS test predictions (b62_out/jobs) added to the
merged test records (b59_out/fam/{enc}/all59_*: STORE = B53's VSA) together with LAMc's (B60, b60_out/jobs); B60's
statistics (per-query correctness, cluster bootstrap with entity-within-store clusters resampled within each cell,
region statistic = mean over cells of the cell mean; b36_verdict.boot_region). R1 only, ideal accounting.

python3 b62_verdict.py part ENC      # -> results/b62_part_{enc}.json
python3 b62_verdict.py panel         # -> results/b62_verdict.{json,txt}
python3 b62_verdict.py predictions   # -> results/b62_predictions.txt
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
import b62_run as R62  # noqa: E402

RES = f'{ROOT}/results'
JOBS = f'{ROOT}/b62_out/jobs'
PANEL = R62.PANEL
LLM, SENT = PANEL[:4], PANEL[4:]
NAMES = V60.NAMES
NDIM = {'qwen8': 4096, 'b46': 4096, 'llama': 4096, 'mistral': 4096, 'e5': 768, 'bge': 384, 'minilm': 384}
M_GRID, VN, BPF = [256, 1024], [2, 16], [4, 8, 16]
REG = {'R1': [(M, V, b) for M in M_GRID for V in VN for b in BPF]}
for V in VN:
    REG[f'R1-V{V}'] = [(M, V, b) for M in M_GRID for b in BPF]
RN = ('R1-V2', 'R1-V16', 'R1')
PAIRS = [('STORE', 'NLMSC'), ('STORE', 'RLSC'), ('NLMSC', 'LAMC'), ('RLSC', 'LAMC'), ('STORE', 'RLSCS'),
         ('NLMSC', 'RLSC'), ('STORE', 'LAMC')]
BIG = {('STORE', 'NLMSC'), ('STORE', 'RLSC'), ('NLMSC', 'LAMC'), ('RLSC', 'LAMC')}
ARMS = ['EXACT', 'STORE', 'LAMC', 'NLMSC', 'RLSC', 'RLSCS']


def choice(enc):
    return json.load(open(f'{RES}/b62_choice_{enc}.json'))['choice']


def label(x, y, lo, hi):
    if x == 'STORE':
        return 'store leads' if lo > 0 else (f'{"online LS" if y != "LAMC" else "LAMc"} leads' if hi < 0 else 'level')
    if y == 'LAMC':
        return 'online leads' if lo > 0 else ('LAMc leads' if hi < 0 else 'level')
    return f'{x} leads' if lo > 0 else (f'{y} leads' if hi < 0 else 'level')


def add_b62(recs, enc, M, V, ch):
    rows = {}
    for rep in range(R62.TEST_REPS[M]):
        for prep in R62.PREPS:
            fn = f'{JOBS}/{enc}_test_M{M}_V{V}_r{rep}_{prep}.json.gz'
            if os.path.exists(fn):
                r = json.load(gzip.open(fn, 'rt'))
                rows[rep, prep] = {(x['arm'], tuple(x['cfg']), x['bpf']): x for x in r['rows']}
                rows[rep, prep]['_qgold'] = r['qgold']
    for r in recs:
        for b in BPF:
            for arm in R62.ARMS:
                cfg = tuple(ch[f'{M}|{V}|{b}|{arm}']['cfg'])
                jr = rows.get((r['rep'], cfg[2]))
                if jr is None or (arm, cfg, b) not in jr:
                    return False
                assert jr['_qgold'] == r['qgold'], (enc, M, V, r['rep'])
                x = jr[arm, cfg, b]
                r['pred'][f'{arm}|{b}|ideal'] = x['pred']
                r.setdefault('b62', {})[f'{arm}|{b}|ideal'] = [list(cfg), x['p'], x['bits']]
    return True


def load(enc):
    ch, ch60 = choice(enc), V60.choice(enc)
    data = {}
    for M in M_GRID:
        for V in VN:
            recs = [json.loads(l) for l in gzip.open(f'{ROOT}/b59_out/fam/{enc}/all59_{enc}_M{M}_V{V}.jsonl.gz', 'rt')]
            recs.sort(key=lambda r: r['rep'])
            for r in recs:
                for b in BPF:
                    r['pred'][f'STORE|{b}|ideal'] = r['pred'][f'VSA|{b}|ideal']
            assert V60.add_lamc(recs, enc, M, V, ch60, 0, bpfs=BPF), ('LAMc test jobs missing', enc, M, V)
            assert add_b62(recs, enc, M, V, ch), ('B62 test jobs missing', enc, M, V)
            data[M, V] = recs
    return data, ch


def cost_rows(data, enc, ch):
    """Per R1 cell: write / read / state of STORE, LAMC, NLMSC, RLSC (chosen shapes), RLS state bits / budget."""
    n = NDIM[enc]
    out = {}
    for (M, V, b) in REG['R1']:
        r0 = data[M, V][0]
        cell = {}
        cell['STORE'] = R62.costs('STORE', M, V, n, None, 0, 0,
                                  store={'cfg': r0['cfg'][f'VSA|{b}|ideal'], 'd': r0['vsa_d'][f'VSA|{b}|ideal']})
        cfg60, p60, _ = r0['b60'][f'LAMC|{b}|ideal']
        cell['LAMC'] = R62.costs('LAMC', M, V, n, cfg60, p60, {'V-1': V - 1, 'V': V}[cfg60[1]])
        for arm in R62.ARMS:
            c = ch[f'{M}|{V}|{b}|{arm}']
            cell[arm] = R62.costs(arm, M, V, n, c['cfg'], c['p'], {'V-1': V - 1, 'V': V}[c['cfg'][1]])
        for arm in ('RLSC', 'RLSCS'):
            p = ch[f'{M}|{V}|{b}|{arm}']['p']
            cell[arm]['state_bits'] = R62.state_bits(p)
            cell[arm]['state_over_budget'] = R62.state_bits(p) / (b * M)
            cell[arm]['charged_feasible'] = R62.state_bits(p) + ch[f'{M}|{V}|{b}|{arm}']['bits'] <= b * M
        for arm in ('STORE', 'LAMC', 'NLMSC', 'RLSC'):
            cell[arm]['state32_over_budget'] = 32 * cell[arm]['state'] / (b * M)
        cell['budget'] = b * M
        out[f'{M}|{V}|{b}'] = cell
    return out


def part(enc):
    ei = PANEL.index(enc)
    data, ch = load(enc)
    J = {'cells': {}, 'regions': {}, 'choices': {}, 'costs': {}}
    pqc = {}

    def pq(M, V, key):
        if (M, V, key) not in pqc:
            pqc[M, V, key] = V36.per_query(data[M, V], key)
        return pqc[M, V, key]

    def k_(arm, b):
        return 'EXACT' if arm == 'EXACT' else f'{arm}|{b}|ideal'
    for (M, V, b) in REG['R1']:
        J['cells'][f'{enc}|{M}|{V}|{b}'] = {x: float(pq(M, V, k_(x, b))[0].mean()) for x in ARMS}
    for ri, rn in enumerate(RN):
        for pi, (x, y) in enumerate(PAIRS):
            diffs = []
            for (M, V, b) in REG[rn]:
                v, cl = pq(M, V, k_(x, b))
                diffs.append((v - pq(M, V, k_(y, b))[0], cl))
            s, _, _, dr = V36.boot_region(diffs, 10000 if (x, y) in BIG else 2000, 62000 + ei * 1000 + ri * 20 + pi)
            lo, hi = float(np.quantile(dr, 0.025)), float(np.quantile(dr, 0.975))
            J['regions'][f'{enc}|{rn}|{x}-{y}'] = {'diff': s, 'ci95': [lo, hi], 'label': label(x, y, lo, hi)}
    for arm in R62.ARMS:
        J['choices'][arm] = {f'{M}|{V}|{b}': ch[f'{M}|{V}|{b}|{arm}'] for (M, V, b) in REG['R1']}
    J['costs'] = cost_rows(data, enc, ch)
    json.dump(J, open(f'{RES}/b62_part_{enc}.json', 'w'))
    print('part', enc, flush=True)


def f(r):
    return f'{100 * r["diff"]:+6.2f} [{100 * r["ci95"][0]:+6.2f}, {100 * r["ci95"][1]:+6.2f}]'


def g(x):
    return f'{x:.2g}'


def panel():
    J = {'cells': {}, 'regions': {}, 'choices': {}, 'costs': {}}
    for enc in PANEL:
        P = json.load(open(f'{RES}/b62_part_{enc}.json'))
        J['cells'].update(P['cells'])
        J['regions'].update(P['regions'])
        J['choices'][enc] = P['choices']
        J['costs'][enc] = P['costs']
    R = J['regions']
    L = ['# B62 verdict (prereg/B62_online_ls_write_prereg.md). Accuracy points; 95% cluster bootstrap (entity within store);',
         '# R1 only, ideal accounting. STORE = B53\'s VSA (Hebbian write, matched filter); LAMc = B60 (batch ridge);',
         '# NLMSc = one pass of the normalised delta rule; RLSc = recursive least squares (p x p state not charged);',
         '# RLScS = RLS with the p x p working state charged against the budget.',
         '# Decision rule: "store leads" iff 95% lower bound > 0; "online LS leads" iff upper bound < 0; else "level".', '']
    counts = {}
    for (x, y) in PAIRS:
        L.append(f'## {x} - {y} (ideal)' + (' [PRIMARY]' if (x, y) == ('STORE', 'NLMSC') else ''))
        L.append('| encoder | R1-V2 | label | R1-V16 | label | R1 | label |')
        L.append('| --- | --- | --- | --- | --- | --- | --- |')
        for enc in PANEL:
            row = []
            for rn in RN:
                r = R[f'{enc}|{rn}|{x}-{y}']
                row += [f(r), r['label']]
            L.append(f'| {NAMES[enc]} | ' + ' | '.join(row) + ' |')
        for rn in RN:
            for gn, grp in (('LLM', LLM), ('sentence', SENT)):
                labs = [R[f'{e}|{rn}|{x}-{y}']['label'] for e in grp]
                c = {lb: labs.count(lb) for lb in sorted(set(labs))}
                counts[f'{x}-{y}|{rn}|{gn}'] = c
                L.append(f'  {rn:7s} {gn:8s}: ' + ', '.join(f'{k} {v}' for k, v in c.items()) + f' (of {len(grp)})')
        L.append('')
    J['counts'] = counts
    # costs
    L.append('## Write cost, read cost and working state at the chosen shapes (ideal; median over the 12 R1 cells '
             '[min, max]); multiply-adds; features separately')
    L.append('| encoder | arm | write per fact | write total (M facts) | x STORE (total) | features (write) | read per query '
             '| working state (numbers) | state at 32 bit / budget |')
    L.append('| --- | --- | --- | --- | --- | --- | --- | --- | --- |')
    for enc in PANEL:
        C_ = J['costs'][enc]
        for arm in ('STORE', 'NLMSC', 'RLSC', 'LAMC', 'RLSCS'):
            def med(k):
                v = [C_[c][arm][k] for c in C_]
                return f'{g(np.median(v))} [{g(min(v))}, {g(max(v))}]'
            ratio = [C_[c][arm]['write'] / max(C_[c]['STORE']['write'], 1) for c in C_]
            s32 = [32 * C_[c][arm]['state'] / C_[c]['budget'] for c in C_]
            L.append(f'| {NAMES[enc]} | {arm} | {med("per_fact")} | {med("write")} | {g(np.median(ratio))} '
                     f'[{g(min(ratio))}, {g(max(ratio))}] | {med("feat")} | {med("read")} | {med("state")} | '
                     f'{g(np.median(s32))} [{g(min(s32))}, {g(max(s32))}] |')
    L.append('')
    inf = sum(not J['costs'][e][c]['RLSC']['charged_feasible'] for e in PANEL for c in J['costs'][e])
    tot = sum(len(J['costs'][e]) for e in PANEL)
    ob = [J['costs'][e][c]['RLSC']['state_over_budget'] for e in PANEL for c in J['costs'][e]]
    L.append(f'RLSc with its working state charged (32 x p (p + 1) / 2 bits + stored bits <= budget): infeasible at the '
             f'chosen p in {inf} of {tot} R1 cells; state / budget median {g(np.median(ob))} [{g(min(ob))}, {g(max(ob))}].')
    ps = [J['choices'][e]['RLSCS'][c]['p'] for e in PANEL for c in J['choices'][e]['RLSCS']]
    L.append(f'RLScS (p cut to fit state + stored bits): p median {np.median(ps):.0f} [{min(ps)}, {max(ps)}].')
    J['rls_charged'] = {'infeasible': inf, 'cells': tot}
    L.append('')
    # dev choices
    L.append('## Dev choices over the R1 cells (counts): scale / target / prep:feat / x / y / q')
    comp = {}
    for arm in R62.ARMS:
        for V in VN:
            cs = [J['choices'][e][arm][f'{M}|{V}|{b}']['cfg'] for e in PANEL for M in M_GRID for b in BPF]
            cnt = {}
            for c in cs:
                for lab_ in (c[0], c[1], f'{c[2]}:{c[3]}', f'x={c[4]}', f'y={c[5]}', f'q={c[6]}'):
                    cnt[lab_] = cnt.get(lab_, 0) + 1
            comp[f'{arm}|V{V}'] = cnt
            L.append(f'  {arm:6s} V={V:2d} ({len(cs)} cells): ' + ', '.join(f'{k} {v}' for k, v in sorted(cnt.items())))
    J['composition'] = comp
    L.append('')
    L.append('## Cell accuracies (ideal), R1: encoder M V bpf: EXACT STORE LAMC NLMSC RLSC RLSCS')
    for k, v in J['cells'].items():
        L.append(f'  {k:24s} ' + ' '.join(f'{100 * v[a]:6.2f}' for a in ARMS))
    json.dump(J, open(f'{RES}/b62_verdict.json', 'w'), indent=0)
    open(f'{RES}/b62_verdict.txt', 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L[:80]))


def predictions():
    J = json.load(open(f'{RES}/b62_verdict.json'))
    R, L = J['regions'], []
    b60 = json.load(open(f'{RES}/b60_verdict.json'))['bars']

    def lab(e, rn, pr):
        return R[f'{e}|{rn}|{pr}']['label']
    n = sum(lab(e, 'R1', 'STORE-NLMSC') == 'online LS leads' for e in PANEL)
    s = sum(lab(e, 'R1', 'STORE-NLMSC') == 'store leads' for e in PANEL)
    L.append(f'Q1: {"HELD" if n >= 5 and s == 0 else "FAILED"} - R1 STORE - NLMSc: NLMSc leads on {n} of 7 (>= 5); store '
             f'leads on {s} (0)')
    a = sum(lab(e, 'R1-V2', 'STORE-NLMSC') == 'online LS leads' for e in LLM)
    b = sum(lab(e, 'R1-V2', 'STORE-NLMSC') == 'online LS leads' for e in SENT)
    L.append(f'Q2: {"HELD" if a == 4 and b <= 2 else "FAILED"} - R1-V2 NLMSc leads on {a} of 4 LLMs (4), on {b} of 3 '
             f'sentence encoders (<= 2)')
    n = sum(lab(e, 'R1-V16', 'STORE-NLMSC') == 'online LS leads' for e in PANEL)
    L.append(f'Q3: {"HELD" if n >= 5 else "FAILED"} - R1-V16 NLMSc leads on {n} of 7 (>= 5)')
    d2 = {e: R[f'{e}|R1-V2|NLMSC-LAMC'] for e in PANEL}
    d16 = {e: R[f'{e}|R1-V16|NLMSC-LAMC'] for e in PANEL}
    a = sum(-0.015 <= d2[e]['diff'] <= 0.005 for e in PANEL)
    led = [e for e in PANEL if d16[e]['label'] == 'LAMc leads']
    rng_ok = all(-0.05 <= d16[e]['diff'] <= -0.01 for e in led)
    L.append(f'Q4: {"HELD" if a >= 6 and len(led) >= 5 and rng_ok else "FAILED"} - NLMSc - LAMc R1-V2 within [-1.5, +0.5] '
             f'on {a} of 7 (>= 6): ' + ', '.join(f'{e} {100 * d2[e]["diff"]:+.2f}' for e in PANEL) +
             f'; R1-V16 LAMc leads on {len(led)} (>= 5), each by 1-5: {rng_ok}: ' +
             ', '.join(f'{e} {100 * d16[e]["diff"]:+.2f}' for e in PANEL))
    a = sum(abs(R[f'{e}|R1|RLSC-LAMC']['diff']) <= 0.005 for e in PANEL)
    lm = {'store leads': 'store leads', 'online LS leads': 'LAMc leads', 'level': 'level'}
    same = sum(lm[lab(e, rn, 'STORE-RLSC')] == b60[f'{e}|{rn}'] for e in PANEL for rn in RN)
    L.append(f'Q5: {"HELD" if a >= 6 and same >= 19 else "FAILED"} - |R1 RLSc - LAMc| <= 0.5 on {a} of 7 (>= 6): ' +
             ', '.join(f'{e} {100 * R[f"{e}|R1|RLSC-LAMC"]["diff"]:+.2f}' for e in PANEL) +
             f'; STORE - RLSc labels = B60 STORE - LAMc labels in {same} of 21 (>= 19)')
    ch = J['choices']
    cells = [(e, k) for e in PANEL for k in ch[e]['RLSC']]
    n = sum(ch[e]['RLSC'][k]['cfg'][5] == 1.0 for e, k in cells)
    L.append(f'Q6: {"HELD" if n >= 75 else "FAILED"} - RLSc beta = 1 in {n} of {len(cells)} R1 cells (>= 75)')
    nl = sum(ch[e]['NLMSC'][k]['cfg'][5] == 'last' for e, k in cells)
    v2 = [(e, k) for e, k in cells if k.split('|')[1] == '2']
    v16 = [(e, k) for e, k in cells if k.split('|')[1] == '16']
    m1 = sum(ch[e]['NLMSC'][k]['cfg'][4] == 1.0 for e, k in v2)
    m25 = sum(ch[e]['NLMSC'][k]['cfg'][4] <= 0.25 for e, k in v16)
    L.append(f'Q7: {"HELD" if nl >= 70 and 2 * m1 >= len(v2) and 2 * m25 >= len(v16) else "FAILED"} - NLMSc output last in '
             f'{nl} of 84 (>= 70); mu = 1 in {m1} of {len(v2)} V = 2 cells (>= half); mu <= 0.25 in {m25} of {len(v16)} '
             f'V = 16 cells (>= half)')
    inf = J['rls_charged']['infeasible']
    s = sum(lab(e, 'R1', 'STORE-RLSCS') == 'store leads' for e in PANEL)
    L.append(f'Q8: {"HELD" if inf == 84 and s >= 6 else "FAILED"} - RLSc working state over budget in {inf} of 84 (84); '
             f'R1 STORE - RLScS store leads on {s} of 7 (>= 6)')
    ok = []
    det = []
    for e in PANEL:
        C_ = J['costs'][e]
        rn = np.median([C_[c]['NLMSC']['per_fact'] / C_[c]['STORE']['per_fact'] for c in C_])
        rr = np.median([C_[c]['RLSC']['per_fact'] / C_[c]['STORE']['per_fact'] for c in C_])
        rl = np.median([C_[c]['LAMC']['write'] / C_[c]['NLMSC']['write'] for c in C_])
        ok.append(rn <= 4 and rn >= 0.25 and rr >= 100 and rl >= 10)
        det.append(f'{e} NLMSc/STORE {rn:.2g}, RLSc/STORE {rr:.2g}, LAMc/NLMSc {rl:.2g}')
    L.append(f'Q9: {"HELD" if all(ok) else "FAILED"} - medians per encoder (NLMSc/STORE per fact within x4; RLSc/STORE '
             f'>= 100; LAMc/NLMSc total >= 10): ' + '; '.join(det))
    L.append('Q10: see results/b62_check.txt')
    open(f'{RES}/b62_predictions.txt', 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    {'part': lambda: part(sys.argv[2]), 'panel': panel, 'predictions': predictions}[sys.argv[1]]()
