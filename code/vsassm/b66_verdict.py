"""B66 verdict, part (A) (prereg/B66_reconciliation_prereg.md): the 2 x 2 (code x grid) for first design - SDMt on
Qwen3-8B and Llama, and the random-map replicates (VSARm - SDMTm, VSAm - BEST7m, VSARm - BEST7m) on all 7 encoders.

python3 b66_verdict.py a1          # -> results/b66_A1.{json,txt}
python3 b66_verdict.py a2 [ENC..]  # -> results/b66_A2_part_{enc}.json
python3 b66_verdict.py a2panel     # -> results/b66_A2.{json,txt}
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

RES = os.environ.get('B66_RES', f'{ROOT}/results')
OUT = os.environ.get('B66_OUT', f'{ROOT}/b66_out')
PANEL = ['qwen8', 'b46', 'llama', 'mistral', 'e5', 'bge', 'minilm']
NAMES = {'qwen8': 'Qwen3-8B', 'b46': 'Granite-3.3-8B', 'llama': 'Llama-3.1-8B', 'mistral': 'Mistral-7B',
         'e5': 'e5-base-v2', 'bge': 'bge-small', 'minilm': 'MiniLM-L6'}
M_GRID, VN, BPF, ACCTS = [256, 1024], [2, 16], [4, 8, 16], ['ideal', 'real']
TEST_REPS = {256: 20, 1024: 5}
REG = {'R1': [(M, V, b) for M in M_GRID for V in VN for b in BPF]}
for _V in VN:
    REG[f'R1-V{_V}'] = [(M, _V, b) for M in M_GRID for b in BPF]
SIX = ['IDX', 'SF', 'CM', 'BF', 'FBF', 'SDM']
MAPS = [0, 1, 2, 3]


def lab(lo, hi):
    return 'first leads' if lo > 0 else ('second leads' if hi < 0 else 'level')


def bar(s, lo, hi):
    return 'passes' if (s >= 0.02 and lo > 0) else ('loss' if (s <= -0.02 and hi < 0) else 'tie')


def merged(enc, M, V):
    recs = [json.loads(l) for l in gzip.open(f'{ROOT}/b59_out/fam/{enc}/all59_{enc}_M{M}_V{V}.jsonl.gz', 'rt')]
    recs.sort(key=lambda r: r['rep'])
    return recs


def sdmt_jobs(jobdir, enc, M, V):
    out = {}
    for rep in range(TEST_REPS[M]):
        fn = f'{jobdir}/{enc}_test_M{M}_V{V}_r{rep}.json.gz'
        if not os.path.exists(fn):
            return None
        r = json.load(gzip.open(fn, 'rt'))
        out[rep] = ({(tuple(x['cfg']), x['bpf'], x['acct']): x['pred'] for x in r['rows']}, r['qgold'])
    return out


def add_sdmt(recs, enc, M, V, jobdir, ch, key, name, bname):
    """Adds name|b|a (SDMt test predictions of the chosen cfg) and bname|b|a (BEST7 rule: the family's predictions)."""
    J = sdmt_jobs(jobdir, enc, M, V)
    if J is None:
        return False
    for r in recs:
        rows, qg = J[r['rep']]
        assert qg == r['qgold'], (enc, M, V, r['rep'])
        for b in BPF:
            for a in ACCTS:
                cfg = tuple(ch[f'{M}|{V}|{b}|{a}|{key}']['cfg'])
                r['pred'][f'{name}|{b}|{a}'] = rows[cfg, b, a]
                f = ch[f'{M}|{V}|{b}|{a}|{"BEST7" if key == "SDMT" else "BEST7CR"}']['fam']
                r['pred'][f'{bname}|{b}|{a}'] = r['pred'][f'{name}|{b}|{a}'] if f == key else r['pred'][f'{f}|{b}|{a}']
    return True


def summ(s, dr):
    lo, hi = float(np.quantile(dr, 0.025)), float(np.quantile(dr, 0.975))
    return {'diff': s, 'ci95': [lo, hi], 'label': lab(lo, hi), 'bar': bar(s, lo, hi)}


# ----------------------------------------------------------------------------------------------- A1

def a1_pipeline(enc):
    ch64 = json.load(open(f'{RES}/b64_choice_{enc}.json'))['choice']
    chcr = json.load(open(f'{RES}/b66_choice_crgrid_{enc}.json'))['choice']
    data = {}
    for M in M_GRID:
        for V in VN:
            recs = merged(enc, M, V)
            assert add_sdmt(recs, enc, M, V, f'{ROOT}/b64_out/jobs', ch64, 'SDMT', 'SDMT', 'BEST7')
            assert add_sdmt(recs, enc, M, V, f'{OUT}/b64cr/jobs', chcr, 'SDMTCR', 'SDMTCR', 'BEST7CR')
            data[M, V] = recs
    ei = PANEL.index(enc)
    out = {'cells': {}, 'regions': {}}
    pqc = {}

    def pq(M, V, k):
        if (M, V, k) not in pqc:
            pqc[M, V, k] = V36.per_query(data[M, V], k)
        return pqc[M, V, k]
    pairs = [(x, y) for x in ('VSAR', 'VSA') for y in ('SDMT', 'SDMTCR', 'BEST7', 'BEST7CR')]
    for ai, acct in enumerate(ACCTS):
        for (M, V, b) in REG['R1']:
            out['cells'][f'{acct}|{M}|{V}|{b}'] = {k: float(pq(M, V, f'{k}|{b}|{acct}')[0].mean())
                                                    for k in ('VSAR', 'VSA', 'SDMT', 'SDMTCR', 'BEST7', 'BEST7CR')}
        for ri, (rn, R) in enumerate(REG.items()):
            for pi, (x, y) in enumerate(pairs):
                diffs = [(pq(M, V, f'{x}|{b}|{acct}')[0] - pq(M, V, f'{y}|{b}|{acct}')[0], pq(M, V, f'{x}|{b}|{acct}')[1])
                         for (M, V, b) in R]
                nb = 10000 if acct == 'ideal' else 2000
                s, _, _, dr = V36.boot_region(diffs, nb, 66100 + ei * 1000 + ai * 500 + ri * 20 + pi)
                out['regions'][f'{acct}|{rn}|{x}-{y}'] = summ(s, dr)
    return out


def a1_clean(enc):
    """The clean room's arms from b66_cr.py's test phase; the clean room's own bootstrap (agg.py's: entity multinomial
    weights shared across its 12 conditions, 2,000 replicates, seed 20261123)."""
    d = f'{OUT}/cr/out_{enc}'
    conds = [(M, V, b) for M in M_GRID for V in VN for b in BPF]
    cells, corr = {}, {}
    for k in conds:
        cells[k] = json.load(open(f'{d}/cell_M{k[0]}_V{k[1]}_b{k[2]}.json'))
        corr[k] = dict(np.load(f'{d}/corr_M{k[0]}_V{k[1]}_b{k[2]}.npz'))
    NE, Rn = 4622, 2000
    rng = np.random.default_rng(20261123)
    W = rng.multinomial(NE, np.full(NE, 1.0 / NE), size=Rn).astype(np.float64)
    arms = ['store_first', 'store_final', 'index', 'static', 'cms', 'bloom', 'fly', 'sdm', 'sdmG', 'sdmTcr', 'sdmT64',
            'best6', 'best7cr', 'best7_64']
    bt = {}
    for a in arms:
        bt[a] = {}
        for k in conds:
            w = W[:, corr[k]['qe']]
            bt[a][k] = 100.0 * (w * corr[k][a].astype(np.float64)[None]).sum(1) / w.sum(1)
    out = {'cells': {}, 'regions': {}, 'choices': {}}
    for k in conds:
        out['cells'][f'{k[0]}|{k[1]}|{k[2]}'] = {a: cells[k][a]['test'] for a in arms}
        out['choices'][f'{k[0]}|{k[1]}|{k[2]}'] = {a: cells[k][a].get('cfg', cells[k][a].get('family'))
                                                   for a in ('sdmG', 'sdmTcr', 'sdmT64', 'best6', 'best7cr', 'best7_64')}
    for rn, R in REG.items():
        for x in ('store_first', 'store_final'):
            for y in ('sdmTcr', 'sdmT64', 'best7cr', 'best7_64', 'best6', 'sdm'):
                pt = float(np.mean([cells[k][x]['test'] - cells[k][y]['test'] for k in R])) / 100
                bs = np.mean(np.stack([bt[x][k] - bt[y][k] for k in R]), 0) / 100
                out['regions'][f'ideal|{rn}|{x}-{y}'] = summ(pt, bs)
    # reproduction of the original clean-room run (same seeds)
    o = f'{OUT}/cr/orig/out_{enc}'
    rep = {}
    for k in conds:
        c0 = json.load(open(f'{o}/cell_M{k[0]}_V{k[1]}_b{k[2]}.json'))
        for a in ('store_first', 'store_final', 'index', 'static', 'cms', 'bloom', 'fly', 'sdm'):
            rep[f'{k[0]}|{k[1]}|{k[2]}|{a}'] = abs(c0[a]['test'] - cells[k][a]['test'])
        rep[f'{k[0]}|{k[1]}|{k[2]}|best6'] = abs(c0['best_keyfree']['test'] - cells[k]['best6']['test'])
    out['reproduction_max_abs_diff'] = max(rep.values())
    out['reproduction_n_diff'] = sum(v > 1e-9 for v in rep.values())
    out['reproduction_n'] = len(rep)
    return out


def a1():
    J = {}
    L = ['# B66 (A1): first design - SDMt, two codes x two grids (prereg/B66_reconciliation_prereg.md). R1 = V 2 / 16 x',
         '# 4, 8, 16 bits per fact x M 256 / 1,024; ideal accounting. Points. Pipeline cells: b36_verdict.boot_region',
         '# (entity within store, 10,000 draws); clean-room cells: its own bootstrap (entity multinomial shared across',
         '# its 12 conditions, 2,000 draws). B64 grid = 504 configurations; clean grid = the clean room\'s 32 data-address',
         '# configurations (LA, a 1/4/16/64, onehot/bits, exact counts, share 1/0.5). Labels: first leads / second leads /',
         '# level (B60 rule).', '']
    for enc in ('qwen8', 'llama'):
        P = a1_pipeline(enc)
        Cc = a1_clean(enc)
        J[enc] = {'pipeline': P, 'clean': Cc}
        g = lambda D, k: D['regions'][f'ideal|R1|{k}']  # noqa: E731
        Dpb = g(P, 'VSAR-SDMT')['diff'] * 100
        Dpc = g(P, 'VSAR-SDMTCR')['diff'] * 100
        Dcb = g(Cc, 'store_first-sdmT64')['diff'] * 100
        Dcc = g(Cc, 'store_first-sdmTcr')['diff'] * 100
        grid = ((Dpc - Dpb) + (Dcc - Dcb)) / 2
        code = ((Dcc - Dpc) + (Dcb - Dpb)) / 2
        inter = (Dcc - Dcb) - (Dpc - Dpb)
        if grid >= 2 * abs(code) and Dcb <= 0.5:
            att = 'grid'
        elif code >= 2 * abs(grid) or (abs(code) >= 2 * abs(grid)):
            att = 'code'
        else:
            att = 'both'
        J[enc]['attribution'] = {'D_pipe_b64': Dpb, 'D_pipe_clean': Dpc, 'D_clean_b64': Dcb, 'D_clean_clean': Dcc,
                                 'grid_effect': grid, 'code_effect': code, 'interaction': inter, 'attribution': att}
        f = lambda r: f'{100 * r["diff"]:+6.2f} [{100 * r["ci95"][0]:+6.2f}, {100 * r["ci95"][1]:+6.2f}] {r["label"]}'  # noqa
        L.append(f'## {NAMES[enc]}')
        L.append('| comparison (R1) | pipeline code, B64 grid | pipeline code, clean grid | clean code, B64 grid | clean code, clean grid |')
        L.append('| --- | --- | --- | --- | --- |')
        for nm, pk, pc, ck, cc in (('first - SDMt', 'VSAR-SDMT', 'VSAR-SDMTCR', 'store_first-sdmT64', 'store_first-sdmTcr'),
                                   ('first - best of seven', 'VSAR-BEST7', 'VSAR-BEST7CR', 'store_first-best7_64', 'store_first-best7cr'),
                                   ('final - SDMt', 'VSA-SDMT', 'VSA-SDMTCR', 'store_final-sdmT64', 'store_final-sdmTcr'),
                                   ('final - best of seven', 'VSA-BEST7', 'VSA-BEST7CR', 'store_final-best7_64', 'store_final-best7cr')):
            L.append(f'| {nm} | {f(g(P, pk))} | {f(g(P, pc))} | {f(g(Cc, ck))} | {f(g(Cc, cc))} |')
        for rn in ('R1-V2', 'R1-V16'):
            gg = lambda D, k: D['regions'][f'ideal|{rn}|{k}']  # noqa: E731
            L.append(f'| first - SDMt, {rn} | {f(gg(P, "VSAR-SDMT"))} | {f(gg(P, "VSAR-SDMTCR"))} | '
                     f'{f(gg(Cc, "store_first-sdmT64"))} | {f(gg(Cc, "store_first-sdmTcr"))} |')
        mean = lambda D, k: np.mean([D['cells'][c][k] for c in D['cells'] if c.startswith('ideal|')]) * 100  # noqa
        cm = lambda k: np.mean([Cc['cells'][c][k] for c in Cc['cells']])  # noqa
        L.append(f'  R1 means: pipeline first {mean(P, "VSAR"):.2f}, final {mean(P, "VSA"):.2f}, SDMt B64 grid '
                 f'{mean(P, "SDMT"):.2f}, SDMt clean grid {mean(P, "SDMTCR"):.2f}; clean first {cm("store_first"):.2f}, '
                 f'final {cm("store_final"):.2f}, SDMt B64 grid {cm("sdmT64"):.2f}, SDMt clean grid {cm("sdmTcr"):.2f}, '
                 f'SDM Gaussian {cm("sdmG"):.2f}, best6 (orig. rule) {cm("best6"):.2f}')
        a = J[enc]['attribution']
        L.append(f'  D = first - SDMt (R1): pipeline/B64 {Dpb:+.2f}, pipeline/clean {Dpc:+.2f}, clean/B64 {Dcb:+.2f}, '
                 f'clean/clean {Dcc:+.2f}; grid effect {a["grid_effect"]:+.2f}, code effect {a["code_effect"]:+.2f}, '
                 f'interaction {a["interaction"]:+.2f} -> attribution: {att.upper()}')
        L.append(f'  clean-room re-run reproduces its original run: {Cc["reproduction_n"] - Cc["reproduction_n_diff"]} of '
                 f'{Cc["reproduction_n"]} (arm, cell) test accuracies identical (max |diff| {Cc["reproduction_max_abs_diff"]:.2e})')
        L.append('  clean-room dev choices per cell (SDMt clean grid / SDMt B64 grid / best7 B64 grid):')
        for c, v in Cc['choices'].items():
            L.append(f'    {c}: {"/".join(map(str, v["sdmTcr"][1:]))} | {"/".join(map(str, v["sdmT64"][1:]))} | '
                     f'{v["best7_64"]}')
        chp = json.load(open(f'{RES}/b66_choice_crgrid_{enc}.json'))['choice']
        ch64 = json.load(open(f'{RES}/b64_choice_{enc}.json'))['choice']
        L.append('  pipeline dev choices per cell, ideal (SDMt clean grid / SDMt B64 grid):')
        for (M, V, b) in REG['R1']:
            L.append(f'    {M}|{V}|{b}: {"/".join(map(str, chp[f"{M}|{V}|{b}|ideal|SDMTCR"]["cfg"]))} | '
                     f'{"/".join(map(str, ch64[f"{M}|{V}|{b}|ideal|SDMT"]["cfg"]))}')
        L.append('')
    # pipeline code, clean grid on all 7 encoders (extra)
    L.append('## Pipeline code on the clean grid, all 7 encoders (R1, ideal): first - SDMTCR, first - BEST7CR, final - BEST7CR')
    for enc in PANEL:
        P = J[enc]['pipeline'] if enc in J else a1_pipeline(enc)
        J.setdefault(enc, {})['pipeline'] = P
        g = lambda k: P['regions'][f'ideal|R1|{k}']  # noqa: E731
        L.append(f'  {NAMES[enc]:15s} first - SDMt(clean grid) {100 * g("VSAR-SDMTCR")["diff"]:+.2f} '
                 f'[{100 * g("VSAR-SDMTCR")["ci95"][0]:+.2f}, {100 * g("VSAR-SDMTCR")["ci95"][1]:+.2f}] '
                 f'{g("VSAR-SDMTCR")["label"]}; first - best7(clean) {100 * g("VSAR-BEST7CR")["diff"]:+.2f} '
                 f'{g("VSAR-BEST7CR")["label"]} bar {g("VSAR-BEST7CR")["bar"]}; final - best7(clean) '
                 f'{100 * g("VSA-BEST7CR")["diff"]:+.2f} {g("VSA-BEST7CR")["label"]} bar {g("VSA-BEST7CR")["bar"]}')
    json.dump(J, open(f'{RES}/b66_A1.json', 'w'), indent=1)
    open(f'{RES}/b66_A1.txt', 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))


# ----------------------------------------------------------------------------------------------- A2 maps

def vsarm_test(enc, m, M, V):
    fn = f'{OUT}/vsarm/test_{enc}_m{m}_M{M}_V{V}.jsonl.gz'
    return {r['rep']: r for r in map(json.loads, gzip.open(fn, 'rt'))}


def a2_load(enc):
    ch64 = json.load(open(f'{RES}/b64_choice_{enc}.json'))['choice']
    data = {}
    for M in M_GRID:
        for V in VN:
            recs = merged(enc, M, V)
            assert add_sdmt(recs, enc, M, V, f'{ROOT}/b64_out/jobs', ch64, 'SDMT', 'SDMTm0', 'BEST7m0')
            for r in recs:
                for b in BPF:
                    for a in ACCTS:
                        r['pred'][f'VSARm0|{b}|{a}'] = r['pred'][f'VSAR|{b}|{a}']
            for m in (1, 2, 3):
                chm = json.load(open(f'{RES}/b66_choice_sdmtm{m}_{enc}.json'))['choice']
                assert add_sdmt(recs, enc, M, V, f'{OUT}/sdmtm{m}/jobs', chm, 'SDMT', f'SDMTm{m}', f'BEST7m{m}')
                vt = vsarm_test(enc, m, M, V)
                for r in recs:
                    v = vt[r['rep']]
                    assert v['qgold'] == r['qgold']
                    for b in BPF:
                        for a in ACCTS:
                            r['pred'][f'VSARm{m}|{b}|{a}'] = v['pred'][f'VSARm{m}|{b}|{a}']
            data[M, V] = recs
    return data


def a2_part(enc):
    ei = PANEL.index(enc)
    data = a2_load(enc)
    pqc = {}

    def pq(M, V, k):
        if (M, V, k) not in pqc:
            pqc[M, V, k] = V36.per_query(data[M, V], k)
        return pqc[M, V, k]
    J = {'maps': {}, 'cells': {}}
    pairs = [('VSAR', 'SDMT'), ('VSA', 'BEST7'), ('VSAR', 'BEST7')]
    for ai, acct in enumerate(ACCTS):
        for (M, V, b) in REG['R1']:
            J['cells'][f'{acct}|{M}|{V}|{b}'] = {f'{x}m{m}': float(pq(M, V, f'{x}m{m}|{b}|{acct}')[0].mean())
                                                 for x in ('VSAR', 'VSA', 'SDMT', 'BEST7') for m in MAPS}
        for ri, (rn, R) in enumerate(REG.items()):
            for pi, (x, y) in enumerate(pairs):
                cells = []
                for (M, V, b) in R:
                    cl = pq(M, V, f'{x}m0|{b}|{acct}')[1]
                    cells.append(([pq(M, V, f'{x}m{m}|{b}|{acct}')[0] - pq(M, V, f'{y}m{m}|{b}|{acct}')[0] for m in MAPS], cl))
                nb = 10000 if acct == 'ideal' else 2000
                pts, two, ent = V59.twolevel(cells, nb, 66600 + ei * 1000 + ai * 500 + ri * 20 + pi)
                mm = float(pts.mean())
                lo, hi = float(np.quantile(two, 0.025)), float(np.quantile(two, 0.975))
                J['maps'][f'{acct}|{rn}|{x}-{y}'] = {
                    'per_map': [float(p) for p in pts], 'mean': mm, 'sd': float(np.std(pts, ddof=1)),
                    'ci95_two': [lo, hi], 'label': lab(lo, hi), 'bar': bar(mm, lo, hi),
                    'registered_rank_desc': int(1 + sum(p > pts[0] for p in pts[1:]))}
    json.dump(J, open(f'{RES}/b66_A2_part_{enc}.json', 'w'))
    print('a2 part', enc, flush=True)


def a2panel():
    J = {}
    v64 = json.load(open(f'{RES}/b64_verdict.json'))['regions'] if os.path.exists(f'{RES}/b64_verdict.json') else {}
    L = ['# B66 (A2): random-map replicates (maps 0-3; 0 = B64 / B49 / B53\'s registered map). VSARm = first design with',
         '# the map\'s projections / codes / store seeds (own dev choice); VSAm = final design per map (B57 / B59);',
         '# SDMTm = SDMt with the map\'s location order (own dev choice); BEST7m = six key-free (map 0) + SDMTm.',
         '# Two-level bootstrap over maps (b59_verdict.twolevel, 10,000 draws). Points. Label on the map mean; bar = B36.', '']
    for enc in PANEL:
        J[enc] = json.load(open(f'{RES}/b66_A2_part_{enc}.json'))
    for acct in ('ideal', 'real'):
        for (x, y) in (('VSAR', 'SDMT'), ('VSA', 'BEST7'), ('VSAR', 'BEST7')):
            L.append(f'## {x}m - {y}m ({acct})')
            L.append('| encoder | region | map 0 / 1 / 2 / 3 | map mean [two-level 95%] | SD | label (map mean) | bar | registered label |')
            L.append('| --- | --- | --- | --- | --- | --- | --- | --- |')
            for enc in PANEL:
                for rn in ('R1-V2', 'R1-V16', 'R1'):
                    r = J[enc]['maps'][f'{acct}|{rn}|{x}-{y}']
                    reg = v64.get(f'{enc}|{acct}|{rn}|{x}-{y}', {}).get('label', '')
                    reg = {'store leads': 'first leads', 'memory leads': 'second leads'}.get(reg, reg)
                    L.append(f'| {NAMES[enc]} | {rn} | {" / ".join(f"{100 * p:+.2f}" for p in r["per_map"])} | '
                             f'{100 * r["mean"]:+.2f} [{100 * r["ci95_two"][0]:+.2f}, {100 * r["ci95_two"][1]:+.2f}] | '
                             f'{100 * r["sd"]:.2f} | {r["label"]} | {r["bar"]} | {reg} |')
            L.append('')
    json.dump(J, open(f'{RES}/b66_A2.json', 'w'), indent=1)
    open(f'{RES}/b66_A2.txt', 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))


def main():
    a = sys.argv[1:]
    if a[0] == 'a1':
        a1()
    elif a[0] == 'a2':
        for e in a[1:] or PANEL:
            a2_part(e)
    elif a[0] == 'a2panel':
        a2panel()
    else:
        raise SystemExit(a[0])


if __name__ == '__main__':
    main()
