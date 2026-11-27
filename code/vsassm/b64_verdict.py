"""B64 verdict (prereg/B64_sdm_training_locations_prereg.md): SDMt's test predictions (b64_out/jobs) added to the merged
test records (b59_out/fam/{enc}/all59_*, V 2 / 16), BEST7 = B36's best-of rule over the six key-free families + SDMt;
B36 / B49 / B60's statistics (per-query correctness, cluster bootstrap with entity-within-store clusters resampled
within each cell, region statistic = mean over cells of the cell mean; b36_verdict.boot_region).

python3 b64_verdict.py part ENC      # -> results/b64_part_{enc}.json
python3 b64_verdict.py panel         # -> results/b64_verdict.{json,txt}, results/b64_predictions.txt
"""
import gzip
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import b36_verdict as V36  # noqa: E402

RES = os.environ.get('B64_RES', f'{ROOT}/results')
JOBS = os.environ.get('B64_JOBS', f'{ROOT}/b64_out/jobs')
PANEL = ['qwen8', 'b46', 'llama', 'mistral', 'e5', 'bge', 'minilm']
LLM, SENT = PANEL[:4], PANEL[4:]
NAMES = {'qwen8': 'Qwen3-8B', 'b46': 'Granite-3.3-8B', 'llama': 'Llama-3.1-8B', 'mistral': 'Mistral-7B',
         'e5': 'e5-base-v2', 'bge': 'bge-small', 'minilm': 'MiniLM-L6'}
M_GRID, VN, BPF, ACCTS = [256, 1024], [2, 16], [4, 8, 16], ['ideal', 'real']
REG = {'R1': [(M, V, b) for M in M_GRID for V in VN for b in BPF]}
for V in VN:
    REG[f'R1-V{V}'] = [(M, V, b) for M in M_GRID for b in BPF]
SIX = ['IDX', 'SF', 'CM', 'BF', 'FBF', 'SDM']
SEVEN = SIX + ['SDMT']
PRIMARY = [('VSAR', 'SDMT'), ('VSAR', 'BEST7'), ('VSA', 'BEST7')]
SECOND = [('VSAR', 'BEST36'), ('VSA', 'BEST36'), ('SDMT', 'SDM'), ('VSA', 'SDMT'), ('BEST7', 'BEST36')]
ARMS = ['EXACT', 'VSAR', 'VSA', 'BEST36', 'BEST7'] + SEVEN
NB = 10000


def choice(enc):
    return json.load(open(f'{RES}/b64_choice_{enc}.json'))['choice']


def job_rows(enc, M, V):
    out = {}
    for rep in range({256: 20, 1024: 5}[M]):
        fn = f'{JOBS}/{enc}_test_M{M}_V{V}_r{rep}.json.gz'
        if not os.path.exists(fn):
            continue
        r = json.load(gzip.open(fn, 'rt'))
        out[rep] = {(tuple(x['cfg']), x['bpf'], x['acct']): (x['pred'], x['bits'], x['N']) for x in r['rows']}
        out[rep]['_qgold'] = r['qgold']
    return out


def load(enc):
    ch = choice(enc)
    data = {}
    for M in M_GRID:
        for V in VN:
            recs = [json.loads(l) for l in gzip.open(f'{ROOT}/b59_out/fam/{enc}/all59_{enc}_M{M}_V{V}.jsonl.gz', 'rt')]
            recs.sort(key=lambda r: r['rep'])
            rows = job_rows(enc, M, V)
            for r in recs:
                jr = rows.get(r['rep'])
                assert jr is not None, ('SDMT test job missing', enc, M, V, r['rep'])
                assert jr['_qgold'] == r['qgold'], (enc, M, V, r['rep'])
                for b in BPF:
                    for a in ACCTS:
                        cfg = tuple(ch[f'{M}|{V}|{b}|{a}|SDMT']['cfg'])
                        p, bt, N = jr[cfg, b, a]
                        r['pred'][f'SDMT|{b}|{a}'] = p
                        r.setdefault('b64', {})[f'SDMT|{b}|{a}'] = [list(cfg), bt, N]
                        f = ch[f'{M}|{V}|{b}|{a}|BEST7']['fam']
                        r['pred'][f'BEST7|{b}|{a}'] = r['pred'][f'{f}|{b}|{a}']
            data[M, V] = recs
    return data, ch


def summ(s, dr):
    lo, hi = float(np.quantile(dr, 0.025)), float(np.quantile(dr, 0.975))
    bar = 'passes' if (s >= 0.02 and lo > 0) else ('loss' if (s <= -0.02 and hi < 0) else 'tie')
    return {'diff': s, 'ci95': [lo, hi], 'label': 'store leads' if lo > 0 else ('memory leads' if hi < 0 else 'level'),
            'bar': bar}


def part(enc):
    ei = PANEL.index(enc)
    data, ch = load(enc)
    J = {'cells': {}, 'regions': {}, 'choices': {}, 'best7': {}}
    pqc = {}

    def pq(M, V, key):
        if (M, V, key) not in pqc:
            pqc[M, V, key] = V36.per_query(data[M, V], key)
        return pqc[M, V, key]

    def k_(arm, b, a):
        return 'EXACT' if arm == 'EXACT' else f'{arm}|{b}|{a}'

    for ai, acct in enumerate(ACCTS):
        for (M, V) in data:
            for b in BPF:
                J['cells'][f'{enc}|{acct}|{M}|{V}|{b}'] = {x: float(pq(M, V, k_(x, b, acct))[0].mean()) for x in ARMS}
        for ri, (rn, R) in enumerate(REG.items()):
            for pi, (x, y) in enumerate(PRIMARY + SECOND):
                diffs = []
                for (M, V, b) in R:
                    v, cl = pq(M, V, k_(x, b, acct))
                    w = pq(M, V, k_(y, b, acct))[0]
                    diffs.append((v - w, cl))
                big = acct == 'ideal' and (x, y) in PRIMARY
                s, _, _, dr = V36.boot_region(diffs, NB if big else 2000, 64000 + ei * 1000 + ai * 500 + ri * 20 + pi)
                J['regions'][f'{enc}|{acct}|{rn}|{x}-{y}'] = summ(s, dr)
    for (M, V) in data:
        for b in BPF:
            for a in ACCTS:
                J['choices'][f'{M}|{V}|{b}|{a}'] = {'cfg': ch[f'{M}|{V}|{b}|{a}|SDMT']['cfg'],
                                                    'N': data[M, V][0]['b64'][f'SDMT|{b}|{a}'][2]}
                J['best7'][f'{M}|{V}|{b}|{a}'] = {'BEST7': ch[f'{M}|{V}|{b}|{a}|BEST7']['fam'],
                                                  'BEST36': ch[f'{M}|{V}|{b}|{a}|BEST7']['BEST36']}
    json.dump(J, open(f'{RES}/b64_part_{enc}.json', 'w'))
    print('part', enc, flush=True)


def f(r):
    return f'{100 * r["diff"]:+6.2f} [{100 * r["ci95"][0]:+6.2f}, {100 * r["ci95"][1]:+6.2f}]'


def panel(encs=PANEL):
    J = {'cells': {}, 'regions': {}, 'choices': {}, 'best7': {}}
    for enc in encs:
        P = json.load(open(f'{RES}/b64_part_{enc}.json'))
        J['cells'].update(P['cells'])
        J['regions'].update(P['regions'])
        J['choices'][enc] = P['choices']
        J['best7'][enc] = P['best7']
    R = J['regions']
    L = ['# B64 verdict (prereg/B64_sdm_training_locations_prereg.md). Accuracy points over R1 (V 2 / 16 x 4, 8, 16 bits',
         '# per fact x M 256 / 1,024); 95% cluster bootstrap (entity within store). VSAR = first design (B49, random value',
         '# codes); VSA = final design (B53); SDMT = SDM with training-split hard locations (B64); SDM = B36\'s SDM;',
         '# BEST36 = B36\'s best of six (IDX, SF, CM, BF, FBF, SDM); BEST7 = the six + SDMT, B36\'s rule on dev.',
         '# Label: "store leads" iff lower > 0; "memory leads" iff upper < 0; else "level". Bar (B36 H1): R1 diff >= +2',
         '# and lower > 0 = passes; <= -2 and upper < 0 = loss; else tie.', '']
    for acct in ACCTS:
        for (x, y) in PRIMARY:
            L.append(f'## {x} - {y} ({acct})')
            L.append('| encoder | R1-V2 | label | R1-V16 | label | R1 | label | bar |')
            L.append('| --- | --- | --- | --- | --- | --- | --- | --- |')
            for enc in encs:
                row = []
                for rn in ('R1-V2', 'R1-V16', 'R1'):
                    r = R[f'{enc}|{acct}|{rn}|{x}-{y}']
                    row += [f(r), r['label']]
                row.append(R[f'{enc}|{acct}|R1|{x}-{y}']['bar'])
                L.append(f'| {NAMES[enc]} | ' + ' | '.join(row) + ' |')
            for rn in ('R1-V2', 'R1-V16', 'R1'):
                for gn, grp in (('LLM', LLM), ('sentence', SENT)):
                    g = [e for e in grp if e in encs]
                    c = {lb: sum(R[f'{e}|{acct}|{rn}|{x}-{y}']['label'] == lb for e in g)
                         for lb in ('store leads', 'level', 'memory leads')}
                    L.append(f'  {rn:7s} {gn:8s}: store leads {c["store leads"]}, level {c["level"]}, memory leads '
                             f'{c["memory leads"]} (of {len(g)})')
            if (x, y) != ('VSAR', 'SDMT'):
                for gn, grp in (('all', PANEL), ('LLM', LLM), ('sentence', SENT)):
                    g = [e for e in grp if e in encs]
                    c = {lb: sum(R[f'{e}|{acct}|R1|{x}-{y}']['bar'] == lb for e in g) for lb in ('passes', 'tie', 'loss')}
                    L.append(f'  bar (R1) {gn:8s}: passes {c["passes"]}, tie {c["tie"]}, loss {c["loss"]} (of {len(g)})')
            L.append('')
        L.append(f'## Secondary ({acct})')
        L.append('| encoder | ' + ' | '.join(f'{x} - {y} R1' for x, y in SECOND) + ' |')
        L.append('| --- |' + ' --- |' * len(SECOND))
        for enc in encs:
            L.append(f'| {NAMES[enc]} | ' + ' | '.join(f(R[f"{enc}|{acct}|R1|{x}-{y}"]) for x, y in SECOND) + ' |')
        L.append('')
    # per-cell accuracies (ideal)
    L.append('## Per-cell test accuracy (ideal), points: VSAR / VSA / SDM / SDMT / BEST36 / BEST7 (BEST7 family)')
    for enc in encs:
        L.append(f'  {NAMES[enc]}')
        for M in M_GRID:
            for V in VN:
                for b in BPF:
                    c = J['cells'][f'{enc}|ideal|{M}|{V}|{b}']
                    fam = J['best7'][enc][f'{M}|{V}|{b}|ideal']['BEST7']
                    L.append(f'    M={M:5d} V={V:3d} b={b:3d}: ' + ' / '.join(f'{100 * c[x]:5.1f}' for x in
                             ('VSAR', 'VSA', 'SDM', 'SDMT', 'BEST36', 'BEST7')) + f'  ({fam})')
    L.append('')
    L.append('## BEST7 family chosen on dev (ideal), R1 cells (12 per encoder)')
    for enc in encs:
        cnt = {}
        for k, v in J['best7'][enc].items():
            if k.endswith('|ideal'):
                cnt[v['BEST7']] = cnt.get(v['BEST7'], 0) + 1
        L.append(f'  {NAMES[enc]:15s} ' + ', '.join(f'{k} {v}' for k, v in sorted(cnt.items(), key=lambda t: -t[1])))
    L.append('')
    L.append('## SDMt dev choices (ideal), R1: prep / loc / a / ctr / q / share, N used on test store 0')
    for enc in encs:
        for M in M_GRID:
            for V in VN:
                cs = [J['choices'][enc][f'{M}|{V}|{b}|ideal'] for b in BPF]
                L.append(f'  {NAMES[enc]:15s} M={M:5d} V={V:3d}: ' + ' ; '.join(
                    f'{"/".join(map(str, c["cfg"]))} N={c["N"]}' for c in cs))
    L.append('')
    json.dump(J, open(f'{RES}/b64_verdict.json', 'w'), indent=1)
    open(f'{RES}/b64_verdict.txt', 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))
    predictions(J, encs)


def predictions(J, encs):
    R = J['regions']

    def g(e, rn, x, y, acct='ideal'):
        return R[f'{e}|{acct}|{rn}|{x}-{y}']

    out = []
    llm = [e for e in LLM if e in encs]
    sent = [e for e in SENT if e in encs]
    v = [100 * g(e, 'R1', 'SDMT', 'SDM')['diff'] for e in llm]
    out.append(('P1 SDMT - SDM over R1 >= +5 on each LLM', all(x >= 5 for x in v), v))
    v = [100 * g(e, 'R1', 'VSAR', 'SDMT')['diff'] for e in llm]
    out.append(('P2 VSAR - SDMT over R1 in [-1, +3] on each LLM', all(-1 <= x <= 3 for x in v), v))
    bars = [g(e, 'R1', 'VSAR', 'BEST7')['bar'] for e in llm]
    labs = [g(e, 'R1', 'VSAR', 'BEST7')['label'] for e in llm]
    out.append(('P3 VSAR - BEST7: bar passes on <= 1 of 4 LLMs and store leads on >= 2 of 4',
                bars.count('passes') <= 1 and labs.count('store leads') >= 2, (bars, labs)))
    bars = [g(e, 'R1', 'VSA', 'BEST7')['bar'] for e in llm]
    labs = [g(e, 'R1', 'VSA', 'BEST7')['label'] for e in llm]
    out.append(('P4 VSA - BEST7: store leads on all 4 LLMs and bar passes on >= 3 of 4',
                labs.count('store leads') == len(llm) and bars.count('passes') >= 3, (bars, labs)))
    v2 = [100 * g(e, 'R1-V2', 'VSA', 'BEST7')['diff'] for e in llm]
    v16 = [100 * g(e, 'R1-V16', 'VSA', 'BEST7')['diff'] for e in llm]
    out.append(('P5 VSA - BEST7: R1-V2 >= +4 on all LLMs; R1-V16 in [-2, +4] on each',
                all(x >= 4 for x in v2) and all(-2 <= x <= 4 for x in v16), (v2, v16)))
    if sent:
        labs = [g(e, 'R1', 'VSAR', 'BEST7')['label'] for e in sent]
        bars = [g(e, 'R1', 'VSA', 'BEST7')['bar'] for e in sent]
        out.append(('P6 sentence: VSAR - BEST7 memory leads on all 3; VSA - BEST7 bar passes on none',
                    labs.count('memory leads') == len(sent) and bars.count('passes') == 0, (labs, bars)))
    cnts = []
    for e in llm:
        cnts.append(sum(v['BEST7'] == 'SDMT' for k, v in J['best7'][e].items() if k.endswith('|ideal')))
    out.append(('P7 BEST7 = SDMT in >= 8 of 12 R1 cells of each LLM', all(c >= 8 for c in cnts), cnts))
    cs = [c for e in encs for k, c in J['choices'][e].items() if k.endswith('|ideal')]
    a16 = sum(c['cfg'][2] <= 16 for c in cs) / len(cs)
    la = sum(c['cfg'][1] == 'LA' for c in cs) / len(cs)
    c16 = [c for e in encs for k, c in J['choices'][e].items() if k.endswith('|ideal') and k.split('|')[1] == '16']
    oh = sum(c['cfg'][3] == 'onehot' for c in c16) / len(c16)
    out.append(('P8 dev choices: a <= 16 in >= 75%; LA in >= 50%; onehot in >= 50% of V = 16 cells',
                a16 >= 0.75 and la >= 0.5 and oh >= 0.5, (round(a16, 3), round(la, 3), round(oh, 3), len(cs))))
    b49 = json.load(open(f'{ROOT}/results/b49_verdict.json'))
    dv = {e: 100 * (g(e, 'R1', 'VSAR', 'BEST36')['diff'] - b49['regions'][f'{e}|ideal|R1|BEST36']['diff'])
          for e in encs} if b49 else {}
    out.append(('P9 VSAR - BEST36 reproduces B49 within 0.01 points', bool(dv) and all(abs(x) <= 0.01 for x in dv.values()),
                {e: (round(100 * g(e, 'R1', 'VSAR', 'BEST36')['diff'], 3), round(x, 6)) for e, x in dv.items()}))
    L = ['# B64 predictions (prereg P1-P9; P10 = checks, results/b64_check.txt). Points.']
    for name, ok, val in out:
        L.append(f'{name}: {"HELD" if ok else ("FAILED" if ok is not None else "see check")} {val}')
    open(f'{RES}/b64_predictions.txt', 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))
    _ = b49


def main():
    a = sys.argv[1:]
    if a[0] == 'part':
        part(a[1])
    elif a[0] == 'panel':
        panel(a[1].split(',') if len(a) > 1 else PANEL)
    else:
        raise SystemExit(a[0])


if __name__ == '__main__':
    main()
