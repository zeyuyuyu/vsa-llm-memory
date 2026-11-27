"""B64 sample reading material (prereg 'Checks'): Qwen3-8B and e5-base-v2, the R1 cell with the lowest VSAR - SDMT
(ideal): 10 queries only SDMt answers right and 10 only the store (first design) answers right, with alias, label,
gold, both answers, and the location strings the alias activated (SDMt's chosen configuration, recomputed).

python3 b64_handread.py ENC [ENC ...]   # -> results/b64_sample_material.txt
"""
import gzip
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import b32_common as C  # noqa: E402
import b64_run as R  # noqa: E402


def main(encs):
    V = json.load(open(f'{ROOT}/results/b64_verdict.json'))
    L = []
    for enc in encs:
        ch = json.load(open(f'{ROOT}/results/b64_choice_{enc}.json'))['choice']
        cells = [(M, Vv, b) for M in R.M_GRID for Vv in R.VN for b in R.BPF]
        d = {c: V['cells'][f'{enc}|ideal|{c[0]}|{c[1]}|{c[2]}'] for c in cells}
        M, Vv, b = min(cells, key=lambda c: d[c]['VSAR'] - d[c]['SDMT'])
        cfg = tuple(ch[f'{M}|{Vv}|{b}|ideal|SDMT']['cfg'])
        L.append(f'## {enc}: lowest VSAR - SDMT cell M={M} V={Vv} b={b}: VSAR {100 * d[M, Vv, b]["VSAR"]:.1f}, '
                 f'SDMT {100 * d[M, Vv, b]["SDMT"]:.1f}; SDMt cfg {cfg}')
        recs = [json.loads(l) for l in gzip.open(f'{ROOT}/b59_out/fam/{enc}/all59_{enc}_M{M}_V{Vv}.jsonl.gz', 'rt')]
        recs.sort(key=lambda r: r['rep'])
        pool, preps, Ls, lr = R.setup(enc)
        rng = np.random.default_rng(64)
        only = {'SDMT': [], 'VSAR': []}
        for r in recs:
            j = json.load(gzip.open(f'{ROOT}/b64_out/jobs/{enc}_test_M{M}_V{Vv}_r{r["rep"]}.json.gz', 'rt'))
            row = [x for x in j['rows'] if tuple(x['cfg']) == cfg and x['bpf'] == b and x['acct'] == 'ideal'][0]
            ps, pv, g = np.array(row['pred']), np.array(r['pred'][f'VSAR|{b}|ideal']), np.array(r['qgold'])
            for i in range(len(g)):
                if ps[i] == g[i] and pv[i] != g[i]:
                    only['SDMT'].append((r['rep'], i, row['N']))
                if pv[i] == g[i] and ps[i] != g[i]:
                    only['VSAR'].append((r['rep'], i, row['N']))
        L.append(f'  queries only SDMt right: {len(only["SDMT"])}; only the store right: {len(only["VSAR"])}')
        prep, loc, a = cfg[0], cfg[1], cfg[2]
        for side in ('SDMT', 'VSAR'):
            pick = [only[side][k] for k in sorted(rng.choice(len(only[side]), min(10, len(only[side])), replace=False))]
            L.append(f'  -- only {side} right (10 sampled)')
            for rep, i, N in pick:
                st = C.make_store(pool, 'test', M, Vv, rep)
                r = recs[rep]
                P = preps[prep]
                q = P(st['queries'][i:i + 1])
                k = P(st['keys'][st['qent'][i]:st['qent'][i] + 1])
                Lc = Ls[prep][loc][:N]
                sq = (q @ Lc.T)[0]
                sk = (k @ Lc.T)[0]
                aq = np.argsort(-sq, kind='stable')[:min(a, 3)]
                ak = set(np.argsort(-sk, kind='stable')[:a].tolist())
                shared = len(set(np.argsort(-sq, kind='stable')[:a].tolist()) & ak)
                j = json.load(gzip.open(f'{ROOT}/b64_out/jobs/{enc}_test_M{M}_V{Vv}_r{rep}.json.gz', 'rt'))
                row = [x for x in j['rows'] if tuple(x['cfg']) == cfg and x['bpf'] == b and x['acct'] == 'ideal'][0]
                cosk = float((q @ k.T)[0, 0])
                L.append(f'    alias {st["qalias"][i]!r} -> label {st["names"][st["qent"][i]]!r}; gold {st["qgold"][i]}; '
                         f'SDMT {row["pred"][i]} store {r["pred"][f"VSAR|{b}|ideal"][i]}; cos(alias, label) {cosk:.2f}; '
                         f'alias and label share {shared} of {a} active locations; alias top locations: ' +
                         ', '.join(f'{pool.strings[lr[loc][l]]!r} ({sq[l]:.2f})' for l in aq))
        L.append('')
    open(f'{ROOT}/results/b64_sample_material.txt', 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main(sys.argv[1:] or ['qwen8', 'e5'])
