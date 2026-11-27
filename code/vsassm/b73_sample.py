"""B73 sample reading by the AI-assisted process (prereg 'Checks'): for an encoder, the R1-V16 cell with the lowest
STORE - LAMC on the fresh set; 10 queries only LAMC answers right and 10 only the store answers right (alias, the
stored entity's label, gold value, both answers). Stores re-drawn by b73_run's make_store (no codes needed).

python3 b73_sample.py ENC [ENC ...]   # needs b73_out/asm/{enc}.json.gz and the fresh entity file -> stdout
"""
import gzip
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import b73_run as T  # noqa: E402  (patches b32_common.make_store's seed tag)
import b32_common as C  # noqa: E402


class LitePool:
    def __init__(self, root):
        d = json.load(open(os.path.join(root, 'b32_data', 'entities.json')))
        self.ents = d['entities']
        S = json.load(open(os.path.join(root, 'b32_data', 'emb_strings.json')))
        self.idx = {s: i for i, s in enumerate(S)}
        self.X = np.zeros((len(S), 1), np.float32)
        self.by_split = {s: [e for e in self.ents if e['split'] == s] for s in ('train', 'dev', 'test')}


def main():
    pool = LitePool(T.FRESH)
    for enc in sys.argv[1:]:
        d = json.load(gzip.open(os.path.join(T.OUT, 'asm', f'{enc}.json.gz'), 'rt'))
        best = None
        for M in T.M_GRID:
            recs = d[f'{M}|16']
            for b in T.BPF:
                g = np.concatenate([r['qgold'] for r in recs])
                s = np.concatenate([np.asarray(r['pred'][f'STORE|{b}|ideal']) == np.asarray(r['qgold']) for r in recs])
                l = np.concatenate([np.asarray(r['pred'][f'LAMC|{b}|ideal']) == np.asarray(r['qgold']) for r in recs])
                diff = s.mean() - l.mean()
                if best is None or diff < best[0]:
                    best = (diff, M, b)
        diff, M, b = best
        print(f'## {enc}: lowest R1-V16 cell STORE - LAMC = {100 * diff:+.2f} points at M = {M}, {b} bits per fact')
        rows = {'LAMC only': [], 'STORE only': []}
        for r in d[f'{M}|16']:
            st = C.make_store(pool, 'test', M, 16, r['rep'])
            assert st['qgold'].tolist() == r['qgold'], 'store re-draw differs'
            ps, pl = r['pred'][f'STORE|{b}|ideal'], r['pred'][f'LAMC|{b}|ideal']
            for i, gg in enumerate(r['qgold']):
                lab = st['names'][st['qent'][i]]
                item = (r['rep'], st['qalias'][i], lab, gg, ps[i], pl[i])
                if pl[i] == gg and ps[i] != gg:
                    rows['LAMC only'].append(item)
                elif ps[i] == gg and pl[i] != gg:
                    rows['STORE only'].append(item)
        rng = np.random.default_rng(73)
        for k, lst in rows.items():
            print(f'### {k}: {len(lst)} queries in the cell; 10 drawn at random (seed 73)')
            for j in sorted(rng.choice(len(lst), size=min(10, len(lst)), replace=False)):
                rep, al, lab, gg, a1, a2 = lst[j]
                print(f'  store {rep:2d} | alias {al!r} -> label {lab!r} | gold {gg} | STORE {a1} | LAMC {a2}')
        print()


if __name__ == '__main__':
    main()
