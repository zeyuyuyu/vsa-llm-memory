"""B49 learned map for the learned-hashing competitors (prereg/B49_panel_mechanism_competitors_prereg.md, 'LH'): for
every (s, r) in b49_common.LH_GRID, the shrinkage-LDA map fitted on the training split's alias-label pairs, scored by
B32's dev entity-retrieval criterion (exact cosine; every dev label stored, every dev alias a query). No memory, no
value, no test string. Choice: highest criterion; ties within 0.001 to the smaller r, then the larger s.

python3 b49_lhchoose.py OUT.json TAG=FILETAG ...      # run in a directory whose b32_data holds emb_FILETAG.npy
"""
import json
import sys

import numpy as np

sys.path.insert(0, '/path/to/vsassm')
import b32_common as C  # noqa: E402
import b49_common as B  # noqa: E402


def choose(filetag):
    pool = C.Pool(filetag)
    X = pool.X
    base = {}
    for kind in ('center', 'whiten'):
        P = C.Prep(kind, pool.train_matrix())
        base[kind] = B.criterion(P(X), pool)
    rows = []
    for s in B.LH_GRID['s']:
        rmax = max(B.LH_GRID['r'])
        lp = B.LearnedPrep(X, pool, s, min(rmax, X.shape[1]))
        A = lp.A
        for r in B.LH_GRID['r']:
            if r > X.shape[1]:
                continue
            lp.A = A[:, :r]
            rows.append({'s': s, 'r': r, 'crit': B.criterion(lp(X), pool)})
            print(filetag, s, r, round(rows[-1]['crit'], 4), flush=True)
    top = max(x['crit'] for x in rows)
    tied = sorted([x for x in rows if x['crit'] >= top - 0.001], key=lambda x: (x['r'], -x['s']))
    return {'choice': tied[0], 'grid': rows, 'base_criterion': base}


def main():
    out = {}
    for a in sys.argv[2:]:
        tag, ft = a.split('=')
        out[tag] = {'file_tag': ft, **choose(ft)}
        print(tag, json.dumps(out[tag]['choice']), out[tag]['base_criterion'], flush=True)
        json.dump(out, open(sys.argv[1], 'w'), indent=1)


if __name__ == '__main__':
    main()
