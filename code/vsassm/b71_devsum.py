"""B71: dev-row summaries computed where the dev jobs ran (the link to the box is slow, compute note 4).

For every dev key of b71_run.load_dev (arm, M, V, bpf, cfg[, prep, fmap]) the number of dev stores and the mean over
the stores (sorted rep order, numpy.mean, exactly as b71_run.choose's mean_of) of every accuracy field, and the per-epoch
mean of the update counts; plus a bits audit of every dev row (b71_check.bits' rules). b71_run.choose / b71_verdict.audit
read these summaries when B71_DEVSUM=1.

python3 b71_devsum.py ENC [ENC ...]     # -> B71_OUT/devsum_{enc}.json.gz
"""
import gzip
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

FIELDS = ('acc', 'acc_float', 'acc_keys', 'acc_keys_float')


def summarise(enc):
    import b71_run as E
    acc = E.load_dev(enc)
    out = []
    for k, d in acc.items():
        reps = sorted(d)
        m = {}
        if isinstance(d[reps[0]], dict):
            for f in FIELDS:
                if f in d[reps[0]]:
                    m[f] = float(np.mean([d[i][f] for i in reps]))
            if 'nupd' in d[reps[0]]:
                L = len(d[reps[0]]['nupd'])
                m['nupd'] = [float(np.mean([d[i]['nupd'][e] for i in reps])) for e in range(L)]
        else:
            m['acc'] = float(np.mean([d[i] for i in reps]))
        out.append([list(k[:4]) + [list(k[4])] + list(k[5:]), len(reps), m])
    fn = os.path.join(E.OUT, f'devsum_{enc}.json.gz')
    with gzip.open(fn + '.tmp', 'wt') as fo:
        json.dump(out, fo)
    os.replace(fn + '.tmp', fn)
    print(enc, len(out), 'keys ->', fn, flush=True)


if __name__ == '__main__':
    for e in sys.argv[1:]:
        summarise(e)
