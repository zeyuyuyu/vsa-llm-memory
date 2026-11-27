"""B72 transport of test job files over the slow link (compute note 2; procedure only): pack = every test job file of
the given arms into one JSON index (rows without predictions) + one compressed npz of the predictions (uint8);
unpack = rebuild the job files exactly (same names, same rows, predictions as int lists) into the local b72_out.

python3 b72_pack.py pack OUT.npz ARM_PREFIX [ARM_PREFIX ...]    # e.g. _OPT _R_ (LAMP job names)
python3 b72_pack.py unpack OUT.npz
"""
import gzip
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get('B72_OUT', os.path.join(ROOT, 'b72_out'))


def files(prefixes):
    for sub in ('jobs', os.path.join('lamp', 'jobs')):
        jd = os.path.join(OUT, sub)
        if not os.path.isdir(jd):
            continue
        for fn in sorted(os.listdir(jd)):
            if '_test_' in fn and fn.endswith('.json.gz') and any(p in fn for p in prefixes):
                yield sub, fn


def pack(dst, prefixes):
    index, arrs = [], {}
    for k, (sub, fn) in enumerate(files(prefixes)):
        r = json.load(gzip.open(os.path.join(OUT, sub, fn), 'rt'))
        for i, x in enumerate(r['rows']):
            if 'pred' in x:
                p = np.asarray(x.pop('pred'))
                assert p.min() >= 0 and p.max() < 256
                arrs[f'p{k}_{i}'] = p.astype(np.uint8)
                x['_pred'] = f'p{k}_{i}'
        g = np.asarray(r.pop('qgold'))
        arrs[f'q{k}'] = g.astype(np.uint8)
        index.append({'sub': sub, 'fn': fn, 'rec': r, 'q': f'q{k}'})
    np.savez_compressed(dst, index=np.frombuffer(json.dumps(index).encode(), dtype=np.uint8), **arrs)
    print(len(index), 'files ->', dst, os.path.getsize(dst), 'bytes')


def unpack(src):
    z = np.load(src)
    index = json.loads(bytes(z['index']).decode())
    n = 0
    for e in index:
        r = e['rec']
        r['qgold'] = [int(v) for v in z[e['q']]]
        for x in r['rows']:
            if '_pred' in x:
                x['pred'] = [int(v) for v in z[x.pop('_pred')]]
        d = os.path.join(OUT, e['sub'])
        os.makedirs(d, exist_ok=True)
        fn = os.path.join(d, e['fn'])
        with gzip.open(fn + '.tmp', 'wt') as fo:
            json.dump(r, fo)
        os.replace(fn + '.tmp', fn)
        n += 1
    print(n, 'files unpacked into', OUT)


if __name__ == '__main__':
    a = sys.argv[1:]
    if a[0] == 'pack':
        pack(a[1], a[2:])
    else:
        unpack(a[1])
