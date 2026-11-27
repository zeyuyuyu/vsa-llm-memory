"""B49 (compute, no design change): B32's and B36's choices for one panel encoder once every dev part is done, with the
VSA dev read from the six (prep, fmap) part files of b49_box/vsa_split.py (b49_dev.sh's own b32choose read the stopped
single-process file and is superseded). Same rules and code (b32_run.choose, b36_run.choose)."""
import json
import os
import sys

sys.path.insert(0, '/path/to/work/b49')
import b49_run as R

enc = sys.argv[1]
R32, R36 = R.register(enc)
parts = [f'b32_out/dev_{enc}_vsa_{p}_{f}.jsonl' for p in ('center', 'whiten') for f in ('lin', 'sgn', 'sgn3')]
for fn in parts:
    n = sum(1 for _ in open(fn))
    assert n == 36, (fn, n)
assert sum(1 for _ in open(f'b32_out/dev_{enc}_idx.jsonl')) == 36
R32.choose(enc, ','.join(parts + [f'b32_out/dev_{enc}_idx.jsonl']), f'results/b32_choice_{enc}.json')
R36.choose(enc)
open('logs/steps.log', 'a').write(__import__('time').strftime('%Y-%m-%d %H:%M:%S') +
                                  ' CHOICES_DONE (b32choose on the six VSA part files + idx; B36 choose)\n')
print('choices done', enc)
