"""B49 (compute, no design change): B32's VSA dev grid of one encoder split by (prep, fmap) part into parallel processes,
each writing its own file b32_out/dev_{enc}_vsa_{prep}_{fmap}.jsonl (the lines of that part already written by the
single-process run are copied first, so no store is run twice); B32's choice later reads all part files.
usage (in b49_out/enc_ENC via b49_run.register): python vsa_split.py ENC PREP FMAP"""
import json
import os
import sys

sys.path.insert(0, '/path/to/work/b49')
import b49_run as R

enc, prep, fmap = sys.argv[1:4]
R32, R36 = R.register(enc)
main = f'b32_out/dev_{enc}_vsa.jsonl'
part = f'b32_out/dev_{enc}_vsa_{prep}_{fmap}.jsonl'
pn = f'vsa:{prep}:{fmap}'
if not os.path.exists(part) and os.path.exists(main):
    with open(part + '.tmp', 'w') as fo:
        for l in open(main):
            try:
                r = json.loads(l)
            except ValueError:  # a line cut by stopping the single-process run
                continue
            if r['part'] == pn:
                fo.write(l if l.endswith('\n') else l + '\n')
    os.replace(part + '.tmp', part)
R32.GRID['prep'] = [prep]
R32.GRID['fmap'] = [fmap]
R32.dev(enc, part, 'vsa', 'cpu')
print('part done', enc, prep, fmap, flush=True)
