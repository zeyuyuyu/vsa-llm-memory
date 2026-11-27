"""B35: merge the test runs finished on the A800 box (b35_out/test_box) and on PACE (b35_out/test_pace) into
b35_out/test, as registered: where both machines finished the same run, the copy whose summary.json was written
first is scored and the other is reported as a reproducibility check (per-cell accuracy difference and the share of
questions with the same correctness). Also copies the external decoding and the from-scratch control.
usage: python b35_collect.py
"""
import gzip
import json
import os
import shutil

R = 'b35_out'
SRC = {'box': os.path.join(R, 'test_box'), 'pace': os.path.join(R, 'test_pace')}
DST = os.path.join(R, 'test')
os.makedirs(DST, exist_ok=True)


def rows(d):
    return {(z['N'], z['idx'], z['qtype']): z['correct']
            for z in (json.loads(l) for l in gzip.open(os.path.join(d, 'test.jsonl.gz'), 'rt'))}


runs = {}
for m, root in SRC.items():
    if not os.path.isdir(root):
        continue
    for r in os.listdir(root):
        s = os.path.join(root, r, 'summary.json')
        if r.startswith('test_') and os.path.exists(s) and 'test' in json.load(open(s)):
            runs.setdefault(r, {})[m] = os.path.join(root, r)
sources, repro = {}, {}
for r, ms in sorted(runs.items()):
    first = min(ms, key=lambda m: os.path.getmtime(os.path.join(ms[m], 'summary.json')))
    sources[r] = dict(scored=first, copies=sorted(ms))
    d = os.path.join(DST, r)
    if os.path.exists(d):
        shutil.rmtree(d)
    shutil.copytree(ms[first], d)
    if len(ms) == 2:
        a, b = (json.load(open(os.path.join(ms[m], 'summary.json')))['test'] for m in ('box', 'pace'))
        ra, rb = rows(ms['box']), rows(ms['pace'])
        repro[r] = dict(max_cell_diff=max(abs(a[k] - b[k]) for k in a),
                        same_correctness=sum(ra[k] == rb[k] for k in ra) / len(ra))
for f in ('ext_test.json',):
    for m, root in SRC.items():
        if os.path.exists(os.path.join(root, f)):
            shutil.copy(os.path.join(root, f), os.path.join(DST, f))
for m, root in SRC.items():
    sc = os.path.join(root, 'scratch')
    if os.path.isdir(sc):
        os.makedirs(os.path.join(DST, 'scratch'), exist_ok=True)
        for f in os.listdir(sc):
            if f.startswith('test_scratch_'):
                shutil.copy(os.path.join(sc, f), os.path.join(DST, 'scratch', f))
json.dump(dict(sources=sources, reproducibility=repro), open(os.path.join(DST, 'SOURCES.json'), 'w'), indent=1)
print(len(sources), 'runs;', sum(v['scored'] == 'pace' for v in sources.values()), 'scored from PACE;',
      len(repro), 'run on both machines')
for r, v in repro.items():
    print(f"  {r}: max cell difference {v['max_cell_diff']:.3f}, same correctness {v['same_correctness']:.3f}")
