"""B35 hand-reading helper: print questions with the instance's relevant facts, gold and each run's prediction.

usage: python b35_handread.py <out_root> <split> <runA> [<runB> ...] [--N 32] [--qtype hop1] [--k 10] [--where wrongA]
  where: wrongA (runA wrong), disagree (runA and runB differ in correctness), any.
For each picked question: the question, the facts about the people it names (and the manager for two hop), the gold,
and every run's raw prediction, so one can see whether an error is a near miss (right attribute, wrong value; a value
of another person; a value of the wrong attribute) or noise.
"""
import argparse
import gzip
import json
import os
import random

import b35_data as D

ap = argparse.ArgumentParser()
ap.add_argument('root'); ap.add_argument('split'); ap.add_argument('runs', nargs='+')
ap.add_argument('--N', type=int, default=32); ap.add_argument('--qtype', default='hop1')
ap.add_argument('--k', type=int, default=10); ap.add_argument('--where', default='wrongA')
ap.add_argument('--seed', type=int, default=0)
a = ap.parse_args()
es = {(x['N'], x['idx']): x for x in D.eval_set(a.split)}
R = {}
for r in a.runs:
    R[r] = {(z['N'], z['idx'], z['qtype']): z for z in
            (json.loads(l) for l in gzip.open(os.path.join(a.root, r, f'{a.split}.jsonl.gz'), 'rt'))}
keys = [k for k in R[a.runs[0]] if k[0] == a.N and k[2] == a.qtype]
if a.where == 'wrongA':
    keys = [k for k in keys if not R[a.runs[0]][k]['correct']]
elif a.where == 'disagree':
    keys = [k for k in keys if k in R[a.runs[1]] and R[a.runs[0]][k]['correct'] != R[a.runs[1]][k]['correct']]
random.Random(a.seed).shuffle(keys)
kinds = {}
for k in keys:
    x = es[k[:2]]
    q = x['q1'] if k[2] == 'hop1' else x['q2']
    names = {c[0] for c in q['chain']}
    allv = {v: (e, r) for e, r, v in x['facts']}
    p = R[a.runs[0]][k]['pred'].strip()
    tag = 'correct' if p == q['answer'] else (
        'value of another person (same attribute)' if p in allv and allv[p][1] == q['chain'][-1][1] else
        'value of another attribute/person in the memory' if p in allv else
        'right type, not in memory' if p in D.ATTRS.get(q['chain'][-1][1], []) else 'other')
    kinds[tag] = kinds.get(tag, 0) + 1
for k in keys[:a.k]:
    x = es[k[:2]]
    q = x['q1'] if k[2] == 'hop1' else x['q2']
    names = {c[0] for c in q['chain']}
    print(f"--- N={k[0]} idx={k[1]} {q['text']}  gold={q['answer']}")
    for e, r, v in x['facts']:
        if e in names:
            print('     ', D.fact_text(e, r, v))
    for r in a.runs:
        z = R[r].get(k)
        print(f'   {r}: {z["pred"]!r} ({"ok" if z["correct"] else "wrong"})' if z else f'   {r}: -')
print(f'{len(keys)} picked; error kinds of {a.runs[0]} over all picked:', json.dumps(kinds))
