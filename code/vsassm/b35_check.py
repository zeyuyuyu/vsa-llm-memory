"""B35 second angle: rescore every run from its saved per-question outputs, independently of b35_run.py.

For each run directory (out/<run>/<split>.jsonl.gz) this regenerates the evaluation questions from b35_data (the gold
answers are recomputed by the DICT parse of the question text against the instance's facts, not read from the file),
re-normalises each saved prediction with its own rule (first line, strip spaces and one trailing period, exact
string), and compares (a) the number of questions and their identity with the regenerated set, (b) the per-question
correctness with the saved flag, (c) the per-cell accuracy with the run's summary.json. Any difference is printed.
usage: python b35_check.py <out_dir_with_runs> <split> [run ...]
"""
import gzip
import json
import os
import sys

import b35_data as D


def rescore(pred):
    s = pred.split('\n')[0].strip()
    if s.endswith('.'):
        s = s[:-1]
    return s.strip()


def main():
    root, split = sys.argv[1], sys.argv[2]
    runs = sys.argv[3:] or sorted(d for d in os.listdir(root) if os.path.exists(os.path.join(root, d, f'{split}.jsonl.gz')))
    es = {(x['N'], x['idx']): x for x in D.eval_set(split)}
    bad_total = 0
    for r in runs:
        rows = [json.loads(l) for l in gzip.open(os.path.join(root, r, f'{split}.jsonl.gz'), 'rt')]
        summ = json.load(open(os.path.join(root, r, 'summary.json')))
        arm = summ['config']['arm']
        qkeys = ('q1',) if arm == 'vsaq' else ('q1', 'q2')
        expect = {(x['N'], x['idx'], x[q]['qtype']) for x in es.values() for q in qkeys}
        got = {(z['N'], z['idx'], z['qtype']) for z in rows}
        bad = []
        if expect != got or len(rows) != len(expect):
            bad.append(f'question set differs: expected {len(expect)}, got {len(rows)} rows / {len(got)} distinct')
        cell = {}
        flips = 0
        for z in rows:
            x = es[(z['N'], z['idx'])]
            q = x['q1'] if z['qtype'] == 'hop1' else x['q2']
            if q['text'] != z['question']:
                bad.append(f"question text differs at {z['N']},{z['idx']}")
            gold = D.dict_answer(x, q)
            c = int(rescore(z['pred']) == gold)
            flips += c != z['correct']
            cell.setdefault(f"N{z['N']}_{z['qtype']}", []).append(c)
        if flips:
            bad.append(f'{flips} correctness flags differ')
        acc = {k: sum(v) / len(v) for k, v in cell.items()}
        for k, v in acc.items():
            if abs(v - summ[split].get(k, -1)) > 1e-9:
                bad.append(f'{k}: rescored {v:.4f} vs summary {summ[split].get(k)}')
        bad_total += len(bad)
        print(r, 'OK' if not bad else 'MISMATCH', json.dumps({k: round(v, 4) for k, v in sorted(acc.items())}))
        for b in bad[:10]:
            print('   ', b)
    print('total mismatches:', bad_total)
    return bad_total


if __name__ == '__main__':
    sys.exit(1 if main() else 0)
