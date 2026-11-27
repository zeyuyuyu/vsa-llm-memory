"""Recompute the equal-cell requirements N* (cells per stored pair for >= 99.5% of the 4,776 probed facts read exactly,
at that budget and every larger evaluated budget) from the archived per-store test outputs.

B61: sparse-block store (H-SBC), with explaining away (H-SIC), Reed-Muller item codebook (H-HAD).
B63: error-corrected tables with an insertable cuckoo directory (O-LDPC, O-BCH).
At each budget only the design chosen on development data (inputs/b6?_choice.json) is counted; duplicate
evaluations of other designs at the same budget, made for other conditions, are skipped.
The dense store U (76 ... 1,722) and the static tables (19 ... 724) come from B38/B59, whose verdict code is deposited.
Usage: python3 device_cells.py
"""
import json, gzip, math, os, collections
H = os.path.dirname(os.path.abspath(__file__))
CONDS = ['k1', 'k2', 'k4', 'e05', 'k8', 'k16', 'k32']
def nstar(acc, arm):
    out = []
    for c in CONDS:
        I = sorted(i for (a, cc, i) in acc if a == arm and cc == c)
        ns = next((i for i in I if all(acc[(arm, c, k)][0] / acc[(arm, c, k)][1] >= 0.995 for k in I if k >= i)), None)
        out.append(round(8 * 2 ** (ns / 4)) if ns is not None else None)
    return out
# B61
d = json.load(gzip.open(os.path.join(H, 'inputs', 'b61_test1.json.gz')))
D = json.load(open(os.path.join(H, 'inputs', 'b61_choice.json')))['designs']
acc = collections.defaultdict(lambda: [0, 0])
for job in d:
    for r in job['res']:
        i = round(4 * math.log2(r['B'] / 8))
        for rule, v in r['rules'].items():
            if rule in ('H-SBC', 'H-SIC'):
                lam = D.get(rule, {}).get(r['cond'], {}).get(str(i))
                if lam is None or abs((r.get('lam') or -1) - lam) > 1e-9:
                    continue
            a = acc[(rule, r['cond'], i)]; a[0] += v['exact']; a[1] += v['n']
print('%-8s' % 'memory' + ''.join('%7s' % c for c in CONDS))
for arm in ('H-SBC', 'H-SIC', 'H-HAD'):
    print('%-8s' % arm + ''.join('%7s' % x for x in nstar(acc, arm)))
# B63
d = json.load(gzip.open(os.path.join(H, 'inputs', 'b63_test1.json.gz')))
D = json.load(open(os.path.join(H, 'inputs', 'b63_choice.json')))['designs']
acc = collections.defaultdict(lambda: [0, 0])
for job in d:
    for r in job['res']:
        i = round(4 * math.log2(r['B'] / 8))
        ch = D[r['arm']].get(r['cond'], {}).get(str(i))
        if ch is None:
            continue
        f, R = ch
        if f != r['f'] or (R is not None and (r['R'] is None or abs(r['R'] - R) > 1e-9)):
            continue
        e, n = (0, job['nq']) if r.get('infeasible') else (r['exact'], r['n'])
        a = acc[(r['arm'], r['cond'], i)]; a[0] += e; a[1] += n
for arm in ('O-LDPC', 'O-BCH'):
    print('%-8s' % arm + ''.join('%7s' % x for x in nstar(acc, arm)))
