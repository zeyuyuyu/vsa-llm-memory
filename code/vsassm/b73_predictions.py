"""B73 (equal bits): the registered predictions Q1-Q7 scored from results/b73_verdict.json (Q8 = the checks, read from
results/b73_check_*.json). -> results/b73_predictions.txt"""
import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
V = json.load(open(os.path.join(ROOT, 'results', 'b73_verdict.json')))
C = V['claims']
L = ['# B73 (equal bits) predictions (prereg Q1-Q8), scored from results/b73_verdict.json', '']


def held(b):
    return 'HELD' if b else 'FAILED'


L.append(f"Q1 C1 confirmed: {C['C1']['verdict']} -> {held(C['C1']['verdict'] == 'CONFIRMED')}")
L.append(f"Q2 C2 confirmed: {C['C2']['verdict']} -> {held(C['C2']['verdict'] == 'CONFIRMED')}")
n3 = C['C3']['label_ok']
L.append(f"Q3 C3 NLMSC leads on >= 6 of 7: {n3} of {C['C3']['n']} -> {held(n3 >= 6)}; point prediction 'confirmed': "
         f"{C['C3']['verdict']} -> {held(C['C3']['verdict'] == 'CONFIRMED')}; "
         f"Llama label {C['C3']['rows'].get('llama', {}).get('label_one')}")
L.append(f"Q4 C4 NOT confirmed: {C['C4']['verdict']} -> {held(C['C4']['verdict'] != 'CONFIRMED')}; VSAPCX leads on "
         f"{C['C4']['x_leads']} (predicted 0) -> {held(C['C4']['x_leads'] == 0)}")
same = tot = 0
for k in ('C1', 'C2', 'C3', 'C4'):
    for r in C[k]['rows'].values():
        tot += 1
        same += r['label_one'] == r['label_two']
L.append(f'Q5 two-level label = one-level in >= 90% of the (claim x encoder) R1 cases: {same} of {tot} '
         f'({100 * same / max(tot, 1):.0f}%) -> {held(same >= 0.9 * tot)}')
vs = C.get('VSAR-SDMT', {'rows': {}})
n6 = sum(r['label_one'] == 'SDMT' for r in vs['rows'].values())
L.append(f'Q6 VSAR - SDMT: SDMT ahead on >= 5 of 7: {n6} of {len(vs["rows"])} -> {held(n6 >= 5)}')
worst = []
for enc, acc in V['acc'].items():
    for arm, v in acc.items():
        p = V['popqa_acc'].get(f'{enc}|{arm}')
        if v is not None and p is not None:
            worst.append((abs(v - p), enc, arm, v, p))
worst.sort(reverse=True)
L.append(f'Q7 mean R1 accuracy within 10 points of PopQA (arms with a PopQA value: STORE, LAMC, NLMSC): '
         f'{sum(w[0] <= 10 for w in worst)} of {len(worst)}; largest |difference| {worst[0][0]:.2f} '
         f'({worst[0][1]} {worst[0][2]}: {worst[0][3]:.2f} vs {worst[0][4]:.2f}) -> {held(all(w[0] <= 10 for w in worst))}'
         if worst else 'Q7 not scorable')
chk = {}
for k in ('score', 'bits', 'same', 'score_sent', 'bits_sent', 'same_sent', 'score_llm', 'bits_llm', 'same_llm'):
    fn = os.path.join(ROOT, 'results', f'b73_check_{k}.json')
    if os.path.exists(fn):
        chk[k] = json.load(open(fn))
L.append('Q8 checks: ' + '; '.join(f'{k}: ' + ', '.join(f'{a}={b}' for a, b in v.items() if a not in ('notes', 'details', 'per_enc'))
                                    for k, v in chk.items()))
open(os.path.join(ROOT, 'results', 'b73_predictions.txt'), 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))
