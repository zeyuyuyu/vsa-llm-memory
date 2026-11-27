"""B74 predictions scorer (prereg/B74_factorial_labels_matched_prereg.md, "Predictions"): reads results/b74_part_*.json
and the check files; -> results/b74_predictions.txt.  Written before any test output exists."""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
RES = os.environ.get('B74_RES', f'{ROOT}/results')
PANEL = ['qwen8', 'b46', 'llama', 'mistral', 'e5', 'bge', 'minilm']
LLM = PANEL[:4]


def main():
    P = {e: json.load(open(f'{RES}/b74_part_{e}.json')) for e in PANEL if os.path.exists(f'{RES}/b74_part_{e}.json')}
    L = []

    def put(name, held, detail):
        L.append(f'{name}: {"HELD" if held else "FAILED"} - {detail}')
    if all('A' in J for J in P.values()):
        A = {e: J['A'] for e, J in P.items()}
        g = lambda e, nm: A[e][f'R1|{nm}']  # noqa: E731
        c1 = sum(g(e, 'W_T')['diff'] > g(e, 'W_R')['diff'] for e in A)
        c2 = sum(g(e, 'I_T')['one_label'] == '+' for e in A)
        c3 = sum(g(e, 'W_P')['diff'] > g(e, 'W_R')['diff'] for e in A)
        c4 = sum(g(e, 'I_P')['one_label'] == '+' for e in A)
        put('PA1', c1 >= 5 and c2 >= 4 and c3 >= 5 and c4 >= 4,
            f'W_T > W_R on {c1}/7 (>= 5), I_T above 0 on {c2} (>= 4); W_P > W_R on {c3} (>= 5), I_P above 0 on {c4} (>= 4)')
        c1 = sum(g(e, 'F_P|H')['diff'] < 0 for e in A)
        c2 = sum(g(e, 'F_T|H')['diff'] >= 0 for e in A)
        put('PA2', c1 >= 5 and c2 >= 4, f'F_P|H < 0 on {c1}/7 (>= 5); F_T|H >= 0 on {c2}/7 (>= 4)')
        fp = sum(P[e]['A_attr']['R1|P']['rule'] == 'features larger (identified)' for e in P)
        ft = sum(P[e]['A_attr']['R1|T']['rule'] == 'features larger (identified)' for e in P)
        sup = fp >= 6 and ft >= 6
        put('PA3', fp <= 2 and ft <= 4 and not sup,
            f'"features larger (identified)" at P on {fp}/7 (<= 2), at T on {ft}/7 (<= 4); statement '
            f'{"SUPPORTED" if sup else "UNSUPPORTED"} (predicted UNSUPPORTED)')
        c = sum(P[e]['A_attr']['R1|T']['shapley']['write'] >= P[e]['A_attr']['R1|T']['shapley']['features'] for e in P)
        put('PA4', c >= 4, f'Shapley write >= features at T on {c}/7 (>= 4)')
        c1 = sum(g(e, 'LT-LAMC')['one_label'] == '-' for e in A)
        c2 = sum(abs(g(e, 'LT-LAMC')['diff']) <= 0.025 for e in A)
        put('PA5', c1 >= 4 and c2 >= 5, f'LAMc leads LT on {c1}/7 (>= 4); |LT - LAMC| <= 2.5 on {c2}/7 (>= 5)')
    fn = f'{RES}/b74_check_repro.json'
    if os.path.exists(fn):
        r = json.load(open(fn))
        put('PA6', r['hp_dev'][0] == r['hp_dev'][1] and r['ht_test'][0] == r['ht_test'][1],
            f'HP dev rows equal B60 VSAPc rows {r["hp_dev"][0]}/{r["hp_dev"][1]} (max |d| {r["hp_dev"][2]:.2e}); HT test '
            f'stores equal B71 VSAPR {r["ht_test"][0]}/{r["ht_test"][1]} ({r["ht_test"][2]} differing predictions)')
    if all('B_paired' in J for J in P.values()):
        arms = ['STORE', 'VSAS', 'VSAPC', 'VSAPCX', 'LAMC', 'NLMSC']
        bad = [(e, a) for e, J in P.items() for a in arms
               if not J['B_paired'][f'R1|{a}']['label'] > J['B_paired'][f'R1|{a}']['alias']]
        put('PB1', not bad, f'label > alias recall for every arm and encoder; exceptions {bad}')
        c1 = sum(J['B_paired']['R1|STORE-LAMC']['delta'] > 0 for J in P.values())
        c2 = sum(J['B_paired']['R1|STORE-NLMSC']['delta'] > 0 for J in P.values())
        put('PB2', c1 >= 5 and c2 >= 5, f'Delta(STORE - LAMC) > 0 on {c1}/7 (>= 5); Delta(STORE - NLMSC) > 0 on {c2}/7 (>= 5)')
        c = sum((J['B_paired']['R1|LAMC']['label'] - J['B_paired']['R1|LAMC']['alias'])
                > (J['B_paired']['R1|STORE']['label'] - J['B_paired']['R1|STORE']['alias']) for J in P.values())
        put('PB3', c >= 5, f'generalisation loss LAMc > store on {c}/7 (>= 5)')
    if all('C' in P[e] for e in LLM if e in P):
        ok1, ok2, med, ok3, det = 0, 0, [], True, []
        for e in LLM:
            C = P[e]['C']
            v = [C[f'R1|STORE-N72_{d}']['diff'] for d in range(20)]
            full = C['R1|STORE-NLMSC']['diff']
            ok1 += sum(x < 0 for x in v) >= 18
            ok2 += abs(np.median(v) - full) <= 0.01
            w = [C[f'R1|N72_{d}-NLMSC']['diff'] for d in range(20)]
            med.append(float(np.median(w)))
            nl = sum(C[f'R1|VSAPCX-N72_{d}']['one_label'] == '-' for d in range(20))
            det.append((e, nl))
            if e == 'mistral':
                ok3 = ok3 and nl >= 18
            if e in ('qwen8', 'llama'):
                ok3 = ok3 and nl <= 5
        put('PC1', ok1 == 4 and ok2 >= 3, f'STORE - NLMSC72 < 0 in >= 18/20 draws on {ok1}/4; median within 1 point on {ok2}/4 (>= 3)')
        put('PC2', all(-0.015 <= m <= -0.002 for m in med), f'NLMSC72 - NLMSC medians {[round(100 * m, 2) for m in med]} (in [-1.5, -0.2])')
        put('PC3', ok3, f'draws with NLMSc72 leading VSAPcX (one-level): {det} (Mistral >= 18, Qwen3-8B / Llama <= 5)')
    chk = [f'{RES}/b74_check_{k}.json' for k in ('scorer', 'bits', 'determ')]
    if all(os.path.exists(f) for f in chk):
        ok = all(json.load(open(f)).get('pass') for f in chk)
        rd = []
        for e in PANEL:
            f = f'{RES}/b74_rederive_{e}.json'
            if os.path.exists(f):
                rd += [x['agree'] for x in json.load(open(f))]
        ok = ok and rd and min(rd) >= 0.99
        put('PX', bool(ok), f'scorer / bits / determinism pass flags and re-derivation min {min(rd) if rd else None}')
    open(f'{RES}/b74_predictions.txt', 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
