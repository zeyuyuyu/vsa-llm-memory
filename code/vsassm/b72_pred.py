"""B72 predictions (prereg 'Predictions') evaluated from results/b72_verdict.json -> results/b72_predictions.txt."""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
RES = os.environ.get('B72_RES', f'{ROOT}/results')
PANEL = ['qwen8', 'b46', 'llama', 'mistral', 'e5', 'bge', 'minilm']
LLM = PANEL[:4]
A = 'ideal'


def main():
    P = json.load(open(f'{RES}/b72_verdict.json'))
    L = ['# B72 predictions (prereg), evaluated', '']

    def e(enc, nm, rn='R1', grp=None):
        for g in ([grp] if grp else ('primary', 'secondary', 'B')):
            x = P[enc].get(g, {}).get(f'{A}|{rn}|{nm}')
            if x is not None:
                return x
        return None

    def lab(enc, nm, rn='R1'):
        x = e(enc, nm, rn)
        return x['one']['label'] if x else None

    encs = [x for x in PANEL if x in P]
    # QA1
    leads = sum(lab(x, 'OPTR-STORE') == 'OPTR leads' for x in encs)
    absok = all(abs(e(x, 'OPTR-STORE')['diff']) <= 0.025 for x in encs)
    smaller = sum(e(x, 'OPTR-STORE')['diff'] < (e(x, 'OPTR-STORE')['diff'] - e(x, 'OPTR-LSFR')['diff']) for x in encs)
    L.append(f'QA1 OPTR leads STORE on {leads}/7 (>= 4); |diff| <= 2.5 on all: {absok}; OPTR - STORE < LSFR - STORE on '
             f'{smaller}/7 (>= 5) -> {"HELD" if leads >= 4 and absok and smaller >= 5 else "FAILED"}')
    # QA2
    a = sum(lab(x, 'OPTR-LAMC') == 'LAMC leads' for x in encs)
    b = sum(lab(x, 'OPTX-LAMC') == 'LAMC leads' for x in encs)
    c = sum(lab(x, 'OPTR-LAMC', 'R1-V16') == 'LAMC leads' for x in encs)
    d = sum(lab(x, 'OPTX-LAMC', 'R1-V16') == 'LAMC leads' for x in encs)
    L.append(f'QA2 LAMc leads OPTR {a}/7, OPTX {b}/7 over R1; at R1-V16 {c}/7, {d}/7 (all 7) -> '
             f'{"HELD" if min(a, b, c, d) == 7 else "FAILED"}')
    # QA3
    a = sum(lab(x, 'OPTR-NLMSC') == 'NLMSC leads' for x in encs)
    b = sum(lab(x, 'OPTX-NLMSC') == 'NLMSC leads' for x in encs)
    L.append(f'QA3 NLMSc leads OPTR {a}/7 (>= 4), OPTX {b}/7 (>= 3) -> {"HELD" if a >= 4 and b >= 3 else "FAILED"}')
    # QA4
    a = sum(lab(x, 'OPTX-VSAPCX') == 'OPTX leads' for x in encs)
    L.append(f'QA4 OPTX leads VSAPcX {a}/7 (>= 4) -> {"HELD" if a >= 4 else "FAILED"}')
    # QA5
    big, fmtle = 0, 0
    for x in encs:
        t = [e(x, nm)['diff'] for nm in ('STORE-OPTR', 'OPTR-LSFR', 'LSFR-LAMC')]
        big += int(np.argmax(np.abs(t)) == 2)
        fmtle += int(t[1] <= 0)
    L.append(f'QA5 features term largest on {big}/7 (>= 5); format term <= 0 on {fmtle}/7 (>= 5) -> '
             f'{"HELD" if big >= 5 and fmtle >= 5 else "FAILED"}')
    # QA6
    a = sum(-0.005 <= e(x, 'OPTQ-OPTR')['diff'] <= 0.01 for x in encs)
    L.append(f'QA6 OPTQ - OPTR in [-0.5, +1.0] on {a}/7 (>= 5) -> {"HELD" if a >= 5 else "FAILED"}')
    # QC1 / QC2
    same, nlead, rej, r1, r2 = 0, 0, 0, [], []
    for x in encs:
        for nm in ('STORE-LAMC', 'VSAPCX-LAMC', 'STORE-NLMSC', 'VSAPCX-NLMSC'):
            c = P[x]['C'][f'R1|{nm}']
            o = e(x, nm)
            same += int(c['t_label'] == o['one']['label'])
            if o['one']['label'] != 'level' and abs(o['diff']) >= 0.01:
                nlead += 1
                rej += int(c['perm_p'] < 0.05)
            w = c['t_ci95'][1] - c['t_ci95'][0]
            r1.append(w / (o['one']['ci95'][1] - o['one']['ci95'][0]))
            r2.append(w / (o['two']['ci95'][1] - o['two']['ci95'][0]))
    L.append(f'QC1 t label = one-level label in {same}/28 (>= 26); permutation rejects in {rej}/{nlead} leads with '
             f'|diff| >= 1 (all) -> {"HELD" if same >= 26 and rej == nlead else "FAILED"}')
    L.append(f'QC2 width ratio t / one-level mean {np.mean(r1):.2f} (> 1.0); t / two-level mean {np.mean(r2):.2f} '
             f'(< 1.0) -> {"HELD" if np.mean(r1) > 1 and np.mean(r2) < 1 else "FAILED"}')
    # QD1-QD3
    a = sum(abs(e(x, 'LAMCP-LAMC')['diff']) <= 0.005 for x in encs)
    ch = {x: json.load(open(f'{RES}/b72_choice_{x}.json'))['choice'] for x in encs}
    nsign = sum(v['src'] == 'LAMP' for x in encs for k, v in ch[x].items() if k.endswith('|LAMP'))
    L.append(f'QD1 |LAMc+ - LAMc| <= 0.5 on {a}/7 (>= 5); LAMc+ takes a sign map in {nsign}/84 cells (<= 21) -> '
             f'{"HELD" if a >= 5 and nsign <= 21 else "FAILED"}')
    a = sum(lab(x, 'STORE-LAMCP') == 'LAMCP leads' for x in encs)
    L.append(f'QD2 LAMc+ leads STORE {a}/7 (7) -> {"HELD" if a == 7 else "FAILED"}')
    pos = [(x, k) for x in encs for k, v in P[x]['hetero'].items() if v['diff'] > 0]
    ld = [(x, k) for x in encs for k, v in P[x]['hetero'].items() if v['label'] == 'STORE leads']
    allv2 = all(k.split('|')[1] == '2' for _, k in pos)
    L.append(f'QD3 store point > LAMc in {len(pos)}/84 (<= 8), all at V = 2: {allv2}; store leads in {len(ld)} (<= 3) -> '
             f'{"HELD" if len(pos) <= 8 and allv2 and len(ld) <= 3 else "FAILED"}')
    # QB1-QB3
    llm = [x for x in LLM if x in P]
    ok = True
    parts = []
    for tg in ('k3', 'k4', 'k5', ''):
        n = sum(lab(x, f'STORE{tg}-LAMC{tg}') == f'LAMC{tg} leads' for x in llm)
        parts.append(f'{tg or "inf"} {n}')
        ok &= n >= 3
    L.append(f'QB1 LAMc leads STORE (of 4) at K: {", ".join(parts)} (>= 3 each) -> {"HELD" if ok else "FAILED"}')
    n3 = sum(lab(x, 'VSARk3-SDMTk3') == 'VSARk3 leads' for x in llm)
    n4 = sum(lab(x, 'VSARk4-SDMTk4') == 'VSARk4 leads' for x in llm)
    n5 = sum(lab(x, 'VSARk5-SDMTk5') != 'SDMTk5 leads' for x in llm)
    L.append(f'QB2 first design leads SDMt at K = 1e3 {n3}/4, 1e4 {n4}/4 (>= 3 each); SDMt lead lost at 1e5 {n5}/4 (>= 2)'
             f' -> {"HELD" if n3 >= 3 and n4 >= 3 and n5 >= 2 else "FAILED"}')
    n = sum(lab(x, 'STOREk5-NLMSCk5') == 'NLMSCk5 leads' for x in llm)
    L.append(f'QB3 NLMSc leads STORE at K = 1e5 {n}/4 (>= 3) -> {"HELD" if n >= 3 else "FAILED"}')
    open(f'{RES}/b72_predictions.txt', 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
