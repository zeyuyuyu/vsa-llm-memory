"""B49 predictions (prereg/B49 section 8) checked mechanically against results/b49_verdict.json and
results/b49_mech_verdict.json. -> results/b49_predictions.txt"""
import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
J = json.load(open(f'{ROOT}/results/b49_verdict.json'))
JM = json.load(open(f'{ROOT}/results/b49_mech_verdict.json')) if os.path.exists(f'{ROOT}/results/b49_mech_verdict.json') \
    else {}
G = json.load(open(f'{ROOT}/b49_out/lh_choice_all.json')) if os.path.exists(f'{ROOT}/b49_out/lh_choice_all.json') else {}
LLM, SENT = ['qwen8', 'b46', 'llama', 'mistral'], ['e5', 'bge', 'minilm']
PANEL = LLM + SENT
M2 = 0.02


def reg(e, vs, name='R1', acct='ideal'):
    return J['regions'][f'{e}|{acct}|{name}|{vs}']


def passes(r):
    return r['diff'] >= M2 and r['ci95'][0] > 0


L = ['# B49 predictions (prereg/B49_panel_mechanism_competitors_prereg.md, section 8)', '']
res = {}
# P1
p1 = passes(reg('llama', 'BEST36')) and passes(reg('mistral', 'BEST36')) and not passes(reg('bge', 'BEST36')) and \
    not passes(reg('minilm', 'BEST36'))
res['P1'] = p1
L.append(f'P1 (llama, mistral pass vs BEST36; bge, minilm do not): {"HELD" if p1 else "FAILED"} ' +
         str({e: round(100 * reg(e, "BEST36")["diff"], 2) for e in ('llama', 'mistral', 'bge', 'minilm')}))
# P2
pl = [e for e in PANEL if passes(reg(e, 'BEST'))]
p2 = 2 <= len(pl) <= 4 and all(e in LLM for e in pl)
res['P2'] = p2
L.append(f'P2 (2-4 encoders pass vs BEST, all LLM): {"HELD" if p2 else "FAILED"}; passing: {pl}')
# P3
close = [e for e in PANEL if abs(reg(e, 'LAM')['diff']) <= 0.03]
lam_best = {e: sum(1 for k, c in J['cells'].items() if k.startswith(f'{e}|ideal|') and int(k.split('|')[3]) in (2, 16)
                   and int(k.split('|')[4]) in (4, 8, 16) and c['BEST_fam'] == 'LAM') for e in PANEL}
p3 = len(close) >= 5 and max(lam_best.values()) <= 4
res['P3'] = p3
L.append(f'P3 (|VSA - LAM| <= 3 on >= 5 encoders; LAM BEST in <= 4 R1 cells): {"HELD" if p3 else "FAILED"}; '
         f'VSA - LAM R1: ' + str({e: round(100 * reg(e, "LAM")["diff"], 2) for e in PANEL}) + f'; LAM BEST cells {lam_best}')
# P4
lhi = [e for e in PANEL if reg(e, 'IDX')['diff'] - reg(e, 'LHI')['diff'] >= 0.02]
crit_up = {e: (G[e]['choice']['crit'] > G[e]['base_criterion']['center']) for e in G if e in PANEL} if G else {}
p4 = len(lhi) >= 4 and (all(crit_up.values()) if crit_up else False)
res['P4'] = p4
L.append(f'P4 (learned map raises the criterion over centred on every encoder; LHI beats IDX by >= 2 on >= 4): '
         f'{"HELD" if p4 else "FAILED"}; criterion up: {crit_up}; LHI - IDX over R1: ' +
         str({e: round(100 * (reg(e, "IDX")["diff"] - reg(e, "LHI")["diff"]), 2) for e in PANEL}))
# P5
import numpy as np  # noqa: E402
gap = np.mean([reg(e, 'BEST36')['diff'] - reg(e, 'BEST')['diff'] for e in PANEL])
res['P5'] = gap >= 0.01
L.append(f'P5 (VSA - BEST at least 1 point lower than VSA - BEST36 on average): {"HELD" if gap >= 0.01 else "FAILED"} '
         f'(mean gap {100 * gap:+.2f})')
# P6
vl = {e: -reg(e, 'VSAL')['diff'] for e in LLM}  # stored as VSAL - VSA under the key VSAL
vl = {e: J['regions'][f'{e}|ideal|R1|VSAL']['diff'] for e in LLM}
p6 = all(-0.03 <= x <= 0.03 for x in vl.values())
res['P6'] = p6
L.append(f'P6 (VSA-L - VSA within +-3 on every LLM): {"HELD" if p6 else "FAILED"} ' +
         str({e: round(100 * x, 2) for e, x in vl.items()}))
# P7
U = J['useful']
vsa_reach = [k for k, x in U.items() if x['reached']['VSA']]
inf_llm = [k for k, x in U.items() if k.split('|')[0] in LLM and x['VSAINF'] < x['U']]
best_sent = [k for k, x in U.items() if k.split('|')[0] in SENT and x['reached']['BEST']]
best_llm = [k for k, x in U.items() if k.split('|')[0] in LLM and x['reached']['BEST']]
p7 = len(vsa_reach) == 0 and len(inf_llm) == 16 and len(best_sent) >= 3 and len(best_llm) <= 4
res['P7'] = p7
L.append(f'P7 (VSA reaches U in none of 28; VSA-inf < U in all 16 LLM settings; BEST reaches U in >= 3 of 12 sentence '
         f'and <= 4 of 16 LLM settings): {"HELD" if p7 else "FAILED"}; VSA reaches U in {vsa_reach}; VSA-inf < U in '
         f'{len(inf_llm)} of 16 LLM settings; BEST reaches U in {len(best_sent)} sentence, {len(best_llm)} LLM settings')
# P8
if JM:
    t1, t2, t3 = JM.get('qwen8|T1|BEST'), JM.get('e5|T2|BEST'), JM.get('qwen8|T3|BEST')
    parts = []
    if t1:
        parts.append(('T1 Delta > -2', t1['delta'] > -M2))
    if t2:
        parts.append(('T2 Delta < +2', t2['delta'] < M2))
    if t3:
        parts.append(('T3 Delta <= -2, CI < 0, margin <= +3.7', t3['delta'] <= -M2 and t3['ci95'][1] < 0 and
                      t3['margin_T'][0] <= 0.037))
    res['P8'] = all(x for _, x in parts) if len(parts) == 3 else None
    L.append(f'P8 (mechanism, HD): {"HELD" if res["P8"] else ("not all runs" if res["P8"] is None else "FAILED")} ' +
             str(parts))
# P9
if 'geom_assoc' in J:
    rho = J['geom_assoc']['whitened PR'][0]
    res['P9'] = rho >= 0.6
    L.append(f'P9 (Spearman margin vs whitened PR >= 0.6): {"HELD" if rho >= 0.6 else "FAILED"} (rho {rho:+.3f})')
L.append('')
L.append(f'held: {sum(1 for v in res.values() if v)} of {len([v for v in res.values() if v is not None])} '
         f'(P10, checks: see results/b49_check.txt)')
open(f'{ROOT}/results/b49_predictions.txt', 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))
