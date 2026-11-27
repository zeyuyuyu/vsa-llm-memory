"""B53 summary tables from results/b53_verdict.json (+ results/b53_check.txt) -> results/b53_summary.txt"""
import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
J = json.load(open(f'{ROOT}/results/b53_verdict.json'))
J49 = json.load(open(f'{ROOT}/results/b49_verdict.json'))
P = ['qwen8', 'b46', 'llama', 'mistral', 'e5', 'bge', 'minilm']
N = {'qwen8': 'Qwen3-8B', 'b46': 'Granite-3.3-8B', 'llama': 'Llama-3.1-8B', 'mistral': 'Mistral-7B', 'e5': 'e5-base-v2',
     'bge': 'bge-small', 'minilm': 'MiniLM-L6'}


def f(r):
    return f'{100 * r["diff"]:+.2f} [{100 * r["ci95"][0]:+.2f}, {100 * r["ci95"][1]:+.2f}]'


def g(e, rn, x, acct='ideal'):
    return f(J['regions'][f'{e}|{acct}|{rn}|{x}'])


L = ['# B53 summary (prereg/B53_value_codes_prereg.md). Ideal accounting unless stated; points; 95% cluster bootstrap.',
     '# VSA = store with antipodal (V 2) / Sylvester-Hadamard orthogonal (V 16) value codes, re-chosen on dev; VSAR = B49\'s',
     '# random-code store; competitors and their test predictions are B49\'s. R2 (V 1,024) is unchanged (random codes).', '',
     '## Primary: VSA - LAM', '',
     '| encoder | R1 B53 | label | R1 B49 | R1 V=2 | R1 V=16 | R0 | RH | R2 (unchanged) |',
     '| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
for e in P:
    L.append(f'| {N[e]} | {g(e, "R1", "LAM")} | {J["bars"][e]["label"]} | {f(J49["regions"][f"{e}|ideal|R1|LAM"])} | '
             f'{g(e, "R1-V2", "LAM")} | {g(e, "R1-V16", "LAM")} | {g(e, "R0", "LAM")} | {g(e, "RH", "LAM")} | '
             f'{g(e, "R2", "LAM")} |')
c = J['bars']['count']
L += ['', f'VSA >= LAM over R1 (interval not below 0): {c["ge"]["all"]} of 7 (LLM {c["ge"]["llm"]} of 4, sentence '
      f'{c["ge"]["sent"]} of 3); VSA ahead: {c["ahead"]["all"]} of 7.', '',
      '## Per V and region, VSA - LAM', '', '| encoder | R0 V=2 | R0 V=16 | RH V=2 | RH V=16 |', '| --- | --- | --- | --- | --- |']
for e in P:
    L.append(f'| {N[e]} | ' + ' | '.join(g(e, rn, 'LAM') for rn in ('R0-V2', 'R0-V16', 'RH-V2', 'RH-V16')) + ' |')
L += ['', '## Secondary: VSA - BEST (all ten families)', '',
      '| encoder | R1 B53 | H1-B49 | R1 B49 | R1 V=2 | R1 V=16 | R0 | RH | R2 (unchanged) |',
      '| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
for e in P:
    L.append(f'| {N[e]} | {g(e, "R1", "BEST")} | {J["bars"][e]["H1_B49"]} | {f(J49["regions"][f"{e}|ideal|R1|BEST"])} | '
             f'{g(e, "R1-V2", "BEST")} | {g(e, "R1-V16", "BEST")} | {g(e, "R0", "BEST")} | {g(e, "RH", "BEST")} | '
             f'{g(e, "R2", "BEST")} |')
L += ['', '## R1, other comparators and real accounting', '',
      '| encoder | vs BEST36 | vs BEST without LAM | vs ORACLE | real: vs LAM | real: vs BEST |', '| --- | --- | --- | --- | --- | --- |']
for e in P:
    L.append(f'| {N[e]} | {g(e, "R1", "BEST36")} | {g(e, "R1", "BESTNL")} | {g(e, "R1", "ORACLE")} | '
             f'{g(e, "R1", "LAM", "real")} | {g(e, "R1", "BEST", "real")} |')
L += ['', '## Effect of the codes: VSA (B53) - VSAR (B49 random codes), same stores, projections and seeds', '',
      '| encoder | R1 | R1 V=2 | R1 V=16 | R0 V=2 | R0 V=16 | RH V=2 | RH V=16 |', '| --- | --- | --- | --- | --- | --- | --- | --- |']
for e in P:
    L.append(f'| {N[e]} | ' + ' | '.join(g(e, rn, 'VSAR') for rn in ('R1', 'R1-V2', 'R1-V16', 'R0-V2', 'R0-V16', 'RH-V2',
                                                                       'RH-V16')) + ' |')
L += ['', 'Half the coordinates (VSA at b vs VSAR at 2b, R1 V=2, mean of 6 cells): ' +
      ', '.join(f'{N[e]} {100 * x:+.2f}' for e, x in J['half'].items()), '',
      '## R1 mean accuracy (points), V = 2 and V = 16 separately', '',
      '| encoder | V | EXACT | VSAR | VSA | LAM | BEST |', '| --- | --- | --- | --- | --- | --- | --- |']
for e in P:
    for V in (2, 16):
        cs = [J['cells'][f'{e}|ideal|{M}|{V}|{b}'] for M in (256, 1024) for b in (4, 8, 16)]
        m = lambda k: 100 * sum(k(c) for c in cs) / len(cs)
        L.append(f'| {N[e]} | {V} | {m(lambda c: c["EXACT"]):.1f} | {m(lambda c: c["acc"]["VSAR"]):.1f} | '
                 f'{m(lambda c: c["VSA"]):.1f} | {m(lambda c: c["acc"]["LAM"]):.1f} | {m(lambda c: c["acc"]["BEST"]):.1f} |')
L += ['', '## Tuning-grid size per arm (dev configurations per cell, V 2 / 16)', '',
      ', '.join(f'{k} {v}' for k, v in J['tuning'].items()) +
      '; LHI / LHC / LHS also choose their learned map among 16 per encoder; BEST picks among the ten families\' choices.',
      '', '## Predictions', '']
for k, (ok, s) in J['predictions'].items():
    L.append(f'- {k}: {"held" if ok else "FAILED"} ({s})')
if os.path.exists(f'{ROOT}/results/b53_check.txt'):
    L += ['', '## Checks', ''] + [f'- {x}' for x in open(f'{ROOT}/results/b53_check.txt').read().splitlines() if x.strip()]
open(f'{ROOT}/results/b53_summary.txt', 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))
