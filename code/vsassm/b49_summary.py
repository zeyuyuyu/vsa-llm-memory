"""B49 summary tables from results/b49_verdict.json and results/b49_mech_verdict.json -> results/b49_summary_tables.txt"""
import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
J = json.load(open(f'{ROOT}/results/b49_verdict.json'))
JM = json.load(open(f'{ROOT}/results/b49_mech_verdict.json')) if os.path.exists(f'{ROOT}/results/b49_mech_verdict.json') \
    else {}
PANEL = [e for e in ['qwen8', 'b46', 'llama', 'mistral', 'e5', 'bge', 'minilm'] if f'{e}|ideal|R1|BEST' in J['regions']]
NAMES = {'qwen8': 'Qwen3-8B L9 mean', 'b46': 'Granite-3.3-8B L30 mean', 'llama': 'Llama-3.1-8B L24 mean',
         'mistral': 'Mistral-7B L24 PromptEOL', 'e5': 'e5-base-v2', 'bge': 'bge-small-en-v1.5', 'minilm': 'all-MiniLM-L6-v2'}


def f(r):
    return f'{100 * r["diff"]:+.2f} [{100 * r["ci95"][0]:+.2f}, {100 * r["ci95"][1]:+.2f}]'


L = ['# B49 summary tables (ideal accounting unless stated; points; 95% cluster bootstrap)', '',
     '## Panel: region VSA - BEST (all ten competitor families, dev-chosen per cell)', '',
     '| encoder | R1 (V 2/16, 4-16 bpf) | R0 (1-2 bpf) | RH (32-64 bpf) | R2 (V 1,024) | H1-B49 |', '| --- | --- | --- | --- | --- | --- |']
for e in PANEL:
    b = J['bars'][e]
    L.append(f'| {NAMES[e]} | ' + ' | '.join(f(J['regions'][f'{e}|ideal|{rn}|BEST']) for rn in ('R1', 'R0', 'RH', 'R2')) +
             f' | {"PASS" if b["H1_B49"] else ("LOSS" if b["loss"] else "tie")} |')
p = J['bars']['panel']
L += ['', f'Panel: {p["pass_total"]} of {len(PANEL)} pass; LLM families {p["pass_llm"]} of {p["n_llm"]}; sentence encoders '
      f'{p["pass_sentence"]} of {p["n_sentence"]}.', '',
      '## R1 against other comparators', '',
      '| encoder | vs BEST36 (B36 set) | vs BEST without LAM | vs ORACLE | real accounting | vs LAM | vs LHI | vs LHC | vs LHS | VSA-L - VSA |',
      '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
for e in PANEL:
    R = lambda vs, a='ideal': J['regions'][f'{e}|{a}|R1|{vs}']
    L.append(f'| {NAMES[e]} | {f(R("BEST36"))} | {f(R("BESTNL"))} | {f(R("ORACLE"))} | {f(R("BEST", "real"))} | '
             f'{f(R("LAM"))} | {f(R("LHI"))} | {f(R("LHC"))} | {f(R("LHS"))} | {f(R("VSAL"))} |')
L += ['', '## R1 mean accuracy and BEST families', '']
for e in PANEL:
    cs = [J['cells'][f'{e}|ideal|{M}|{V}|{b}'] for M in (256, 1024) for V in (2, 16) for b in (4, 8, 16)]
    fams = {}
    for c in cs:
        fams[c['BEST_fam']] = fams.get(c['BEST_fam'], 0) + 1
    m = lambda k: 100 * sum(c[k] for c in cs) / len(cs)
    L.append(f'- {NAMES[e]}: VSA {m("VSA"):.1f}, BEST {m("BEST"):.1f}, EXACT {m("EXACT"):.1f}; BEST family counts {fams}')
L += ['', '## Useful accuracy (U = 0.9 x EXACT); bits per fact to reach U on the 1-64 grid', '',
      '| encoder | M | V | U | VSA | BEST | VSA-inf | LAM-inf | families reaching U |', '| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
for e in PANEL:
    for M in (256, 1024):
        for V in (2, 16):
            x = J['useful'][f'{e}|{M}|{V}']
            fr = ','.join(a for a, ok in x['reached'].items() if ok and a not in ('VSA', 'BEST', 'BEST36')) or '-'
            L.append(f'| {e} | {M} | {V} | {100 * x["U"]:.1f} | {x["VSA"][1]}{x["VSA"][0]:.1f} | {x["BEST"][1]}{x["BEST"][0]:.1f}'
                     f' | {100 * x["VSAINF"]:.1f} | {100 * x["LAMINF"]:.1f} | {fr} |')
L += ['', '## Item 5: R1 cells where a single competitor family leads the VSA (>= 2 points, Bonferroni interval < 0)', '']
for e in PANEL:
    r1 = [x for x in J['item5'][e] if x.split()[1] in ('V2', 'V16') and x.split()[2].rstrip(':') in ('b4', 'b8', 'b16')]
    L.append(f'- {e}: {"; ".join(r1) or "none"} (all cells: {len(J["item5"][e])})')
if 'geom_assoc' in J:
    L += ['', '## Geometry association (Spearman over the panel, R1 VSA - BEST)', '']
    for k, (rho, pv) in J['geom_assoc'].items():
        L.append(f'- {k}: rho {rho:+.3f} (p {pv:.3f})')
if JM:
    L += ['', '## Mechanism interventions (R1; Delta = transformed - original margin, same queries)', '',
          '| run | margin original | margin transformed | Delta | changed |', '| --- | --- | --- | --- | --- |']
    for k, v in JM.items():
        if k == 'reading' or not k.endswith('|BEST'):
            continue
        m0, mt = v['margin_0'], v['margin_T']
        L.append(f'| {k} | {100 * m0[0]:+.2f} [{100 * m0[1]:+.2f}, {100 * m0[2]:+.2f}] | {100 * mt[0]:+.2f} '
                 f'[{100 * mt[1]:+.2f}, {100 * mt[2]:+.2f}] | {100 * v["delta"]:+.2f} [{100 * v["ci95"][0]:+.2f}, '
                 f'{100 * v["ci95"][1]:+.2f}] | {v["changed"]} |')
    L.append(f'\nReading: {JM.get("reading", "primary runs incomplete")}')
open(f'{ROOT}/results/b49_summary_tables.txt', 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))
