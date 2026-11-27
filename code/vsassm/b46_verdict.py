"""B46 verdict (prereg/B46_second_llm_family_prereg.md): B36's verdict code (b36_verdict.main, unchanged) on the B46
codes, then the B46 bars and a side-by-side table with B36's Qwen3-8B and e5 results.

python3 b46_verdict.py [--boot 10000]     # -> results/b46_verdict.{json,txt}

b36_verdict.main runs in b46_out/ with its encoder list set to ['b46', 'e5'] and PRIMARY_ENC 'b46'; the e5 test files
are B36's own (linked into b46_out/b36_out/test), so its e5 rows must equal B36's e5 rows exactly (checked here: the
cell / region / rate seeds depend only on the encoder's position in the list, which is the same as in B36). Its text
up to its own bar section is kept; the B36 bar section (worded for Qwen3-8B / e5) is replaced by the B46 bars.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
WD = os.environ.get('B46_WD', os.path.join(ROOT, 'b46_out'))  # B46_WD: smoke tests only
RES = os.path.join(ROOT, 'results') if 'B46_WD' not in os.environ else os.path.join(WD, 'results')  # smoke: stay in WD
MARGIN = 0.02


def main():
    os.chdir(WD)
    sys.path.insert(0, ROOT)
    for M in (256, 1024):
        for V in (2, 16, 128, 1024):
            fn = f'b36_out/test/all_e5_M{M}_V{V}.jsonl.gz'
            if not os.path.exists(fn):
                os.symlink(os.path.join(ROOT, 'b36_out/test', os.path.basename(fn)), fn)
    import b36_verdict as V36
    V36.ENCS = ['b46', 'e5']
    V36.PRIMARY_ENC = 'b46'
    sys.argv = ['b36_verdict.py', 'b36_out/test'] + sys.argv[1:]
    V36.main()
    J = json.load(open('results/b36_verdict.json'))
    txt = open('results/b36_verdict.txt').read()
    J36 = json.load(open(os.path.join(ROOT, 'results', 'b36_verdict.json')))  # B36's own verdict

    # 1. the wrapper reproduces B36's e5 rows exactly
    mism = []
    for part in ('cells', 'regions', 'rates'):
        for k, v in J36[part].items():
            if k.startswith('e5|') and J[part].get(k) != v:
                mism.append((part, k))
    n_e5 = sum(1 for part in ('cells', 'regions', 'rates') for k in J36[part] if k.startswith('e5|'))

    # 2. B46 bars
    def reg(Jx, enc, acct, name, vs='BEST'):
        return Jx['regions'][f'{enc}|{acct}|{name}|{vs}']

    def passes(r):
        return r['diff'] >= MARGIN and r['ci95'][0] > 0

    def lost(r):
        return r['diff'] <= -MARGIN and r['ci95'][1] < 0

    r1 = reg(J, 'b46', 'ideal', 'R1')
    r1r = reg(J, 'b46', 'real', 'R1')
    r0 = reg(J, 'b46', 'ideal', 'R0')
    r1o = reg(J, 'b46', 'ideal', 'R1', 'ORACLE')
    H1, H1_loss, H1_real = passes(r1), lost(r1), passes(r1r)
    cells = {k: c for k, c in J['cells'].items() if k.startswith('b46|ideal|')}
    wins = [k for k, c in cells.items() if c['label'] == 'VSA WIN']
    losses = [k for k, c in cells.items() if c['label'] == 'BEST WIN']
    useful = {k: cells[k]['VSA'] - 1 / int(k.split('|')[3]) >= 0.10 for k in wins}
    if H1:
        reading = ('PASS: in R1 the VSA store beats the best dev-chosen competitor on the B46 codes, so the R1 win holds '
                   'on hidden-state codes of two LLM families (Qwen3-8B layer 9 and the B46 encoder), not on e5')
        if not passes(r1o):
            reading += '; against the test-selected competitor (ORACLE) the R1 win is not established'
    elif H1_loss:
        reading = ('LOSS: a classical memory beats the VSA store in R1 on the B46 codes; win on 1 of 3 readers '
                   '(Qwen3-8B), loss on the B46 encoder, tie on e5')
    elif H1_real:
        reading = 'PASS ONLY UNDER REAL ACCOUNTING: no win claimed for the B46 encoder'
    else:
        reading = ('TIE: the R1 win does not replicate on the B46 codes; it is specific to Qwen3-8B layer 9 (win on 1 of '
                   '3 readers)')
    J['b46_bars'] = {'H1_B46_R1': H1, 'H1_B46_loss': H1_loss, 'H1_B46_real': H1_real, 'R0_bar': passes(r0),
                     'R1_oracle_bar': passes(r1o), 'cells_vsa_win_bonferroni': wins,
                     'cells_best_win_bonferroni': losses, 'useful_wins': useful, 'reading': reading,
                     'e5_rows_reproduced': {'checked': n_e5, 'mismatched': mism}}
    J.pop('bars', None)
    J.pop('reading', None)
    head = txt.split('## Registered bars')[0].rstrip() + '\n'
    head = head.replace('# B36 verdict (prereg/B36_keyfree_equal_bits_prereg.md)',
                        '# B46 verdict (prereg/B46_second_llm_family_prereg.md): B36 protocol on the B46 codes '
                        '(encoder b46); e5 rows = B36 re-computed')
    enc = json.load(open('b46_encoder.json'))
    L = [head, f'## B46 encoder: {enc["file_tag"]} (md5 {enc["md5"]})', '',
         '## Registered bars (prereg/B46; primary: encoder b46, ideal accounting; BEST = dev-chosen competitor per cell)',
         f'  H1-B46 (R1, VSA - BEST >= +2 points, 95% CI > 0): {"PASS" if H1 else "FAIL"}  '
         f'({100 * r1["diff"]:+.2f} [{100 * r1["ci95"][0]:+.2f}, {100 * r1["ci95"][1]:+.2f}])',
         f'  reversed (<= -2, CI < 0): {"yes" if H1_loss else "no"}',
         f'  H1 bar under real accounting (secondary): {"yes" if H1_real else "no"}  '
         f'({100 * r1r["diff"]:+.2f} [{100 * r1r["ci95"][0]:+.2f}, {100 * r1r["ci95"][1]:+.2f}])',
         f'  R0 (1-2 bits per fact; secondary): {"VSA ahead by the bar" if passes(r0) else "not ahead by the bar"}  '
         f'({100 * r0["diff"]:+.2f} [{100 * r0["ci95"][0]:+.2f}, {100 * r0["ci95"][1]:+.2f}])',
         f'  R1 against ORACLE (test-selected competitor): {100 * r1o["diff"]:+.2f} '
         f'[{100 * r1o["ci95"][0]:+.2f}, {100 * r1o["ci95"][1]:+.2f}]',
         f'  Bonferroni VSA-win cells (56): {len(wins)}: ' + ', '.join(wins),
         f'  of these useful (VSA >= chance + 10 points): {sum(useful.values())} of {len(useful)}',
         f'  Bonferroni BEST-win cells: {len(losses)}: ' + ', '.join(losses),
         f'  e5 rows reproduced from B36: {n_e5 - len(mism)} of {n_e5} identical',
         f'  READING: {reading}', '']

    # 3. side by side: B46 vs B36's Qwen3-8B and e5 (B36's own verdict file)
    for acct in ('ideal', 'real'):
        L.append(f'## Side by side ({acct} accounting): region VSA - X [95% CI]; mean accuracy over the region\'s cells')
        for name in ('R1', 'R0', 'RH', 'R2'):
            for vs in ('BEST', 'IDX', 'ORACLE', 'SF', 'CM', 'BF', 'FBF', 'SDM'):
                row = []
                for lab, Jx, e in (('B46', J, 'b46'), ('Qwen3-8B', J36, 'qwen8'), ('e5', J36, 'e5')):
                    r = reg(Jx, e, acct, name, vs)
                    row.append(f'{lab} {100 * r["diff"]:+6.2f} [{100 * r["ci95"][0]:+6.2f},{100 * r["ci95"][1]:+6.2f}]')
                L.append(f'  {name} VSA - {vs:6s}: ' + '   '.join(row))
            accs = []
            for lab, Jx, e in (('B46', J, 'b46'), ('Qwen3-8B', J36, 'qwen8'), ('e5', J36, 'e5')):
                cs = [c for k, c in Jx['cells'].items() if k.startswith(f'{e}|{acct}|') and
                      _in(name, k)]
                m = lambda f: 100 * sum(c[f] for c in cs) / len(cs)
                accs.append(f'{lab}: VSA {m("VSA"):5.2f} BEST {m("BEST"):5.2f} EXACT {m("EXACT"):5.2f}')
            L.append(f'  {name} mean accuracy  ' + '   '.join(accs))
        L.append('')
    # BEST family counts in R1 (dev choice) per encoder
    for lab, Jx, e in (('B46', J, 'b46'), ('Qwen3-8B', J36, 'qwen8'), ('e5', J36, 'e5')):
        cnt = {}
        for k, c in Jx['cells'].items():
            if k.startswith(f'{e}|ideal|') and _in('R1', k):
                cnt[c['BEST_fam']] = cnt.get(c['BEST_fam'], 0) + 1
        L.append(f'  BEST family in the 12 R1 cells (ideal), {lab}: {cnt}')
    L.append('')
    json.dump(J, open(os.path.join(RES, 'b46_verdict.json'), 'w'), indent=1)
    open(os.path.join(RES, 'b46_verdict.txt'), 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L[1:]))


def _in(name, key):
    _, _, M, V, b = key.split('|')
    V, b = int(V), int(b)
    if name == 'R1':
        return V in (2, 16) and b in (4, 8, 16)
    if name == 'R0':
        return V in (2, 16) and b in (1, 2)
    if name == 'RH':
        return V in (2, 16) and b in (32, 64)
    return V == 1024


if __name__ == '__main__':
    main()
