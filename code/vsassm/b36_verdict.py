"""B36 verdict (prereg/B36_keyfree_equal_bits_prereg.md): scores the merged per-query test predictions and applies the
registered bars mechanically.

python3 b36_verdict.py OUTDIR [--boot 10000]   # reads OUTDIR/all_{enc}_M{M}_V{V}.jsonl.gz (enc in qwen8, e5)
                                                # writes results/b36_verdict.{json,txt}
"""
import gzip
import json
import math
import sys

import numpy as np

M_GRID, V_GRID, BPF = [256, 1024], [2, 16, 128, 1024], [1, 2, 4, 8, 16, 32, 64]
ENCS = ['qwen8', 'e5']
ACCTS = ['ideal', 'real']
PRIMARY_ENC, PRIMARY_ACCT = 'qwen8', 'ideal'
COMP = ['IDX', 'SF', 'CM', 'BF', 'FBF', 'SDM']
R1 = [(M, V, b) for M in M_GRID for V in (2, 16) for b in (4, 8, 16)]    # B32's region (primary bar)
R0 = [(M, V, b) for M in M_GRID for V in (2, 16) for b in (1, 2)]        # the new low-budget cells
R2 = [(M, 1024, b) for M in M_GRID for b in BPF]                         # many values
RH = [(M, V, b) for M in M_GRID for V in (2, 16) for b in (32, 64)]      # few values, high budget
MARGIN = 0.02
N_CELLS = len(M_GRID) * len(V_GRID) * len(BPF)
LIFTS = [0.25, 0.5]


def load(outdir, enc):
    return {(M, V): [json.loads(l) for l in gzip.open(f'{outdir}/all_{enc}_M{M}_V{V}.jsonl.gz', 'rt')]
            for M in M_GRID for V in V_GRID}


def per_query(recs, arm):
    ok, cl = [], []
    for r in recs:
        ok.append((np.array(r['pred'][arm]) == np.array(r['qgold'])).astype(np.float64))
        cl += [f'{r["rep"]}:{e}' for e in r['qent']]
    return np.concatenate(ok), np.array(cl)


def clusters(cl):
    u, inv = np.unique(cl, return_inverse=True)
    return len(u), inv


def boot_region(diffs, B, seed):
    """diffs: list of (per-query difference, cluster ids) per cell; statistic = mean over cells of the cell mean;
    clusters (entity within store) resampled within each cell. Returns (stat, lo, hi, draws)."""
    rng = np.random.default_rng(seed)
    stat = float(np.mean([d.mean() for d, _ in diffs]))
    draws = np.zeros(B)
    for d, cl in diffs:
        k, inv = clusters(cl)
        s = np.bincount(inv, weights=d, minlength=k)
        n = np.bincount(inv, minlength=k).astype(np.float64)
        idx = rng.integers(0, k, size=(B, k))
        draws += s[idx].sum(1) / n[idx].sum(1)
    draws /= len(diffs)
    return stat, float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975)), draws


def sign_test(a, b):
    from scipy.stats import binomtest
    w, l = int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
    return w, l, (float(binomtest(w, w + l, 0.5).pvalue) if w + l else 1.0)


def rate(accs, target):
    """Bits per fact at which the running maximum of the accuracy curve (over BPF) first reaches target; log-linear
    interpolation between grid points. Returns (value, flag): flag '' exact, '<=' reached at 1 bit per fact,
    '>' never reached by 64."""
    env = np.maximum.accumulate(np.asarray(accs))
    if env[0] >= target:
        return 1.0, '<='
    for i in range(1, len(BPF)):
        if env[i] >= target:
            a0, a1 = env[i - 1], env[i]
            x0, x1 = math.log2(BPF[i - 1]), math.log2(BPF[i])
            x = x0 + (target - a0) / (a1 - a0) * (x1 - x0) if a1 > a0 else x1
            return 2.0 ** x, ''
    return 64.0, '>'


def main():
    outdir = sys.argv[1]
    B = int(sys.argv[sys.argv.index('--boot') + 1]) if '--boot' in sys.argv else 10000
    J = {'cells': {}, 'regions': {}, 'rates': {}, 'bars': {}}
    L = ['# B36 verdict (prereg/B36_keyfree_equal_bits_prereg.md)',
         '# VSA store vs key-free classical memories and the per-fact index at equal bits; BEST = the competitor family',
         '# chosen on dev for each cell; queries are unseen Wikidata aliases; values random', '']
    data = {e: load(outdir, e) for e in ENCS}
    for enc in ENCS:
        for acct in ACCTS:
            L.append(f'## encoder {enc}, accounting {acct}: accuracy (points); VSA - BEST [95% cluster bootstrap] '
                     f'[Bonferroni {N_CELLS} cells]; sign test')
            L.append(f'{"M":>5} {"V":>5} {"bpf":>4} {"nq":>5} {"chance":>6} {"EXACT":>6} ' +
                     ' '.join(f'{c:>6}' for c in COMP) + f' {"VSA":>6} {"BEST":>9} {"diff":>7} {"95% CI":>17} '
                     f'{"Bonf CI":>17} {"sign":>11} label')
            for M in M_GRID:
                for V in V_GRID:
                    recs = data[enc][M, V]
                    ex, cl = per_query(recs, 'EXACT')
                    for b in BPF:
                        v, _ = per_query(recs, f'VSA|{b}|{acct}')
                        x, _ = per_query(recs, f'BEST|{b}|{acct}')
                        fam = recs[0]['cfg'][f'BEST|{b}|{acct}'][0]
                        comp = {c: per_query(recs, f'{c}|{b}|{acct}')[0].mean() for c in COMP}
                        seed = M * 7 + V * 13 + b * 17 + ENCS.index(enc) * 100000 + ACCTS.index(acct) * 50000
                        stat, lo, hi, draws = boot_region([(v - x, cl)], B, seed)
                        alpha = 0.05 / N_CELLS
                        lo2, hi2 = float(np.quantile(draws, alpha / 2)), float(np.quantile(draws, 1 - alpha / 2))
                        w, l_, p = sign_test(v, x)
                        lab = 'VSA WIN' if stat >= MARGIN and lo2 > 0 else (
                            'BEST WIN' if stat <= -MARGIN and hi2 < 0 else 'tie/inconclusive')
                        # VSA against each family's own dev-chosen configuration
                        per_fam = {}
                        for c in COMP:
                            xc, _ = per_query(recs, f'{c}|{b}|{acct}')
                            s2, l2, h2, _ = boot_region([(v - xc, cl)], 2000, seed + 7 + COMP.index(c))
                            per_fam[c] = {'acc': float(xc.mean()), 'diff': s2, 'ci95': [l2, h2]}
                        # sensitivity: the competitor with the highest TEST accuracy in this cell (selection on test)
                        oracle = max(COMP, key=lambda c: comp[c])
                        xo, _ = per_query(recs, f'{oracle}|{b}|{acct}')
                        so, lo_o, hi_o, _ = boot_region([(v - xo, cl)], 2000, seed + 99)
                        J['cells'][f'{enc}|{acct}|{M}|{V}|{b}'] = {
                            'nq': int(len(v)), 'n_clusters': int(clusters(cl)[0]), 'EXACT': float(ex.mean()),
                            'VSA': float(v.mean()), 'BEST': float(x.mean()), 'BEST_fam': fam,
                            'diff': stat, 'ci95': [lo, hi], 'ci_bonf': [lo2, hi2], 'sign': [w, l_], 'p_sign': p,
                            'label': lab, 'per_fam': per_fam,
                            'oracle_fam': oracle, 'oracle_diff': so, 'oracle_ci95': [lo_o, hi_o],
                            'lift_VSA': float((v.mean() - 1 / V) / (ex.mean() - 1 / V)),
                            'lift_BEST': float((x.mean() - 1 / V) / (ex.mean() - 1 / V))}
                        L.append(f'{M:5d} {V:5d} {b:4d} {len(v):5d} {100 / V:6.2f} {100 * ex.mean():6.2f} ' +
                                 ' '.join(f'{100 * comp[c]:6.2f}' for c in COMP) +
                                 f' {100 * v.mean():6.2f} {fam:>3}{100 * x.mean():6.2f} {100 * stat:+7.2f} '
                                 f'[{100 * lo:+6.2f},{100 * hi:+6.2f}] [{100 * lo2:+6.2f},{100 * hi2:+6.2f}] '
                                 f'{w:5d}/{l_:<5d} {lab}')
            L.append('')
            for name, R in (('R1', R1), ('R0', R0), ('RH', RH), ('R2', R2)):
                for vs in ['BEST'] + COMP:
                    diffs = []
                    for (M, V, b) in R:
                        recs = data[enc][M, V]
                        v, cl = per_query(recs, f'VSA|{b}|{acct}')
                        x, _ = per_query(recs, f'{vs}|{b}|{acct}')
                        diffs.append((v - x, cl))
                    stat, lo, hi, _ = boot_region(diffs, B if vs == 'BEST' else 2000,
                                                  1 + ['R1', 'R0', 'RH', 'R2'].index(name) * 10 + (['BEST'] + COMP).index(vs))
                    J['regions'][f'{enc}|{acct}|{name}|{vs}'] = {'diff': stat, 'ci95': [lo, hi], 'cells': len(R)}
                    L.append(f'  region {name} ({len(R)} cells) VSA - {vs:4s}: {100 * stat:+6.2f} [{100 * lo:+6.2f}, '
                             f'{100 * hi:+6.2f}]')
                # sensitivity: oracle (test-selected) competitor per cell
                diffs = []
                for (M, V, b) in R:
                    recs = data[enc][M, V]
                    o = J['cells'][f'{enc}|{acct}|{M}|{V}|{b}']['oracle_fam']
                    v, cl = per_query(recs, f'VSA|{b}|{acct}')
                    x, _ = per_query(recs, f'{o}|{b}|{acct}')
                    diffs.append((v - x, cl))
                stat, lo, hi, _ = boot_region(diffs, 2000, 77 + ['R1', 'R0', 'RH', 'R2'].index(name))
                J['regions'][f'{enc}|{acct}|{name}|ORACLE'] = {'diff': stat, 'ci95': [lo, hi], 'cells': len(R)}
                L.append(f'  region {name} ({len(R)} cells) VSA - ORACLE (best competitor on test, per cell): '
                         f'{100 * stat:+6.2f} [{100 * lo:+6.2f}, {100 * hi:+6.2f}]')
            L.append('')
            # rate at equal accuracy
            L.append(f'## encoder {enc}, accounting {acct}: bits per fact needed to reach chance + lift x (EXACT - '
                     f'chance); ratio BEST / VSA [95% bootstrap]')
            for M in M_GRID:
                for V in V_GRID:
                    recs = data[enc][M, V]
                    ex, cl = per_query(recs, 'EXACT')
                    k, inv = clusters(cl)
                    arms = ['VSA', 'BEST'] + COMP
                    okm = {a: np.stack([per_query(recs, f'{a}|{b}|{acct}')[0] for b in BPF]) for a in arms}
                    n = np.bincount(inv, minlength=k).astype(np.float64)
                    sums = {a: np.stack([np.bincount(inv, weights=okm[a][i], minlength=k) for i in range(len(BPF))])
                            for a in arms}
                    exs = np.bincount(inv, weights=ex, minlength=k)
                    rng = np.random.default_rng(M + V + ENCS.index(enc) * 7 + ACCTS.index(acct) * 3)
                    for lift in LIFTS:
                        tgt = 1 / V + lift * (ex.mean() - 1 / V)
                        rr = {a: rate(okm[a].mean(1), tgt) for a in arms}
                        ratios = []
                        for _ in range(1000):
                            idx = rng.integers(0, k, size=k)
                            nn = n[idx].sum()
                            t2 = 1 / V + lift * (exs[idx].sum() / nn - 1 / V)
                            rv = rate(sums['VSA'][:, idx].sum(1) / nn, t2)[0]
                            rb = rate(sums['BEST'][:, idx].sum(1) / nn, t2)[0]
                            ratios.append(math.log2(rb / rv))
                        lo, hi = np.quantile(ratios, [0.025, 0.975])
                        ratio = rr['BEST'][0] / rr['VSA'][0]
                        # censoring: VSA reached at 1 bpf or BEST never reached -> the ratio is a lower bound;
                        # VSA never reached or BEST reached at 1 bpf -> an upper bound; both censored -> undefined
                        lower = rr['VSA'][1] == '<=' or rr['BEST'][1] == '>'
                        upper = rr['VSA'][1] == '>' or rr['BEST'][1] == '<='
                        rflag = '?' if (lower and upper) else ('>=' if lower else ('<=' if upper else ''))
                        J['rates'][f'{enc}|{acct}|{M}|{V}|{lift}'] = {
                            'target': tgt, 'rates': {a: list(rr[a]) for a in arms}, 'ratio_BEST_over_VSA': ratio,
                            'ratio_flag': rflag, 'ratio_ci95': [2 ** lo, 2 ** hi]}
                        L.append(f'  M={M:5d} V={V:5d} lift {lift:.2f} (acc {100 * tgt:5.1f}): ' +
                                 '  '.join(f'{a} {rr[a][1]}{rr[a][0]:.2f}' for a in arms) +
                                 f'   BEST/VSA {rflag}{ratio:.2f} [{2 ** lo:.2f}, {2 ** hi:.2f}]')
            L.append('')
    # registered bars
    def reg(enc, acct, name, vs='BEST'):
        return J['regions'][f'{enc}|{acct}|{name}|{vs}']
    r1 = reg(PRIMARY_ENC, PRIMARY_ACCT, 'R1')
    H1 = r1['diff'] >= MARGIN and r1['ci95'][0] > 0
    H1_loss = r1['diff'] <= -MARGIN and r1['ci95'][1] < 0
    r1e = reg('e5', PRIMARY_ACCT, 'R1')
    H1_rep = r1e['diff'] >= MARGIN and r1e['ci95'][0] > 0
    H1_rep_loss = r1e['diff'] <= -MARGIN and r1e['ci95'][1] < 0
    r1r = reg(PRIMARY_ENC, 'real', 'R1')
    H1_real = r1r['diff'] >= MARGIN and r1r['ci95'][0] > 0
    r0 = reg(PRIMARY_ENC, PRIMARY_ACCT, 'R0')
    H0b = r0['diff'] >= MARGIN and r0['ci95'][0] > 0
    wins = [k for k, c in J['cells'].items() if k.startswith(f'{PRIMARY_ENC}|{PRIMARY_ACCT}|') and c['label'] == 'VSA WIN']
    losses = [k for k, c in J['cells'].items() if k.startswith(f'{PRIMARY_ENC}|{PRIMARY_ACCT}|') and c['label'] == 'BEST WIN']
    useful = {k: J['cells'][k]['VSA'] - 1 / int(k.split('|')[3]) >= 0.10 for k in wins}
    J['bars'] = {'H1_vsa_beats_best_R1': H1, 'H1_best_beats_vsa_R1': H1_loss, 'H1_replicates_e5': H1_rep,
                 'H1_e5_reversed': H1_rep_loss, 'H1_real_accounting': H1_real, 'H0b_low_budget_R0': H0b,
                 'cells_vsa_win_bonferroni': wins, 'cells_best_win_bonferroni': losses, 'useful_wins': useful}
    if H1 and H1_rep:
        reading = 'WIN: the VSA beats the best dev-chosen key-free competitor (or index) in R1 on both encoders'
    elif H1:
        reading = 'WIN ON ONE READER: the VSA beats the best competitor in R1 on the Qwen3-8B codes but not on e5'
    elif H1_loss:
        reading = 'LOSS: a key-free classical memory (or the index) beats the VSA in R1'
    else:
        reading = 'TIE: in R1 the VSA and the best competitor are within the margin or inconclusive'
    J['reading'] = reading
    L += ['## Registered bars (primary: encoder qwen8, ideal accounting; BEST = dev-chosen competitor per cell)',
          f'  H1 (R1, VSA - BEST >= +2 points, 95% CI > 0): {"PASS" if H1 else "FAIL"}  '
          f'({100 * r1["diff"]:+.2f} [{100 * r1["ci95"][0]:+.2f}, {100 * r1["ci95"][1]:+.2f}])',
          f'  H1 replicated on e5 (same bar): {"yes" if H1_rep else "no"}  '
          f'({100 * r1e["diff"]:+.2f} [{100 * r1e["ci95"][0]:+.2f}, {100 * r1e["ci95"][1]:+.2f}])',
          f'  H1 under real accounting (secondary): {"yes" if H1_real else "no"}  '
          f'({100 * r1r["diff"]:+.2f} [{100 * r1r["ci95"][0]:+.2f}, {100 * r1r["ci95"][1]:+.2f}])',
          f'  R0 (1-2 bits per fact, V in 2/16; secondary): {"VSA ahead" if H0b else "not ahead by the bar"}  '
          f'({100 * r0["diff"]:+.2f} [{100 * r0["ci95"][0]:+.2f}, {100 * r0["ci95"][1]:+.2f}])',
          f'  Bonferroni VSA-win cells ({N_CELLS}): {len(wins)}: ' + ', '.join(wins),
          f'  of these useful (VSA >= chance + 10 points): {sum(useful.values())} of {len(useful)}',
          f'  Bonferroni BEST-win cells: {len(losses)}: ' + ', '.join(losses),
          f'  READING: {reading}', '']
    json.dump(J, open('results/b36_verdict.json', 'w'), indent=1)
    open('results/b36_verdict.txt', 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
