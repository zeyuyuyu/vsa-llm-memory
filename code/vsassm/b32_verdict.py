"""B32 verdict (prereg/B32_vector_form_equal_bits_prereg.md): scores the per-query test predictions and applies the
registered bars mechanically.

python3 b32_verdict.py OUTDIR [--boot 10000]    # reads OUTDIR/preds_{enc}_M{M}_V{V}.jsonl.gz for enc in qwen8, e5
                                                 # writes results/b32_verdict.json and results/b32_verdict.txt
"""
import gzip
import json
import math
import sys

import numpy as np

M_GRID, V_GRID, BPF = [256, 1024], [2, 16, 128, 1024], [4, 8, 16, 32, 64]
ENCS = ['qwen8', 'e5']
PRIMARY_ENC, PRIMARY_ACCT = 'qwen8', 'ideal'
R1 = [(M, V, b) for M in M_GRID for V in (2, 16) for b in (4, 8, 16)]       # registered region of a VSA win
R2 = [(M, 1024, b) for M in M_GRID for b in BPF]                             # registered region of an index win
MARGIN = 0.02
N_CELLS = len(M_GRID) * len(V_GRID) * len(BPF)


def load(outdir, enc):
    cells = {}
    for M in M_GRID:
        for V in V_GRID:
            fn = f'{outdir}/preds_{enc}_M{M}_V{V}.jsonl.gz'
            recs = [json.loads(l) for l in gzip.open(fn, 'rt')]
            cells[M, V] = recs
    return cells


def per_query(recs, arm):
    """Concatenate one arm's per-query correctness over the stores of a cell; cluster id = (rep, entity)."""
    ok, cl = [], []
    for r in recs:
        p = np.array(r['pred'][arm])
        g = np.array(r['qgold'])
        ok.append((p == g).astype(np.float64))
        cl += [f'{r["rep"]}:{e}' for e in r['qent']]
    return np.concatenate(ok), np.array(cl)


def cluster_index(cl):
    u, inv = np.unique(cl, return_inverse=True)
    return len(u), inv


def boot_mean_diff(diffs_by_cell, B, seed):
    """diffs_by_cell: list of (per-query difference, cluster ids). Statistic: mean over cells of the cell's mean
    difference. Clusters resampled with replacement within each cell, independently. Returns (stat, lo, hi, draws)."""
    rng = np.random.default_rng(seed)
    stat = float(np.mean([d.mean() for d, _ in diffs_by_cell]))
    draws = np.zeros(B)
    for d, cl in diffs_by_cell:
        k, inv = cluster_index(cl)
        s = np.bincount(inv, weights=d, minlength=k)
        n = np.bincount(inv, minlength=k).astype(np.float64)
        idx = rng.integers(0, k, size=(B, k))
        draws += s[idx].sum(1) / n[idx].sum(1)
    draws /= len(diffs_by_cell)
    return stat, float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975)), draws


def sign_test(a, b):
    from scipy.stats import binomtest
    w, l = int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
    p = binomtest(w, w + l, 0.5).pvalue if w + l else 1.0
    return w, l, float(p)


def main():
    outdir = sys.argv[1]
    B = int(sys.argv[sys.argv.index('--boot') + 1]) if '--boot' in sys.argv else 10000
    J = {'cells': {}, 'regions': {}, 'bars': {}}
    lines = ['# B32 verdict (prereg/B32_vector_form_equal_bits_prereg.md)',
             '# VSA vs the dev-tuned vector index at equal bits; queries are unseen Wikidata aliases; values random', '']
    data = {e: load(outdir, e) for e in ENCS}
    for enc in ENCS:
        for acct in ('ideal', 'raw'):
            lines.append(f'## encoder {enc}, accounting {acct}: accuracy (points), VSA - IDX [95% cluster bootstrap], '
                         f'sign test (VSA right only / IDX right only)')
            lines.append(f'{"M":>5} {"V":>5} {"bpf":>4} {"nq":>5} {"chance":>6} {"EXACT":>6} {"IDX":>6} {"VSA":>6} '
                         f'{"diff":>7} {"95% CI":>17} {"99.875% CI":>17} {"sign":>11} {"p":>8} label')
            for M in M_GRID:
                for V in V_GRID:
                    recs = data[enc][M, V]
                    ex, cl = per_query(recs, 'EXACT')
                    for b in BPF:
                        v, _ = per_query(recs, f'VSA|{b}|{acct}')
                        x, _ = per_query(recs, f'IDX|{b}|{acct}')
                        d = v - x
                        cs = M * 7 + V * 13 + b * 17 + (0 if enc == 'qwen8' else 100000) + (0 if acct == 'ideal' else 50000)
                        stat, lo, hi, draws = boot_mean_diff([(d, cl)], B, cs)
                        lo2, hi2 = float(np.quantile(draws, 0.000625)), float(np.quantile(draws, 0.999375))
                        w, l, p = sign_test(v, x)
                        if stat >= MARGIN and lo2 > 0:
                            lab = 'VSA WIN'
                        elif stat <= -MARGIN and hi2 < 0:
                            lab = 'IDX WIN'
                        else:
                            lab = 'tie/inconclusive'
                        key = f'{enc}|{acct}|{M}|{V}|{b}'
                        J['cells'][key] = {'nq': int(len(d)), 'n_clusters': int(cluster_index(cl)[0]),
                                           'EXACT': float(ex.mean()), 'IDX': float(x.mean()), 'VSA': float(v.mean()),
                                           'diff': stat, 'ci95': [lo, hi], 'ci_bonf': [lo2, hi2], 'sign': [w, l],
                                           'p_sign': p, 'label': lab,
                                           'lift_VSA': float((v.mean() - 1 / V) / (ex.mean() - 1 / V)),
                                           'lift_IDX': float((x.mean() - 1 / V) / (ex.mean() - 1 / V))}
                        lines.append(f'{M:5d} {V:5d} {b:4d} {len(d):5d} {100 / V:6.2f} {100 * ex.mean():6.2f} '
                                     f'{100 * x.mean():6.2f} {100 * v.mean():6.2f} {100 * stat:+7.2f} '
                                     f'[{100 * lo:+6.2f},{100 * hi:+6.2f}] [{100 * lo2:+6.2f},{100 * hi2:+6.2f}] '
                                     f'{w:5d}/{l:<5d} {p:8.1e} {lab}')
            lines.append('')
            for name, R in (('R1', R1), ('R2', R2)):
                diffs = []
                for (M, V, b) in R:
                    recs = data[enc][M, V]
                    v, cl = per_query(recs, f'VSA|{b}|{acct}')
                    x, _ = per_query(recs, f'IDX|{b}|{acct}')
                    diffs.append((v - x, cl))
                stat, lo, hi, _ = boot_mean_diff(diffs, B, 1 if name == 'R1' else 2)
                J['regions'][f'{enc}|{acct}|{name}'] = {'diff': stat, 'ci95': [lo, hi], 'cells': len(R)}
                lines.append(f'  region {name} ({len(R)} cells, mean of cell differences): VSA - IDX {100 * stat:+.2f} '
                             f'[{100 * lo:+.2f}, {100 * hi:+.2f}]')
            lines.append('')
    # registered bars (primary encoder and accounting) and readings
    def reg(enc, acct, name):
        return J['regions'][f'{enc}|{acct}|{name}']
    r1, r2 = reg(PRIMARY_ENC, PRIMARY_ACCT, 'R1'), reg(PRIMARY_ENC, PRIMARY_ACCT, 'R2')
    H1 = r1['diff'] >= MARGIN and r1['ci95'][0] > 0
    H1_loss = r1['diff'] <= -MARGIN and r1['ci95'][1] < 0
    H2 = r2['diff'] <= -MARGIN and r2['ci95'][1] < 0
    r1s = reg('e5', PRIMARY_ACCT, 'R1')
    H1_rep = r1s['diff'] >= MARGIN and r1s['ci95'][0] > 0
    r1raw = reg(PRIMARY_ENC, 'raw', 'R1')
    H1_raw = r1raw['diff'] >= MARGIN and r1raw['ci95'][0] > 0
    wins = [k for k, c in J['cells'].items() if k.startswith(f'{PRIMARY_ENC}|{PRIMARY_ACCT}|') and c['label'] == 'VSA WIN']
    losses = [k for k, c in J['cells'].items() if k.startswith(f'{PRIMARY_ENC}|{PRIMARY_ACCT}|') and c['label'] == 'IDX WIN']
    useful = {k: J['cells'][k]['VSA'] - 1 / int(k.split('|')[3]) >= 0.10 for k in wins}
    J['bars'] = {'useful_wins (VSA >= chance + 10 points)': useful, 'H1_vsa_wins_R1': H1, 'H1_index_wins_R1': H1_loss, 'H2_index_wins_R2': H2,
                 'H1_replicates_e5': H1_rep, 'H1_under_raw_accounting': H1_raw,
                 'cells_vsa_win_bonferroni': wins, 'cells_index_win_bonferroni': losses}
    if H1:
        reading = ('WIN: in the registered low-budget, few-values region the vector form beats the equally engineered '
                   'index at equal bits' + (' and the win replicates with the second encoder' if H1_rep else
                                             ', but the win does not replicate with the second encoder'))
    elif H1_loss:
        reading = 'LOSS: the index beats the vector form even in the region registered for a VSA win'
    else:
        reading = 'TIE: in the registered region the vector form and the index are within the margin or inconclusive'
    reading += ('; with 1,024 values the index wins (H2 holds)' if H2 else
                '; H2 fails: with 1,024 values the index does not win by the margin')
    J['reading'] = reading
    lines += ['## Registered bars (primary: encoder qwen8, ideal accounting)',
              f'  H1 (R1, VSA - IDX >= +2 points, 95% CI > 0): {"PASS" if H1 else "FAIL"}  '
              f'({100 * r1["diff"]:+.2f} [{100 * r1["ci95"][0]:+.2f}, {100 * r1["ci95"][1]:+.2f}])',
              f'  H2 (R2, VSA - IDX <= -2 points, 95% CI < 0): {"PASS" if H2 else "FAIL"}  '
              f'({100 * r2["diff"]:+.2f} [{100 * r2["ci95"][0]:+.2f}, {100 * r2["ci95"][1]:+.2f}])',
              f'  H1 replicated with e5 (same bar): {"yes" if H1_rep else "no"}  '
              f'({100 * r1s["diff"]:+.2f} [{100 * r1s["ci95"][0]:+.2f}, {100 * r1s["ci95"][1]:+.2f}])',
              f'  H1 under raw accounting (secondary): {"yes" if H1_raw else "no"}  '
              f'({100 * r1raw["diff"]:+.2f} [{100 * r1raw["ci95"][0]:+.2f}, {100 * r1raw["ci95"][1]:+.2f}])',
              f'  cells with a VSA win at the Bonferroni level ({N_CELLS} cells): {len(wins)}: ' + ', '.join(wins),
              f'  of these, useful (VSA >= chance + 10 points): {sum(useful.values())} of {len(useful)}',
              f'  cells with an index win at the Bonferroni level: {len(losses)}: ' + ', '.join(losses),
              f'  READING: {reading}', '']
    # context: string look-ups and EXACT (unconstrained memory), primary encoder
    lines.append('## Unconstrained memory (context): accuracy (points), pooled over V within M')
    for M in M_GRID:
        for V in V_GRID:
            recs = data[PRIMARY_ENC][M, V]
            row = []
            for arm in ('EXACT', 'DICT', 'TRI', 'WR'):
                a, _ = per_query(recs, arm)
                row.append(f'{arm} {100 * a.mean():6.2f}')
                J['cells'].setdefault(f'context|{M}|{V}', {})[arm] = float(a.mean())
            ex_e5, _ = per_query(data['e5'][M, V], 'EXACT')
            row.append(f'EXACT(e5) {100 * ex_e5.mean():6.2f}')
            J['cells'][f'context|{M}|{V}']['EXACT_e5'] = float(ex_e5.mean())
            lines.append(f'  M={M:5d} V={V:5d} ' + '  '.join(row))
    json.dump(J, open('results/b32_verdict.json', 'w'), indent=1)
    open('results/b32_verdict.txt', 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
