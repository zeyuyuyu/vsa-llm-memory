"""B35 verdict generator (prereg/B35_soft_token_vsa_prereg.md): tables, paired CIs, bar readings, prediction check.

Reads the test runs' per-question outputs (<root>/<run>/test.jsonl.gz), their summaries and the external decoding
(<root>/ext_test.json); writes the text verdict to stdout and a JSON of every number to --json.
Run names: test_vsa_B{B}_s{s}, test_lmt_B{B}_s{s} (s = 0, 1), test_ttrunc_B{B}_s0, test_vsaq_B{B}_s0, test_tfull_s0,
test_tfull_zeroshot.
usage: python b35_verdict.py <root> --json results/b35_scores.json
"""
import argparse
import gzip
import json
import math
import os

import numpy as np

import b35_data as D

BS = (1, 2, 4, 8)
NS = D.NS
SEEDS = (0, 1)
NBOOT = 10000
CHANCE = 1 / 32
FLOOR = 0.10
WIN = 0.05
TIE = 0.02
TIE_CI = 0.05
TEXT_MARGIN = 0.10
REF_GAP = 0.02


def load(root, run):
    p = os.path.join(root, run, 'test.jsonl.gz')
    if not os.path.exists(p):
        return None
    out = {}
    for l in gzip.open(p, 'rt'):
        z = json.loads(l)
        out[(z['N'], z['idx'], z['qtype'])] = z['correct']
    return out


def vec(rows, N, qt):
    n = D.SPLIT_SIZES['test']
    return np.array([rows[(N, i, qt)] for i in range(n)], dtype=float)


def arm_vecs(root, runs, N, qt):
    """(seeds, 600) correctness; None if any run is missing."""
    rs = [load(root, r) for r in runs]
    if any(r is None for r in rs):
        return None
    return np.stack([vec(r, N, qt) for r in rs])


def boot_ci(d, seed=35):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), (NBOOT, len(d)))
    m = d[idx].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def classify(v, l, ref):
    """v, l: (seeds, n) arrays for VSA and learned tokens; ref: full-text accuracy of the cell."""
    av, al = v.mean(), l.mean()
    d = v.mean(0) - l.mean(0)
    delta = float(d.mean())
    lo, hi = boot_ci(d)
    per_seed = [float(v[s].mean() - l[s].mean()) for s in range(min(len(v), len(l)))]
    if max(av, al) <= FLOOR:
        c = 'FLOOR'
    elif ref is not None and min(av, al) >= ref - REF_GAP:
        c = 'REFERENCE'
    elif delta >= WIN and lo > 0 and all(x > 0 for x in per_seed):
        c = 'VSA WINS'
    elif delta <= -WIN and hi < 0 and all(x < 0 for x in per_seed):
        c = 'VSA LOSES'
    elif abs(delta) <= TIE and lo >= -TIE_CI and hi <= TIE_CI:
        c = 'TIE'
    else:
        c = 'UNRESOLVED'
    return dict(vsa=float(av), lmt=float(al), delta=delta, ci=[lo, hi], per_seed=per_seed, cls=c)


def headline(cells):
    el = [c for c in cells if c['cls'] not in ('FLOOR', 'REFERENCE')]
    w = sum(c['cls'] == 'VSA WINS' for c in el)
    lo = sum(c['cls'] == 'VSA LOSES' for c in el)
    if not el:
        return 'NO READING (every cell at the floor or at the reference)'
    if w >= 2 and lo == 0:
        return 'SUPPORTED: the VSA structure helps at an equal budget'
    if lo >= 2 and w == 0:
        return 'REFUTED: learned memory tokens do better at an equal budget'
    if all(c['cls'] == 'TIE' for c in el):
        return 'NULL: no difference at an equal budget'
    return f'MIXED ({w} wins, {lo} losses, {len(el)} eligible cells)'


def p_theory(V, s):
    from scipy.stats import norm
    t = np.linspace(-10, 10, 4001)
    trap = getattr(np, 'trapezoid', None) or np.trapz
    return float(trap(norm.pdf(t) * norm.cdf(t + s) ** (V - 1), t))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root')
    ap.add_argument('--json', default=None)
    ap.add_argument('--codes', default='rand')
    a = ap.parse_args()
    R = a.root
    out = dict(cells={}, text={}, capacity={}, vsaq={}, ref={})
    print('B35 verdict (test split; accuracy = exact match; means over seeds; 95% paired bootstrap CIs over the 600 '
          'instances of each N)\n')
    # references
    ref = {}
    for N in NS:
        for qt in ('hop1', 'hop2'):
            t = arm_vecs(R, ['test_tfull_s0'], N, qt)
            z = arm_vecs(R, ['test_tfull_zeroshot'], N, qt)
            ref[(N, qt)] = float(t.mean()) if t is not None else None
            out['ref'][f'N{N}_{qt}'] = dict(tfull=ref[(N, qt)], zeroshot=float(z.mean()) if z is not None else None)
    print('Full text with LoRA (zero-shot):', ', '.join(
        f"N={N} {qt} {out['ref'][f'N{N}_{qt}']['tfull']:.3f} ({out['ref'][f'N{N}_{qt}']['zeroshot'] or float('nan'):.3f})"
        for N in NS for qt in ('hop1', 'hop2') if out['ref'][f'N{N}_{qt}']['tfull'] is not None))
    for qt in ('hop1', 'hop2'):
        print(f'\n== {qt}: VSA vs learned memory tokens at equal budget ' + ('(PRIMARY)' if qt == 'hop1' else
                                                                             '(secondary)'))
        print(f"{'B':>2} {'N':>4} {'N/B':>5} {'VSA':>6} {'LMT':>6} {'delta':>7} {'CI':>17} {'T-B':>6} {'Vq':>6} "
              f"{'ext':>6} {'theory':>6}  class")
        cells = []
        for B in BS:
            for N in NS:
                v = arm_vecs(R, [f'test_vsa_B{B}_s{s}' for s in SEEDS], N, qt)
                l = arm_vecs(R, [f'test_lmt_B{B}_s{s}' for s in SEEDS], N, qt)
                if v is None or l is None:
                    continue
                c = classify(v, l, ref[(N, qt)])
                tb = arm_vecs(R, [f'test_ttrunc_B{B}_s0'], N, qt)
                c['ttrunc'] = float(tb.mean()) if tb is not None else None
                if tb is not None:
                    for nm, x in (('vsa', v), ('lmt', l)):
                        d = x.mean(0) - tb[0]
                        lo, hi = boot_ci(d)
                        c[f'{nm}_minus_text'] = dict(delta=float(d.mean()), ci=[lo, hi],
                                                     beats=bool(d.mean() >= TEXT_MARGIN and lo > 0))
                vq = arm_vecs(R, [f'test_vsaq_B{B}_s0'], N, qt) if qt == 'hop1' else None
                c['vsaq'] = float(vq.mean()) if vq is not None else None
                ext = json.load(open(os.path.join(R, 'ext_test.json'))) if os.path.exists(
                    os.path.join(R, 'ext_test.json')) else {}
                e = ext.get(f'{a.codes}_B{B}_N{N}')
                c['ext'] = e[qt] if e else None
                c['theory'] = e[f'theory_{qt}'] if e else None
                c['B'], c['N'] = B, N
                cells.append(c)
                out['cells'][f'{qt}_B{B}_N{N}'] = c
                f = lambda x: f'{x:6.3f}' if x is not None else '     -'
                print(f"{B:>2} {N:>4} {N / B:>5.0f} {c['vsa']:6.3f} {c['lmt']:6.3f} {c['delta']:+7.3f} "
                      f"[{c['ci'][0]:+.3f},{c['ci'][1]:+.3f}] {f(c['ttrunc'])} {f(c['vsaq'])} {f(c['ext'])} "
                      f"{f(c['theory'])}  {c['cls']}")
        h = headline(cells)
        out[f'headline_{qt}'] = h
        print(f'Reading ({qt}): {h}')
        bt = [(c['B'], c['N'], nm) for c in cells for nm in ('vsa', 'lmt') if c.get(f'{nm}_minus_text', {}).get('beats')]
        out[f'beats_text_{qt}'] = bt
        print(f'Soft memory beats text cut to the budget (>= +{TEXT_MARGIN:.2f}, CI > 0): VSA in '
              f"{sum(x[2] == 'vsa' for x in bt)}/{len(cells)} cells, learned tokens in "
              f"{sum(x[2] == 'lmt' for x in bt)}/{len(cells)}")
    # capacity
    print('\n== Capacity: exact external decoding of the VSA bundles vs theory (test split)')
    p = os.path.join(R, 'ext_test.json')
    if os.path.exists(p):
        ext = json.load(open(p))
        ok_all = True
        for k, e in ext.items():
            if not k.startswith(a.codes + '_B'):
                continue
            for qt in ('hop1', 'hop2'):
                th, x, n = e[f'theory_{qt}'], e[qt], e['n']
                tol = max(0.03, 3 * math.sqrt(max(th * (1 - th), 1e-9) / n))
                ok = abs(x - th) <= tol if 0.05 <= th <= 0.995 else True
                ok_all &= ok
                out['capacity'][f'{k}_{qt}'] = dict(ext=x, theory=th, tol=tol, ok=bool(ok))
        out['capacity_matches_theory'] = bool(ok_all)
        print('ext within tolerance of theory in every informative cell:', ok_all)
    # LM efficiency
    print('\n== How much of what the bundle holds does the LM read? (single hop, VSA arm)')
    for c in [out['cells'][k] for k in out['cells'] if k.startswith('hop1')]:
        if c['ext'] is not None:
            c['reads_bundle'] = bool(c['vsa'] >= c['ext'] - 0.05)
            print(f"B={c['B']} N={c['N']} N/B={c['N'] / c['B']:.0f}: LM {c['vsa']:.3f} of ext {c['ext']:.3f}"
                  f" ({'reads what it holds' if c['reads_bundle'] else 'short of it'})")
    # VSA-q: the LM names what unbinding outside it retrieves
    print('\n== VSA-q (unbinding outside the LM; single hop) vs exact clean-up of the same bundles')
    for c in [out['cells'][k] for k in out['cells'] if k.startswith('hop1')]:
        if c.get('vsaq') is not None and c['ext'] is not None:
            c['vsaq_names'] = bool(c['vsaq'] >= c['ext'] - 0.05)
            print(f"B={c['B']} N={c['N']} N/B={c['N'] / c['B']:.0f}: Vq {c['vsaq']:.3f} vs ext {c['ext']:.3f} "
                  f"(theory {c['theory']:.3f}) -> {'names what unbinding retrieves' if c['vsaq_names'] else 'short of it'}")
    # zero-shot full text, lenient
    print('\n== Full text zero-shot (frozen LM), lenient scoring (gold as a whole word in the first line)')
    p = os.path.join(R, 'test_tfull_zeroshot', 'test.jsonl.gz')
    if os.path.exists(p):
        import re
        rows = [json.loads(l) for l in gzip.open(p, 'rt')]
        for N in NS:
            for qt in ('hop1', 'hop2'):
                r = [z for z in rows if z['N'] == N and z['qtype'] == qt]
                ok = [bool(re.search(r'\b' + re.escape(z['gold']) + r'\b', z['pred'].split('\n')[0])) for z in r]
                out['ref'][f'N{N}_{qt}']['zeroshot_lenient'] = float(np.mean(ok))
        print(', '.join(f"N={N} {qt} {out['ref'][f'N{N}_{qt}']['zeroshot_lenient']:.3f}" for N in NS
                        for qt in ('hop1', 'hop2')))
    # from-scratch control
    print('\n== From-scratch transformer reader (CPU; B = 4; mean of seeds)')
    for arm in ('vsa', 'vsakey', 'vsaq', 'lmt'):
        fs = [os.path.join(R, 'scratch', f'test_scratch_{arm}_B4_s{s}.json') for s in SEEDS]
        fs = [f for f in fs if os.path.exists(f)]
        if not fs:
            continue
        ds = [json.load(open(f))['test'] for f in fs]
        m = {k: float(np.mean([d[k] for d in ds])) for k in ds[0]}
        out.setdefault('scratch', {})[arm] = dict(n_seeds=len(ds), **m)
        print(f"{arm:7s} ({len(ds)} seeds): " + ', '.join(f'{k} {v:.3f}' for k, v in sorted(m.items())) +
              f"  -> {'learns to read' if m['N8_hop1'] >= 0.5 else 'does not learn to read'} (N=8 single hop >= 0.5)")
    # machines
    hosts = {}
    for d in sorted(os.listdir(R)):
        sp = os.path.join(R, d, 'summary.json')
        if d.startswith('test_') and os.path.exists(sp):
            s_ = json.load(open(sp))
            hosts[d] = f"{s_.get('host', '?')} / {s_.get('gpu', '?')}"
    out['machines'] = hosts
    print('\n== Machines:', '; '.join(f'{k}: {v}' for k, v in hosts.items()))
    predictions(out, R)
    if a.json:
        json.dump(out, open(a.json, 'w'), indent=1)
    return out


def predictions(out, R):
    """Mechanical check of the registered predictions (reporting only; prereg/B35_soft_token_vsa_prereg.md)."""
    C = out['cells']
    res = {}
    def cells(qt):
        return [C[k] for k in C if k.startswith(qt)]
    dict_ok = True
    for x in D.eval_set('test'):
        for q in ('q1', 'q2'):
            dict_ok &= D.dict_answer(x, x[q]) == x[q]['answer']
    res['P1 DICT 1.000'] = dict_ok
    cap = out.get('capacity', {})
    hop1_ext = {k: v for k, v in cap.items() if k.endswith('_hop1')}
    res['P2 ext matches theory everywhere'] = bool(out.get('capacity_matches_theory')) and all(
        v['ext'] >= 0.99 for k, v in hop1_ext.items() if not k.startswith('rand_B1_N128'))
    rf = out['ref']
    res['P3 T-full >= .99 / .98'] = all(rf[f'N{N}_hop1']['tfull'] >= 0.99 and rf[f'N{N}_hop2']['tfull'] >= 0.98
                                        for N in NS)
    res['P4 T-B <= .08'] = all(c['ttrunc'] is not None and c['ttrunc'] <= 0.08 for c in cells('hop'))
    res['P5 V <= .10 everywhere'] = all(c['vsa'] <= 0.10 for c in cells('hop'))
    res['P6 L <= .10 (B8 N8 <= .25)'] = all(c['lmt'] <= (0.25 if (c['B'], c['N']) == (8, 8) else 0.10)
                                           for c in cells('hop'))
    el = [c for c in cells('hop1') if c['cls'] not in ('FLOOR', 'REFERENCE')]
    res['P7 headline NO READING (<= 1 eligible)'] = len(el) <= 1
    bt = out.get('beats_text_hop1', [])
    res['P8 no soft arm beats text (V 0, L <= 1)'] = sum(x[2] == 'vsa' for x in bt) == 0 and sum(
        x[2] == 'lmt' for x in bt) <= 1
    res['P9 V short of its bundle by >= .85 where ext >= .95'] = all(
        c['ext'] - c['vsa'] >= 0.85 for c in cells('hop1') if c['ext'] is not None and c['ext'] >= 0.95)
    vq = {(c['B'], c['N']): c['vsaq'] for c in cells('hop1') if c.get('vsaq') is not None}
    ex = {(c['B'], c['N']): c['ext'] for c in cells('hop1')}
    res['P10 Vq >= .95 at N=8'] = all(vq.get((B, 8), 0) >= 0.95 for B in BS)
    res['P11 Vq N=32 >= .80 (B>=4); N=128 short by >= .2 (B<=4); rises with B at N=128'] = (
        all(vq.get((B, 32), 0) >= 0.80 for B in (4, 8)) and
        all(ex[(B, 128)] - vq.get((B, 128), 1) >= 0.2 for B in (1, 2, 4)) and
        all(vq.get((BS[i], 128), 0) <= vq.get((BS[i + 1], 128), 0) for i in range(len(BS) - 1)))
    sc = out.get('scratch', {})
    if sc:
        res['P12 scratch: vsa, vsakey <= .10; vsaq >= .95; lmt >= .15 and >= vsa + .10 (N=8)'] = (
            sc['vsa']['N8_hop1'] <= 0.10 and sc['vsakey']['N8_hop1'] <= 0.10 and sc['vsaq']['N8_hop1'] >= 0.95
            and sc['lmt']['N8_hop1'] >= 0.15 and sc['lmt']['N8_hop1'] >= sc['vsa']['N8_hop1'] + 0.10)
    z = [rf[f'N{N}_hop1'].get('zeroshot_lenient') for N in NS]
    if all(v is not None for v in z):
        res['P13 zero-shot lenient <= .6'] = all(v <= 0.6 for v in z)
    out['predictions'] = res
    print('\n== Registered predictions (mechanical check; P14 second scorer and P15 GPU-hours are checked separately)')
    for k, v in res.items():
        print(f"  {'HELD  ' if v else 'FAILED'} {k}")


if __name__ == '__main__':
    main()
