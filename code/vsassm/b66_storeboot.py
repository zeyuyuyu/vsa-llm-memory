"""B66 part (D) (prereg/B66_*_prereg.md): store-level intervals for the headline equal-bit contrasts of B60, B62 and
B64, from the SAVED test predictions only (no new test run).

The verdicts of B60 / B62 / B64 resample entity clusters (entity within store) inside each cell; the test stores
themselves (20 per cell at M = 256, 5 at M = 1,024) are held fixed.  Here the test stores are resampled too.

Data (per encoder; the per-query test predictions the verdicts used, assembled on the same merged records):
  b59_out/fam/{enc}/all59_{enc}_M{M}_V{V}.jsonl.gz (STORE = B53's VSA = 'VSA', VSAR, VSAS ...),
  + b60_verdict.add_lamc (LAMC), b60_verdict.add_vsapx (VSAPC, VSAPCX), b62_verdict.add_b62 (NLMSC, ideal only),
  + B64's SDMT / BEST7 (b64_verdict.load's logic: b64_out/jobs + results/b64_choice_{enc}.json).
Contrasts (x - y, accuracy points): B60 STORE-LAMC, VSAPCX-LAMC, VSAR-LAMC, VSAS-LAMC, VSAPC-LAMC; B62 STORE-NLMSC;
B64 VSAR-SDMT, VSAR-BEST7, VSA-BEST7.  Regions R1 (12 cells), R1-V2, R1-V16.  Ideal accounting (10,000 draws) primary,
real (2,000 draws) secondary (NLMSC has no real-accounting run: reported missing).
Point statistic: mean over cells of the cell mean of the per-query difference (b36_verdict.boot_region's).
Intervals (95%, percentile):
  one-level  b36_verdict.boot_region (entity clusters of the cell resampled; stores fixed), as in the verdicts;
  two-level  (registered) per draw and per cell: the cell's S test stores drawn with replacement, then, for every drawn
             store, its own k_s entity clusters drawn with replacement; cell draw = sum of the drawn clusters'
             difference sums / sum of their query counts; region draw = mean over cells; cells independent;
  store-only (secondary diagnostic) the S stores drawn with replacement, every drawn store's queries kept as they are.
  The two-stage scheme double-counts the within-store sampling variance when stores do not differ (its variance is
  about (2 - 1/S) times the one-level variance, width x1.34-1.40 for S = 5-20; Davison & Hinkley 1997, sec. 3.8), so it
  is conservative; the store-only scheme is the first-stage bootstrap (about unbiased for many stores, coarse at S = 5).
Labels (B60's rule): "x leads" iff lower > 0, "y leads" iff upper < 0, else "level".  B36's bar for VSAR-BEST7 and
VSA-BEST7: diff >= +2 points and lower > 0 passes; <= -2 and upper < 0 loss; else tie.

python3 b66_storeboot.py selftest              # fabricated records only
python3 b66_storeboot.py timing                # full-size cost on fabricated records (one encoder)
python3 b66_storeboot.py run [--encs a,b,..]    # -> results/b66_D_part_{enc}.json   (real saved test outputs)
python3 b66_storeboot.py panel                  # -> results/b66_D_verdict.{json,txt}
python3 b66_storeboot.py check [--encs ..]      # plain-loop point statistics vs the part files -> results/b66_D_check.txt
"""
import gzip
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import b36_verdict as V36  # noqa: E402

RES = os.environ.get('B66_RES', f'{ROOT}/results')
PANEL = ['qwen8', 'b46', 'llama', 'mistral', 'e5', 'bge', 'minilm']
LLM, SENT = PANEL[:4], PANEL[4:]
NAMES = {'qwen8': 'Qwen3-8B', 'b46': 'Granite-3.3-8B', 'llama': 'Llama-3.1-8B', 'mistral': 'Mistral-7B',
         'e5': 'e5-base-v2', 'bge': 'bge-small', 'minilm': 'MiniLM-L6'}
M_GRID, VN, BPF, ACCTS = [256, 1024], [2, 16], [4, 8, 16], ['ideal', 'real']
TEST_REPS = {256: 20, 1024: 5}
REG = {'R1': [(M, V, b) for M in M_GRID for V in VN for b in BPF]}
for _V in VN:
    REG[f'R1-V{_V}'] = [(M, _V, b) for M in M_GRID for b in BPF]
RN = ('R1-V2', 'R1-V16', 'R1')
CONTRASTS = [('STORE', 'LAMC', 'B60'), ('VSAPCX', 'LAMC', 'B60'), ('VSAR', 'LAMC', 'B60'), ('VSAS', 'LAMC', 'B60'),
             ('VSAPC', 'LAMC', 'B60'), ('STORE', 'NLMSC', 'B62'), ('VSAR', 'SDMT', 'B64'), ('VSAR', 'BEST7', 'B64'),
             ('VSA', 'BEST7', 'B64')]
BAR = {('VSAR', 'BEST7'), ('VSA', 'BEST7')}
NB = {'ideal': 10000, 'real': 2000}
CHUNK = 4_000_000  # random cluster indices per chunk (memory bound)


# ----------------------------------------------------------------------------------------------- real data (run only)

def load(enc):
    """Merged test records per (M, V) with every arm of CONTRASTS added (only called by run / check)."""
    import b60_verdict as V60
    import b62_verdict as V62
    import b64_verdict as V64
    ch60, ch62, ch64 = V60.choice(enc), V62.choice(enc), V64.choice(enc)
    data = {}
    for M in M_GRID:
        for V in VN:
            recs = [json.loads(l) for l in gzip.open(f'{ROOT}/b59_out/fam/{enc}/all59_{enc}_M{M}_V{V}.jsonl.gz', 'rt')]
            recs.sort(key=lambda r: r['rep'])
            assert [r['rep'] for r in recs] == list(range(TEST_REPS[M])), (enc, M, V, [r['rep'] for r in recs])
            for r in recs:
                for b in BPF:
                    for a in ACCTS:
                        r['pred'][f'STORE|{b}|{a}'] = r['pred'][f'VSA|{b}|{a}']
            assert V60.add_lamc(recs, enc, M, V, ch60, 0, bpfs=tuple(BPF)), ('LAMC test jobs missing', enc, M, V)
            assert V62.add_b62(recs, enc, M, V, ch62), ('B62 test jobs missing', enc, M, V)
            rows = V64.job_rows(enc, M, V)
            for r in recs:
                jr = rows.get(r['rep'])
                assert jr is not None and jr['_qgold'] == r['qgold'], ('SDMT test job', enc, M, V, r['rep'])
                for b in BPF:
                    for a in ACCTS:
                        cfg = tuple(ch64[f'{M}|{V}|{b}|{a}|SDMT']['cfg'])
                        r['pred'][f'SDMT|{b}|{a}'] = jr[cfg, b, a][0]
                        f = ch64[f'{M}|{V}|{b}|{a}|BEST7']['fam']
                        r['pred'][f'BEST7|{b}|{a}'] = r['pred'][f'{f}|{b}|{a}']
            data[M, V] = recs
    assert V60.add_vsapx(data, enc), ('VSAPc test files missing', enc)
    return data


# ----------------------------------------------------------------------------------------------- statistics

def correct(recs, arm):
    """Per-query correctness (float) concatenated over the cell's stores, or None if any store lacks the arm."""
    out = []
    for r in recs:
        p = r['pred'].get(arm)
        if p is None:
            return None
        out.append((np.asarray(p) == np.asarray(r['qgold'])).astype(np.float64))
    return np.concatenate(out)


def cell_layout(recs):
    """Per store: (query slice, local cluster index per query, k_s); and the one-level cluster ids (per_query's)."""
    lay, o = [], 0
    cl = []
    for r in recs:
        qe = np.asarray(r['qent'])
        u, inv = np.unique(qe, return_inverse=True)
        lay.append((o, o + len(qe), inv, len(u)))
        o += len(qe)
        cl += [f'{r["rep"]}:{e}' for e in r['qent']]
    return lay, np.array(cl)


def cell_draws(lay, diffs, B, rng):
    """diffs: {name: per-query difference}. Returns ({name: two-level draws (B,)}, {name: store-only draws (B,)})."""
    S = len(lay)
    names = list(diffs)
    per = []  # per store: counts per cluster, {name: sums per cluster}
    for (a, b, inv, k) in lay:
        n = np.bincount(inv, minlength=k).astype(np.float64)
        per.append((n, {nm: np.bincount(inv, weights=diffs[nm][a:b], minlength=k) for nm in names}))
    sidx = rng.integers(0, S, size=(B, S))
    tn = np.zeros(B)
    ts = {nm: np.zeros(B) for nm in names}
    sn = np.zeros(B)
    ss = {nm: np.zeros(B) for nm in names}
    for s in range(S):
        n, sums = per[s]
        k = len(n)
        bs = np.nonzero(sidx == s)[0]  # one entry per (draw, slot) occupied by store s
        cnt = np.bincount(bs, minlength=B).astype(np.float64)
        sn += cnt * n.sum()
        for nm in names:
            ss[nm] += cnt * sums[nm].sum()
        step = max(1, CHUNK // max(1, k))
        for c0 in range(0, len(bs), step):
            bb = bs[c0:c0 + step]
            idx = rng.integers(0, k, size=(len(bb), k)).astype(np.int32)
            tn += np.bincount(bb, weights=n[idx].sum(1), minlength=B)
            for nm in names:
                ts[nm] += np.bincount(bb, weights=sums[nm][idx].sum(1), minlength=B)
    return {nm: ts[nm] / tn for nm in names}, {nm: ss[nm] / sn for nm in names}


def lab(x, y, lo, hi):
    return f'{x} leads' if lo > 0 else (f'{y} leads' if hi < 0 else 'level')


def bar(s, lo, hi):
    return 'passes' if (s >= 0.02 and lo > 0) else ('loss' if (s <= -0.02 and hi < 0) else 'tie')


def ci(dr):
    return [float(np.quantile(dr, 0.025)), float(np.quantile(dr, 0.975))]


def analyse(data, ei, contrasts, accts=ACCTS, nb=NB, regions=REG):
    """data: {(M, V): records}; returns {'regions': {...}, 'missing': [...], 'cells': {...}}."""
    J = {'regions': {}, 'missing': [], 'cells': {}}
    for ai, acct in enumerate(accts):
        B = nb[acct]
        cellres = {}  # (M, V, b) -> {name: (point, diff, cl, two, so)}
        cells = sorted({c for R in regions.values() for c in R})
        for (M, V, b) in cells:
            recs = data[M, V]
            lay, cl = cell_layout(recs)
            diffs = {}
            for (x, y, _) in contrasts:
                cx, cy = correct(recs, f'{x}|{b}|{acct}'), correct(recs, f'{y}|{b}|{acct}')
                if cx is None or cy is None:
                    miss = [a_ for a_, c_ in ((x, cx), (y, cy)) if c_ is None]
                    J['missing'].append(f'{acct}|{M}|{V}|{b}|{x}-{y}: no {"/".join(miss)}')
                    continue
                diffs[f'{x}-{y}'] = cx - cy
            rng = np.random.default_rng(V36_seed(ei, ai, M, V, b))
            two, so = cell_draws(lay, diffs, B, rng) if diffs else ({}, {})
            cellres[M, V, b] = {nm: (float(d.mean()), d, cl, two[nm], so[nm]) for nm, d in diffs.items()}
            for nm, d in diffs.items():
                J['cells'][f'{acct}|{M}|{V}|{b}|{nm}'] = float(d.mean())
        for ri, (rn, R) in enumerate(regions.items()):
            for pi, (x, y, src) in enumerate(contrasts):
                nm = f'{x}-{y}'
                if not all(nm in cellres[c] for c in R):
                    continue
                pts = [cellres[c][nm][0] for c in R]
                s1, lo1, hi1, _ = V36.boot_region([(cellres[c][nm][1], cellres[c][nm][2]) for c in R], B,
                                                  66000 + ei * 1000 + ai * 500 + ri * 20 + pi)
                s = float(np.mean(pts))
                assert abs(s - s1) < 1e-12, (s, s1)
                two = np.mean([cellres[c][nm][3] for c in R], 0)
                so = np.mean([cellres[c][nm][4] for c in R], 0)
                c2, c3 = ci(two), ci(so)
                e = {'src': src, 'diff': s, 'n_cells': len(R),
                     'one': {'ci95': [lo1, hi1], 'label': lab(x, y, lo1, hi1)},
                     'two': {'ci95': c2, 'label': lab(x, y, *c2)},
                     'store_only': {'ci95': c3, 'label': lab(x, y, *c3)},
                     'width_ratio_two': (c2[1] - c2[0]) / (hi1 - lo1) if hi1 > lo1 else float('nan'),
                     'width_ratio_store_only': (c3[1] - c3[0]) / (hi1 - lo1) if hi1 > lo1 else float('nan'),
                     'two_draw_mean': float(two.mean())}
                e['label_changed'] = e['one']['label'] != e['two']['label']
                if (x, y) in BAR:
                    e['one']['bar'] = bar(s, lo1, hi1)
                    e['two']['bar'] = bar(s, *c2)
                    e['store_only']['bar'] = bar(s, *c3)
                J['regions'][f'{acct}|{rn}|{nm}'] = e
    return J


def V36_seed(ei, ai, M, V, b):
    return 660000 + ei * 10000 + ai * 5000 + M + V * 3 + b * 7


# ----------------------------------------------------------------------------------------------- run / panel / check

def run(encs):
    for enc in encs:
        t0 = time.time()
        data = load(enc)
        t1 = time.time()
        J = analyse(data, PANEL.index(enc), CONTRASTS)
        J['enc'] = enc
        J['sec'] = {'load': round(t1 - t0, 1), 'boot': round(time.time() - t1, 1)}
        json.dump(J, open(f'{RES}/b66_D_part_{enc}.json', 'w'))
        print(f'part {enc} load {t1 - t0:.0f}s boot {time.time() - t1:.0f}s; missing {len(J["missing"])}', flush=True)


def fmt(e, k):
    lo, hi = e[k]['ci95']
    return f'[{100 * lo:+6.2f}, {100 * hi:+6.2f}] {e[k]["label"]}'


CLAIMS = [
    ('(1) B60: LAMc leads the store (STORE - LAMC) over R1 on all 7', [('STORE', 'LAMC', 'R1', PANEL)]),
    ('(2) B60: LAMc leads the store at R1-V16 on all 7', [('STORE', 'LAMC', 'R1-V16', PANEL)]),
    ('(3) B60: LAMc leads the store at R1-V2 on the 4 language models, e5 and bge',
     [('STORE', 'LAMC', 'R1-V2', LLM + ['e5', 'bge'])]),
    ('(4) B60: LAMc leads VSAPcX over R1 on all 7', [('VSAPCX', 'LAMC', 'R1', PANEL)]),
    ('(5) "least squares beat every Hebbian store variant on every encoder": LAMc leads STORE, VSAR, VSAS, VSAPC, '
     'VSAPCX over R1 on all 7', [(x, 'LAMC', 'R1', PANEL) for x in ('STORE', 'VSAR', 'VSAS', 'VSAPC', 'VSAPCX')]),
    ('(6) B62: NLMSc leads the store over R1 on all 7', [('STORE', 'NLMSC', 'R1', PANEL)]),
    ('(7) B64: SDMt leads the first design (VSAR - SDMT) over R1 on all 7', [('VSAR', 'SDMT', 'R1', PANEL)]),
    ('(8) B64: the best of seven leads the first design (VSAR - BEST7) over R1 on all 7', [('VSAR', 'BEST7', 'R1', PANEL)]),
]
# (9): VSA - BEST7 per-encoder labels as B64 reported them
CLAIM9 = {'qwen8': 'VSA leads', 'b46': 'level', 'llama': 'VSA leads', 'mistral': 'BEST7 leads', 'e5': 'BEST7 leads',
          'bge': 'BEST7 leads', 'minilm': 'BEST7 leads'}


def claims(R, encs, kind):
    """Returns lines; kind in ('one', 'two', 'store_only')."""
    out = []
    for title, parts in CLAIMS:
        fails = []
        for (x, y, rn, es) in parts:
            for e in es:
                if e not in encs:
                    continue
                r = R.get(f'{e}|ideal|{rn}|{x}-{y}')
                if r is None:
                    fails.append(f'{NAMES[e]} {x}-{y} missing')
                elif r[kind]['label'] != f'{y} leads':
                    fails.append(f'{NAMES[e]} {x}-{y} {rn}: {100 * r["diff"]:+.2f} {fmt(r, kind)}')
        out.append(f'{title}: {"HOLDS" if not fails else "FAILS"}' + ('' if not fails else ' (' + '; '.join(fails) + ')'))
    diff9, nbar = [], {'passes': 0, 'tie': 0, 'loss': 0}
    for e in encs:
        r = R.get(f'{e}|ideal|R1|VSA-BEST7')
        if r is None:
            diff9.append(f'{NAMES[e]} missing')
            continue
        nbar[r[kind]['bar']] += 1
        if r[kind]['label'] != CLAIM9[e]:
            diff9.append(f'{NAMES[e]}: {100 * r["diff"]:+.2f} {fmt(r, kind)} (B64: {CLAIM9[e]})')
    out.append('(9) B64: VSA - BEST7 over R1 labels as reported (store leads Qwen3-8B, Llama; level Granite; memory leads '
               f'Mistral + 3 sentence encoders): {"HOLDS" if not diff9 else "CHANGES"}' +
               ('' if not diff9 else ' (' + '; '.join(diff9) + ')') +
               f'; bar passes {nbar["passes"]}, tie {nbar["tie"]}, loss {nbar["loss"]} (of {len(encs)})')
    return out


def panel(encs=PANEL):
    R, miss = {}, {}
    for enc in encs:
        P = json.load(open(f'{RES}/b66_D_part_{enc}.json'))
        for k, v in P['regions'].items():
            R[f'{enc}|{k}'] = v
        miss[enc] = P['missing']
    L = ['# B66 part (D): store-level intervals (prereg/B66_*_prereg.md). Saved test predictions only; accuracy points.',
         '# point = mean over cells of the cell mean; one-level = b36_verdict.boot_region (entity clusters within each',
         '# cell resampled, test stores fixed; the verdicts\' interval); two-level (registered) = test stores resampled,',
         '# then entity clusters within each drawn store; store-only (diagnostic) = test stores resampled, queries kept.',
         '# Ideal 10,000 draws, real 2,000. Label: "x leads" iff lower > 0, "y leads" iff upper < 0, else "level".',
         '# Note: with no between-store variation the two-level variance is about (2 - 1/S) x the one-level variance',
         '# (width x1.34-1.40 for S = 5-20), so the two-level interval is conservative by construction.', '']
    for acct in ACCTS:
        for (x, y, src) in CONTRASTS:
            nm = f'{x}-{y}'
            if not any(f'{e}|{acct}|R1|{nm}' in R for e in encs):
                L.append(f'## {nm} ({src}, {acct}): not available (missing arm)')
                L.append('')
                continue
            L.append(f'## {nm} ({src}, {acct})')
            hb = ' | bar one / two' if (x, y) in BAR else ''
            L.append(f'| encoder | region | point | one-level 95% | two-level 95% | store-only 95% | label changed{hb} |')
            L.append('| --- | --- | --- | --- | --- | --- | ---' + (' | ---' if hb else '') + ' |')
            for e in encs:
                for rn in RN:
                    r = R.get(f'{e}|{acct}|{rn}|{nm}')
                    if r is None:
                        continue
                    row = [NAMES[e], rn, f'{100 * r["diff"]:+6.2f}', fmt(r, 'one'), fmt(r, 'two'), fmt(r, 'store_only'),
                           'YES' if r['label_changed'] else 'no']
                    if hb:
                        row.append(f'{r["one"]["bar"]} / {r["two"]["bar"]}')
                    L.append('| ' + ' | '.join(row) + ' |')
            L.append('')
    L.append('## Width of the two-level (and store-only) 95% interval relative to the one-level interval (ideal; mean '
             'over encoders, [min, max])')
    for (x, y, src) in CONTRASTS:
        nm = f'{x}-{y}'
        for rn in RN:
            w2 = [R[k]['width_ratio_two'] for k in (f'{e}|ideal|{rn}|{nm}' for e in encs) if k in R]
            w3 = [R[k]['width_ratio_store_only'] for k in (f'{e}|ideal|{rn}|{nm}' for e in encs) if k in R]
            if w2:
                L.append(f'  {nm:14s} {rn:7s} two-level x{np.mean(w2):.2f} [{min(w2):.2f}, {max(w2):.2f}]   '
                         f'store-only x{np.mean(w3):.2f} [{min(w3):.2f}, {max(w3):.2f}]')
    L.append('')
    for kind, title in (('one', 'one-level (the verdicts\' interval; continuity)'), ('two', 'TWO-LEVEL (registered)'),
                        ('store_only', 'store-only (diagnostic)')):
        L.append(f'## "On every encoder" claims under the {title} interval (ideal)')
        L += ['  ' + s for s in claims(R, encs, kind)]
        L.append('')
    ch = [k for k, v in R.items() if v['label_changed'] and '|ideal|' in k]
    L.append(f'## Label changes one-level -> two-level (ideal): {len(ch)} of '
             f'{sum("|ideal|" in k for k in R)} (encoder, region, contrast) cases')
    for k in sorted(ch):
        e, acct, rn, nm = k.split('|')
        r = R[k]
        L.append(f'  {NAMES[e]} {rn} {nm}: {100 * r["diff"]:+.2f}; one {fmt(r, "one")} -> two {fmt(r, "two")}')
    L.append('')
    L.append('## Missing arms (reported, not imputed)')
    for e in encs:
        m = sorted({s.split(': ')[0].split('|', 4)[-1] + ' (' + s.split('|')[0] + ')' for s in miss[e]})
        L.append(f'  {NAMES[e]}: {", ".join(m) if m else "none"}')
    open(f'{RES}/b66_D_verdict.txt', 'w').write('\n'.join(L) + '\n')
    json.dump({'regions': R, 'missing': miss}, open(f'{RES}/b66_D_verdict.json', 'w'), indent=1)
    print('\n'.join(L))


def check(encs):
    """Plain-loop recomputation (pure Python loops) of every point statistic against the part files."""
    L = ['# B66 part (D) check: plain-loop point statistics (pure Python loops over the records) vs b66_D_part_*.json']
    worst = 0.0
    n = 0
    for enc in encs:
        P = json.load(open(f'{RES}/b66_D_part_{enc}.json'))
        data = load(enc)
        for acct in ACCTS:
            for (x, y, _) in CONTRASTS:
                nm = f'{x}-{y}'
                for rn, R in REG.items():
                    k = f'{acct}|{rn}|{nm}'
                    if k not in P['regions']:
                        continue
                    tot = 0.0
                    for (M, V, b) in R:
                        s, c = 0.0, 0
                        for r in data[M, V]:
                            px, py, g = r['pred'][f'{x}|{b}|{acct}'], r['pred'][f'{y}|{b}|{acct}'], r['qgold']
                            for i in range(len(g)):
                                s += (1.0 if px[i] == g[i] else 0.0) - (1.0 if py[i] == g[i] else 0.0)
                                c += 1
                        tot += s / c
                    pt = tot / len(R)
                    d = abs(pt - P['regions'][k]['diff'])
                    worst = max(worst, d)
                    n += 1
        L.append(f'{enc}: checked through {n} values so far, max |diff| {worst:.2e}')
        print(L[-1], flush=True)
    L.append(f'ALL: {n} point statistics, max |difference| {worst:.2e} -> {"PASS" if worst < 1e-9 else "FAIL"} (bar 1e-9)')
    open(f'{RES}/b66_D_check.txt', 'w').write('\n'.join(L) + '\n')
    print(L[-1])


# ----------------------------------------------------------------------------------------------- selftest (fabricated)

def fabricate(rng, store_sd, pa=0.50, pb=0.45, ent_sd=0.0):
    """Fabricated records with the real shapes: S stores per (M, V), entities with 1-3 alias queries each; arm A's
    per-query correctness ~ Bernoulli(pa + store effect + entity effect), arm B ~ Bernoulli(pb + entity effect)."""
    data = {}
    for M in M_GRID:
        S = TEST_REPS[M]
        ne = {256: 140, 1024: 560}[M]
        for V in VN:
            recs = []
            for rep in range(S):
                u = rng.normal(0, store_sd)  # store effect on A - B (interaction)
                qent = np.repeat(np.arange(ne), rng.integers(1, 4, ne))
                ee = rng.normal(0, ent_sd, ne)[qent]
                g = np.zeros(len(qent), int)
                pred = {}
                for b in BPF:
                    for a in ACCTS:
                        ca = rng.random(len(qent)) < np.clip(pa + u + ee, 0, 1)
                        cb = rng.random(len(qent)) < np.clip(pb + ee, 0, 1)
                        pred[f'A|{b}|{a}'] = np.where(ca, 0, 1).tolist()
                        pred[f'B|{b}|{a}'] = np.where(cb, 0, 1).tolist()
                recs.append({'rep': rep, 'qent': qent.tolist(), 'qgold': g.tolist(), 'pred': pred})
            data[M, V] = recs
    return data


def selftest():
    rng = np.random.default_rng(20261123)
    con = [('A', 'B', 'fab'), ('A', 'C', 'fab')]  # 'C' missing on purpose
    nb = {'ideal': 2000, 'real': 500}
    out = {}
    for name, sd, esd in (('a: no store effect', 0.0, 0.0), ('a2: no store effect, entity effects', 0.0, 0.2),
                          ('b: store effect sd 0.05', 0.05, 0.0)):
        data = fabricate(rng, sd, ent_sd=esd)
        t0 = time.time()
        J = analyse(data, 0, con, nb=nb)
        dt = time.time() - t0
        e = J['regions']['ideal|R1|A-B']
        out[name] = e
        assert any('A-C: no C' in m for m in J['missing']), J['missing'][:3]
        assert 'ideal|R1|A-C' not in J['regions']
        # (c) the point statistic equals boot_region's and a plain loop's; the two-level draws are centred on it
        recs = {k: v for k, v in data.items()}
        tot = 0.0
        for (M, V, b) in REG['R1']:
            s, c = 0.0, 0
            for r in recs[M, V]:
                for i in range(len(r['qgold'])):
                    s += (r['pred'][f'A|{b}|ideal'][i] == r['qgold'][i]) - (r['pred'][f'B|{b}|ideal'][i] == r['qgold'][i])
                    c += 1
            tot += s / c
        assert abs(tot / 12 - e['diff']) < 1e-12, (tot / 12, e['diff'])
        assert abs(e['two_draw_mean'] - e['diff']) < 0.003, (e['two_draw_mean'], e['diff'])
        print(f'{name}: point {100 * e["diff"]:+.2f}; one {fmt(e, "one")}; two {fmt(e, "two")}; store-only '
              f'{fmt(e, "store_only")}; width two/one x{e["width_ratio_two"]:.2f}, store-only/one '
              f'x{e["width_ratio_store_only"]:.2f}; {dt:.1f}s for 2 accts x 12 cells (2,000 / 500 draws)', flush=True)
    a, a2, b = out['a: no store effect'], out['a2: no store effect, entity effects'], out['b: store effect sd 0.05']
    # theory without store effect: two-level variance ~ (2 - 1/S) one-level -> width ~ x1.34-1.40; store-only ~ x0.9-1.0
    assert 1.2 <= a['width_ratio_two'] <= 1.6, a['width_ratio_two']
    assert 0.75 <= a['width_ratio_store_only'] <= 1.15, a['width_ratio_store_only']
    assert 1.2 <= a2['width_ratio_two'] <= 1.6, a2['width_ratio_two']
    assert b['width_ratio_two'] > a['width_ratio_two'] + 0.3, (b['width_ratio_two'], a['width_ratio_two'])
    assert b['width_ratio_store_only'] > a['width_ratio_store_only'] + 0.4, (b['width_ratio_store_only'],
                                                                              a['width_ratio_store_only'])
    # the one-level interval of (a) reproduces boot_region directly (same seed path)
    print('selftest ok: (a) no store effect: two-level wider by the double-counting factor only; store-only ~ one-level; '
          '(b) store effect: both store-level intervals much wider; (c) point = boot_region = plain loop; missing arms '
          'reported, not imputed')


def timing():
    """Full-size cost on fabricated records: 9 contrasts (aliases of A - B), 10,000 / 2,000 draws, one encoder."""
    rng = np.random.default_rng(7)
    data = fabricate(rng, 0.02)
    for recs in data.values():
        for r in recs:
            for k in [k for k in r['pred'] if k.startswith('A|')]:
                for j in range(1, 9):
                    r['pred'][f'A{j}' + k[1:]] = r['pred'][k]
    con = [('A', 'B', 'fab')] + [(f'A{j}', 'B', 'fab') for j in range(1, 9)]
    t0 = time.time()
    J = analyse(data, 0, con)
    print(f'timing: 9 contrasts x 3 regions x 2 accts (10,000 / 2,000 draws), one fabricated encoder: '
          f'{time.time() - t0:.0f}s; {len(J["regions"])} region entries', flush=True)


def main():
    a = sys.argv[1:]
    encs = a[a.index('--encs') + 1].split(',') if '--encs' in a else PANEL
    if a[0] == 'selftest':
        selftest()
    elif a[0] == 'timing':
        timing()
    elif a[0] == 'run':
        run(encs)
    elif a[0] == 'panel':
        panel(encs)
    elif a[0] == 'check':
        check(encs)
    else:
        raise SystemExit(a[0])


if __name__ == '__main__':
    main()
