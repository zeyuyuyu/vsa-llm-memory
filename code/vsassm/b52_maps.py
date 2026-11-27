"""B52 part (b) (prereg/B52_robustness_layers_maps_1day_prereg.md): B36's region-R1 compression result re-run with
three NEW random maps / value codes. Qwen3-8B codes, ideal accounting, R1 = V in {2, 16} x bpf in {4, 8, 16} x
M in {256, 1024} (12 cells), B32's dev and test stores.

Per map m (tag 'qwen8#b52m<m>'; map 0 = tag 'qwen8' = the original B32 / B36 seeds, used ONLY for the reproduction
gates):
  VSA : every random component of the store is drawn from the map's tag (projections G and value codes C per
        (tag, prep, fmap, factor, block), quantizer tie signs per (tag, store)); B32's grid (prep x fmap x q x stored
        share = 72) on the 18 dev stores of R1, B32's choice rule (best dev mean, ties within 0.1 points to the first in
        grid order) per cell, then the test stores once.
  BEST: B36's chosen competitor family and configuration per R1 cell (results/b36_choice_qwen8.json; no re-selection),
        with every random component of that family drawn from the map's tag: the projection pool (SimHash, PQ sub-spaces,
        fly cells, SDM addresses), the PQ codebook initialisations (refitted on the training split), SDM value codes,
        tie jitter, and the index's LSH hyperplanes (IndexFamily seed).

  python b52_maps.py pqfit  1,2,3                  # PQ banks (centre prep) of the new pools, training split only
  python b52_maps.py dev    0,1,2,3 --workers 8    # VSA dev grid per map -> b52_out/vsa_dev_m<m>.jsonl
  python b52_maps.py choose 0,1,2,3                # -> b52_out/vsa_choice_m<m>.json (+ gate: map 0 vs B32's choices)
  python b52_maps.py gate0                         # map-0 code path reproduces B36's test predictions (4 stores)
  python b52_maps.py test   1,2,3 --workers 8      # VSA and BEST on the R1 test stores -> b52_out/test_m<m>.jsonl.gz
  python b52_maps.py verdict                       # -> results/b52b_verdict.{txt,json}
  python b52_maps.py check                         # second angle: numpy float64 VSA re-derivation (10% of stores),
                                                   # independent re-scoring
"""
import gzip, itertools, json, math, os, sys, time

import numpy as np

import b32_common as C
import b32_run as R32
import b36_run as R36
import b36_common as K

ENC = 'qwen8'
M_GRID, V_R1, BPF = [256, 1024], [2, 16], [4, 8, 16]
CELLS = [(M, V, b) for M in M_GRID for V in V_R1 for b in BPF]
ACCT = 'ideal'
TAG = {0: 'qwen8', 1: 'qwen8#b52m1', 2: 'qwen8#b52m2', 3: 'qwen8#b52m3'}
NEW = [1, 2, 3]
OUT = 'b52_out'
PQDIR = {0: 'b36_out/pqcache'}
B36_CHOICE = 'results/b36_choice_qwen8.json'
B36_TEST = 'b36_out/test'


def idx_seed(m):
    return 0 if m == 0 else C.seed_of('b52idx', m)


# ----------------------------------------------------------------------------------------------- setups

_G = {}


def base():
    if 'base' not in _G:
        pool = C.Pool(R32.ENCS[ENC])
        Tr = pool.train_matrix()
        _G['base'] = (pool, Tr, {p: C.Prep(p, Tr) for p in R32.GRID['prep']})
    return _G['base']


class MapSetup(R36.Setup):
    """B36's Setup with every seed drawn from the map's tag (the data are the qwen8 codes)."""

    def __init__(self, m):
        pool, Tr, preps = base()
        self.enc = TAG[m]
        self.pool, self.preps = pool, preps
        self.ppool = K.Pool(TAG[m], pool.X.shape[1])
        d = PQDIR.get(m, f'{OUT}/pqcache')
        os.makedirs(d, exist_ok=True)
        self.banks = {p: K.PQBank(self.ppool, preps[p](Tr), f'{TAG[m]}_{p}', d) for p in R36.PREPS}
        self.idx = {p: C.IndexFamily('lsh', preps[p](Tr), seed=idx_seed(m)) for p in R32.GRID['prep']}


def setup(m):
    if ('S', m) not in _G:
        _G[('S', m)] = MapSetup(m)
    return _G[('S', m)]


def stores(split, cells=None):
    pool = base()[0]
    reps = R32.DEV_REPS if split == 'dev' else R32.TEST_REPS
    MV = sorted({(M, V) for M, V, _ in (cells or CELLS)})
    return [(M, V, rep) for M, V in MV for rep in range(reps[M])]


def make(split, M, V, rep):
    return C.make_store(base()[0], split, M, V, rep)


# ----------------------------------------------------------------------------------------------- VSA

def _vsa_dev_task(args):
    m, prep, fmap = args
    import torch
    torch.set_num_threads(int(os.environ.get('B52_THREADS', '2')))
    pool, _, preps = base()
    feat = R32.Feat(TAG[m], prep, fmap, pool.X.shape[1], 'cpu')
    rows = []
    for M, V, rep in stores('dev'):
        st = make('dev', M, V, rep)
        cfgs = R32.vsa_cfgs(st, BPF, [ACCT], R32.GRID['q'], R32.GRID['f'])
        P = preps[prep]
        pr = R32.vsa_grid(P(st['keys']), st['vals'], P(st['queries']), V, feat, cfgs, R32.store_seed(TAG[m], st),
                          R32.store_order(st))
        g = st['qgold']
        for (bpf, acct, q, f), p in pr.items():
            rows.append({'M': M, 'V': V, 'rep': rep, 'cfg': [prep, fmap, q, f], 'bpf': bpf, 'acct': acct,
                         'acc': float((p == g).mean())})
    return m, prep, fmap, rows


def pmap(fn, jobs, workers):
    if workers <= 1:
        return map(fn, jobs)
    import multiprocessing as mp
    return mp.get_context('fork').Pool(workers, maxtasksperchild=2).imap_unordered(fn, jobs)


def dev(maps, workers):
    jobs = [(m, p, f) for m in maps for p in R32.GRID['prep'] for f in R32.GRID['fmap']
            if not os.path.exists(f'{OUT}/vsa_dev_m{m}_{p}_{f}.json')]
    base()
    for m, p, f, rows in pmap(_vsa_dev_task, jobs, workers):
        json.dump(rows, open(f'{OUT}/vsa_dev_m{m}_{p}_{f}.json', 'w'))
        print(time.strftime('%H:%M:%S'), 'dev map', m, p, f, len(rows), flush=True)


def choose(maps):
    for m in maps:
        acc = {}
        for p in R32.GRID['prep']:
            for f in R32.GRID['fmap']:
                for r in json.load(open(f'{OUT}/vsa_dev_m{m}_{p}_{f}.json')):
                    acc.setdefault((r['M'], r['V'], r['bpf'], tuple(r['cfg'])), {})[r['rep']] = r['acc']
        out = {}
        for M, V, b in CELLS:
            lst = [(float(np.mean(list(d.values()))), c) for (M_, V_, b_, c), d in acc.items() if (M_, V_, b_) == (M, V, b)]
            assert all(len(d) == R32.DEV_REPS[M] for (M_, V_, b_, c), d in acc.items() if (M_, V_, b_) == (M, V, b))
            assert len(lst) == 72, len(lst)
            top = max(a for a, _ in lst)
            tied = sorted([c for a, c in lst if a >= top - 0.001], key=R32.vsa_key)
            out[f'{M}|{V}|{b}'] = {'cfg': list(tied[0]), 'dev_acc': [a for a, c in lst if c == tied[0]][0],
                                   'best_dev_acc': top, 'n_tied': len(tied)}
        json.dump({'map': m, 'tag': TAG[m], 'choice': out}, open(f'{OUT}/vsa_choice_m{m}.json', 'w'), indent=1)
        print('map', m, 'choices:', {k: v['cfg'] for k, v in out.items()})
    if 0 in maps:   # gate: the map-0 dev path reproduces B32's committed choices
        b36 = json.load(open(B36_CHOICE))['choice']
        mine = json.load(open(f'{OUT}/vsa_choice_m0.json'))['choice']
        same = {k: b36[f'{k}|{ACCT}|VSA']['cfg'] == v['cfg'] for k, v in mine.items()}
        print('GATE map-0 VSA choices equal B32/B36 in', sum(same.values()), 'of', len(same), 'R1 cells', same)
        json.dump(same, open(f'{OUT}/gate_choice_m0.json', 'w'), indent=1)


def b36_best():
    ch = json.load(open(B36_CHOICE))['choice']
    out = {}
    for M, V, b in CELLS:
        fam = ch[f'{M}|{V}|{b}|{ACCT}|BEST']['fam']
        out[M, V, b] = (fam, tuple(ch[f'{M}|{V}|{b}|{ACCT}|{fam}']['cfg']))
    return out


def vsa_preds(m, st, chosen):
    """chosen: {bpf: [prep, fmap, q, f]} -> {bpf: predictions}."""
    pool, _, preps = base()
    groups = {}
    for b, (prep, fmap, q, f) in chosen.items():
        groups.setdefault((prep, fmap), []).append((b, q, f))
    out = {}
    for (prep, fmap), lst in groups.items():
        key = ('F', m, prep, fmap)
        if key not in _G:
            _G[key] = R32.Feat(TAG[m], prep, fmap, pool.X.shape[1], 'cpu')
        cfgs = [(b, C.vsa_dim(b * st['M'], q, ACCT), q, max(1, int(math.floor(f * st['M'])))) for b, q, f in lst]
        P = preps[prep]
        pr = R32.vsa_grid(P(st['keys']), st['vals'], P(st['queries']), st['V'], _G[key], cfgs,
                          R32.store_seed(TAG[m], st), R32.store_order(st))
        out.update(pr)
    return out


def best_preds(m, st, best):
    """best: {bpf: (family, cfg)} -> {bpf: predictions} with the map's seeds."""
    S = setup(m)
    out = {}
    nonidx = {b: fc for b, fc in best.items() if fc[0] != 'IDX'}
    if nonidx:
        fams = sorted({f for f, _ in nonidx.values()})
        sel = {}
        for f, c in nonidx.values():
            sel.setdefault(f, set()).add(c)
        res = R36.run_store(S, st, fams, sorted(nonidx), [ACCT], select=sel)
        for b, (f, c) in nonidx.items():
            out[b] = np.asarray(res[f, c, b, ACCT][0])
    for b, (f, c) in best.items():
        if f == 'IDX':
            prep, family, share = c
            assert family == 'lsh'
            p, _ = R32.run_idx(st, S.preps, {(prep, 'lsh'): S.idx[prep]}, prep, 'lsh', b, ACCT, share)
            out[b] = np.asarray(p)
    return out


def _test_task(args):
    m, M, V, rep = args
    import torch
    torch.set_num_threads(int(os.environ.get('B52_THREADS', '2')))
    st = make('test', M, V, rep)
    ch = json.load(open(f'{OUT}/vsa_choice_m{m}.json'))['choice']
    vp = vsa_preds(m, st, {b: ch[f'{M}|{V}|{b}']['cfg'] for b in BPF})
    bb = b36_best()
    bp = best_preds(m, st, {b: bb[M, V, b] for b in BPF})
    rec = {'map': m, 'M': M, 'V': V, 'rep': rep, 'qgold': st['qgold'].tolist(), 'qent': st['qent'].tolist(), 'pred': {}}
    for b in BPF:
        rec['pred'][f'VSA|{b}'] = np.asarray(vp[b]).tolist()
        rec['pred'][f'BEST|{b}'] = bp[b].tolist()
    return rec


def test(maps, workers, only=None, outname='test'):
    jobs = [(m, M, V, rep) for m in maps for M, V, rep in stores('test')]
    if only:
        jobs = [j for j in jobs if j[1:] in only]
    for m in maps:
        setup(m)
        for p in R36.PREPS:
            if setup(m).banks[p].cent is None and p == 'center':
                raise SystemExit(f'PQ bank of map {m} ({p}) not fitted: run pqfit first')
    recs = {}
    for r in pmap(_test_task, jobs, workers):
        recs.setdefault(r['map'], []).append(r)
        print(time.strftime('%H:%M:%S'), outname, 'map', r['map'], r['M'], r['V'], r['rep'], flush=True)
    for m, lst in recs.items():
        with gzip.open(f'{OUT}/{outname}_m{m}.jsonl.gz', 'wt') as fo:
            for r in sorted(lst, key=lambda x: (x['M'], x['V'], x['rep'])):
                fo.write(json.dumps(r) + '\n')


def pqfit(maps):
    for m in maps:
        S = setup(m)
        t0 = time.time()
        if S.banks['center'].cent is None:
            S.banks['center'].fit()
        print('pq fitted map', m, 'center', f'{time.time() - t0:.0f}s', flush=True)


def gate0():
    """Map 0 (original seeds) through this code path must reproduce B36's test predictions (VSA and BEST, every R1
    bpf) on rep 0 of each (M, V)."""
    only = [(M, V, 0) for M in M_GRID for V in V_R1]
    test([0], 1, only=only, outname='gate0')
    res = {}
    for r in (json.loads(l) for l in gzip.open(f'{OUT}/gate0_m0.jsonl.gz', 'rt')):
        b36 = [json.loads(l) for l in gzip.open(f'{B36_TEST}/all_{ENC}_M{r["M"]}_V{r["V"]}.jsonl.gz', 'rt')]
        o = [x for x in b36 if x['rep'] == r['rep']][0]
        for b in BPF:
            for arm in ('VSA', 'BEST'):
                a, z = np.array(r['pred'][f'{arm}|{b}']), np.array(o['pred'][f'{arm}|{b}|{ACCT}'])
                res[f'{r["M"]}|{r["V"]}|{b}|{arm}'] = float((a == z).mean())
    print('GATE map-0 per-query agreement with B36 test predictions:', res)
    json.dump(res, open(f'{OUT}/gate0_agreement.json', 'w'), indent=1)


# ----------------------------------------------------------------------------------------------- verdict

def load_test(m):
    if m == 'b36':
        out = []
        for M in M_GRID:
            for V in V_R1:
                for o in (json.loads(l) for l in gzip.open(f'{B36_TEST}/all_{ENC}_M{M}_V{V}.jsonl.gz', 'rt')):
                    out.append({'M': M, 'V': V, 'rep': o['rep'], 'qgold': o['qgold'], 'qent': o['qent'],
                                'pred': {f'{a}|{b}': o['pred'][f'{a}|{b}|{ACCT}'] for a in ('VSA', 'BEST') for b in BPF}})
        return out
    return [json.loads(l) for l in gzip.open(f'{OUT}/test_m{m}.jsonl.gz', 'rt')]


def cell_arrays(recs, M, V, b, vsa_src, best_src):
    """Per-query correctness of VSA (from recs vsa_src) and BEST (from best_src), cluster ids (rep:entity)."""
    def get(rs, arm):
        ok, cl = [], []
        for r in sorted([r for r in rs if r['M'] == M and r['V'] == V], key=lambda x: x['rep']):
            ok.append((np.array(r['pred'][f'{arm}|{b}']) == np.array(r['qgold'])).astype(np.float64))
            cl += [f'{r["rep"]}:{e}' for e in r['qent']]
        return np.concatenate(ok), np.array(cl)
    a, cl = get(recs[vsa_src], 'VSA')
    z, cl2 = get(recs[best_src], 'BEST')
    assert (cl == cl2).all()
    return a, z, cl


def verdict(B=10000):
    import b36_verdict as V36
    recs = {m: load_test(m) for m in NEW}
    recs['b36'] = load_test('b36')
    L, J = [], {'maps': {}}

    def P(s=''):
        L.append(s); print(s)
    P('# B52 (b) verdict (prereg/B52_robustness_layers_maps_1day_prereg.md): region R1 (Qwen3-8B codes, ideal accounting),')
    P('# VSA - BEST with three new random maps / value codes; generated by b52_maps.py verdict on %s UTC'
      % time.strftime('%Y-%m-%d %H:%M', time.gmtime()))
    for m in NEW:
        ch = json.load(open(f'{OUT}/vsa_choice_m{m}.json'))['choice']
        P(f'  map {m} VSA dev choices: ' + '; '.join(f'{k} {v["cfg"]}' for k, v in ch.items()))
    # per-map region statistics, and the shared-cluster draws for the two-level bootstrap
    rng = np.random.default_rng(52)
    cellinfo = []
    for M, V, b in CELLS:
        a0, z0, cl = cell_arrays(recs, M, V, b, 'b36', 'b36')
        k, inv = V36.clusters(cl)
        n = np.bincount(inv, minlength=k).astype(np.float64)
        idx = rng.integers(0, k, size=(B, k))
        cellinfo.append((M, V, b, inv, k, n, idx))
    D = {}      # (vsa_src, best_src) -> (stat, per-draw region diffs (B,), per-cell table rows)
    for vs, bs in [(m, m) for m in NEW] + [(m, 'b36') for m in NEW] + [('b36', 'b36')]:
        draws = np.zeros(B)
        rows = []
        for (M, V, b, inv, k, n, idx) in cellinfo:
            a, z, cl = cell_arrays(recs, M, V, b, vs, bs)
            d = a - z
            s = np.bincount(inv, weights=d, minlength=k)
            dr = s[idx].sum(1) / n[idx].sum(1)
            draws += dr
            rows.append((M, V, b, 100 * a.mean(), 100 * z.mean(), 100 * d.mean(), 100 * np.quantile(dr, 0.025),
                         100 * np.quantile(dr, 0.975)))
        draws = 100 * draws / len(cellinfo)
        stat = float(np.mean([r[5] for r in rows]))
        D[vs, bs] = (stat, draws, rows)
    P('\n## Per map: R1 VSA - BEST (points), 95% cluster bootstrap (entity within store, resampled within each cell, '
      f'{B} draws)')
    P('    map  BEST used                               R1 diff    95% CI           VSA R1 mean  BEST R1 mean')
    for vs, bs in [(m, m) for m in NEW] + [(m, 'b36') for m in NEW] + [('b36', 'b36')]:
        stat, draws, rows = D[vs, bs]
        lo, hi = np.quantile(draws, 0.025), np.quantile(draws, 0.975)
        lab = {('b36', 'b36'): 'B36 as published (map 0, B36 BEST)'}.get((vs, bs),
              f'BEST re-seeded with map {bs}' if bs != 'b36' else 'BEST as in B36 (original seeds)')
        P(f'    {str(vs):>4} {lab:<38} {stat:+7.2f}   [{lo:+6.2f}, {hi:+6.2f}]   {np.mean([r[3] for r in rows]):6.2f}'
          f'       {np.mean([r[4] for r in rows]):6.2f}')
        J['maps'][f'{vs}|{bs}'] = {'R1': stat, 'ci': [float(lo), float(hi)],
                                   'cells': [dict(zip(['M', 'V', 'bpf', 'VSA', 'BEST', 'diff', 'lo', 'hi'], map(float, r)))
                                             for r in rows]}
    P('\n## Per cell (points): VSA / BEST / diff for each new map (BEST re-seeded with the same map)')
    for i, (M, V, b, *_r) in enumerate(cellinfo):
        P(f'  M={M:5d} V={V:3d} bpf={b:3d}  ' + '  '.join(
            f'm{m}: {D[m, m][2][i][3]:5.2f}/{D[m, m][2][i][4]:5.2f}/{D[m, m][2][i][5]:+6.2f}' for m in NEW) +
          f'   B36: {D["b36", "b36"][2][i][3]:5.2f}/{D["b36", "b36"][2][i][4]:5.2f}/{D["b36", "b36"][2][i][5]:+6.2f}')
    # across maps
    stats = np.array([D[m, m][0] for m in NEW])
    Dm = np.stack([D[m, m][1] for m in NEW])                       # (3, B): shared entity draws per map
    pick = np.random.default_rng(53).integers(0, len(NEW), size=(B, len(NEW)))
    two = Dm[pick, np.arange(B)[:, None]].mean(1)
    sd = float(stats.std(ddof=1))
    from scipy import stats as sst
    tq = sst.t.ppf(0.975, len(NEW) - 1)
    ent = Dm.mean(0)                                                # entity-only draws of the map-averaged diff
    comb = np.sqrt(np.var(ent, ddof=1) + sd ** 2 / len(NEW))
    J['across'] = {'mean': float(stats.mean()), 'sd_between_maps': sd, 'per_map': stats.tolist(),
                   'two_level_ci': [float(np.quantile(two, 0.025)), float(np.quantile(two, 0.975))],
                   'entity_only_ci': [float(np.quantile(ent, 0.025)), float(np.quantile(ent, 0.975))],
                   't_ci_maps_only': [float(stats.mean() - tq * sd / math.sqrt(3)), float(stats.mean() + tq * sd / math.sqrt(3))],
                   'normal_combined_ci': [float(stats.mean() - 1.96 * comb), float(stats.mean() + 1.96 * comb)]}
    A = J['across']
    P('\n## Across the three new maps (BEST re-seeded with each map)')
    P(f'  mean R1 VSA - BEST {A["mean"]:+.2f}; per map ' + ', '.join(f'{x:+.2f}' for x in stats) +
      f'; between-map SD {sd:.2f}')
    P(f'  two-level bootstrap (maps with replacement, entity clusters shared across maps): '
      f'[{A["two_level_ci"][0]:+.2f}, {A["two_level_ci"][1]:+.2f}]  (registered primary interval)')
    P(f'  entity clusters only (map-averaged difference): [{A["entity_only_ci"][0]:+.2f}, {A["entity_only_ci"][1]:+.2f}]')
    P(f'  normal, entity variance + between-map variance / 3: [{A["normal_combined_ci"][0]:+.2f}, {A["normal_combined_ci"][1]:+.2f}]')
    P(f'  t interval over the 3 map means (maps only, df 2): [{A["t_ci_maps_only"][0]:+.2f}, {A["t_ci_maps_only"][1]:+.2f}]')
    four = np.array(list(stats) + [D['b36', 'b36'][0]])
    P(f'  with B36 as a fourth map: mean {four.mean():+.2f}, SD {four.std(ddof=1):.2f}, range [{four.min():+.2f}, {four.max():+.2f}]')
    J['four_maps'] = {'mean': float(four.mean()), 'sd': float(four.std(ddof=1)), 'min': float(four.min()), 'max': float(four.max())}
    per_ok = [J['maps'][f'{m}|{m}']['R1'] >= 2.0 and J['maps'][f'{m}|{m}']['ci'][0] > 0 for m in NEW]
    robust = all(per_ok) and A['two_level_ci'][0] > 0
    J['bar'] = {'per_map_pass': per_ok, 'two_level_lo_gt_0': A['two_level_ci'][0] > 0, 'ROBUST': robust}
    P(f'\n## Bar: every map R1 >= +2.0 with its cluster CI above 0 ({per_ok}) and the two-level interval above 0 '
      f'({A["two_level_ci"][0] > 0}) -> {"ROBUST" if robust else "NOT ROBUST"}')
    os.makedirs('results', exist_ok=True)
    open('results/b52b_verdict.txt', 'w').write('\n'.join(L) + '\n')
    json.dump(J, open('results/b52b_verdict.json', 'w'), indent=1)


# ----------------------------------------------------------------------------------------------- second angle

def rederive_vsa(m, st, cfg, b):
    """numpy float64 re-implementation of the VSA read with the map's seeds (same generators, own arithmetic)."""
    pool, _, preps = base()
    prep, fmap, q, f = cfg
    M, V = st['M'], st['V']
    d = C.vsa_dim(b * M, q, ACCT)
    ns = max(1, int(math.floor(f * M)))
    order = R32.store_order(st)
    keep = order[:ns]
    P = preps[prep]
    Kc, Qc = P(st['keys']).astype(np.float64), P(st['queries']).astype(np.float64)
    vals = st['vals']
    nf = 1 if fmap in ('lin', 'sgn') else int(fmap[3:])
    sc = np.zeros((Qc.shape[0], V))
    for j0 in range(0, d, R32.CH):
        w = min(R32.CH, d - j0)
        Gs = [np.random.default_rng(C.seed_of('G', TAG[m], prep, fmap, k, j0, R32.CH)).standard_normal(
            (Kc.shape[1], R32.CH), dtype=np.float32)[:, :w].astype(np.float64) for k in range(nf)]
        r = np.random.default_rng(C.seed_of('C', TAG[m], prep, fmap, j0, R32.CH))
        Cv = np.where(r.random((R32.V_MAX, R32.CH), dtype=np.float32) < 0.5, -1.0, 1.0)[:V, :w]

        def phi(Y):
            o = np.ones((Y.shape[0], w))
            for G in Gs:
                Z = Y @ G
                o = o * (Z if fmap == 'lin' else np.where(Z >= 0, 1.0, -1.0))
            return o
        S = (phi(Kc[keep]) * Cv[vals[keep]]).sum(0)
        if q == 1:
            Sq = np.sign(S)
            z = Sq == 0
            if z.any():
                Sq[z] = np.random.default_rng(C.seed_of('Q', R32.store_seed(TAG[m], st), d, q, ns, j0)).choice([-1.0, 1.0], size=int(z.sum()))
        else:
            step = C.MAX_STEP[q] * (S.std() or 1.0)
            Lq = 2 ** q
            Sq = (np.clip(np.floor(S / step), -Lq // 2, Lq // 2 - 1) + 0.5) * step
        sc += (phi(Qc) * Sq) @ Cv.T
    return sc.argmax(1)


def _check_task(args):
    m, M, V, rep, b, cfg, pred = args
    st = make('test', M, V, rep)
    p2 = rederive_vsa(m, st, cfg, b)
    return m, M, V, rep, b, float((np.asarray(pred) == p2).mean()), len(p2)


def check(workers=8):
    rng = np.random.default_rng(5252)
    jobs = []
    for m in NEW:
        ch = json.load(open(f'{OUT}/vsa_choice_m{m}.json'))['choice']
        recs = load_test(m)
        for M in M_GRID:
            for V in V_R1:
                reps = sorted({r['rep'] for r in recs if r['M'] == M and r['V'] == V})
                pick = rng.choice(reps, size=max(1, len(reps) // 10), replace=False)
                for r in recs:
                    if r['M'] == M and r['V'] == V and r['rep'] in pick:
                        for b in BPF:
                            jobs.append((m, M, V, r['rep'], b, ch[f'{M}|{V}|{b}']['cfg'], r['pred'][f'VSA|{b}']))
    res = list(pmap(_check_task, jobs, workers))
    agree = {}
    for m, M, V, rep, b, ag, n in res:
        agree.setdefault(m, []).append((ag, n))
    out = {'vsa_rederive': {str(m): {'stores_x_bpf': len(v), 'queries': sum(n for _, n in v),
                                     'agreement': float(sum(a * n for a, n in v) / sum(n for _, n in v)),
                                     'min_store_agreement': float(min(a for a, _ in v))} for m, v in agree.items()}}
    # independent re-scoring: plain loops over saved predictions vs gold, cell accuracies vs the verdict's
    J = json.load(open('results/b52b_verdict.json'))
    mx = 0.0
    for m in NEW:
        recs = load_test(m)
        for c in J['maps'][f'{m}|{m}']['cells']:
            for arm, kk in (('VSA', 'VSA'), ('BEST', 'BEST')):
                ok = tot = 0
                for r in recs:
                    if r['M'] == c['M'] and r['V'] == c['V']:
                        for p_, g_ in zip(r['pred'][f'{arm}|{int(c["bpf"])}'], r['qgold']):
                            ok += int(p_ == g_); tot += 1
                mx = max(mx, abs(100.0 * ok / tot - c[kk]))
    out['rescore_max_abs_diff_points'] = mx
    json.dump(out, open('results/b52b_check.json', 'w'), indent=1)
    print(json.dumps(out, indent=1))


def handread(n=10):
    """10 queries of the R1 cell with the largest between-map range of VSA accuracy, where the best map's VSA is right
    and the worst map's VSA is wrong: alias, stored label, gold value, both answers, alias-label cosine (centred)."""
    J = json.load(open('results/b52b_verdict.json'))
    cells = [[J['maps'][f'{m}|{m}']['cells'][i] for m in NEW] for i in range(len(CELLS))]
    i = max(range(len(CELLS)), key=lambda j: max(c['VSA'] for c in cells[j]) - min(c['VSA'] for c in cells[j]))
    M, V, b = CELLS[i]
    hi = NEW[int(np.argmax([c['VSA'] for c in cells[i]]))]
    lo = NEW[int(np.argmin([c['VSA'] for c in cells[i]]))]
    R = {m: {r['rep']: r for r in load_test(m) if r['M'] == M and r['V'] == V} for m in (hi, lo)}
    _, _, preps = base()
    L = [f'# B52 (b) hand-reading: cell M={M} V={V} bpf={b}; VSA accuracy per map ' +
         ', '.join(f'm{m} {c["VSA"]:.2f}' for m, c in zip(NEW, cells[i])) + f'; queries map {hi} right, map {lo} wrong']
    k = 0
    for rep in sorted(R[hi]):
        st = make('test', M, V, rep)
        Pp = preps['center']
        Kc, Qc = Pp(st['keys']), Pp(st['queries'])
        a, z = R[hi][rep]['pred'][f'VSA|{b}'], R[lo][rep]['pred'][f'VSA|{b}']
        for q in range(len(st['qgold'])):
            if a[q] == st['qgold'][q] and z[q] != st['qgold'][q]:
                e = st['qent'][q]
                cos = float(Kc[e] @ Qc[q] / (np.linalg.norm(Kc[e]) * np.linalg.norm(Qc[q])))
                L.append(f'  rep {rep} alias "{st["qalias"][q]}" -> label "{st["names"][e]}" gold {st["qgold"][q]}; '
                         f'map {hi} says {a[q]}, map {lo} says {z[q]}; alias-label cosine (centred) {cos:.2f}')
                k += 1
                if k >= n:
                    break
        if k >= n:
            break
    open('results/b52b_handread.txt', 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    a = sys.argv[1:]
    if a[0] == 'handread':
        handread(); sys.exit(0)
    W = int(a[a.index('--workers') + 1]) if '--workers' in a else 8
    os.makedirs(OUT, exist_ok=True)
    if a[0] == 'pqfit':
        pqfit([int(x) for x in a[1].split(',')])
    elif a[0] == 'dev':
        dev([int(x) for x in a[1].split(',')], W)
    elif a[0] == 'choose':
        choose([int(x) for x in a[1].split(',')])
    elif a[0] == 'gate0':
        gate0()
    elif a[0] == 'test':
        test([int(x) for x in a[1].split(',')], W)
    elif a[0] == 'verdict':
        verdict()
    elif a[0] == 'check':
        check(W)
