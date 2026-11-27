"""B66 part (C), matched grids (prereg/B66_*_prereg.md): does the store's lead or deficit depend on the size of each
competitor's tuning grid?  The store (B53's final design) searches 72 configurations on dev, LAMc 288, SDMt 504 and
B36's six key-free families 468 together.  Here every family is re-chosen on dev from 72-configuration sub-grids.

Families (registered full grids, each with its own dev rule, unchanged):
  STORE  B53's store, 72 = prep x fmap (lin, sgn, sgn3) x q x stored share; B32's rule (b32_run.choose: highest mean
         dev accuracy, within 0.1 points tied, first by b32_run.vsa_key).  Every draw is its full grid.
  LAMC   B60's LAMc, 288 (b60_run.grid); B60's rule (b60_run.choose_one).
  SDMT   B64's SDMt, 504 (b64_run.GRID); B36's rule as in b64_run.choose.
  KF6    B36's six key-free families IDX (B32's index, 48), SF (176), CM (132), BF (60), FBF (12), SDM (40) = 468;
         B36's per-family rule (IDX: B32's rule with b32_run.idx_key; others b36_run.choose with grid_rank), then B36's
         BEST rule over the families (highest dev accuracy of each family's chosen configuration, exact ties in the
         order IDX, SF, CM, BF, FBF, SDM).
Draws: for F in (STORE, LAMC, SDMT, KF6) and d = 0..19, 72 configurations without replacement from F's full grid,
numpy default_rng(b32_common.seed_of('B66C', F, d)), kept in grid order; the same sub-grid for every cell and encoder.
KF6's draw is from the union of the six grids (union order IDX, SF, CM, BF, FBF, SDM, each in its grid order);
families with no drawn configuration are absent from the BEST rule.  BEST7_72 (draw d) = B36's BEST rule over KF6's
draw-d family choices and SDMT's draw-d choice (SDMT last in the tie order, as b64_run.choose).
Cells: the 12 R1 cells (M 256 / 1,024 x V 2 / 16 x 4, 8, 16 bits per fact) of the 4 language models, ideal accounting.

Test: only the draw-chosen configurations whose test predictions are not saved already (the full-grid choices' are:
STORE / VSAR / the six families / BEST36 in the merged b59_out records, LAMC in b60_out/jobs, SDMT in b64_out/jobs).
For KF6 only the configuration of the family that the draw's BEST rule picks is needed.  Each family's own code runs
them, unchanged: SDMT b64_run.run_job (spec cfgs), LAMC b60_run.run_job (spec cfgs), KF6 b36_run.run_store (select;
B36's full budget list and accountings, as B36's test, because B36's families size some structures by the largest
budget) and b32_run.run_idx, in the encoder's B36 working directory (repo root, b46_out, b49_out/enc_{enc}).

python3 b66_grids.py selftest            # fabricated rows + full-grid reproduction of the committed choices (dev rows)
python3 b66_grids.py draws               # dev only -> b66_out/grids/choices.json, need.json
python3 b66_grids.py plan                # -> b66_out/grids/plan_test.jsonl
python3 b66_grids.py runplan [--workers N] [--kinds SDMT,LAMC,KF6,KF6IDX]   # KF6IDX needs faiss (box b49 env)
python3 b66_grids.py verdict             # -> results/b66_C_verdict.{json,txt}
"""
import gzip
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import b32_common as C  # noqa: E402

RES = os.environ.get('B66_RES', f'{ROOT}/results')
OUT = os.environ.get('B66_OUT', os.path.join(ROOT, 'b66_out'))
GD = os.path.join(OUT, 'grids')
LLM = ['qwen8', 'b46', 'llama', 'mistral']
NAMES = {'qwen8': 'Qwen3-8B', 'b46': 'Granite-3.3-8B', 'llama': 'Llama-3.1-8B', 'mistral': 'Mistral-7B'}
M_GRID, VN, BPF = [256, 1024], [2, 16], [4, 8, 16]
CELLS = [(M, V, b) for M in M_GRID for V in VN for b in BPF]
DEV_REPS = {256: 6, 1024: 3}
TEST_REPS = {256: 20, 1024: 5}
NDRAW, KSUB = 20, 72
FAMILIES = ['STORE', 'LAMC', 'SDMT', 'KF6']
SIX = ['IDX', 'SF', 'CM', 'BF', 'FBF', 'SDM']
SEVEN = SIX + ['SDMT']
PREPS = ['center', 'whiten']
TIE = 0.001  # 0.1 points

# encoder working directories of B32 / B36 (dev rows, committed B36-protocol choices, fit / PQ caches)
WDIR = {'qwen8': ROOT, 'b46': os.path.join(ROOT, 'b46_out'),
        'llama': os.path.join(ROOT, 'b49_out', 'enc_llama'), 'mistral': os.path.join(ROOT, 'b49_out', 'enc_mistral')}
B36_CHOICE = {'qwen8': f'{ROOT}/results/b36_choice_qwen8.json', 'b46': f'{ROOT}/results/b46_choice_b36.json',
              'llama': f'{ROOT}/b49_out/enc_llama/results/b36_choice_llama.json',
              'mistral': f'{ROOT}/b49_out/enc_mistral/results/b36_choice_mistral.json'}


def idx_dev_file(enc):
    return os.path.join(WDIR[enc], 'b32_out', f'dev_{enc}_idx.jsonl')


def six_dev_file(enc):
    return os.path.join(WDIR[enc], 'b36_out', f'dev_{enc}.jsonl')


# ----------------------------------------------------------------------------------------------- grids

def store_grid():
    import b32_run as R
    g = [(p, f, q, s) for p in R.GRID['prep'] for f in R.GRID['fmap'] for q in R.GRID['q'] for s in R.GRID['f']]
    return sorted(g, key=R.vsa_key)


def lamc_grid(V):
    import b60_run as R60
    return [tuple(c) for c in R60.grid(V)]


def sdmt_grid():
    import b64_run as R64
    return [tuple(c) for c in R64.GRID]


def idx_grid():
    import b32_run as R
    g = [(p, f, s) for p in R.GRID['prep'] for f in R.GRID['family'] for s in R.GRID['f']]
    return sorted(g, key=R.idx_key)


def fam6_grid(fam, V):
    """B36's grid of one family in its grid order (prep first, then b36_run.GRID order); cfg = (prep,) + cfg."""
    if fam == 'IDX':
        return idx_grid()
    import b36_run as R36
    return [(p,) + tuple(c) for p in R36.PREPS for c in R36.fam_grid(fam, V)]


def kf6_grid(V):
    return [(f, c) for f in SIX for c in fam6_grid(f, V)]


def full_grid(F, V):
    return {'STORE': store_grid, 'LAMC': lambda: lamc_grid(V), 'SDMT': sdmt_grid, 'KF6': lambda: kf6_grid(V)}[F]()


def draw_idx(F, d, n):
    if n <= KSUB:
        return list(range(n))
    rng = np.random.default_rng(C.seed_of('B66C', F, d))
    return sorted(int(i) for i in rng.choice(n, KSUB, replace=False))


def sub_grid(F, d, V):
    g = full_grid(F, V)
    return [g[i] for i in draw_idx(F, d, len(g))]


# ----------------------------------------------------------------------------------------------- dev rows

def load_dev(enc):
    """{family: {(M, V, b): {cfg: mean dev accuracy}}}, means computed as each family's own choose() computes them
    (B32 / B36: mean over the values in file order; B60 / B64: over sorted store indices)."""
    out = {}
    # STORE: B53's dev rows, files in b53_run.PARTS order (B32's rule reads them in that order)
    acc = {}
    for p in ('center', 'whiten'):
        for f in ('lin', 'sgn', 'sgn3'):
            for l in open(f'{ROOT}/b53_out/{enc}/dev_{enc}_{p}_{f}.jsonl'):
                r = json.loads(l)
                if r['V'] not in VN:
                    continue
                for x in r['rows']:
                    if x['acct'] == 'ideal' and x['bpf'] in BPF:
                        acc.setdefault((r['M'], r['V'], x['bpf']), {}).setdefault(tuple(x['cfg']), {})[r['rep']] = x['acc']
    out['STORE'] = _means(acc, 'insertion')
    # LAMC: B60's dev jobs (map 0)
    acc = {}
    jd = f'{ROOT}/b60_out/jobs'
    for fn in sorted(os.listdir(jd)):
        if not fn.startswith(f'{enc}_m0_dev_'):
            continue
        r = json.load(gzip.open(os.path.join(jd, fn), 'rt'))
        s = r['spec']
        if s['V'] not in VN:
            continue
        for x in r['rows']:
            if x['acct'] == 'ideal' and x['bpf'] in BPF:
                acc.setdefault((s['M'], s['V'], x['bpf']), {}).setdefault(tuple(x['cfg']), {})[s['rep']] = x['acc']
    out['LAMC'] = _means(acc, 'sorted')
    # SDMT: B64's dev jobs
    acc = {}
    jd = f'{ROOT}/b64_out/jobs'
    for fn in sorted(os.listdir(jd)):
        if not fn.startswith(f'{enc}_dev_'):
            continue
        r = json.load(gzip.open(os.path.join(jd, fn), 'rt'))
        s = r['spec']
        for x in r['rows']:
            if x['acct'] == 'ideal' and x['bpf'] in BPF:
                acc.setdefault((s['M'], s['V'], x['bpf']), {}).setdefault(tuple(x['cfg']), {})[s['rep']] = x['acc']
    out['SDMT'] = _means(acc, 'sorted')
    # KF6: IDX from B32's index dev file, SF..SDM from B36's dev file
    acc = {}
    for l in open(idx_dev_file(enc)):
        r = json.loads(l)
        if r['V'] not in VN:
            continue
        for x in r['rows']:
            if x['arm'] == 'IDX' and x['acct'] == 'ideal' and x['bpf'] in BPF:
                acc.setdefault((r['M'], r['V'], x['bpf']), {}).setdefault(('IDX', tuple(x['cfg'])), {})[r['rep']] = x['acc']
    for l in open(six_dev_file(enc)):
        r = json.loads(l)
        if r['V'] not in VN:
            continue
        for x in r['rows']:
            if x['acct'] == 'ideal' and x['bpf'] in BPF:
                acc.setdefault((r['M'], r['V'], x['bpf']), {}).setdefault((x['fam'], tuple(x['cfg'])), {})[r['rep']] = x['acc']
    out['KF6'] = _means(acc, 'insertion')
    return out


def _means(acc, how):
    out = {}
    for cell, d in acc.items():
        M = cell[0]
        out[cell] = {}
        for cfg, reps in d.items():
            if len(reps) != DEV_REPS[M]:
                raise SystemExit(f'incomplete dev rows {cell} {cfg}: {len(reps)}')
            vals = [reps[k] for k in sorted(reps)] if how == 'sorted' else list(reps.values())
            out[cell][cfg] = float(np.mean(vals))
    return out


# ----------------------------------------------------------------------------------------------- dev rules

def pick(devc, cfgs):
    """Highest mean dev accuracy; within 0.1 points tied, first in the given (grid) order. -> (cfg, dev_acc, top)."""
    lst = []
    for c in cfgs:
        if c not in devc:
            raise SystemExit(f'missing dev row {c}')
        lst.append((devc[c], c))
    top = max(a for a, _ in lst)
    for a, c in lst:
        if a >= top - TIE:
            return c, a, top


def pick_kf6(devc, cfgs):
    """B36's per-family rule on the given (fam, cfg) list, then B36's BEST rule over the present families."""
    by = {}
    for f in SIX:
        fc = [c for (g, c) in cfgs if g == f]
        if fc:
            c, a, _ = pick({c: devc[(f, c)] for c in fc}, fc)
            by[f] = (c, a)
    cands = [(by[f][1], -SIX.index(f), f) for f in SIX if f in by]
    a, _, f = max(cands)
    return {'fam': f, 'cfg': list(by[f][0]), 'dev_acc': a, 'by_fam': {g: [list(v[0]), v[1]] for g, v in by.items()}}


def best7(kf, sd):
    """B36's BEST rule over KF6's family choices (kf['by_fam']) and SDMT's choice sd = (cfg, dev_acc)."""
    dbf = {g: v[1] for g, v in kf['by_fam'].items()}
    dbf['SDMT'] = sd[1]
    a, _, f = max((dbf[g], -SEVEN.index(g), g) for g in SEVEN if g in dbf)
    return f


def committed(enc):
    """The committed full-grid choices (dev only): {(M, V, b): {family: cfg}} + six-family choices and BEST36 / BEST7."""
    ch53 = json.load(open(f'{RES}/b53_choice_{enc}.json'))['choice']
    ch60 = json.load(open(f'{RES}/b60_choice_{enc}.json'))['choice']
    ch64 = json.load(open(f'{RES}/b64_choice_{enc}.json'))['choice']
    ch36 = json.load(open(B36_CHOICE[enc]))['choice']
    ch49 = json.load(open(f'{RES}/b49_choice_{enc}.json'))['choice']
    out = {}
    for (M, V, b) in CELLS:
        k = f'{M}|{V}|{b}'
        out[M, V, b] = {'STORE': tuple(ch53[f'{k}|ideal|VSA']['cfg']), 'LAMC': tuple(ch60[f'{k}|ideal|LAMC']['cfg']),
                        'SDMT': tuple(ch64[f'{k}|ideal|SDMT']['cfg']),
                        'six': {f: tuple(ch36[f'{k}|ideal|{f}']['cfg']) for f in SIX},
                        'BEST36': ch49[f'{k}|ideal|BEST36']['fam'], 'BEST7': ch64[f'{k}|ideal|BEST7']['fam']}
    return out


def full_choices(enc, dev):
    """Full-grid choices recomputed from the dev rows; asserted equal to the committed ones."""
    cm = committed(enc)
    res = {}
    for (M, V, b) in CELLS:
        cell = (M, V, b)
        r = {}
        for F in ('STORE', 'LAMC', 'SDMT'):
            c, a, _ = pick(dev[F][cell], full_grid(F, V))
            assert c == cm[cell][F], (enc, cell, F, c, cm[cell][F])
            r[F] = (c, a)
        kf = pick_kf6(dev['KF6'][cell], full_grid('KF6', V))
        for f in SIX:
            assert tuple(kf['by_fam'][f][0]) == cm[cell]['six'][f], (enc, cell, f, kf['by_fam'][f], cm[cell]['six'][f])
        assert kf['fam'] == cm[cell]['BEST36'], (enc, cell, kf['fam'], cm[cell]['BEST36'])
        b7 = best7(kf, r['SDMT'])
        assert b7 == cm[cell]['BEST7'], (enc, cell, b7, cm[cell]['BEST7'])
        r['KF6'] = kf
        r['BEST7'] = b7
        res[cell] = r
    return res


# ----------------------------------------------------------------------------------------------- draws (dev only)

def ckey(c):
    return json.dumps(list(c))


def draws():
    os.makedirs(GD, exist_ok=True)
    J = {'n_draw': NDRAW, 'k': KSUB, 'grid_sizes': {F: len(full_grid(F, 2)) for F in FAMILIES},
         'subgrids': {F: [draw_idx(F, d, len(full_grid(F, 2))) for d in range(NDRAW)] for F in FAMILIES}, 'enc': {}}
    need = {}
    for enc in LLM:
        dev = load_dev(enc)
        full = full_choices(enc, dev)
        E = {}
        for cell in CELLS:
            M, V, b = cell
            fc = full[cell]
            ent = {'full': {'STORE': list(fc['STORE'][0]), 'LAMC': list(fc['LAMC'][0]), 'SDMT': list(fc['SDMT'][0]),
                            'KF6': fc['KF6'], 'BEST7': fc['BEST7']},
                   'draws': []}
            for d in range(NDRAW):
                x = {}
                for F in ('STORE', 'LAMC', 'SDMT'):
                    c, a, _ = pick(dev[F][cell], sub_grid(F, d, V))
                    x[F] = {'cfg': list(c), 'dev_acc': a}
                    if F != 'STORE' and c != fc[F][0]:
                        need.setdefault(F, set()).add((enc, M, V, b, ckey(c)))
                    assert F != 'STORE' or c == fc[F][0]  # STORE: every draw is its full grid
                kf = pick_kf6(dev['KF6'][cell], sub_grid('KF6', d, V))
                x['KF6'] = kf
                if list(kf['cfg']) != list(fc['KF6']['by_fam'][kf['fam']][0]):
                    need.setdefault('KF6', set()).add((enc, M, V, b, kf['fam'] + '|' + ckey(kf['cfg'])))
                x['BEST7'] = best7(kf, (tuple(x['SDMT']['cfg']), x['SDMT']['dev_acc']))
                ent['draws'].append(x)
            E[f'{M}|{V}|{b}'] = ent
        J['enc'][enc] = E
        print('draws', enc, flush=True)
    json.dump(J, open(os.path.join(GD, 'choices.json'), 'w'), indent=1)
    N = {F: sorted(list(v)) for F, v in need.items()}
    cnt = {F: len(v) for F, v in N.items()}
    jobs = {}
    for F, v in N.items():
        groups = {(e, M, V) for (e, M, V, b, c) in v}
        jobs[F] = sum(TEST_REPS[M] for (_, M, _) in groups)
    json.dump({'need': N, 'counts': cnt, 'store_runs_upper': jobs}, open(os.path.join(GD, 'need.json'), 'w'), indent=1)
    print('need (distinct (enc, M, V, bpf, cfg)):', cnt, ' test-store runs:', jobs)


# ----------------------------------------------------------------------------------------------- test jobs

def plan():
    N = json.load(open(os.path.join(GD, 'need.json')))['need']
    fo = open(os.path.join(GD, 'plan_test.jsonl'), 'w')
    n = 0
    for F in ('SDMT', 'LAMC', 'KF6'):
        groups = {}
        for (e, M, V, b, c) in N.get(F, []):
            if F == 'KF6':
                fam, cs = c.split('|', 1)
                groups.setdefault((e, M, V), []).append([fam, json.loads(cs), b])
            elif F == 'LAMC':
                cfg = json.loads(c)
                groups.setdefault((e, M, V, cfg[2]), []).append([cfg, b, 'ideal'])
            else:
                groups.setdefault((e, M, V), []).append([json.loads(c), b, 'ideal'])
        for key in sorted(groups):
            e, M, V = key[:3]
            for rep in range(TEST_REPS[M]):
                if F == 'SDMT':
                    spec = {'enc': e, 'split': 'test', 'M': M, 'V': V, 'rep': rep, 'cfgs': groups[key]}
                elif F == 'LAMC':
                    spec = {'enc': e, 'map': 0, 'split': 'test', 'M': M, 'V': V, 'rep': rep, 'prep': key[3],
                            'cfgs': groups[key]}
                else:  # B36's five families run anywhere; the index (B32 code) needs faiss (the box's b49 env)
                    for kind, nd in (('KF6', [x for x in groups[key] if x[0] != 'IDX']),
                                     ('KF6IDX', [x for x in groups[key] if x[0] == 'IDX'])):
                        if nd:
                            spec = {'enc': e, 'split': 'test', 'M': M, 'V': V, 'rep': rep, 'need': nd, 'kind': kind}
                            fo.write(json.dumps({'kind': kind, 'spec': spec}) + '\n')
                            n += 1
                    continue
                fo.write(json.dumps({'kind': F, 'spec': spec}) + '\n')
                n += 1
    fo.close()
    print(n, 'test jobs ->', os.path.join(GD, 'plan_test.jsonl'))


_KF = {}


def kf6_env(enc):
    """B32 / B36 code in the encoder's B36 working directory (as b46_run / b49_run register it); cached per worker."""
    if enc in _KF:
        os.chdir(WDIR[enc])
        return _KF[enc]
    _KF.clear()
    os.chdir(WDIR[enc])
    os.environ['B32_FIT_CACHE'] = 'b32_out/fitcache'
    import b32_run as R32
    import b36_run as R36
    if enc != 'qwen8':
        fn = 'b46_encoder.json' if enc == 'b46' else 'b49_encoder.json'
        tag = json.load(open(fn))['file_tag']
        R32.ENCS[enc] = tag
        R36.ENCS[enc] = tag
    S = R36.Setup(enc)
    pool, preps = S.pool, S.preps
    Tr = pool.train_matrix()
    _KF[enc] = {'R32': R32, 'R36': R36, 'S': S, 'Tr': Tr, 'idx': {}}
    return _KF[enc]


def kf6_job(spec):
    enc, M, V, rep = spec['enc'], spec['M'], spec['V'], spec['rep']
    fn = os.path.join(GD, 'kf6_jobs', f'{enc}_test_M{M}_V{V}_r{rep}_{spec["kind"]}.json.gz')
    if os.path.exists(fn):
        return fn
    cwd = os.getcwd()
    t0 = time.time()
    try:
        E = kf6_env(enc)
        R32, R36, S = E['R32'], E['R36'], E['S']
        st = C.make_store(S.pool, 'test', M, V, rep)
        rec = {'spec': spec, 'qgold': st['qgold'].tolist(), 'pred': {}}
        sel = {}
        for fam, cfg, b in spec['need']:
            if fam != 'IDX':
                sel.setdefault(fam, set()).add(tuple(cfg))
        if sel:
            res = R36.run_store(S, st, R36.FAMS, R36.BPF, R36.ACCTS, select=sel)
        for fam, cfg, b in spec['need']:
            if fam == 'IDX':
                prep, family, f = cfg
                if (prep, family) not in E['idx']:
                    E['idx'][prep, family] = C.IndexFamily(family, S.preps[prep](E['Tr']), cache_tag=f'{enc}_{prep}')
                p, _ = R32.run_idx(st, S.preps, E['idx'], prep, family, b, 'ideal', f)
            else:
                p, _ = res[fam, tuple(cfg), b, 'ideal']
            rec['pred'][f'{fam}|{ckey(cfg)}|{b}'] = np.asarray(p).astype(int).tolist()
    finally:
        os.chdir(cwd)
    rec['sec'] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    with gzip.open(fn + '.tmp', 'wt') as fo:
        json.dump(rec, fo)
    os.replace(fn + '.tmp', fn)
    print(f'kf6 {enc} M={M} V={V} rep={rep} {rec["sec"]}s {len(spec["need"])} preds', flush=True)
    return fn


def _run(job):
    try:
        k, spec = job['kind'], job['spec']
        if k == 'SDMT':
            import b64_run as R64
            R64.OUT = os.path.join(GD, 'sdmt')
            return R64.run_job(spec)
        if k == 'LAMC':
            import b60_run as R60
            R60.OUT = os.path.join(GD, 'lamc')
            return R60.run_job(spec)
        return kf6_job(spec)
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        return f'FAILED {job["kind"]} {json.dumps(job["spec"])[:200]} {e}'


def runplan(workers, kinds):
    jobs = [json.loads(l) for l in open(os.path.join(GD, 'plan_test.jsonl'))]
    jobs = [j for j in jobs if j['kind'] in kinds]
    jobs.sort(key=lambda j: (j['kind'], j['spec']['enc'], -j['spec']['M']))
    print(len(jobs), 'jobs', flush=True)
    if workers <= 1:
        for j in jobs:
            r = _run(j)
            if str(r).startswith('FAILED'):
                print(r, flush=True)
        return
    import multiprocessing as mp
    with mp.get_context('fork').Pool(workers) as pl:
        for r in pl.imap_unordered(_run, jobs, chunksize=4):
            if str(r).startswith('FAILED'):
                print(r, flush=True)


# ----------------------------------------------------------------------------------------------- verdict

def load_test(enc):
    """Merged test records (b64_verdict.load: VSA, VSAR, the six, BEST36, SDMT, BEST7) + LAMC (b60_verdict.add_lamc)."""
    import b60_verdict as V60
    import b64_verdict as V64
    data, _ = V64.load(enc)
    ch60 = json.load(open(f'{RES}/b60_choice_{enc}.json'))['choice']
    for (M, V) in data:
        assert V60.add_lamc(data[M, V], enc, M, V, ch60, 0, bpfs=BPF), ('LAMc test jobs missing', enc, M, V)
    return data


def job_preds(enc, M, V):
    """Draw-chosen configurations' test predictions: {(kind, cfgkey, b, rep): pred}."""
    out = {}
    for rep in range(TEST_REPS[M]):
        fn = os.path.join(GD, 'sdmt', 'jobs', f'{enc}_test_M{M}_V{V}_r{rep}.json.gz')
        if os.path.exists(fn):
            for x in json.load(gzip.open(fn, 'rt'))['rows']:
                if x['acct'] == 'ideal':
                    out['SDMT', ckey(x['cfg']), x['bpf'], rep] = x['pred']
        for prep in PREPS:
            fn = os.path.join(GD, 'lamc', 'jobs', f'{enc}_m0_test_M{M}_V{V}_r{rep}_{prep}.json.gz')
            if os.path.exists(fn):
                for x in json.load(gzip.open(fn, 'rt'))['rows']:
                    if x['acct'] == 'ideal':
                        out['LAMC', ckey(x['cfg']), x['bpf'], rep] = x['pred']
        for kind in ('KF6', 'KF6IDX'):
            fn = os.path.join(GD, 'kf6_jobs', f'{enc}_test_M{M}_V{V}_r{rep}_{kind}.json.gz')
            if os.path.exists(fn):
                for k, p in json.load(gzip.open(fn, 'rt'))['pred'].items():
                    fam, rest = k.split('|', 1)
                    cs, b = rest.rsplit('|', 1)
                    out['KF6', fam + '|' + cs, int(b), rep] = p
    return out


def cell_acc(recs, getp):
    """Pooled accuracy over the cell's test stores; None if a needed prediction is missing (job not run / cut)."""
    ok = n = 0
    for r in recs:
        try:
            p = np.asarray(getp(r))
        except KeyError:
            return None
        g = np.asarray(r['qgold'])
        ok += int((p == g).sum())
        n += len(g)
    return ok / n


def verdict():
    J = json.load(open(os.path.join(GD, 'choices.json')))
    out = {'families': {}, 'contrasts': {}, 'cells': {}}
    for enc in LLM:
        data = load_test(enc)
        acc = {}  # (arm, d or 'full', cell) -> accuracy
        for (M, V, b) in CELLS:
            recs = data[M, V]
            jp = job_preds(enc, M, V)
            ent = J['enc'][enc][f'{M}|{V}|{b}']
            for arm in ('VSA', 'VSAR', 'LAMC', 'SDMT', 'BEST36', 'BEST7'):
                acc[arm, 'full', (M, V, b)] = cell_acc(recs, lambda r, a=arm: r['pred'][f'{a}|{b}|ideal'])
            fulls = {'LAMC': ent['full']['LAMC'], 'SDMT': ent['full']['SDMT']}
            for d, x in enumerate(ent['draws']):
                for F in ('LAMC', 'SDMT'):
                    if x[F]['cfg'] == fulls[F]:
                        acc[F, d, (M, V, b)] = acc[F, 'full', (M, V, b)]
                    else:
                        acc[F, d, (M, V, b)] = cell_acc(recs, lambda r, F=F, c=x[F]['cfg']: jp[F, ckey(c), b, r['rep']])
                kf = x['KF6']
                f = kf['fam']
                if kf['cfg'] == [*ent['full']['KF6']['by_fam'][f][0]]:
                    getk = (lambda r, f=f: r['pred'][f'{f}|{b}|ideal'])
                else:
                    getk = (lambda r, f=f, c=kf['cfg']: jp['KF6', f + '|' + ckey(c), b, r['rep']])
                acc['KF6', d, (M, V, b)] = cell_acc(recs, getk)
                acc['BEST7', d, (M, V, b)] = acc['SDMT', d, (M, V, b)] if x['BEST7'] == 'SDMT' else acc['KF6', d, (M, V, b)]
                acc['BEST7fam', d, (M, V, b)] = x['BEST7']
            acc['KF6', 'full', (M, V, b)] = acc['BEST36', 'full', (M, V, b)]
            out['cells'][f'{enc}|{M}|{V}|{b}'] = {a: acc[a, 'full', (M, V, b)] for a in
                                                   ('VSA', 'VSAR', 'LAMC', 'SDMT', 'BEST36', 'BEST7')}

        def reg(arm, d):
            v = [acc[arm, d, c] for c in CELLS]
            return None if any(x is None for x in v) else float(np.mean(v))
        for F, arm in (('STORE', 'VSA'), ('LAMC', 'LAMC'), ('SDMT', 'SDMT'), ('KF6', 'KF6'), ('BEST7', 'BEST7')):
            full = reg(arm if F != 'KF6' else 'BEST36', 'full')
            dr = [full] * NDRAW if F == 'STORE' else [reg(arm, d) for d in range(NDRAW)]
            out['families'][f'{enc}|{F}'] = {'full': full, 'draws': dr}
        out['families'][f'{enc}|BEST7']['sdmt_cells'] = [sum(acc['BEST7fam', d, c] == 'SDMT' for c in CELLS)
                                                         for d in range(NDRAW)]
        for name, x, y in (('STORE-LAMC72', 'VSA', 'LAMC'), ('VSAR-SDMT72', 'VSAR', 'SDMT'),
                           ('VSA-SDMT72', 'VSA', 'SDMT'), ('VSAR-BEST7_72', 'VSAR', 'BEST7'),
                           ('VSA-BEST7_72', 'VSA', 'BEST7')):
            full = reg(x, 'full') - reg(y, 'full')
            dr = [None if reg(y, d) is None else reg(x, 'full') - reg(y, d) for d in range(NDRAW)]
            out['contrasts'][f'{enc}|{name}'] = {'full': full, 'draws': dr}
        print('verdict', enc, flush=True)
    json.dump(out, open(f'{RES}/b66_C_verdict.json', 'w'), indent=1)

    def q(v):
        v = 100 * np.asarray([x for x in v if x is not None])
        if not len(v):
            return 'no complete draw'
        return (f'{v.min():6.2f} {np.quantile(v, .25):6.2f} {np.median(v):6.2f} {np.quantile(v, .75):6.2f} '
                f'{v.max():6.2f} {v.mean():6.2f} (n={len(v)})')
    L = ['# B66 part C: matched (72-configuration) tuning grids. R1 test accuracy (points; mean over the 12 R1 cells of',
         '# the cell accuracy), ideal accounting, 4 language models. full = the registered full-grid dev choice; draws =',
         f'# {NDRAW} random 72-configuration sub-grids per family, each family\'s own dev rule within the draw.',
         '# Columns of the draw distribution: min, 25%, median, 75%, max, mean. STORE has 72 configurations, so every',
         '# draw is its full grid. KF6 = best of B36\'s six key-free families; BEST7 = KF6 + SDMT.', '',
         '## Families: R1 test accuracy', '| encoder | family | full grid | draws: min 25% median 75% max mean |',
         '| --- | --- | --- | --- |']
    for enc in LLM:
        for F in ('STORE', 'LAMC', 'SDMT', 'KF6', 'BEST7'):
            r = out['families'][f'{enc}|{F}']
            L.append(f'| {NAMES[enc]} | {F} | {100 * r["full"]:6.2f} | {q(r["draws"])} |')
    L += ['', '## Contrasts with the competitor at a 72-configuration grid (store arm at its registered grid)',
          '| encoder | contrast | full grid | draws: min 25% median 75% max mean | draws with the full grid\'s sign |',
          '| --- | --- | --- | --- | --- |']
    for enc in LLM:
        for name in ('STORE-LAMC72', 'VSAR-SDMT72', 'VSA-SDMT72', 'VSAR-BEST7_72', 'VSA-BEST7_72'):
            r = out['contrasts'][f'{enc}|{name}']
            ok = [x for x in r['draws'] if x is not None]
            same = sum(np.sign(x) == np.sign(r['full']) for x in ok)
            L.append(f'| {NAMES[enc]} | {name} | {100 * r["full"]:+6.2f} | {q(r["draws"])} | {same} of {len(ok)} |')
    L += ['', '## BEST7_72: number of R1 cells (of 12) in which the draw\'s BEST rule picks SDMT, per draw']
    for enc in LLM:
        L.append(f'  {NAMES[enc]}: {out["families"][f"{enc}|BEST7"]["sdmt_cells"]}')
    L += ['', 'Draws whose region value is missing (a needed test job not run, e.g. a registered cut) are left out; n',
          'gives the number of complete draws.']
    open(f'{RES}/b66_C_verdict.txt', 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))


# ----------------------------------------------------------------------------------------------- selftest

def selftest(real=True):
    rng = np.random.default_rng(1)
    # 1. draws: 72 distinct, inside the grid, deterministic, grid order; STORE = its full grid
    for F in FAMILIES:
        n = len(full_grid(F, 2))
        for d in range(NDRAW):
            ix = draw_idx(F, d, n)
            assert ix == draw_idx(F, d, n) and len(set(ix)) == min(n, KSUB) and ix == sorted(ix)
            assert 0 <= ix[0] and ix[-1] < n
        if F == 'STORE':
            assert n == KSUB and sub_grid(F, 0, 2) == full_grid(F, 2)
    sizes = {F: len(full_grid(F, V)) for F in FAMILIES for V in VN}
    assert len(full_grid('LAMC', 2)) == 288 and len(full_grid('SDMT', 2)) == 504 and len(full_grid('KF6', 2)) == 468
    assert len(full_grid('KF6', 16)) == 468 and len(full_grid('STORE', 2)) == 72
    print('draws ok; grid sizes', {F: len(full_grid(F, 2)) for F in FAMILIES}, sizes and '')
    # 2. fabricated dev rows: pick on the full grid = a brute-force argmax with the 0.1-point tie rule
    for F in ('LAMC', 'SDMT', 'STORE'):
        g = full_grid(F, 16)
        for t in range(50):
            dv = {c: float(rng.integers(0, 30)) / 100 for c in g}  # coarse values: many ties
            c, a, top = pick(dv, g)
            first = next(x for x in g if dv[x] >= max(dv.values()) - TIE)
            assert c == first and a == dv[c] and top == max(dv.values())
            sub = sub_grid(F, t % NDRAW, 16)
            cs, _, _ = pick(dv, sub)
            assert cs in sub and dv[cs] >= max(dv[x] for x in sub) - TIE
    # 3. KF6 two-stage on fabricated rows = a direct per-family / BEST implementation; draw of the full grid = full
    g = full_grid('KF6', 2)
    for t in range(50):
        dv = {c: float(rng.integers(0, 40)) / 100 for c in g}
        kf = pick_kf6(dv, g)
        best = {}
        for f in SIX:
            fc = [c for (h, c) in g if h == f]
            m = max(dv[(f, c)] for c in fc)
            cc = next(c for c in fc if dv[(f, c)] >= m - TIE)
            best[f] = (dv[(f, cc)], cc)
        fam = max(SIX, key=lambda f: (best[f][0], -SIX.index(f)))
        assert kf['fam'] == fam and tuple(kf['cfg']) == best[fam][1], (kf, fam, best[fam])
        sub = sub_grid('KF6', t % NDRAW, 2)
        ks = pick_kf6(dv, sub)
        assert (ks['fam'], tuple(ks['cfg'])) in sub
    print('fabricated-row rules ok (pick, sub-grid pick, KF6 two-stage, BEST)')
    # 4. full-grid reproduction of the committed choices from the dev rows (dev only)
    if real:
        for enc in LLM:
            t0 = time.time()
            dev = load_dev(enc)
            for F in FAMILIES:
                for cell in CELLS:
                    n_exp = len(full_grid(F, cell[1]))
                    assert len(dev[F][cell]) == n_exp, (enc, F, cell, len(dev[F][cell]), n_exp)
            full_choices(enc, dev)
            print(f'{enc}: dev rows complete (72 / 288 / 504 / 468 per cell); full-grid choices reproduce B53 VSA, '
                  f'B60 LAMC, B64 SDMT, B36 six families, B49 BEST36, B64 BEST7 in 12 / 12 R1 cells '
                  f'({time.time() - t0:.0f}s)', flush=True)
    print('selftest ok')


def main():
    a = sys.argv[1:]
    if a[0] == 'selftest':
        selftest(real='--fake-only' not in a)
    elif a[0] == 'draws':
        draws()
    elif a[0] == 'plan':
        plan()
    elif a[0] == 'runplan':
        wk = int(a[a.index('--workers') + 1]) if '--workers' in a else 1
        kinds = a[a.index('--kinds') + 1].split(',') if '--kinds' in a else ['SDMT', 'LAMC', 'KF6', 'KF6IDX']
        runplan(wk, kinds)
    elif a[0] == 'verdict':
        verdict()
    else:
        raise SystemExit(a[0])


if __name__ == '__main__':
    main()
