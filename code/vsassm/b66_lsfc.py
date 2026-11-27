"""B66 part (B) (prereg/B66_*_prereg.md): LSfc = least squares with LAMc's corrected options (B60) fitted on the STORE's
own features at the same bits; separates the write rule (Hebbian vs least squares) from the features.

LSfc is b60_run.run_lamc, called UNCHANGED (ridge W = F^T (F F^T + lam I)^-1 T, lam = alpha x mean diag(F F^T); all M
facts; targets 'V-1' / 'V'; scales 'col' / 'flat'; quantizer per scale group; ideal bits c p H_q + 32 x groups, real q
instead of H_q; p = the largest that fits, maximal), with a feature object whose feats() return the store's features:
  R (kind 'R', run_lamc feat 'rp')   the final-design store's random map of the cell: B53's dev-chosen (prep, fmap) for
           (M, V, bpf, acct) (results/b53_choice_{enc}.json, B53 acct 'raw' = 'real'); phi = b32_run.Feat(enc, prep,
           fmap, n, 'cpu').phi on the store's own projection blocks (same seeds; B53's Feat53 has the same projections,
           only its value codes differ): lin = Y G, sgn = sign(Y G) (>= 0 -> +1), sgn3 = product of 3 signs.
           pmax = 65,536 (LAMc's rp cap).
  P (kind 'P', run_lamc feat 'pca')  the store's principal map, corrected basis (VSAPc, B60): VSAPc's dev-chosen (prep,
           pfmap) for the cell (results/b60_choicepc_{enc}.json, '|VSA' keys); phi = (Y - mu) @ U[:, j0:j0 + w] (plin)
           or its signs (psgn, >= 0 -> +1); center: B57's basis (b49_common.LamFeat of the centre-prepped training
           codes: U, mu); whiten: b60_vsapc.CorrectedBasis (U = U0, the raw codes' centred training covariance's
           eigenvectors, descending; mu = 0).  pmax = n.
Grid per kind (order = tie order): scale (col, flat) > target (V-1, V) > alpha (1e-4, 1e-2, 0.1, 1, 10, 100) > q (1, 2,
4): 72 configurations.  Arms: LSFR, LSFP, LSFX = per cell the dev-best of LSFR's and LSFP's choices (within 0.1 points
tied -> LSFR).  Seeds: run_lamc's seed = C.seed_of('b66lsf', enc, kind, split, M, V, rep).
R1 budgets (4, 8, 16 bits per fact), V in {2, 16}, M in {256, 1024}, both accountings, B49 / B60's stores.

python3 b66_lsfc.py selftest
python3 b66_lsfc.py timing ENC M V                 # one TRAINING-split store, every configuration of both kinds
python3 b66_lsfc.py plan dev|test [--encs ..]      # -> b66_out/lsf/plan_{dev|test}.jsonl
python3 b66_lsfc.py runplan PLAN --workers N
python3 b66_lsfc.py choose ENC                     # -> results/b66_lsf_choice_{enc}.json
python3 b66_lsfc.py verdict [--encs ..]            # -> results/b66_B_verdict.{json,txt}
python3 b66_lsfc.py check [--encs ..]              # -> results/b66_B_check.txt
"""
import os

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import gzip  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import b32_common as C  # noqa: E402
import b32_run as R  # noqa: E402
import b49_common as B  # noqa: E402
import b60_run as L60  # noqa: E402

OUTROOT = os.environ.get('B66_OUT', os.path.join(ROOT, 'b66_out'))
OUT = os.path.join(OUTROOT, 'lsf')
RES = os.environ.get('B66_RES', os.path.join(ROOT, 'results'))  # B66 outputs
INRES = os.path.join(ROOT, 'results')  # earlier experiments' choice files (inputs)
DATA = L60.DATA
PANEL = L60.PANEL
LLM, SENT = PANEL[:4], PANEL[4:]
NAMES = {'qwen8': 'Qwen3-8B', 'b46': 'Granite-3.3-8B', 'llama': 'Llama-3.1-8B', 'mistral': 'Mistral-7B',
         'e5': 'e5-base-v2', 'bge': 'bge-small', 'minilm': 'MiniLM-L6'}
M_GRID, VN, BPF, ACCTS = [256, 1024], [2, 16], [4, 8, 16], ['ideal', 'real']
A3 = {'ideal': 'ideal', 'real': 'raw'}
DEV_REPS, TEST_REPS = L60.DEV_REPS, L60.TEST_REPS
PREPS = ['center', 'whiten']
SCALES, TARGETS, ALPHAS, QS = ['col', 'flat'], ['V-1', 'V'], [1e-4, 1e-2, 0.1, 1.0, 10.0, 100.0], [1, 2, 4]
GRID = [(s, t, a, q) for s in SCALES for t in TARGETS for a in ALPHAS for q in QS]  # 72, tie order
KINDS = {'R': 'rp', 'P': 'pca'}
ARM = {'R': 'LSFR', 'P': 'LSFP'}
REG = {'R1': [(M, V, b) for M in M_GRID for V in VN for b in BPF]}
for _V in VN:
    REG[f'R1-V{_V}'] = [(M, _V, b) for M in M_GRID for b in BPF]
PAIRS = [('STORE', 'LSFR'), ('LSFR', 'LAMC'), ('VSAPC', 'LSFP'), ('LSFP', 'LAMC'), ('STORE', 'LSFX'), ('LSFX', 'LAMC')]
CELL_ARMS = ['STORE', 'LSFR', 'LSFP', 'LSFX', 'LAMC', 'VSAPC', 'VSAPCX']
NB = 10000


def cfg5(g, kind):
    return (g[0], g[1], KINDS[kind], g[2], g[3])


# ----------------------------------------------------------------------------------------------- setup and features

_SET = {}


def setup(enc):
    if enc not in _SET:
        cwd = os.getcwd()
        os.chdir(DATA)
        pool = C.Pool(L60.ENC_TAG[enc])
        os.chdir(cwd)
        Tr = pool.train_matrix()
        preps = {p: C.Prep(p, Tr) for p in PREPS}
        _SET.clear()
        _SET[enc] = {'pool': pool, 'preps': preps, 'Tr': Tr, 'basis': {}}
    return _SET[enc]


def basis(enc, prep):
    """(U, mu) of the store's principal map: center = B57's LamFeat basis of the centre-prepped training codes;
    whiten = b60_vsapc.CorrectedBasis (U0 of the raw codes, mu = 0)."""
    S = setup(enc)
    if prep not in S['basis']:
        if prep == 'center':
            lf = B.LamFeat(enc, 'center', S['preps']['center'](S['Tr']))
            U, mu = lf.U, lf.mu.astype(np.float32)
        else:
            Tr = np.asarray(S['Tr'])
            mu0 = Tr.mean(0)
            w, U = np.linalg.eigh(np.cov((Tr - mu0).T))
            U = U[:, np.argsort(-w)].astype(np.float32)
            mu = np.zeros(Tr.shape[1], np.float32)
        S['basis'][prep] = (U, mu)
    return S['basis'][prep]


class FeatStoreR:
    """The final-design store's random map (b32_run.Feat's projection blocks, same seeds)."""

    def __init__(self, enc, prep, fmap, n):
        self.f = R.Feat(enc, prep, fmap, n, 'cpu')
        self.f.torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '1')))
        self.n = n

    def pmax(self, feat):
        assert feat == 'rp', feat
        return L60.P_MAX

    def feats(self, Y, feat, j0, w):
        assert feat == 'rp' and j0 % C.CH == 0 and w <= C.CH, (feat, j0, w)
        t = self.f.torch
        Gs, _ = self.f.block(j0)
        Yt = t.from_numpy(np.ascontiguousarray(Y, dtype=np.float32))
        return self.f.phi(Yt, Gs, w).numpy().astype(np.float32)


class FeatStoreP:
    """The store's principal map (b57_run.FeatP with b60_vsapc's corrected whiten basis), torch float32 like FeatP."""

    def __init__(self, enc, prep, pfmap):
        import torch
        torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '1')))
        self.torch = torch
        U, mu = basis(enc, prep)
        self.n = U.shape[0]
        self.Ut = torch.from_numpy(np.ascontiguousarray(U))
        self.mut = torch.from_numpy(np.ascontiguousarray(mu))
        assert pfmap in ('plin', 'psgn'), pfmap
        self.pfmap = pfmap

    def pmax(self, feat):
        assert feat == 'pca', feat
        return self.n

    def feats(self, Y, feat, j0, w):
        assert feat == 'pca' and j0 + w <= self.n, (feat, j0, w, self.n)
        t = self.torch
        Yt = t.from_numpy(np.ascontiguousarray(Y, dtype=np.float32))
        P = (Yt - self.mut) @ self.Ut[:, j0:j0 + w]
        if self.pfmap == 'psgn':
            P = t.where(P >= 0, 1.0, -1.0)
        return P.numpy().astype(np.float32)


def make_fc(enc, kind, prep, fmap):
    n = setup(enc)['pool'].X.shape[1]
    return FeatStoreR(enc, prep, fmap, n) if kind == 'R' else FeatStoreP(enc, prep, fmap)


def store_feats(enc, kind):
    """{(M, V, bpf, acct): (prep, fmap)}: the store's chosen features per R1 cell (R: B53's VSA; P: VSAPc's VSA)."""
    fn = f'{INRES}/b53_choice_{enc}.json' if kind == 'R' else f'{INRES}/b60_choicepc_{enc}.json'
    ch = json.load(open(fn))['choice']
    out = {}
    for M in M_GRID:
        for V in VN:
            for b in BPF:
                for a in ACCTS:
                    c = ch[f'{M}|{V}|{b}|{A3[a]}|VSA']['cfg']
                    out[M, V, b, a] = (c[0], c[1])
    return out


# ----------------------------------------------------------------------------------------------- jobs

def job_name(s):
    return f'{s["enc"]}_{s["kind"]}_{s["split"]}_M{s["M"]}_V{s["V"]}_r{s["rep"]}_{s["prep"]}_{s["fmap"]}'


def run_job(spec):
    """spec: enc, kind, split, M, V, rep, prep, fmap, ba = [[bpf, acct], ...] (the R1 cells whose chosen features are
    (prep, fmap)), cfgs (None = the 72-configuration grid; else [[cfg4, bpf, acct], ...]). -> OUT/jobs/{name}.json.gz"""
    fn = os.path.join(OUT, 'jobs', job_name(spec) + '.json.gz')
    if os.path.exists(fn):
        return fn
    t0 = time.time()
    enc, kind, split, M, V, rep, prep, fmap = (spec[k] for k in ('enc', 'kind', 'split', 'M', 'V', 'rep', 'prep',
                                                                 'fmap'))
    S = setup(enc)
    st = C.make_store(S['pool'], split, M, V, rep)
    P = S['preps'][prep]
    Yk, Yq = P(st['keys']), P(st['queries'])
    fc = make_fc(enc, kind, prep, fmap)
    seed = C.seed_of('b66lsf', enc, kind, split, M, V, rep)
    g = st['qgold']
    want = {(b, a) for b, a in spec['ba']}
    o = {}
    if spec.get('cfgs') is None:
        cfgs = [cfg5(x, kind) for x in GRID]
        o = L60.run_lamc(st, Yk, Yq, fc, cfgs, sorted({b for b, _ in want}), ACCTS, seed)
        keep = {(c5, b, a) for c5 in cfgs for (b, a) in want}
    else:
        byc = {}
        for c4, b, a in spec['cfgs']:
            byc.setdefault(cfg5(tuple(c4), kind), set()).add((b, a))
        keep = set()
        for c5, ba in byc.items():
            o.update(L60.run_lamc(st, Yk, Yq, fc, [c5], sorted({b for b, _ in ba}), ACCTS, seed))
            keep |= {(c5, b, a) for (b, a) in ba}
    rec = {'spec': spec, 'qgold': g.tolist(), 'pmax': fc.pmax(KINDS[kind]), 'n': int(S['pool'].X.shape[1]), 'rows': []}
    for (c5, b, a), (pred, bt, p) in o.items():
        if (c5, b, a) not in keep:
            continue
        row = {'cfg': [c5[0], c5[1], c5[3], c5[4]], 'bpf': b, 'acct': a, 'acc': float((pred == g).mean()),
               'bits': bt, 'p': p}
        if split == 'test':
            row['pred'] = np.asarray(pred).astype(int).tolist()
        rec['rows'].append(row)
    rec['sec'] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    with gzip.open(fn + '.tmp', 'wt') as fo:
        json.dump(rec, fo)
    os.replace(fn + '.tmp', fn)
    print(f'job {job_name(spec)} {rec["sec"]}s {len(rec["rows"])} rows', flush=True)
    return fn


def _run(spec):
    try:
        return run_job(spec)
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        return f'FAILED {job_name(spec)} {e}'


def runplan(planfile, workers):
    """Jobs handed out by encoder (scheduling only; results depend on registered seeds only)."""
    specs = [json.loads(l) for l in open(planfile)]
    specs = [s for s in specs if not os.path.exists(os.path.join(OUT, 'jobs', job_name(s) + '.json.gz'))]
    specs.sort(key=lambda s: (PANEL.index(s['enc']), s['kind'], -s['M'], s['V']))
    print(f'{len(specs)} jobs to run', flush=True)
    if workers <= 1:
        for s in specs:
            print(_run(s), flush=True)
        return
    import multiprocessing as mp
    with mp.get_context('fork').Pool(workers) as pl:
        for r in pl.imap_unordered(_run, specs, chunksize=4):
            if str(r).startswith('FAILED'):
                print(r, flush=True)


def make_plan(kind, encs):
    os.makedirs(OUT, exist_ok=True)
    fn = os.path.join(OUT, f'plan_{kind}.jsonl')
    n = 0
    with open(fn, 'w') as fo:
        for enc in encs:
            ch = choice(enc) if kind == 'test' else None
            for kd in ('R', 'P'):
                sf = store_feats(enc, kd)
                for V in VN:
                    for M in M_GRID:
                        groups = {}
                        for b in BPF:
                            for a in ACCTS:
                                groups.setdefault(sf[M, V, b, a], []).append([b, a])
                        for (prep, fmap), ba in sorted(groups.items()):
                            cfgs = None
                            if kind == 'test':
                                cfgs = []
                                for b, a in ba:
                                    c = ch[f'{M}|{V}|{b}|{a}|{ARM[kd]}']
                                    assert tuple(c['feat']) == (prep, fmap), (enc, kd, M, V, b, a)
                                    cfgs.append([c['cfg'], b, a])
                            for rep in range(DEV_REPS[M] if kind == 'dev' else TEST_REPS[M]):
                                spec = {'enc': enc, 'kind': kd, 'split': kind, 'M': M, 'V': V, 'rep': rep,
                                        'prep': prep, 'fmap': fmap, 'ba': ba, 'cfgs': cfgs}
                                fo.write(json.dumps(spec) + '\n')
                                n += 1
    print(n, 'jobs ->', fn)


# ----------------------------------------------------------------------------------------------- dev choice

def choice(enc):
    return json.load(open(f'{RES}/b66_lsf_choice_{enc}.json'))['choice']


def choose(enc):
    """B60's rule per (M, V, bpf, acct) and kind: highest mean dev accuracy over the cell's dev stores (6 / 3); within
    0.1 points tied, first in GRID order.  LSFX: LSFP iff its dev accuracy exceeds LSFR's by more than 0.1 points."""
    acc = {}
    jd = os.path.join(OUT, 'jobs')
    for fn in sorted(os.listdir(jd)):
        if not fn.startswith(f'{enc}_') or '_dev_' not in fn:
            continue
        r = json.load(gzip.open(os.path.join(jd, fn), 'rt'))
        s = r['spec']
        if s['enc'] != enc:
            continue
        for x in r['rows']:
            k = (s['kind'], s['M'], s['V'], x['bpf'], x['acct'], s['prep'], s['fmap'], tuple(x['cfg']))
            assert s['rep'] not in acc.get(k, {}), ('duplicate dev row', k, s['rep'])
            acc.setdefault(k, {})[s['rep']] = x['acc']
    ch = {}
    for kd in ('R', 'P'):
        sf = store_feats(enc, kd)
        for M in M_GRID:
            for V in VN:
                for b in BPF:
                    for a in ACCTS:
                        prep, fmap = sf[M, V, b, a]
                        lst = []
                        for g in GRID:
                            d = acc.get((kd, M, V, b, a, prep, fmap, g))
                            if d is None or len(d) != DEV_REPS[M]:
                                raise SystemExit(f'missing / incomplete dev rows {enc} {kd} {M} {V} {b} {a} {g}')
                            lst.append((float(np.mean([d[k] for k in sorted(d)])), g))
                        top = max(x for x, _ in lst)
                        tied = [c for x, c in lst if x >= top - 0.001]
                        dacc = [x for x, c in lst if c == tied[0]][0]
                        ch[f'{M}|{V}|{b}|{a}|{ARM[kd]}'] = {'cfg': list(tied[0]), 'feat': [prep, fmap], 'dev_acc': dacc,
                                                            'best_dev_acc': top, 'n_tied': len(tied)}
    for M in M_GRID:
        for V in VN:
            for b in BPF:
                for a in ACCTS:
                    dr = ch[f'{M}|{V}|{b}|{a}|LSFR']['dev_acc']
                    dp = ch[f'{M}|{V}|{b}|{a}|LSFP']['dev_acc']
                    ch[f'{M}|{V}|{b}|{a}|LSFX'] = {'src': 'LSFP' if dp > dr + 0.001 else 'LSFR',
                                                   'dev': {'LSFR': dr, 'LSFP': dp}}
    fn = f'{RES}/b66_lsf_choice_{enc}.json'
    json.dump({'enc': enc, 'choice': ch}, open(fn, 'w'), indent=1, sort_keys=True)
    print(len(ch), 'choices ->', fn)


# ----------------------------------------------------------------------------------------------- verdict

def test_rows(enc, kd, M, V):
    """{(rep, prep, fmap): {(cfg4, bpf, acct): (pred, bits, p)}, '_qgold'} from the test jobs."""
    out = {}
    jd = os.path.join(OUT, 'jobs')
    pre = f'{enc}_{kd}_test_M{M}_V{V}_r'
    for fn in sorted(os.listdir(jd)):
        if not fn.startswith(pre):
            continue
        r = json.load(gzip.open(os.path.join(jd, fn), 'rt'))
        s = r['spec']
        d = {(tuple(x['cfg']), x['bpf'], x['acct']): (x['pred'], x['bits'], x['p']) for x in r['rows']}
        d['_qgold'] = r['qgold']
        out[s['rep'], s['prep'], s['fmap']] = d
    return out


def add_lsf(recs, enc, M, V, ch):
    for kd in ('R', 'P'):
        rows = test_rows(enc, kd, M, V)
        arm = ARM[kd]
        for r in recs:
            for b in BPF:
                for a in ACCTS:
                    c = ch[f'{M}|{V}|{b}|{a}|{arm}']
                    jr = rows.get((r['rep'], c['feat'][0], c['feat'][1]))
                    assert jr is not None, ('LSF test job missing', enc, kd, M, V, r['rep'])
                    assert jr['_qgold'] == r['qgold'], (enc, kd, M, V, r['rep'])
                    p, bt, pp = jr[tuple(c['cfg']), b, a]
                    r['pred'][f'{arm}|{b}|{a}'] = p
                    r.setdefault('b66', {})[f'{arm}|{b}|{a}'] = [c['feat'] + list(c['cfg']), bt, pp]
    for r in recs:
        for b in BPF:
            for a in ACCTS:
                src = ch[f'{M}|{V}|{b}|{a}|LSFX']['src']
                r['pred'][f'LSFX|{b}|{a}'] = r['pred'][f'{src}|{b}|{a}']


def load(enc):
    """B60's arms (STORE = B53's VSA, LAMC, VSAPC, VSAPCX; b60_verdict's own functions) for V 2 / 16 and the R1
    budgets, plus LSFR / LSFP / LSFX."""
    import b60_verdict as V60
    ch60 = V60.choice(enc)
    ch = choice(enc)
    data = {}
    for M in M_GRID:
        for V in VN:
            recs = [json.loads(l) for l in gzip.open(f'{ROOT}/b59_out/fam/{enc}/all59_{enc}_M{M}_V{V}.jsonl.gz', 'rt')]
            recs.sort(key=lambda r: r['rep'])
            for r in recs:
                for b in BPF:
                    for a in ACCTS:
                        r['pred'][f'STORE|{b}|{a}'] = r['pred'][f'VSA|{b}|{a}']
            assert V60.add_lamc(recs, enc, M, V, ch60, 0, bpfs=BPF), ('LAMC test jobs missing', enc, M, V)
            add_lsf(recs, enc, M, V, ch)
            data[M, V] = recs
    assert V60.add_vsapx(data, enc), ('VSAPc test predictions missing', enc)
    return data, ch


def lab(x, y, lo, hi):
    return f'{x} leads' if lo > 0 else (f'{y} leads' if hi < 0 else 'level')


def summ(x, y, s, dr):
    lo, hi = float(np.quantile(dr, 0.025)), float(np.quantile(dr, 0.975))
    return {'diff': s, 'ci95': [lo, hi], 'label': lab(x, y, lo, hi),
            'side': 'first' if lo > 0 else ('second' if hi < 0 else 'level')}


def verdict_part(enc):
    import b36_verdict as V36
    ei = PANEL.index(enc)
    data, ch = load(enc)
    J = {'cells': {}, 'regions': {}, 'choices': {}}
    pqc = {}

    def pq(M, V, key):
        if (M, V, key) not in pqc:
            pqc[M, V, key] = V36.per_query(data[M, V], key)
        return pqc[M, V, key]

    for ai, acct in enumerate(ACCTS):
        for (M, V, b) in REG['R1']:
            J['cells'][f'{enc}|{acct}|{M}|{V}|{b}'] = {x: float(pq(M, V, f'{x}|{b}|{acct}')[0].mean())
                                                       for x in CELL_ARMS}
        for ri, (rn, Rg) in enumerate(REG.items()):
            for pi, (x, y) in enumerate(PAIRS):
                diffs = []
                for (M, V, b) in Rg:
                    v, cl = pq(M, V, f'{x}|{b}|{acct}')
                    w = pq(M, V, f'{y}|{b}|{acct}')[0]
                    diffs.append((v - w, cl))
                s, _, _, dr = V36.boot_region(diffs, NB if acct == 'ideal' else 2000,
                                              66000 + ei * 1000 + ai * 500 + ri * 20 + pi)
                J['regions'][f'{enc}|{acct}|{rn}|{x}-{y}'] = summ(x, y, s, dr)
    for (M, V, b) in REG['R1']:
        for a in ACCTS:
            k = f'{M}|{V}|{b}|{a}'
            J['choices'][f'{enc}|{k}'] = {'LSFR': ch[f'{k}|LSFR']['feat'] + ch[f'{k}|LSFR']['cfg'],
                                          'LSFP': ch[f'{k}|LSFP']['feat'] + ch[f'{k}|LSFP']['cfg'],
                                          'LSFX': ch[f'{k}|LSFX']['src'],
                                          'p': {arm: data[M, V][0]['b66'][f'{arm}|{b}|{a}'][2] for arm in ('LSFR', 'LSFP')}}
    return J


def verdict(encs=PANEL):
    J = {'cells': {}, 'regions': {}, 'choices': {}}
    for enc in encs:
        P = verdict_part(enc)
        for k in J:
            J[k].update(P[k])
        print('verdict part', enc, flush=True)
    json.dump(J, open(f'{RES}/b66_B_verdict.json', 'w'), indent=1)
    Rr = J['regions']

    def f(r):
        return f'{100 * r["diff"]:+6.2f} [{100 * r["ci95"][0]:+6.2f}, {100 * r["ci95"][1]:+6.2f}]'

    L = ['# B66 part (B) verdict: LSfc = least squares with LAMc\'s corrected options (B60) on the STORE\'s own features.',
         '# STORE = B53\'s final design (Hebbian write, matched filter); LSFR = LSfc on the store\'s random map of the cell',
         '# (B53\'s chosen prep / fmap); LSFP = LSfc on the store\'s principal map (VSAPc\'s chosen prep / pfmap, corrected',
         '# basis); LSFX = per cell the dev-best of LSFR / LSFP (ties LSFR); LAMC = B60\'s corrected least squares on its',
         '# own dev-chosen features; VSAPC = B60\'s store on the corrected principal basis (B57 / B56 read rules).',
         '# Accuracy points over R1 (V 2 / 16 x 4, 8, 16 bits per fact x M 256 / 1,024); 95% cluster bootstrap (entity',
         '# within store, resampled within each cell; b36_verdict.boot_region); ideal 10,000 draws, real 2,000.',
         '# Label (B60\'s rule): "X leads" iff lower > 0 (first arm) / upper < 0 (second arm); else "level".', '']
    for acct in ACCTS:
        for (x, y) in PAIRS:
            L.append(f'## {x} - {y} ({acct})')
            L.append('| encoder | R1-V2 | label | R1-V16 | label | R1 | label |')
            L.append('| --- | --- | --- | --- | --- | --- | --- |')
            for enc in encs:
                row = []
                for rn in ('R1-V2', 'R1-V16', 'R1'):
                    r = Rr[f'{enc}|{acct}|{rn}|{x}-{y}']
                    row += [f(r), r['label']]
                L.append(f'| {NAMES[enc]} | ' + ' | '.join(row) + ' |')
            for rn in ('R1-V2', 'R1-V16', 'R1'):
                for grp, es in (('LLM', LLM), ('sentence', SENT)):
                    es = [e for e in es if e in encs]
                    sides = [Rr[f'{e}|{acct}|{rn}|{x}-{y}']['side'] for e in es]
                    L.append(f'  {rn:7s} {grp:8s}: {x} leads {sides.count("first")}, level {sides.count("level")}, '
                             f'{y} leads {sides.count("second")} (of {len(es)})')
            L.append('')
    L.append('## Per-cell test accuracy (ideal), points: ' + ' / '.join(CELL_ARMS))
    for enc in encs:
        L.append(f'  {NAMES[enc]}')
        for (M, V, b) in REG['R1']:
            c = J['cells'][f'{enc}|ideal|{M}|{V}|{b}']
            ch = J['choices'][f'{enc}|{M}|{V}|{b}|ideal']
            L.append(f'    M={M:5d} V={V:3d} b={b:3d}: ' + ' / '.join(f'{100 * c[x]:5.1f}' for x in CELL_ARMS) +
                     f'   LSFR {ch["LSFR"]} p={ch["p"]["LSFR"]}; LSFP {ch["LSFP"]} p={ch["p"]["LSFP"]}; LSFX={ch["LSFX"]}')
    L.append('')
    L.append('## Dev choice composition (ideal, R1, pooled over the encoders shown)')
    for V in VN:
        for arm in ('LSFR', 'LSFP'):
            cs = [J['choices'][f'{e}|{M}|{V}|{b}|ideal'][arm] for e in encs for M in M_GRID for b in BPF]
            for i, nm in ((2, 'scale'), (3, 'target'), (4, 'alpha'), (5, 'q'), (0, 'prep'), (1, 'fmap')):
                cnt = {}
                for c in cs:
                    cnt[str(c[i])] = cnt.get(str(c[i]), 0) + 1
                L.append(f'  V={V:3d} {arm} {nm:6s}: ' + ', '.join(f'{k} {v}' for k, v in sorted(cnt.items())) +
                         f' (of {len(cs)})')
        src = [J['choices'][f'{e}|{M}|{V}|{b}|ideal']['LSFX'] for e in encs for M in M_GRID for b in BPF]
        L.append(f'  V={V:3d} LSFX picks: LSFR {src.count("LSFR")}, LSFP {src.count("LSFP")} (of {len(src)})')
    open(f'{RES}/b66_B_verdict.txt', 'w').write('\n'.join(L) + '\n')
    print('->', f'{RES}/b66_B_verdict.txt')


# ----------------------------------------------------------------------------------------------- checks

def own_bits(p, c, q, acct, scale):
    if p <= 0:
        return 0.0
    per = {1: 1.0, 2: 1.9037, 3: 2.7606, 4: 3.6024}[q] if acct == 'ideal' else float(q)
    groups = c * (-(-p // 8192)) if scale == 'col' else -(-(c * p) // 8192)
    return c * p * per + 32 * groups


def own_quant(x, q):
    """Independent float64 quantizer: q = 1 sign x mean |x| (no exact zeros expected); q >= 2 uniform mid-rise with
    Max's step on the group's standard deviation (b32_common's MAX_STEP constants)."""
    if q == 1:
        s = np.where(x >= 0, 1.0, -1.0)
        return s * (np.abs(x).mean() or 1.0)
    step = {2: 0.9957, 4: 0.3352}[q] * (float(x.std()) or 1.0)
    L = 2 ** q
    return (np.clip(np.floor(x / step), -L // 2, L // 2 - 1) + 0.5) * step


def own_feats_R(enc, prep, fmap, Y, p):
    """Float64 store random features (own code; the matrices are the map's definition: b32_run.Feat's seeds)."""
    n = Y.shape[1]
    K = 1 if fmap in ('lin', 'sgn') else int(fmap[3:])
    Y = Y.astype(np.float64)
    out = np.empty((Y.shape[0], p))
    for j0 in range(0, p, 8192):
        w = min(8192, p - j0)
        f = None
        for m in range(K):
            G = np.random.default_rng(C.seed_of('G', enc, prep, fmap, m, j0, 8192)).standard_normal(
                (n, 8192), dtype=np.float32)[:, :w].astype(np.float64)
            P = Y @ G
            h = P if fmap == 'lin' else np.where(P >= 0, 1.0, -1.0)
            f = h if f is None else f * h
        out[:, j0:j0 + w] = f
    return out


def rederive(st, Yk, Yq, F, Fq, cfg4, V):
    scale, t, alpha, q = cfg4
    M = st['M']
    vals = np.asarray(st['vals'])
    if t == 'V-1':
        T = np.eye(V)[vals][:, 1:] - 1.0 / V
    else:
        T = np.eye(V)[vals] - 1.0 / V
    Kg = F @ F.T
    lam = alpha * float(np.trace(Kg)) / M
    A = np.linalg.solve(Kg + lam * np.eye(M), T)
    W = F.T @ A  # p x c
    p, c = W.shape
    Wq = np.empty_like(W)
    if scale == 'col':
        for j0 in range(0, p, 8192):
            for k in range(c):
                Wq[j0:j0 + 8192, k] = own_quant(W[j0:j0 + 8192, k], q)
    else:
        fl = W.T.ravel()
        o = np.empty_like(fl)
        for i0 in range(0, len(fl), 8192):
            o[i0:i0 + 8192] = own_quant(fl[i0:i0 + 8192], q)
        Wq = o.reshape(c, p).T
    S = Fq @ Wq
    if t == 'V-1':
        S = np.hstack([-S.sum(1, keepdims=True), S])
    return S.argmax(1)


def check(encs=PANEL):
    L = ['# B66 part (B) checks (b66_lsfc.py check)', '']
    J = json.load(open(f'{RES}/b66_B_verdict.json'))
    # (1) plain-loop scorer
    mx, nv = 0.0, 0
    jd = os.path.join(OUT, 'jobs')
    import b60_verdict as V60
    for enc in encs:
        ch = choice(enc)
        ch60 = V60.choice(enc)
        cellacc = {}
        perq = {}
        for M in M_GRID:
            for V in VN:
                recs = [json.loads(l) for l in gzip.open(f'{ROOT}/b59_out/fam/{enc}/all59_{enc}_M{M}_V{V}.jsonl.gz', 'rt')]
                recs.sort(key=lambda r: r['rep'])
                lamrows = V60.job_rows(enc, 0, M, V)
                for b in BPF:
                    for a in ACCTS:
                        lists = {x: [] for x in ('LSFR', 'LSFP', 'LSFX', 'STORE', 'LAMC')}
                        for r in recs:
                            g = r['qgold']
                            for kd in ('R', 'P'):
                                arm = ARM[kd]
                                c = ch[f'{M}|{V}|{b}|{a}|{arm}']
                                fn = os.path.join(jd, f'{enc}_{kd}_test_M{M}_V{V}_r{r["rep"]}_{c["feat"][0]}_{c["feat"][1]}.json.gz')
                                jr = json.load(gzip.open(fn, 'rt'))
                                hit = [x for x in jr['rows'] if x['cfg'] == c['cfg'] and x['bpf'] == b and x['acct'] == a]
                                assert len(hit) == 1
                                pr = hit[0]['pred']
                                lists[arm] += [1.0 if pr[i] == g[i] else 0.0 for i in range(len(g))]
                            src = ch[f'{M}|{V}|{b}|{a}|LSFX']['src']
                            n0 = len(g)
                            lists['LSFX'] += lists[src][-n0:]
                            sp = r['pred'][f'VSA|{b}|{a}']
                            lists['STORE'] += [1.0 if sp[i] == g[i] else 0.0 for i in range(n0)]
                            lc = tuple(ch60[f'{M}|{V}|{b}|{a}|LAMC']['cfg'])
                            lp = lamrows[r['rep'], lc[2]][lc, b, a][0]
                            lists['LAMC'] += [1.0 if lp[i] == g[i] else 0.0 for i in range(n0)]
                        for x, lst in lists.items():
                            acc = sum(lst) / len(lst)
                            cellacc[enc, a, M, V, b, x] = acc
                            perq[enc, a, M, V, b, x] = lst
                            if x in ('LSFR', 'LSFP', 'LSFX', 'STORE', 'LAMC'):
                                d = abs(acc - J['cells'][f'{enc}|{a}|{M}|{V}|{b}'][x])
                                mx, nv = max(mx, d), nv + 1
        for a in ACCTS:
            for (x, y) in PAIRS:
                if 'VSAPC' in (x, y):
                    continue
                for rn, Rg in REG.items():
                    tot = 0.0
                    for (M, V, b) in Rg:
                        u, w = perq[enc, a, M, V, b, x], perq[enc, a, M, V, b, y]
                        tot += sum(u[i] - w[i] for i in range(len(u))) / len(u)
                    d = abs(tot / len(Rg) - J['regions'][f'{enc}|{a}|{rn}|{x}-{y}']['diff'])
                    mx, nv = max(mx, d), nv + 1
    L.append(f'(1) plain-loop scorer: {nv} values (cell accuracies of LSFR / LSFP / LSFX / STORE / LAMC and region point '
             f'statistics of the pairs without VSAPC), max |difference| {mx:.2e} ({"PASS" if mx < 1e-9 else "FAIL"})')
    # (2) bits audit
    nrow = nover = nnotmax = nmis = 0
    for fn in sorted(os.listdir(jd)):
        if not fn.endswith('.json.gz') or fn.split('_')[0] not in encs:
            continue
        r = json.load(gzip.open(os.path.join(jd, fn), 'rt'))
        s = r['spec']
        pm = r['pmax']
        for x in r['rows']:
            scale, t, alpha, q = x['cfg']
            c = s['V'] - 1 if t == 'V-1' else s['V']
            Bud = x['bpf'] * s['M']
            p = x['p']
            ob = own_bits(p, c, q, x['acct'], scale)
            nrow += 1
            nmis += abs(ob - x['bits']) > 1e-6
            nover += ob > Bud
            nnotmax += (p < pm and own_bits(p + 1, c, q, x['acct'], scale) <= Bud)
    L.append(f'(2) bits audit: {nrow} dev + test rows; logged bits != own formula {nmis}; over budget {nover}; not maximal '
             f'{nnotmax} ({"PASS" if nover == 0 and nnotmax == 0 and nmis == 0 else "FAIL"})')
    # (3) float64 re-derivation: test rep 0 per (enc, M, V), b 4 and 16, ideal, LSFR's chosen configuration
    L.append('(3) float64 re-derivation (own features from the map\'s seeds, own dual ridge, own quantizer, own read), '
             'test rep 0, ideal, LSFR:')
    worst = 1.0
    for enc in encs:
        ch = choice(enc)
        S = setup(enc)
        for M in M_GRID:
            for V in VN:
                st = C.make_store(S['pool'], 'test', M, V, 0)
                for b in (4, 16):
                    c = ch[f'{M}|{V}|{b}|ideal|LSFR']
                    prep, fmap = c['feat']
                    fn = os.path.join(jd, f'{enc}_R_test_M{M}_V{V}_r0_{prep}_{fmap}.json.gz')
                    jr = json.load(gzip.open(fn, 'rt'))
                    hit = [x for x in jr['rows'] if x['cfg'] == c['cfg'] and x['bpf'] == b and x['acct'] == 'ideal'][0]
                    p = hit['p']
                    if p == 0:
                        L.append(f'    {enc} M={M} V={V} b={b}: p = 0 (default value), skipped')
                        continue
                    Pp = S['preps'][prep]
                    Yk, Yq = Pp(st['keys']), Pp(st['queries'])
                    F = own_feats_R(enc, prep, fmap, Yk, p)
                    Fq = own_feats_R(enc, prep, fmap, Yq, p)
                    pr = rederive(st, Yk, Yq, F, Fq, tuple(c['cfg']), V)
                    ag = float((pr == np.asarray(hit['pred'])).mean())
                    worst = min(worst, ag)
                    L.append(f'    {enc} M={M} V={V} b={b} {prep}:{fmap} {c["cfg"]} p={p}: agreement {ag:.4f}')
    L.append(f'    min agreement {worst:.4f} ({"PASS" if worst >= 0.99 else "FAIL"}; bar >= 0.99)')
    open(f'{RES}/b66_B_check.txt', 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))


# ----------------------------------------------------------------------------------------------- selftests

def selftest():
    import b53_run as R53
    rng = np.random.default_rng(0)
    # (a) toy: near-orthogonal keys, exact queries, large budget -> every fact read with the store's random maps
    M, V, n = 200, 16, 256
    Kk = rng.standard_normal((M, n)).astype(np.float32)
    Kk /= np.linalg.norm(Kk, axis=1, keepdims=True)
    vals = rng.integers(0, V, M)
    st = {'M': M, 'V': V, 'vals': vals}
    for fmap in ('lin', 'sgn', 'sgn3'):
        fc = FeatStoreR('selftest', 'center', fmap, n)
        cfgs = [cfg5(g, 'R') for g in GRID if g[2] == 1e-4 and g[3] == 4]
        o = L60.run_lamc(st, Kk, Kk, fc, cfgs, [256], ACCTS, 1)
        for (c5, b, a), (pr, bt, p) in o.items():
            assert bt <= b * M and bt == own_bits(p, V - 1 if c5[1] == 'V-1' else V, c5[4], a, c5[0])
            assert p == L60.P_MAX or own_bits(p + 1, V - 1 if c5[1] == 'V-1' else V, c5[4], a, c5[0]) > b * M
            acc = float((pr == vals).mean())
            assert acc >= 0.95, (fmap, c5, a, acc)
    print('selftest (a) toy: random-map LSfc reads every fact at 256 bits per fact; bits within budget and maximal')
    # (b) training-split store, qwen8: features are the store's own (store read rebuilt from them = R.vsa_grid's)
    enc = 'qwen8'
    S = setup(enc)
    n = S['pool'].X.shape[1]
    st = C.make_store(S['pool'], 'train', 1024, 16, 0)
    for prep, fmap in (('whiten', 'lin'), ('center', 'sgn'), ('whiten', 'sgn3')):
        P = S['preps'][prep]
        Yk, Yq = P(st['keys']), P(st['queries'])
        fc = FeatStoreR(enc, prep, fmap, n)
        # identical to b32_common.features (the numpy twin of the store's map, the same seeds), both blocks
        for j0 in (0, 8192):
            a1 = fc.feats(Yk[:64], 'rp', j0, 8192)
            a2 = C.features(Yk[:64], fmap, f'{enc}|{prep}|{fmap}', j0, 8192)
            if fmap == 'lin':
                assert np.allclose(a1, a2, atol=1e-4), (prep, fmap, j0, float(np.abs(a1 - a2).max()))
            else:
                assert (a1 == a2).mean() >= 0.999, (prep, fmap, j0, float((a1 == a2).mean()))
        # the store's read rebuilt from these features + B53's value codes = R.vsa_grid's predictions
        feat53 = R53.Feat53(enc, prep, fmap, n, 'cpu')
        feat53.V = 16
        order = R.store_order(st)
        sseed = R.store_seed(enc, st)
        for q in (1, 4):
            d = C.vsa_dim(16 * 1024, q, 'ideal')
            ns = 1024
            ref = R.vsa_grid(Yk, st['vals'], Yq, 16, feat53, [('k', d, q, ns)], sseed, order)['k']
            t = feat53.torch
            score = np.zeros((Yq.shape[0], 16))
            for j0 in range(0, d, C.CH):
                w = min(C.CH, d - j0)
                _, Cfull = feat53.block(j0)
                Cv = Cfull[:16, :w].numpy()
                FK = fc.feats(Yk, 'rp', j0, w)
                FQ = fc.feats(Yq, 'rp', j0, w)
                Sb = (FK[order[:ns]] * Cv[np.asarray(st['vals'])[order[:ns]]]).sum(0)
                Sq = R.quantize_t(t.from_numpy(Sb.astype(np.float32)), q,
                                  np.random.default_rng(C.seed_of('Q', sseed, d, q, ns, j0)), t).numpy()
                score += ((FQ * Sq) @ Cv.T).astype(np.float64)
            ag = float((score.argmax(1) == ref).mean())
            assert ag >= 0.99, (prep, fmap, q, ag)
            print(f'selftest (b) {enc} train store M=1024 V=16 {prep}:{fmap} q={q} d={d}: store read from LSfc\'s '
                  f'features vs R.vsa_grid agreement {ag:.4f}')
        # (c) col / V-1 with B49's alphas reproduces b49_common.run_lam on the same features
        for alpha in (1e-2, 1.0):
            for q in (1, 2, 4):
                o49 = B.run_lam(st, Yk, Yq, fc, [('rp', alpha, q, 'onehot')], [4, 16], ['ideal'], 1)
                o66 = L60.run_lamc(st, Yk, Yq, fc, [('col', 'V-1', 'rp', alpha, q)], [4, 16], ['ideal'], 1)
                for b in (4, 16):
                    ag = float((o49[('rp', alpha, q, 'onehot'), b, 'ideal'][0] ==
                                o66[('col', 'V-1', 'rp', alpha, q), b, 'ideal'][0]).mean())
                    assert ag >= 0.995, (prep, fmap, alpha, q, b, ag)
        print(f'selftest (c) {enc} train store {prep}:{fmap}: col / V-1 = b49_common.run_lam (>= 0.995 on 12 configs)')
    # (d) principal features: e5, both preps, = b57_run.FeatP with b60_vsapc's corrected setup
    import b60_vsapc as PC  # noqa: F401  (patches b57_run.setup with the corrected whiten basis)
    import b57_run as R57
    enc = 'e5'
    S = setup(enc)
    n = S['pool'].X.shape[1]
    _, _, Tr57, lfs = R57.setup(enc)
    os.chdir(ROOT)
    assert np.array_equal(np.asarray(Tr57), np.asarray(S['Tr']))
    st = C.make_store(S['pool'], 'train', 256, 2, 0)
    for prep in PREPS:
        Yk = S['preps'][prep](st['keys'])
        for pf in ('plin', 'psgn'):
            fp = R57.FeatP(enc, prep, pf, n, 'cpu', lfs[prep])
            fp.V = 2
            Gs, _ = fp.block(0)
            ref = fp.phi(fp.torch.from_numpy(Yk), Gs, n).numpy()
            mine = FeatStoreP(enc, prep, pf).feats(Yk, 'pca', 0, n)
            if pf == 'plin':
                assert np.allclose(ref, mine, atol=1e-5), (prep, pf, float(np.abs(ref - mine).max()))
            else:
                assert (ref == mine).mean() >= 0.999, (prep, pf, float((ref == mine).mean()))
        # LSfc on these features: bits within budget, maximal
        fc = FeatStoreP(enc, prep, 'plin')
        o = L60.run_lamc(st, Yk, S['preps'][prep](st['queries']), fc, [cfg5(g, 'P') for g in GRID], BPF, ACCTS, 2)
        for (c5, b, a), (pr, bt, p) in o.items():
            c = 1 if c5[1] == 'V-1' else 2
            assert bt <= b * 256 and (p == n or own_bits(p + 1, c, c5[4], a, c5[0]) > b * 256)
    print(f'selftest (d) {enc}: principal features = b57_run.FeatP with b60_vsapc\'s basis (both preps, plin / psgn); '
          f'LSfc rows within budget and maximal')
    # (e) store feature choices: B53's choice = the merged records' cfg field (rep 0; cfg only, no predictions read)
    nchk = 0
    for enc in PANEL:
        sf = store_feats(enc, 'R')
        store_feats(enc, 'P')
        for M in M_GRID:
            for V in VN:
                r = json.loads(gzip.open(f'{ROOT}/b59_out/fam/{enc}/all59_{enc}_M{M}_V{V}.jsonl.gz', 'rt').readline())
                for b in BPF:
                    for a in ACCTS:
                        assert tuple(r['cfg'][f'VSA|{b}|{a}'][:2]) == sf[M, V, b, a], (enc, M, V, b, a)
                        nchk += 1
    print(f'selftest (e) B53 choice (prep, fmap) = merged records\' VSA cfg on {nchk} R1 cells; VSAPc choices present')
    print('selftest ok')


def timing(enc, M, V):
    S = setup(enc)
    st = C.make_store(S['pool'], 'train', M, V, 0)
    for kd in ('R', 'P'):
        sf = store_feats(enc, kd)
        prep, fmap = sf[M, V, 16, 'ideal']
        P = S['preps'][prep]
        t0 = time.time()
        fc = make_fc(enc, kd, prep, fmap)
        t1 = time.time()
        o = L60.run_lamc(st, P(st['keys']), P(st['queries']), fc, [cfg5(g, kd) for g in GRID], BPF, ACCTS, 3)
        pm = max(v[2] for v in o.values())
        print(f'timing {enc} train M={M} V={V} kind {kd} {prep}:{fmap}: feature setup {t1 - t0:.1f}s, 72 configs x 3 '
              f'bpf x 2 accts = {len(o)} rows in {time.time() - t1:.1f}s (max p {pm})', flush=True)


def main():
    a = sys.argv[1:]
    encs = a[a.index('--encs') + 1].split(',') if '--encs' in a else PANEL
    if a[0] == 'selftest':
        selftest()
    elif a[0] == 'timing':
        timing(a[1], int(a[2]), int(a[3]))
    elif a[0] == 'job':
        run_job(json.load(open(a[1])))
    elif a[0] == 'plan':
        make_plan(a[1], encs)
    elif a[0] == 'runplan':
        wk = int(a[a.index('--workers') + 1]) if '--workers' in a else 1
        runplan(a[1], wk)
    elif a[0] == 'choose':
        choose(a[1])
    elif a[0] == 'verdict':
        verdict(encs)
    elif a[0] == 'check':
        check(encs)
    else:
        raise SystemExit(a[0])


if __name__ == '__main__':
    main()
