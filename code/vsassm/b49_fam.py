"""B49 new competitors (prereg/B49_panel_mechanism_competitors_prereg.md, section 3) on one encoder's B32 stores:
LAM (least-squares linear associative memory), LHI / LHC / LHS (learned hashing: per-fact index, LSH bucket sketch,
static function on the learned codes), VSAL (secondary: B32's VSA on the learned codes), and the unbudgeted references
VSAINF / LAMINF. Runs in b49_out/ (its b32_data holds every panel encoder's codes); outputs in b49_out/fam/{enc}/.

python3 b49_fam.py ENC prefit                       # PQ / index fits of the learned space (training split only)
python3 b49_fam.py ENC selftest
python3 b49_fam.py ENC dev [--workers N] [--fam LAM,LHI,LHC,LHS,VSAL,REF]
python3 b49_fam.py ENC choose                       # -> results/b49_choice_{enc}.json (+ BEST over all ten families)
python3 b49_fam.py ENC test [--workers N]
python3 b49_fam.py ENC merge                        # B36-format merged file + new families -> fam/{enc}/test/all49_*

Environment B49_GRID_V / B49_GRID_BPF / B49_DEV_ONLY_R1 restrict the grids (mechanism runs only, b49_mech.py).
"""
import gzip
import itertools
import json
import math
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import b32_common as C  # noqa: E402
import b36_common as K  # noqa: E402
import b49_common as B  # noqa: E402

ENC_TAG = {'qwen8': 'qwen8_mean_L9_f16', 'e5': 'e5_base', 'b46': 'b46_granite33i_mean_L30_f16',
           'llama': 'llama31i_mean_L24_f16', 'mistral': 'mistral7i_eol_L24_f16', 'bge': 'bge_small', 'minilm': 'minilm'}
# where B36's merged test files and B36-protocol choices of each encoder are
B36_DIRS = {'qwen8': (f'{ROOT}/b36_out/test', f'{ROOT}/results/b36_choice_qwen8.json'),
            'e5': (f'{ROOT}/b36_out/test', f'{ROOT}/results/b36_choice_e5.json'),
            'b46': (f'{ROOT}/b46_out/b36_out/test', f'{ROOT}/results/b46_choice_b36.json')}
for _e in ('llama', 'mistral', 'bge', 'minilm'):
    B36_DIRS[_e] = (f'{ROOT}/b49_out/enc_{_e}/b36_out/test', f'{ROOT}/b49_out/enc_{_e}/results/b36_choice_{_e}.json')

SEED = {}  # mechanism runs '{base}_{T}': fixed random components seeded by the base encoder's tag
for _b in ('qwen8', 'e5', 'b46', 'bge'):
    for _t in ('T1', 'T2', 'T3'):
        ENC_TAG[f'{_b}_{_t}'] = f'{_b}_{_t}'
        SEED[f'{_b}_{_t}'] = _b
        B36_DIRS[f'{_b}_{_t}'] = (f'{ROOT}/b49_out/mech_{_b}_{_t}/b36_out/test',
                                  f'{ROOT}/b49_out/mech_{_b}_{_t}/results/b36_choice_{_b}.json')


def b36name(enc):
    return SEED.get(enc, enc)


M_GRID = [256, 1024]
V_GRID = [int(x) for x in os.environ.get('B49_GRID_V', '2,16,128,1024').split(',')]
BPF = [int(x) for x in os.environ.get('B49_GRID_BPF', '1,2,4,8,16,32,64').split(',')]
ACCTS = ['ideal', 'real']
PREPS = ['center', 'whiten']
DEV_REPS = {256: 6, 1024: 3}
TEST_REPS = {256: 20, 1024: 5}
NEW = ['LAM', 'LHI', 'LHC', 'LHS']
SEC = ['VSAL']
REF = ['VSAINF', 'LAMINF']
OLD = ['IDX', 'SF', 'CM', 'BF', 'FBF', 'SDM']
ALL10 = OLD + NEW
ALPHAS = [1e-4, 1e-2, 1.0, 100.0]


def lam_grid(V):
    tg = ['onehot'] + (['code32', 'code128'] if V >= 128 else [])
    return [(p, ft, a, q, t) for p in PREPS for ft in ('rp', 'pca') for a in ALPHAS for q in (1, 2, 4) for t in tg]


def lhi_grid(V):
    return [('lh', f, s) for f in ('lsh', 'itq', 'sq1', 'pq4', 'pq8') for s in (1.0, 0.5, 0.25, 0.125)]


def b36_grid(fam, V):
    global K_GRID
    if K_GRID is None:
        import b36_run as R36
        K_GRID = R36.GRID
    g = K_GRID[fam]
    if fam == 'CM' and V > 16:
        g = [c for c in g if c[2] == 'plural']
    return [('lh',) + tuple(c) for c in g]


K_GRID = None  # set from b36_run.GRID at import of b36_run (no circularity: b36_run imports only b32 / b36 common)


def vsal_grid(V):
    return [('lh', fm, q, f) for fm in ('lin', 'sgn', 'sgn3') for q in (1, 2, 4) for f in (1.0, 0.5, 0.25, 0.125)]


def ref_grid(fam):
    if fam == 'VSAINF':
        return [(p,) for p in PREPS]
    return [(p, a) for p in PREPS for a in ALPHAS]


def grid(fam, V):
    return {'LAM': lam_grid, 'LHI': lhi_grid, 'LHC': lambda v: b36_grid('CM', v), 'LHS': lambda v: b36_grid('SF', v),
            'VSAL': vsal_grid}[fam](V) if fam not in REF else ref_grid(fam)


def wdir(enc):
    return f'{ROOT}/b49_out/fam/{enc}'


# ----------------------------------------------------------------------------------------------- setup

class Setup:
    def __init__(self, enc, tag=None, seed_enc=None):
        global K_GRID
        import b36_run as R36
        K_GRID = R36.GRID
        self.enc = enc
        self.senc = seed_enc or enc  # tag that seeds the fixed random components (mechanism runs: the base encoder)
        self.pool = C.Pool(tag or ENC_TAG[enc])
        Tr = self.pool.train_matrix()
        self.preps = {p: C.Prep(p, Tr) for p in PREPS}
        self.lamf = {p: B.LamFeat(self.senc, p, self.preps[p](Tr)) for p in PREPS}
        lc = json.load(open(f'{wdir(enc)}/lh_choice.json'))
        self.lh = B.LearnedPrep(self.pool.X, self.pool, lc['s'], lc['r'])
        self.r = lc['r']
        Tl = self.lh(Tr)
        os.environ['B32_FIT_CACHE'] = f'{wdir(enc)}/fitcache'
        self.lhfam = {('lh', f): C.IndexFamily(f, Tl, cache_tag=f'{self.senc}_lh') for f in
                      ('lsh', 'itq', 'sq1', 'pq4', 'pq8')}
        self.ppool = K.Pool(f'{self.senc}|lh', self.r)
        self.bank = K.PQBank(self.ppool, Tl, f'{self.senc}_lh', f'{wdir(enc)}/pqcache')

    def prefit(self):
        os.makedirs(f'{wdir(self.enc)}/pqcache', exist_ok=True)
        if self.bank.cent is None:
            self.bank.fit()
        for (_, f), fam in self.lhfam.items():
            if f in ('pq4', 'pq8'):
                nb = 4 if f == 'pq4' else 8
                for m in C.divisors(self.r):
                    if m * nb <= 600:  # every m a key of <= 64 bits per fact can reach at a stored share >= 1/8
                        fam._fit(m)
            elif f == 'itq':
                pass  # ITQ fits are cheap and per process (in memory)


# ----------------------------------------------------------------------------------------------- one store

def run_new(S, st, fams, bpfs, accts, select=None):
    """Every configuration (or select = {fam: set of cfg}) of the given families on one store.
    Returns {(fam, cfg, bpf, acct): (pred, bits)}; reference families have bpf = acct = None."""
    import b32_run as R32
    M, V = st['M'], st['V']
    res = {}
    seed = C.seed_of('b49', S.senc, st['split'], M, V, st['rep'])

    def cf(f):
        return [c for c in grid(f, V) if select is None or tuple(c) in select.get(f, set())]

    # LAM (per prep)
    if 'LAM' in fams:
        for prep in PREPS:
            cs = [c for c in cf('LAM') if c[0] == prep]
            if not cs:
                continue
            P = S.preps[prep]
            o = B.run_lam(st, P(st['keys']), P(st['queries']), S.lamf[prep], [c[1:] for c in cs], bpfs, accts, seed)
            for (c1, b, a), v in o.items():
                res['LAM', (prep,) + tuple(c1), b, a] = v
    # references
    for prep in (PREPS if 'REF' in fams else []):
        P = S.preps[prep]
        Kk, Qq = P(st['keys']).astype(np.float64), P(st['queries']).astype(np.float64)
        Sm = Qq @ Kk.T
        if True:
            oh = np.eye(V)[st['vals']]
            if select is None or (prep,) in select.get('VSAINF', set()):
                res['VSAINF', (prep,), None, None] = ((Sm @ oh).argmax(1), 0.0)
            G = Kk @ Kk.T
            Y = oh - 1.0 / V
            for a in ALPHAS:
                if select is None or (prep, a) in select.get('LAMINF', set()):
                    A = np.linalg.solve(G + a * np.eye(M), Y)
                    res['LAMINF', (prep, a), None, None] = ((Sm @ A).argmax(1), 0.0)
    # learned codes
    need_l = [f for f in ('LHI', 'LHC', 'LHS', 'VSAL') if f in fams and cf(f)]
    if need_l:
        Yk, Yq = S.lh(st['keys']), S.lh(st['queries'])
        if 'LHI' in need_l:
            preps = {'lh': S.lh}
            for c in cf('LHI'):
                for b in bpfs:
                    for a in accts:
                        p, kb = R32.run_idx(st, preps, S.lhfam, 'lh', c[1], b, 'ideal' if a == 'ideal' else 'raw', c[2])
                        res['LHI', tuple(c), b, a] = (p, float(kb))
        if 'LHC' in need_l or 'LHS' in need_l:
            Pp = S.ppool.project(np.vstack([Yk, Yq]), K.POOL)
            ptag = f'{S.senc}|{st["split"]}|{M}|{V}|{st["rep"]}|lh'
            if 'LHC' in need_l:
                o = K.run_cm(st, Pp, S.bank, [c[1:] for c in cf('LHC')], bpfs, accts, ptag)
                for (c1, b, a), v in o.items():
                    res['LHC', ('lh',) + tuple(c1), b, a] = v
            if 'LHS' in need_l:
                order = np.random.default_rng(C.seed_of('subset', st['split'], M, V, st['rep'])).permutation(M)
                o = K.run_sf(st, Pp, S.bank, [c[1:] for c in cf('LHS')], bpfs, accts, ptag, order)
                for (c1, b, a), v in o.items():
                    res['LHS', ('lh',) + tuple(c1), b, a] = v
            del Pp
        if 'VSAL' in need_l:
            cs = cf('VSAL')
            for fm in sorted({c[1] for c in cs}):
                feat = S.vfeat.get(fm) if hasattr(S, 'vfeat') else None
                if feat is None:
                    if not hasattr(S, 'vfeat'):
                        S.vfeat = {}
                    feat = S.vfeat[fm] = R32.Feat(S.senc, 'lh', fm, S.r, 'cpu')
                sub = [c for c in cs if c[1] == fm]
                qs = sorted({c[2] for c in sub})
                fs = sorted({c[3] for c in sub}, reverse=True)
                acc32 = ['ideal' if a == 'ideal' else 'raw' for a in accts]
                o = R32.run_vsa(st, {'lh': S.lh}, 'lh', feat, bpfs, acc32, qs, fs, S.senc)
                for (b, a, q, f), p in o.items():
                    c = ('lh', fm, q, f)
                    if c in sub:
                        a2 = 'ideal' if a == 'ideal' else 'real'
                        res['VSAL', c, b, a2] = (p, float(C.vsa_bits(C.vsa_dim(b * M, q, a), q, a)))
    return res


# ----------------------------------------------------------------------------------------------- dev

_S = None


def _init(enc):
    global _S
    _S = Setup(enc, ENC_TAG[enc], SEED.get(enc, enc))


def _dev_one(args):
    M, V, rep, fams = args
    st = C.make_store(_S.pool, 'dev', M, V, rep)
    t0 = time.time()
    res = run_new(_S, st, fams, BPF, ACCTS)
    g = st['qgold']
    rows = [{'fam': f, 'cfg': list(c), 'bpf': b, 'acct': a, 'acc': float((p == g).mean()), 'bits': float(bits)}
            for (f, c, b, a), (p, bits) in res.items()]
    return {'enc': _S.enc, 'M': M, 'V': V, 'rep': rep, 'nq': len(g), 'fams': fams, 'rows': rows,
            'sec': round(time.time() - t0, 1)}


def dev(enc, fams, workers):
    out = f'{wdir(enc)}/dev_{enc}.jsonl'
    done = set()
    if os.path.exists(out):
        for l in open(out):
            r = json.loads(l)
            done.add((r['M'], r['V'], r['rep'], tuple(r['fams'])))
    jobs = [(M, V, rep, fams) for M in M_GRID for V in V_GRID for rep in range(DEV_REPS[M])
            if (M, V, rep, tuple(fams)) not in done]
    jobs.sort(key=lambda j: -j[0] * (1 + math.log2(j[1])))
    import multiprocessing as mp
    fo = open(out, 'a')
    if workers <= 1:
        _init(enc)
        it = map(_dev_one, jobs)
    else:
        pl = mp.get_context('fork').Pool(workers, initializer=_init, initargs=(enc,), maxtasksperchild=4)
        it = pl.imap_unordered(_dev_one, jobs)
    for r in it:
        fo.write(json.dumps(r) + '\n')
        fo.flush()
        print(f'dev {enc} {",".join(fams)} M={r["M"]} V={r["V"]} rep={r["rep"]} {r["sec"]}s', flush=True)


# ----------------------------------------------------------------------------------------------- choice

def grid_rank(fam, c, V):
    return grid(fam, V).index(tuple(c))


def choose(enc, b36_choice=None, out=None):
    """B36's rule per (M, V, bpf, acct) and family (highest mean dev accuracy over the dev stores; within 0.1 points
    tied, first in grid order); references per (M, V). BEST = the family with the highest dev accuracy of its chosen
    configuration among ALL10 (ties: ALL10 order); BESTNL the same without LAM; BEST36 copied from B36's choice."""
    acc, reps = {}, {}
    for l in open(f'{wdir(enc)}/dev_{enc}.jsonl'):
        r = json.loads(l)
        for x in r['rows']:
            k = (r['M'], r['V'], x['bpf'], x['acct'], x['fam'], tuple(x['cfg']))
            acc.setdefault(k, {})[r['rep']] = x['acc']
    best = {}
    for k, d in acc.items():
        M, V, bpf, acct, fam, cfg = k
        if len(d) != DEV_REPS[M]:
            raise SystemExit(f'incomplete dev results for {k}: {len(d)}')
        best.setdefault((M, V, bpf, acct, fam), []).append((float(np.mean(list(d.values()))), cfg))
    ch = {}
    for (M, V, bpf, acct, fam), lst in best.items():
        top = max(a for a, _ in lst)
        tied = sorted([c for a, c in lst if a >= top - 0.001], key=lambda c: grid_rank(fam, c, V))
        ch[f'{M}|{V}|{bpf}|{acct}|{fam}'] = {'cfg': list(tied[0]), 'dev_acc': [a for a, c in lst if c == tied[0]][0],
                                            'best_dev_acc': top, 'n_tied': len(tied), 'n_cfg': len(lst)}
    old = json.load(open(b36_choice or B36_DIRS[enc][1]))['choice']
    for M in M_GRID:
        for V in V_GRID:
            for bpf in BPF:
                for acct in ACCTS:
                    for f in OLD:
                        ch[f'{M}|{V}|{bpf}|{acct}|{f}'] = {k: old[f'{M}|{V}|{bpf}|{acct}|{f}'][k] for k in
                                                          ('cfg', 'dev_acc')}
                    ch[f'{M}|{V}|{bpf}|{acct}|VSA'] = {k: old[f'{M}|{V}|{bpf}|{acct}|VSA'][k] for k in ('cfg', 'dev_acc')}
                    for name, fl in (('BEST', ALL10), ('BESTNL', [f for f in ALL10 if f != 'LAM'])):
                        cands = [(ch[f'{M}|{V}|{bpf}|{acct}|{f}']['dev_acc'], -ALL10.index(f), f) for f in fl]
                        a, _, f = max(cands)
                        ch[f'{M}|{V}|{bpf}|{acct}|{name}'] = {'fam': f, 'dev_acc': a,
                                                             'dev_by_fam': {x[2]: x[0] for x in cands}}
                    ch[f'{M}|{V}|{bpf}|{acct}|BEST36'] = {'fam': old[f'{M}|{V}|{bpf}|{acct}|BEST']['fam']}
    fn = out or f'{ROOT}/results/b49_choice_{enc}.json'
    json.dump({'enc': enc, 'choice': ch}, open(fn, 'w'), indent=1, sort_keys=True)
    print(f'{len(ch)} choices written to {fn}')


# ----------------------------------------------------------------------------------------------- test

def _test_one(args):
    M, V, rep, sel = args
    st = C.make_store(_S.pool, 'test', M, V, rep)
    t0 = time.time()
    res = run_new(_S, st, list(sel['fams']), BPF, ACCTS, select=sel['cfgs'])
    rec = {'enc': _S.enc, 'M': M, 'V': V, 'rep': rep, 'qgold': st['qgold'].tolist(), 'pred': {}, 'cfg': {}, 'bits': {}}
    for key, (f, cfg) in sel['need'].items():
        parts = key.split('|')
        if f in REF:
            p, bits = res[f, tuple(cfg), None, None]
        else:
            p, bits = res[f, tuple(cfg), int(parts[1]), parts[2]]
        rec['pred'][key] = p.tolist()
        rec['cfg'][key] = list(cfg)
        rec['bits'][key] = bits
    rec['sec'] = round(time.time() - t0, 1)
    return rec


def test(enc, workers, choicefile=None):
    ch = json.load(open(choicefile or f'{ROOT}/results/b49_choice_{enc}.json'))
    assert ch['enc'] == enc
    ch = ch['choice']
    od = f'{wdir(enc)}/test'
    os.makedirs(od, exist_ok=True)
    jobs = []
    for M in M_GRID:
        for V in V_GRID:
            fn = f'{od}/new49_{enc}_M{M}_V{V}.jsonl.gz'
            if os.path.exists(fn):
                continue
            need, cfgs = {}, {}
            for f in NEW + SEC:
                for b in BPF:
                    for a in ACCTS:
                        c = tuple(ch[f'{M}|{V}|{b}|{a}|{f}']['cfg'])
                        need[f'{f}|{b}|{a}'] = (f, c)
                        cfgs.setdefault(f, set()).add(c)
            for f in REF:
                c = tuple(ch[f'{M}|{V}|None|None|{f}']['cfg'])
                need[f] = (f, c)
                cfgs.setdefault(f, set()).add(c)
            fams = set(NEW + SEC + ['REF'])
            sel = {'need': need, 'cfgs': cfgs, 'fams': fams}
            for rep in range(TEST_REPS[M]):
                jobs.append((M, V, rep, sel))
    import multiprocessing as mp
    if workers <= 1:
        _init(enc)
        it = map(_test_one, jobs)
    else:
        pl = mp.get_context('fork').Pool(workers, initializer=_init, initargs=(enc,), maxtasksperchild=4)
        it = pl.imap_unordered(_test_one, jobs)
    parts = {}
    for r in it:
        parts.setdefault((r['M'], r['V']), []).append(r)
        print(f'test {enc} M={r["M"]} V={r["V"]} rep={r["rep"]} {r["sec"]}s', flush=True)
        M, V = r['M'], r['V']
        if len(parts[M, V]) == TEST_REPS[M]:
            fn = f'{od}/new49_{enc}_M{M}_V{V}.jsonl.gz'
            with gzip.open(fn + '.tmp', 'wt') as fo:
                for x in sorted(parts[M, V], key=lambda x: x['rep']):
                    fo.write(json.dumps(x) + '\n')
            os.replace(fn + '.tmp', fn)


def merge(enc, choicefile=None, b36dir=None):
    """B36's merged record of every test store + the new families, BEST (all ten), BESTNL, BEST36, references."""
    ch = json.load(open(choicefile or f'{ROOT}/results/b49_choice_{enc}.json'))['choice']
    od = f'{wdir(enc)}/test'
    for M in M_GRID:
        for V in V_GRID:
            old = [json.loads(l) for l in gzip.open(f'{b36dir or B36_DIRS[enc][0]}/all_{b36name(enc)}_M{M}_V{V}.jsonl.gz',
                                                    'rt')]
            new = [json.loads(l) for l in gzip.open(f'{od}/new49_{enc}_M{M}_V{V}.jsonl.gz', 'rt')]
            with gzip.open(f'{od}/all49_{enc}_M{M}_V{V}.jsonl.gz', 'wt') as fo:
                for o, n in zip(old, new):
                    assert o['rep'] == n['rep'] and o['qgold'] == n['qgold'], (M, V, o['rep'])
                    rec = {k: o[k] for k in ('enc', 'M', 'V', 'rep', 'uris', 'vals', 'qent', 'qalias', 'qgold')}
                    pred, cfg, bits = dict(o['pred']), dict(o['cfg']), dict(o['bits'])
                    for k in n['pred']:
                        pred[k] = n['pred'][k]
                        cfg[k] = n['cfg'][k]
                        bits[k] = n['bits'][k]
                    for b in BPF:
                        for a in ACCTS:
                            pred[f'BEST36|{b}|{a}'] = o['pred'][f'BEST|{b}|{a}']
                            cfg[f'BEST36|{b}|{a}'] = o['cfg'][f'BEST|{b}|{a}']
                            for name in ('BEST', 'BESTNL'):
                                f = ch[f'{M}|{V}|{b}|{a}|{name}']['fam']
                                pred[f'{name}|{b}|{a}'] = pred[f'{f}|{b}|{a}']
                                cfg[f'{name}|{b}|{a}'] = [f] + list(cfg[f'{f}|{b}|{a}'])
                    rec.update({'pred': pred, 'cfg': cfg, 'bits': bits, 'keybits': o.get('keybits', {})})
                    fo.write(json.dumps(rec) + '\n')
    print('merged', enc)


# ----------------------------------------------------------------------------------------------- selftest

def selftest():
    rng = np.random.default_rng(0)
    # LAM: near-orthogonal keys, exact queries, a generous budget: every fact read; budgets respected
    M, V, n = 200, 16, 512
    Kk = rng.standard_normal((M, n)).astype(np.float32)
    Kk /= np.linalg.norm(Kk, axis=1, keepdims=True)
    vals = rng.integers(0, V, M)
    st = {'M': M, 'V': V, 'vals': vals}
    lf = B.LamFeat('selftest', 'center', Kk)
    for feat in ('rp', 'pca'):
        o = B.run_lam(st, Kk, Kk, lf, [(feat, 1e-4, 4, 'onehot'), (feat, 100.0, 1, 'onehot')], [64, 256], ACCTS, 1)
        for (cfg, b, a), (p, bits) in o.items():
            assert bits <= b * M, (cfg, b, a, bits)
        acc = float((o[(feat, 1e-4, 4, 'onehot'), 256, 'ideal'][0] == vals).mean())
        assert acc >= 0.95, (feat, acc)
    # code targets at V = 1,024
    V2 = 1024
    vals2 = rng.integers(0, V2, M)
    o = B.run_lam({'M': M, 'V': V2, 'vals': vals2}, Kk, Kk, lf, [('pca', 1e-4, 4, 'code128')], [256], ['ideal'], 1)
    acc2 = float((o[('pca', 1e-4, 4, 'code128'), 256, 'ideal'][0] == vals2).mean())
    assert acc2 >= 0.9, acc2
    # contrast coding: V = 2 sign symmetric
    T, D = B.value_targets(2, 'onehot', 0)
    assert np.allclose(T @ D, np.eye(2) - 0.5)
    print(f'selftest ok (LAM exact-key reads: {acc:.3f}; code128 at V = 1,024: {acc2:.3f})')


def main():
    enc, a = sys.argv[1], sys.argv[2:]
    os.chdir(f'{ROOT}/b49_out')
    os.makedirs(wdir(enc), exist_ok=True)
    wk = int(a[a.index('--workers') + 1]) if '--workers' in a else 1
    if a[0] == 'selftest':
        selftest()
    elif a[0] == 'prefit':
        S = Setup(enc, ENC_TAG[enc], SEED.get(enc, enc))
        S.prefit()
        print('prefit done', enc, flush=True)
    elif a[0] == 'dev':
        fams = a[a.index('--fam') + 1].split(',') if '--fam' in a else NEW + SEC + ['REF']
        dev(enc, fams, wk)
    elif a[0] == 'choose':
        choose(enc)
    elif a[0] == 'test':
        test(enc, wk)
    elif a[0] == 'merge':
        merge(enc)
    else:
        raise SystemExit(a[0])


if __name__ == '__main__':
    main()
