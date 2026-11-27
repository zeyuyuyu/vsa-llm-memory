"""B36 (prereg/B36_keyfree_equal_bits_prereg.md): key-free classical memories (static-function retrieval, LSH bucket
sketch, Bloom filter, fly Bloom filter, sparse distributed memory) against B32's VSA store and per-fact index, at equal
bits, on B32's dev and test stores.

python3 b36_run.py selftest
python3 b36_run.py dev    ENC OUT.jsonl [--fam SF,CM,BF,FBF,SDM] [--workers N] [--stores M:V:rep,...]
python3 b36_run.py dev32  ENC OUTPREFIX          # B32's VSA and index grids at 1 and 2 bits per fact (B32 code)
python3 b36_run.py choose ENC                    # per cell and family: best dev mean; BEST competitor per cell
python3 b36_run.py test   ENC CHOICE.json OUTDIR [--workers N]
python3 b36_run.py test32 ENC OUTDIR             # B32's VSA / index chosen configurations at 1 and 2 bits per fact
python3 b36_run.py merge  ENC OUTDIR             # -> OUTDIR/all_{enc}_M{M}_V{V}.jsonl.gz (every memory, every cell)
"""
import gzip
import itertools
import json
import math
import os
import sys
import time

import numpy as np

import b32_common as C
import b36_common as K

ENCS = {'qwen8': 'qwen8_mean_L9_f16', 'e5': 'e5_base'}
M_GRID, V_GRID = [256, 1024], [2, 16, 128, 1024]
BPF = [1, 2, 4, 8, 16, 32, 64]
BPF_NEW = [1, 2]          # B32's grids are run at these; B32's own runs cover 4-64
ACCTS = ['ideal', 'real']  # B32's 'raw' is the 'real' accounting of the VSA and the index
PREPS = ['center', 'whiten']
DEV_REPS = {256: 6, 1024: 3}
TEST_REPS = {256: 20, 1024: 5}
FAMS = ['SF', 'CM', 'BF', 'FBF', 'SDM']
PQCACHE = 'b36_out/pqcache'

GRID = {
    'SF': [(h, k, fp, T) for h, ks in (('sim', [6, 8, 10, 12, 14, 16]), ('pq', [4, 8, 12, 16, 20]))
           for k in ks for fp in [0, 1, 2, 4] for T in [1, 4]],
    'CM': [(h, k, ct, T) for h, ks in (('sim', [2, 4, 6, 8, 10, 12, 14]), ('pq', [4, 8, 12, 16]))
           for k in ks for ct in ['plural', 'pres', 'count'] for T in [1, 4]],
    'BF': [(h, k, L) for h, ks in (('sim', [8, 12, 16, 20]), ('pq', [8, 16])) for k in ks for L in [1, 2, 4, 8, 16]],
    'FBF': [(rho,) for rho in [1, 2, 4, 8, 16, 32]],
    'SDM': [(p, nd, q) for p in [0.003, 0.01, 0.03, 0.1, 0.3] for nd in [1, 8] for q in ['tern', 'cnt']],
}


def fam_grid(fam, V):
    g = GRID[fam]
    if fam == 'CM' and V > 16:
        g = [c for c in g if c[2] == 'plural']
    return g


# ----------------------------------------------------------------------------------------------- per store

class Setup:
    def __init__(self, enc):
        self.enc = enc
        self.pool = C.Pool(ENCS[enc])
        Tr = self.pool.train_matrix()
        self.preps = {p: C.Prep(p, Tr) for p in PREPS}
        self.ppool = K.Pool(enc, self.pool.X.shape[1])
        os.makedirs(PQCACHE, exist_ok=True)
        self.banks = {p: K.PQBank(self.ppool, self.preps[p](Tr), f'{enc}_{p}', PQCACHE) for p in PREPS}


def store_order(st):
    return np.random.default_rng(C.seed_of('subset', st['split'], st['M'], st['V'], st['rep'])).permutation(st['M'])


TIMES = {}


def run_store(S, st, fams, bpfs, accts, select=None):
    """Every configuration of the given families (or, with select = {fam: set of (prep, cfg)}, only those) on one
    store. Returns {(fam, (prep,) + cfg, bpf, acct): (pred, bits)}."""
    M, V = st['M'], st['V']
    tag = f'{S.enc}|{st["split"]}|{M}|{V}|{st["rep"]}'
    order = store_order(st)
    Bmax = max(bpfs) * M
    res = {}
    for prep in PREPS:
        cf = {f: [c for c in fam_grid(f, V) if select is None or (prep,) + tuple(c) in select.get(f, set())]
              for f in fams}
        if not any(cf.values()):
            continue
        Y = S.preps[prep](np.vstack([st['keys'], st['queries']]))
        P = S.ppool.project(Y, K.POOL)
        bank = S.banks[prep]
        ptag = f'{tag}|{prep}'
        for f in fams:
            if not cf[f]:
                continue
            t0 = time.time()
            if f == 'SF':
                o = K.run_sf(st, P, bank, cf[f], bpfs, accts, ptag, order)
            elif f == 'CM':
                o = K.run_cm(st, P, bank, cf[f], bpfs, accts, ptag)
            elif f == 'BF':
                o = K.run_bf(st, P, bank, cf[f], bpfs, accts, ptag)
            elif f == 'FBF':
                o = K.run_fbf(st, P, cf[f], bpfs, accts, ptag)
            elif f == 'SDM':
                o = K.run_sdm(st, Y, P, S.ppool, cf[f], bpfs, accts, S.enc)
            for (cfg, bpf, acct), v in o.items():
                res[f, (prep,) + tuple(cfg), bpf, acct] = v
            TIMES[f] = TIMES.get(f, 0.0) + time.time() - t0
        del P
    return res


# ----------------------------------------------------------------------------------------------- dev

_S = None


def _init(enc):
    global _S
    _S = Setup(enc)


def _dev_one(args):
    M, V, rep, fams = args
    st = C.make_store(_S.pool, 'dev', M, V, rep)
    t0 = time.time()
    TIMES.clear()
    res = run_store(_S, st, fams, BPF, ACCTS)
    g = st['qgold']
    rows = [{'fam': f, 'cfg': list(cfg), 'bpf': b, 'acct': a, 'acc': float((p == g).mean()), 'bits': float(bits)}
            for (f, cfg, b, a), (p, bits) in res.items()]
    return {'enc': _S.enc, 'M': st['M'], 'V': V, 'rep': rep, 'nq': len(g), 'fams': fams, 'rows': rows,
            'sec': round(time.time() - t0, 1), 'sec_fam': {k: round(v, 1) for k, v in TIMES.items()}}


def dev(enc, out, fams, workers, stores=None):
    done = set()
    if os.path.exists(out):
        for l in open(out):
            r = json.loads(l)
            done.add((r['M'], r['V'], r['rep'], tuple(r['fams'])))
    jobs = [(M, V, rep, fams) for M in M_GRID for V in V_GRID for rep in range(DEV_REPS[M])]
    if stores:
        jobs = [j for j in jobs if f'{j[0]}:{j[1]}:{j[2]}' in stores]
    jobs = [j for j in jobs if (j[0], j[1], j[2], tuple(fams)) not in done]
    jobs.sort(key=lambda j: -j[0] * (1 + math.log2(j[1])))  # large stores first
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
        print(f'dev {enc} {",".join(fams)} M={r["M"]} V={r["V"]} rep={r["rep"]} {r["sec"]}s {r.get("sec_fam")}', flush=True)


def dev32(enc, prefix):
    """B32's VSA and index dev grids at 1 and 2 bits per fact, by B32's own code (b32_run.dev) with its bpf list
    replaced; same dev stores, same fixed projections, same index fits."""
    import b32_run as R
    R.GRID['bpf'] = list(BPF_NEW)
    R.dev(enc, f'{prefix}_vsa.jsonl', 'vsa', 'cpu')
    R.dev(enc, f'{prefix}_idx.jsonl', 'idx', 'cpu')


# ----------------------------------------------------------------------------------------------- choice

FAM_ORDER = ['IDX', 'SF', 'CM', 'BF', 'FBF', 'SDM']


def grid_rank(fam, cfg):
    return (PREPS.index(cfg[0]), GRID[fam].index(tuple(cfg[1:])))


def choose(enc):
    """Per (M, V, bpf, acct) and family: the configuration with the highest mean dev accuracy over the dev stores
    (ties within 0.1 points: first in grid order). VSA and IDX: B32's choices (bpf >= 4) and B32's rule on the new
    1-2 bits-per-fact dev grid. BEST: per cell, the competitor family (IDX, SF, CM, BF, FBF, SDM) with the highest
    dev accuracy of its chosen configuration."""
    acc, reps = {}, {}
    for l in open(f'b36_out/dev_{enc}.jsonl'):
        r = json.loads(l)
        for x in r['rows']:
            k = (r['M'], r['V'], x['bpf'], x['acct'], x['fam'], tuple(x['cfg']))
            acc.setdefault(k, {})[r['rep']] = x['acc']
            reps.setdefault((r['M'], r['V'], x['fam']), set()).add(r['rep'])
    best = {}
    for k, d in acc.items():
        M, V, bpf, acct, fam, cfg = k
        if len(d) != DEV_REPS[M]:
            raise SystemExit(f'incomplete dev results for {k}: {len(d)}')
        best.setdefault((M, V, bpf, acct, fam), []).append((float(np.mean(list(d.values()))), cfg))
    out = {}
    for (M, V, bpf, acct, fam), lst in best.items():
        top = max(a for a, _ in lst)
        tied = sorted([c for a, c in lst if a >= top - 0.001], key=lambda c: grid_rank(fam, c))
        out[f'{M}|{V}|{bpf}|{acct}|{fam}'] = {'cfg': list(tied[0]), 'dev_acc': [a for a, c in lst if c == tied[0]][0],
                                             'best_dev_acc': top, 'n_tied': len(tied), 'n_cfg': len(lst)}
    # B32 arms: bpf >= 4 from B32's committed choice; 1-2 from B32's rule on the new grid
    import b32_run as R
    R.GRID['bpf'] = list(BPF_NEW)
    tmp = f'b36_out/choice32_{enc}.json'
    R.choose(enc, f'b36_out/dev32_{enc}_vsa.jsonl,b36_out/dev32_{enc}_idx.jsonl', tmp)
    c32new = json.load(open(tmp))['choice']
    c32old = json.load(open(f'results/b32_choice_{enc}.json'))['choice']
    amap = {'ideal': 'ideal', 'real': 'raw'}
    for M in M_GRID:
        for V in V_GRID:
            for bpf in BPF:
                for acct in ACCTS:
                    src = c32new if bpf in BPF_NEW else c32old
                    for arm in ('VSA', 'IDX'):
                        e = src[f'{M}|{V}|{bpf}|{amap[acct]}|{arm}']
                        out[f'{M}|{V}|{bpf}|{acct}|{arm}'] = {'cfg': e['cfg'], 'dev_acc': e['dev_acc'],
                                                             'best_dev_acc': e['best_dev_acc'], 'n_tied': e['n_tied'],
                                                             'source': 'b32' if bpf not in BPF_NEW else 'b36dev32'}
                    cands = [(out[f'{M}|{V}|{bpf}|{acct}|{f}']['dev_acc'], -FAM_ORDER.index(f), f) for f in FAM_ORDER
                             if f'{M}|{V}|{bpf}|{acct}|{f}' in out]
                    a, _, f = max(cands)
                    out[f'{M}|{V}|{bpf}|{acct}|BEST'] = {'fam': f, 'dev_acc': a,
                                                        'dev_by_fam': {x[2]: x[0] for x in cands}}
            e = c32old[f'{M}|{V}|None|None|EXACT']
            out[f'{M}|{V}|EXACT'] = e
    fn = f'results/b36_choice_{enc}.json'
    json.dump({'enc': enc, 'choice': out}, open(fn, 'w'), indent=1, sort_keys=True)
    print(f'{len(out)} choices written to {fn}')


# ----------------------------------------------------------------------------------------------- test

def _test_one(args):
    M, V, rep, sel, outdir = args
    st = C.make_store(_S.pool, 'test', M, V, rep)
    t0 = time.time()
    res = run_store(_S, st, FAMS, BPF, ACCTS, select=sel['cfgs'])
    rec = {'enc': _S.enc, 'M': st['M'], 'V': V, 'rep': rep, 'qgold': st['qgold'].tolist(), 'pred': {}, 'cfg': {},
           'bits': {}}
    for key, (f, cfg) in sel['need'].items():  # key = f'{fam}|{bpf}|{acct}'
        _, b, a = key.split('|')
        p, bits = res[f, tuple(cfg), int(b), a]
        rec['pred'][key] = p.tolist()
        rec['cfg'][key] = list(cfg)
        rec['bits'][key] = bits
    rec['sec'] = round(time.time() - t0, 1)
    return rec


def test(enc, choicefile, outdir, workers):
    ch = json.load(open(choicefile))
    assert ch['enc'] == enc
    ch = ch['choice']
    os.makedirs(outdir, exist_ok=True)
    jobs = []
    for M in M_GRID:
        for V in V_GRID:
            fn = f'{outdir}/new_{enc}_M{M}_V{V}.jsonl.gz'
            if os.path.exists(fn):
                continue
            need, cfgs = {}, {}
            for f in FAMS:
                for b in BPF:
                    for a in ACCTS:
                        c = tuple(ch[f'{M}|{V}|{b}|{a}|{f}']['cfg'])
                        need[f'{f}|{b}|{a}'] = (f, c)
                        cfgs.setdefault(f, set()).add(c)
            sel = {'need': need, 'cfgs': cfgs}
            for rep in range(TEST_REPS[M]):
                jobs.append((M, V, rep, sel, outdir))
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
            fn = f'{outdir}/new_{enc}_M{M}_V{V}.jsonl.gz'
            with gzip.open(fn + '.tmp', 'wt') as fo:
                for x in sorted(parts[M, V], key=lambda x: x['rep']):
                    fo.write(json.dumps(x) + '\n')
            os.replace(fn + '.tmp', fn)


def test32(enc, outdir):
    """B32's chosen VSA / index configurations at 1 and 2 bits per fact on the test stores (b32_run.test, same code)."""
    import b32_run as R
    R.GRID['bpf'] = list(BPF_NEW)
    ch = json.load(open(f'results/b36_choice_{enc}.json'))['choice']
    amap = {'ideal': 'ideal', 'real': 'raw'}
    c = {}
    for M in M_GRID:
        for V in V_GRID:
            for bpf in BPF_NEW:
                for acct in ACCTS:
                    for arm in ('VSA', 'IDX'):
                        c[f'{M}|{V}|{bpf}|{amap[acct]}|{arm}'] = {'cfg': ch[f'{M}|{V}|{bpf}|{acct}|{arm}']['cfg']}
            c[f'{M}|{V}|None|None|EXACT'] = ch[f'{M}|{V}|EXACT']
    fn = f'{outdir}/choice32_{enc}.json'
    json.dump({'enc': enc, 'choice': c}, open(fn, 'w'))
    R.test(enc, fn, f'{outdir}/b32arms', 'vsa', 'cpu')
    R.test(enc, fn, f'{outdir}/b32arms', 'idx', 'cpu')


def merge(enc, outdir):
    """One record per test store with every memory's predictions in every cell: B32's VSA / IDX / EXACT (bpf >= 4
    from b32_out/test, 1-2 from OUTDIR/b32arms), the new families, and BEST (the dev-chosen competitor family)."""
    ch = json.load(open(f'results/b36_choice_{enc}.json'))['choice']
    amap = {'ideal': 'ideal', 'real': 'raw'}
    for M in M_GRID:
        for V in V_GRID:
            old = [json.loads(l) for l in gzip.open(f'b32_out/test/preds_{enc}_M{M}_V{V}.jsonl.gz', 'rt')]
            a32 = [json.loads(l) for l in gzip.open(f'{outdir}/b32arms/idx_{enc}_M{M}_V{V}.jsonl.gz', 'rt')]
            v32 = [json.loads(l) for l in gzip.open(f'{outdir}/b32arms/vsa_{enc}_M{M}_V{V}.jsonl.gz', 'rt')]
            new = [json.loads(l) for l in gzip.open(f'{outdir}/new_{enc}_M{M}_V{V}.jsonl.gz', 'rt')]
            with gzip.open(f'{outdir}/all_{enc}_M{M}_V{V}.jsonl.gz', 'wt') as fo:
                for o, x, y, n in zip(old, a32, v32, new):
                    assert o['rep'] == x['rep'] == y['rep'] == n['rep'] and o['qgold'] == n['qgold'] == x['qgold']
                    rec = {k: o[k] for k in ('enc', 'M', 'V', 'rep', 'uris', 'vals', 'qent', 'qalias', 'qgold')}
                    pred, cfg, bits = {}, {}, {}
                    pred['EXACT'] = o['pred']['EXACT']
                    pred['EXACT_ENT'] = o['pred']['EXACT_ENT']
                    for b in BPF:
                        for a in ACCTS:
                            for arm in ('VSA', 'IDX'):
                                src = o if b not in BPF_NEW else (y if arm == 'VSA' else x)
                                pred[f'{arm}|{b}|{a}'] = src['pred'][f'{arm}|{b}|{amap[a]}']
                                cfg[f'{arm}|{b}|{a}'] = src['cfg'][f'{arm}|{b}|{amap[a]}']
                            for f in FAMS:
                                pred[f'{f}|{b}|{a}'] = n['pred'][f'{f}|{b}|{a}']
                                cfg[f'{f}|{b}|{a}'] = n['cfg'][f'{f}|{b}|{a}']
                                bits[f'{f}|{b}|{a}'] = n['bits'][f'{f}|{b}|{a}']
                            bf = ch[f'{M}|{V}|{b}|{a}|BEST']['fam']
                            pred[f'BEST|{b}|{a}'] = pred[f'{bf}|{b}|{a}']
                            cfg[f'BEST|{b}|{a}'] = [bf] + list(cfg[f'{bf}|{b}|{a}'])
                    rec.update({'pred': pred, 'cfg': cfg, 'bits': bits})
                    rec['keybits'] = {**o.get('keybits', {}), **x.get('keybits', {})}
                    fo.write(json.dumps(rec) + '\n')
    print('merged', enc)


# ----------------------------------------------------------------------------------------------- selftest

def selftest():
    rng = np.random.default_rng(0)
    # 1. ribbon: every stored key reads its word exactly; keys outside read ~uniform words
    for n in (1, 5, 64, 300, 1024):
        keys = np.unique(rng.integers(0, 2 ** 40, size=n))
        words = rng.integers(0, 2 ** 14, size=len(keys))
        rb = K.Ribbon(keys, words, ('selftest', n))
        assert np.array_equal(rb.query(keys).astype(np.int64), words), n
        assert rb.m <= max(math.ceil(len(keys) * 1.03) + 4, K.W + 1) * 1.2, (n, rb.m)
    other = rng.integers(2 ** 41, 2 ** 42, size=20000)
    o = rb.query(other).astype(np.int64) & 1
    assert abs(o.mean() - 0.5) < 0.02, o.mean()
    # 2. families on a synthetic store: budgets respected; exact queries of a small store are read
    M, V, n = 256, 16, 64
    keys = rng.standard_normal((M, n)).astype(np.float32)
    vals = rng.integers(0, V, M)
    qi = rng.integers(0, M, 200)
    st = {'split': 'selftest', 'M': M, 'V': V, 'rep': 0, 'keys': keys, 'vals': vals,
          'queries': keys[qi] + 0.05 * rng.standard_normal((200, n)).astype(np.float32), 'qgold': vals[qi],
          'qent': qi}

    class S0:
        enc = 'selftest'
        preps = {p: (lambda X: X / np.linalg.norm(X, axis=1, keepdims=True)) for p in PREPS}
        ppool = K.Pool('selftest', n)
        banks = {p: K.PQBank(K.Pool('selftest', n), keys / np.linalg.norm(keys, axis=1, keepdims=True), 'st')
                 for p in PREPS}
    sel = {'SF': {('center', 'sim', 10, 4, 1)}, 'CM': {('center', 'sim', 10, 'plural', 1)},
           'BF': {('center', 'sim', 12, 4)}, 'FBF': {('center', 8)}, 'SDM': {('center', 0.03, 1, 'tern')}}
    res = run_store(S0, st, FAMS, [4, 16, 64], ACCTS, select=sel)
    for (f, cfg, b, a), (p, bits) in res.items():
        assert bits <= b * M + 1e-6, (f, cfg, b, a, bits)
    accs = {(f, b, a): float((p == st['qgold']).mean()) for (f, cfg, b, a), (p, bits) in res.items()}
    for f in FAMS:
        assert accs[f, 64, 'ideal'] > 0.5, (f, accs[f, 64, 'ideal'])
    print('selftest ok:', {k: round(v, 3) for k, v in sorted(accs.items()) if k[2] == 'ideal'})


if __name__ == '__main__':
    a = sys.argv[1:]
    wk = int(a[a.index('--workers') + 1]) if '--workers' in a else 1
    if a[0] == 'selftest':
        selftest()
    elif a[0] == 'dev':
        fams = a[a.index('--fam') + 1].split(',') if '--fam' in a else FAMS
        stores = a[a.index('--stores') + 1].split(',') if '--stores' in a else None
        dev(a[1], a[2], fams, wk, stores)
    elif a[0] == 'dev32':
        dev32(a[1], a[2])
    elif a[0] == 'choose':
        choose(a[1])
    elif a[0] == 'test':
        test(a[1], a[2], a[3], wk)
    elif a[0] == 'test32':
        test32(a[1], a[2])
    elif a[0] == 'merge':
        merge(a[1], a[2])
    elif a[0] == 'pqfit':  # fit and cache every PQ sub-quantizer of the pool (training split only)
        S = Setup(a[1])
        for p in PREPS:
            t0 = time.time()
            if S.banks[p].cent is None:
                S.banks[p].fit()
            print('pq fitted', a[1], p, f'{time.time() - t0:.0f}s', flush=True)
