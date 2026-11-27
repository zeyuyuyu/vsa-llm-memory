"""B66 part (A) (prereg/B66_reconciliation_prereg.md), pipeline side.

(A2) B64's code restricted to the clean room's grid: SDMt configurations (prep, 'LA', a in {1, 4, 16, 64}, ctr, 'cnt',
     share in {1, 1/2}) = 32 (the clean room's data-address SDM grid); B36's rule on B64's existing dev rows restricted
     to them, tie order = the clean room's order (prep > a > ctr > share); test = b64_run.run_job on the chosen
     configurations (b66_out/b64cr/jobs).  BEST7cr = B36's BEST rule over the six (B49's dev accuracies) + SDMTcr.
(A3) Random-map replicates m = 1, 2, 3 (B57 / B60's tag '{enc}#b57m{m}'):
     VSARm{m}  the first design (B49's store: b32_run.Feat, random value codes) with the map's projections, value codes,
               store seeds (R.store_seed(tag, st)), B32's 72-configuration dev grid on R1 budgets and B32's rule
               (b57_run.b32_choose's algorithm); test = the chosen configuration (b66_out/vsarm).
     SDMTm{m}  SDMt whose only random component, the fixed location order, is re-drawn: seed
               C.seed_of('B64LOC', loc, '{enc}#b57m{m}') instead of C.seed_of('B64LOC', loc); B64's full 504 grid on dev,
               B36's rule, test = the chosen configuration (b66_out/sdmtm{m}/jobs, b64_run.run_job unchanged).
     VSAm{m}   the final design per map: B57 / B59's existing predictions (merged records).
     BEST7m{m} B36's BEST rule over the six (B49's dev accuracies, fixed: their maps are not re-drawn) + SDMTm{m}.

python3 b66_run.py crgrid ENC                  # -> results/b66_choice_crgrid_{enc}.json
python3 b66_run.py plan crtest [--encs ..]     # -> b66_out/b64cr/plan_test.jsonl
python3 b66_run.py plan sdmtdev MAP [--encs ..] | plan sdmttest MAP [--encs ..]
python3 b66_run.py runb64 PLAN --out DIR --map MAP --workers N     # b64_run.runplan with OUT = DIR, map patch
python3 b66_run.py sdmtchoose ENC MAP          # -> results/b66_choice_sdmtm{MAP}_{enc}.json
python3 b66_run.py vsardev ENC MAP PREP:FMAP   # first design, map MAP, one (prep, fmap) part, dev stores
python3 b66_run.py vsarchoose ENC MAP          # -> results/b66_choice_vsarm{MAP}_{enc}.json
python3 b66_run.py vsartest ENC MAP            # test stores, chosen configurations
python3 b66_run.py selftest
"""
import gzip
import json
import math
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import b32_common as C  # noqa: E402

OUT = os.environ.get('B66_OUT', os.path.join(ROOT, 'b66_out'))
RES = os.environ.get('B66_RES', os.path.join(ROOT, 'results'))
B64JOBS = os.environ.get('B64_JOBS_SRC', os.path.join(ROOT, 'b64_out', 'jobs'))
PANEL = ['qwen8', 'b46', 'llama', 'mistral', 'e5', 'bge', 'minilm']
M_GRID, VN, BPF, ACCTS = [256, 1024], [2, 16], [4, 8, 16], ['ideal', 'real']
DEV_REPS = {256: 6, 1024: 3}
TEST_REPS = {256: 20, 1024: 5}
MAPS = [1, 2, 3]
SIX = ['IDX', 'SF', 'CM', 'BF', 'FBF', 'SDM']


def tag(enc, m):
    return enc if m == 0 else f'{enc}#b57m{m}'


def cr_grid():
    """The clean room's data-address SDM grid in B64's configuration space, the clean room's order."""
    return [(p, 'LA', a, c, 'cnt', s) for p in ('center', 'whiten') for a in (1, 4, 16, 64) for c in ('onehot', 'bits')
            for s in (1.0, 0.5)]


def b64_dev_acc(enc, jobs=B64JOBS):
    acc = {}
    for fn in sorted(os.listdir(jobs)):
        if not fn.startswith(f'{enc}_dev_'):
            continue
        r = json.load(gzip.open(os.path.join(jobs, fn), 'rt'))
        s = r['spec']
        for x in r['rows']:
            acc.setdefault((s['M'], s['V'], x['bpf'], x['acct'], tuple(x['cfg'])), {})[s['rep']] = x['acc']
    return acc


def b36_choose(acc, grid, M, V, b, a):
    lst = []
    for cfg in grid:
        d = acc.get((M, V, b, a, tuple(cfg)))
        if d is None or len(d) != DEV_REPS[M]:
            raise SystemExit(f'missing / incomplete dev rows {M} {V} {b} {a} {cfg}')
        lst.append((float(np.mean([d[k] for k in sorted(d)])), tuple(cfg)))
    top = max(x for x, _ in lst)
    tied = [c for x, c in lst if x >= top - 0.001]
    return tied[0], [x for x, c in lst if c == tied[0]][0], top, len(tied)


def best7(enc, dacc, M, V, b, a, name):
    ch49 = json.load(open(f'{ROOT}/results/b49_choice_{enc}.json'))['choice']
    dbf = {f: ch49[f'{M}|{V}|{b}|{a}|BEST']['dev_by_fam'][f] for f in SIX}
    b36 = ch49[f'{M}|{V}|{b}|{a}|BEST36']['fam']
    c6 = [(dbf[f], -SIX.index(f), f) for f in SIX]
    assert max(c6)[2] == b36
    dbf[name] = dacc
    fams = SIX + [name]
    c7 = [(dbf[f], -fams.index(f), f) for f in fams]
    return {'fam': max(c7)[2], 'dev_acc': max(c7)[0], 'dev_by_fam': dbf, 'BEST36': b36}


def crgrid(enc):
    acc = b64_dev_acc(enc)
    ch = {}
    for M in M_GRID:
        for V in VN:
            for b in BPF:
                for a in ACCTS:
                    cfg, dacc, top, nt = b36_choose(acc, cr_grid(), M, V, b, a)
                    ch[f'{M}|{V}|{b}|{a}|SDMTCR'] = {'cfg': list(cfg), 'dev_acc': dacc, 'best_dev_acc': top, 'n_tied': nt}
                    ch[f'{M}|{V}|{b}|{a}|BEST7CR'] = best7(enc, dacc, M, V, b, a, 'SDMTCR')
    fn = f'{RES}/b66_choice_crgrid_{enc}.json'
    json.dump({'enc': enc, 'choice': ch}, open(fn, 'w'), indent=1, sort_keys=True)
    print(len(ch), 'choices ->', fn)


# ----------------------------------------------------------------------------------------------- SDMt maps

def loc_rows_m(pool, m, enc):
    """b64_run.loc_rows with the order seed of map m (m = 0 reproduces B64)."""
    import b64_run as S
    lab, ali = [], []
    for e in pool.by_split['train']:
        lab.append(pool.idx[e['name']])
        ali += [pool.idx[a] for a in e['aliases']]
    bad = S.other_rows(pool)
    badc = {pool.X[r].tobytes() for r in bad}
    out = {}
    for loc, rows in (('LA', lab + ali), ('L', lab), ('A', ali)):
        u, seen = [], set()
        for r in rows:
            if r not in seen and r not in bad:
                seen.add(r)
                u.append(r)
        sd = C.seed_of('B64LOC', loc) if m == 0 else C.seed_of('B64LOC', loc, tag(enc, m))
        o = np.array(u)[np.random.default_rng(sd).permutation(len(u))]
        keep = np.array([pool.X[r].tobytes() not in badc for r in o], bool)
        out[loc] = o[keep]
    return out


def patch_b64(m):
    import b64_run as S

    def setup(enc):
        if enc not in S._SET:
            cwd = os.getcwd()
            os.chdir(S.DATA)
            pool = C.Pool(S.ENC_TAG[enc])
            os.chdir(cwd)
            Tr = pool.train_matrix()
            preps = {p: C.Prep(p, Tr) for p in S.PREPS}
            lr = loc_rows_m(pool, m, enc)
            Ls = {p: {l: preps[p](pool.X[lr[l]]) for l in S.LOCS} for p in S.PREPS}
            S._SET.clear()
            S._SET[enc] = (pool, preps, Ls, lr)
        return S._SET[enc]
    S.setup = setup
    return S


def plan(kind, encs, m=None):
    if kind == 'crtest':
        d = os.path.join(OUT, 'b64cr')
    else:
        d = os.path.join(OUT, f'sdmtm{m}')
    os.makedirs(d, exist_ok=True)
    fn = os.path.join(d, f'plan_{"dev" if kind == "sdmtdev" else "test"}.jsonl')
    n = 0
    with open(fn, 'w') as fo:
        for enc in encs:
            ch = None
            if kind == 'crtest':
                ch = json.load(open(f'{RES}/b66_choice_crgrid_{enc}.json'))['choice']
                key = 'SDMTCR'
            elif kind == 'sdmttest':
                ch = json.load(open(f'{RES}/b66_choice_sdmtm{m}_{enc}.json'))['choice']
                key = 'SDMT'
            for V in VN:
                for M in M_GRID:
                    reps = DEV_REPS[M] if kind == 'sdmtdev' else TEST_REPS[M]
                    cfgs = None
                    if ch is not None:
                        cfgs = [[ch[f'{M}|{V}|{b}|{a}|{key}']['cfg'], b, a] for b in BPF for a in ACCTS]
                    for rep in range(reps):
                        spec = {'enc': enc, 'split': 'dev' if kind == 'sdmtdev' else 'test', 'M': M, 'V': V,
                                'rep': rep, 'cfgs': cfgs}
                        fo.write(json.dumps(spec) + '\n')
                        n += 1
    print(n, 'jobs ->', fn)


def sdmtchoose(enc, m):
    import b64_run as S
    acc = b64_dev_acc(enc, os.path.join(OUT, f'sdmtm{m}', 'jobs'))
    ch = {}
    for M in M_GRID:
        for V in VN:
            for b in BPF:
                for a in ACCTS:
                    cfg, dacc, top, nt = b36_choose(acc, S.GRID, M, V, b, a)
                    ch[f'{M}|{V}|{b}|{a}|SDMT'] = {'cfg': list(cfg), 'dev_acc': dacc, 'best_dev_acc': top, 'n_tied': nt}
                    ch[f'{M}|{V}|{b}|{a}|BEST7'] = best7(enc, dacc, M, V, b, a, 'SDMT')
    fn = f'{RES}/b66_choice_sdmtm{m}_{enc}.json'
    json.dump({'enc': enc, 'map': m, 'choice': ch}, open(fn, 'w'), indent=1, sort_keys=True)
    print(len(ch), 'choices ->', fn)


# ----------------------------------------------------------------------------------------------- first design per map

PARTS = [(p, f) for p in ('center', 'whiten') for f in ('lin', 'sgn', 'sgn3')]
QS32, FS32 = [1, 2, 4], [1.0, 0.5, 0.25, 0.125]
ACCT32 = ['ideal', 'raw']


def vdir():
    d = os.path.join(OUT, 'vsarm')
    os.makedirs(d, exist_ok=True)
    return d


_STORE = {}


def _setup_store(enc):
    """(pool, preps) as b53_run.setup / b64_run.setup build them (C.Pool on the panel codes, b32_common.Prep fitted on
    the training split), from B64's DATA directory (b49_out/b32_data: every panel code, the md5s of B49)."""
    import b64_run as S
    if enc not in _STORE:
        cwd = os.getcwd()
        os.chdir(S.DATA)
        pool = C.Pool(S.ENC_TAG[enc])
        os.chdir(cwd)
        Tr = pool.train_matrix()
        _STORE.clear()
        _STORE[enc] = (pool, {p: C.Prep(p, Tr) for p in ('center', 'whiten')})
    return _STORE[enc]


def vsardev(enc, m, part):
    import b32_run as R
    prep, fmap = part.split(':')
    pool, preps = _setup_store(enc)
    n = pool.X.shape[1]
    tg = tag(enc, m)
    out = os.path.join(vdir(), f'dev_{enc}_m{m}_{prep}_{fmap}.jsonl')
    done = set()
    if os.path.exists(out):
        done = {(r['M'], r['V'], r['rep']) for r in map(json.loads, open(out))}
    fo = open(out, 'a')
    feat = R.Feat(tg, prep, fmap, n, 'cpu')
    for M in M_GRID:
        for V in VN:
            for rep in range(DEV_REPS[M]):
                if (M, V, rep) in done:
                    continue
                t0 = time.time()
                st = C.make_store(pool, 'dev', M, V, rep)
                cfgs = [((bpf, acct, q, f), C.vsa_dim(bpf * M, q, acct), q, max(1, int(math.floor(f * M))))
                        for bpf in BPF for acct in ACCT32 for q in QS32 for f in FS32]
                P = preps[prep]
                pr = R.vsa_grid(P(st['keys']), st['vals'], P(st['queries']), V, feat, cfgs, R.store_seed(tg, st),
                                R.store_order(st))
                g = st['qgold']
                rows = [{'arm': 'VSAR', 'cfg': [prep, fmap, q, f], 'bpf': bpf, 'acct': acct,
                         'acc': float((p == g).mean())} for (bpf, acct, q, f), p in pr.items()]
                fo.write(json.dumps({'enc': enc, 'map': m, 'M': M, 'V': V, 'rep': rep, 'nq': len(g), 'rows': rows}) + '\n')
                fo.flush()
                print(f'vsardev {enc} m{m} {part} M={M} V={V} rep={rep} {time.time() - t0:.0f}s', flush=True)


def vsarchoose(enc, m):
    """B32's rule (b57_run.b32_choose's algorithm): highest mean dev accuracy; within 0.1 points tied, first in the order
    prep > fmap > q > f."""
    acc = {}
    for p, f in PARTS:
        for l in open(os.path.join(vdir(), f'dev_{enc}_m{m}_{p}_{f}.jsonl')):
            r = json.loads(l)
            for x in r['rows']:
                acc.setdefault((r['M'], r['V'], x['bpf'], x['acct'], tuple(x['cfg'])), {})[r['rep']] = x['acc']
    best = {}
    for (M, V, bpf, acct, cfg), d in acc.items():
        assert len(d) == DEV_REPS[M], (M, V, bpf, acct, cfg, len(d))
        best.setdefault((M, V, bpf, acct), []).append((float(np.mean(list(d.values()))), cfg))
    key = (lambda c: (['center', 'whiten'].index(c[0]), ['lin', 'sgn', 'sgn3'].index(c[1]), QS32.index(c[2]),
                      FS32.index(c[3])))
    ch = {}
    for (M, V, bpf, acct), lst in best.items():
        assert len(lst) == 72, (M, V, bpf, acct, len(lst))
        top = max(a for a, _ in lst)
        tied = sorted([c for a, c in lst if a >= top - 0.001], key=key)
        ch[f'{M}|{V}|{bpf}|{acct}|VSAR'] = {'cfg': list(tied[0]), 'dev_acc': [a for a, c in lst if c == tied[0]][0],
                                            'n_tied': len(tied), 'n_cfg': len(lst)}
    fn = f'{RES}/b66_choice_vsarm{m}_{enc}.json'
    json.dump({'enc': enc, 'map': m, 'choice': ch}, open(fn, 'w'), indent=1, sort_keys=True)
    print(len(ch), 'choices ->', fn)


def vsartest(enc, m):
    import b32_run as R
    ch = json.load(open(f'{RES}/b66_choice_vsarm{m}_{enc}.json'))['choice']
    pool, preps = _setup_store(enc)
    n = pool.X.shape[1]
    tg = tag(enc, m)
    feats = {}
    for M in M_GRID:
        for V in VN:
            fn = os.path.join(vdir(), f'test_{enc}_m{m}_M{M}_V{V}.jsonl.gz')
            if os.path.exists(fn):
                continue
            with gzip.open(fn + '.tmp', 'wt') as fo:
                for rep in range(TEST_REPS[M]):
                    t0 = time.time()
                    st = C.make_store(pool, 'test', M, V, rep)
                    rec = {'enc': enc, 'map': m, 'M': M, 'V': V, 'rep': rep, 'qgold': st['qgold'].tolist(), 'pred': {},
                           'cfg': {}, 'd': {}}
                    groups = {}
                    for bpf in BPF:
                        for acct in ACCT32:
                            c = ch[f'{M}|{V}|{bpf}|{acct}|VSAR']['cfg']
                            groups.setdefault((c[0], c[1]), []).append((bpf, acct, c[2], c[3]))
                            rec['cfg'][f'VSARm{m}|{bpf}|{"real" if acct == "raw" else acct}'] = c
                    for (prep, fmap), lst in groups.items():
                        if (prep, fmap) not in feats:
                            feats[prep, fmap] = R.Feat(tg, prep, fmap, n, 'cpu')
                        cfgs = [((bpf, acct, q, f), C.vsa_dim(bpf * M, q, acct), q, max(1, int(math.floor(f * M))))
                                for bpf, acct, q, f in lst]
                        P = preps[prep]
                        pr = R.vsa_grid(P(st['keys']), st['vals'], P(st['queries']), V, feats[prep, fmap], cfgs,
                                        R.store_seed(tg, st), R.store_order(st))
                        for (key, d, q, ns) in cfgs:
                            a2 = 'real' if key[1] == 'raw' else key[1]
                            rec['pred'][f'VSARm{m}|{key[0]}|{a2}'] = pr[key].tolist()
                            rec['d'][f'VSARm{m}|{key[0]}|{a2}'] = [d, q, ns]
                    fo.write(json.dumps(rec) + '\n')
                    print(f'vsartest {enc} m{m} M={M} V={V} rep={rep} {time.time() - t0:.0f}s', flush=True)
            os.replace(fn + '.tmp', fn)


# ----------------------------------------------------------------------------------------------- selftest

def selftest():
    """Training split only. (1) loc_rows_m(m = 0) equals b64_run.loc_rows (B64's order) and m = 1..3 are permutations
    of the same sets; (2) the first-design path with m = 0 equals b32_run.vsa_grid with B49's key on a TRAINING store
    (trivially the same call), and m = 1 differs from m = 0 (new projections); (3) cr_grid has 32 distinct
    configurations, all inside b64_run.GRID."""
    import b64_run as S
    import b32_run as R
    cwd = os.getcwd()
    os.chdir(S.DATA)
    pool = C.Pool(S.ENC_TAG['qwen8'])
    os.chdir(cwd)
    l0 = S.loc_rows(pool)
    for m in (0, 1, 2, 3):
        lm = loc_rows_m(pool, m, 'qwen8')
        for loc in S.LOCS:
            if m == 0:
                assert (lm[loc] == l0[loc]).all(), loc
            else:
                assert sorted(lm[loc].tolist()) == sorted(l0[loc].tolist()) and not (lm[loc] == l0[loc]).all(), (m, loc)
    g = cr_grid()
    assert len(set(g)) == 32 and all(c in set(S.GRID) for c in g)
    pool2, preps = _setup_store('qwen8')
    st = C.make_store(pool2, 'train', 256, 16, 0)
    n = pool2.X.shape[1]
    P = preps['center']
    cfgs = [((8, 'ideal', 2, 1.0), C.vsa_dim(8 * 256, 2, 'ideal'), 2, 256)]
    out = {}
    for m in (0, 1):
        f = R.Feat(tag('qwen8', m), 'center', 'lin', n, 'cpu')
        out[m] = R.vsa_grid(P(st['keys']), st['vals'], P(st['queries']), 16, f, cfgs, R.store_seed(tag('qwen8', m), st),
                            R.store_order(st))[cfgs[0][0]]
    acc = {m: float((out[m] == st['qgold']).mean()) for m in out}
    agree = float((out[0] == out[1]).mean())
    assert agree < 0.999, agree
    print(f'selftest ok: location orders (m0 = B64, m1-3 permutations); cr_grid 32 in B64 grid; first design on a '
          f'training store acc m0 {acc[0]:.3f} m1 {acc[1]:.3f}, m0/m1 agreement {agree:.3f}')


def main():
    a = sys.argv[1:]
    encs = a[a.index('--encs') + 1].split(',') if '--encs' in a else PANEL
    if a[0] == 'crgrid':
        crgrid(a[1])
    elif a[0] == 'plan':
        plan(a[1], encs, int(a[2]) if a[1].startswith('sdmt') else None)
    elif a[0] == 'runb64':
        m = int(a[a.index('--map') + 1])
        os.environ['B64_OUT'] = a[a.index('--out') + 1]
        import b64_run as S
        S.OUT = a[a.index('--out') + 1]
        patch_b64(m)
        S.runplan(a[1], int(a[a.index('--workers') + 1]), True)
    elif a[0] == 'sdmtchoose':
        sdmtchoose(a[1], int(a[2]))
    elif a[0] == 'vsardev':
        vsardev(a[1], int(a[2]), a[3])
    elif a[0] == 'vsarchoose':
        vsarchoose(a[1], int(a[2]))
    elif a[0] == 'vsartest':
        vsartest(a[1], int(a[2]))
    elif a[0] == 'selftest':
        selftest()
    else:
        raise SystemExit(a[0])


if __name__ == '__main__':
    main()
