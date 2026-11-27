"""B32 (prereg/B32_vector_form_equal_bits_prereg.md): the vector form against an equally engineered vector index at
the same number of bits, on entities queried by unseen Wikidata aliases.

python3 b32_run.py selftest [--device cuda]
python3 b32_run.py dev    ENC OUT.jsonl --part vsa|idx [--device cuda]   # every configuration on the dev stores
python3 b32_run.py choose ENC DEV1.jsonl,DEV2.jsonl OUT.json            # per cell and memory, the best dev mean
python3 b32_run.py test   ENC CHOICE.json OUTDIR --part vsa|idx [--device cuda]
python3 b32_run.py merge  OUTDIR                                          # -> OUTDIR/preds_{enc}_M{M}_V{V}.jsonl.gz

A cell is (M, V, bpf, acct). The VSA's projections and value codes are fixed per (encoder, prep, fmap), drawn per
block of CH = 8192 dimensions from seeds that do not depend on the store, d, q or the stored share, so a vector of d
dimensions is the prefix of a longer one; the stored subsets are prefixes of one permutation of the store
(b32_common.subset_mask). Configurations of one store that share (prep, fmap) share one pass; each configuration's
predictions equal those of running it alone (selftest). Arithmetic in torch float32 (TF32 off), on CPU or GPU.
(b32_common.vsa_answer is the same algorithm with per-store seeds and numpy; it was used by the pilots.)
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

CH = C.CH
GRID = {
    'M': [256, 1024],
    'V': [2, 16, 128, 1024],
    'bpf': [4, 8, 16, 32, 64],
    'acct': ['ideal', 'raw'],
    'prep': ['center', 'whiten'],
    'fmap': ['lin', 'sgn', 'sgn3'],
    'q': [1, 2, 4],
    'f': [1.0, 0.5, 0.25, 0.125],
    'family': ['lsh', 'itq', 'sq1', 'pq4', 'pq8', 'opq8'],
}
DEV_REPS = {256: 6, 1024: 3}
TEST_REPS = {256: 20, 1024: 5}
ENCS = {'qwen8': 'qwen8_mean_L9_f16', 'e5': 'e5_base'}


# ----------------------------------------------------------------------------------------------- VSA, shared pass

V_MAX = 1024  # value codes: the first V rows of one V_MAX x d random sign matrix


class Feat:
    """The VSA's fixed random components for one (encoder, prep, fmap): projection blocks G[m, j0] (n x CH, Gaussian)
    and value-code blocks (V_MAX x CH, random signs), drawn from seeds that depend on (enc, prep, fmap, factor, j0)
    only. They are the same for every store (like any fixed part of a memory) and cached per block. Arithmetic runs in
    torch (float32, TF32 off) on DEVICE."""

    def __init__(self, enc, prep, fmap, n, device):
        import torch
        self.torch, self.dev = torch, device
        torch.backends.cuda.matmul.allow_tf32 = False
        self.key = (enc, prep, fmap)
        self.n, self.fmap = n, fmap
        self.K = 1 if fmap in ('lin', 'sgn') else int(fmap[3:])
        self.cache = {}

    def block(self, j0):
        if j0 not in self.cache:
            t = self.torch
            Gs = [t.from_numpy(np.random.default_rng(C.seed_of('G', *self.key, m, j0, CH)).standard_normal(
                (self.n, CH), dtype=np.float32)).to(self.dev) for m in range(self.K)]
            r = np.random.default_rng(C.seed_of('C', *self.key, j0, CH))
            Cv = t.from_numpy(np.where(r.random((V_MAX, CH), dtype=np.float32) < 0.5, -1.0, 1.0).astype(
                np.float32)).to(self.dev)
            self.cache[j0] = (Gs, Cv)
        return self.cache[j0]

    def phi(self, Y, Gs, w):
        out = None
        for G in Gs:
            P = Y @ G[:, :w]
            f = P if self.fmap == 'lin' else self.torch.where(P >= 0, 1.0, -1.0)
            out = f if out is None else out * f
        return out


def quantize_t(S, q, rng, torch):
    """torch twin of b32_common.quantize (q = 1: sign, zeros to +-1 at random; q >= 2: Max's uniform quantizer on
    S / std(S))."""
    if q == 1:
        out = torch.sign(S)
        z = out == 0
        nz = int(z.sum())
        if nz:
            out[z] = torch.from_numpy(rng.choice([-1.0, 1.0], size=nz).astype(np.float32)).to(S.device)
        return out
    sd = float(S.std(unbiased=False)) or 1.0
    step = C.MAX_STEP[q] * sd
    L = 2 ** q
    return (torch.clamp(torch.floor(S / step), -L // 2, L // 2 - 1) + 0.5) * step


def vsa_grid(K, vals, Qy, V, feat, cfgs, store_seed, order):
    """cfgs: list of (key, d, q, n_stored); facts stored = order[:n_stored]. Returns {key: predictions (numpy)}."""
    t = feat.torch
    Kt = t.from_numpy(np.ascontiguousarray(K, dtype=np.float32)).to(feat.dev)
    Qt = t.from_numpy(np.ascontiguousarray(Qy, dtype=np.float32)).to(feat.dev)
    vt = t.from_numpy(np.asarray(vals, dtype=np.int64)).to(feat.dev)
    scores = {c[0]: t.zeros((Qy.shape[0], V), dtype=t.float64, device=feat.dev) for c in cfgs}
    dmax = max([c[1] for c in cfgs] + [0])
    orders = {}
    for j0 in range(0, dmax, CH):
        w = min(CH, dmax - j0)
        Gs, Cfull = feat.block(j0)
        Cv = Cfull[:V, :w]
        FK = feat.phi(Kt, Gs, w)
        FQ = feat.phi(Qt, Gs, w)
        Bd = FK * Cv[vt]
        cum = {}
        for key, d, q, ns in cfgs:
            if d <= j0:
                continue
            ww = min(w, d - j0)
            if ns not in orders:
                orders[ns] = t.from_numpy(np.asarray(order[:ns], dtype=np.int64)).to(feat.dev)
            if (ns, ww) not in cum:
                cum[ns, ww] = Bd[orders[ns], :ww].sum(0)
            Sq = quantize_t(cum[ns, ww].clone(), q, np.random.default_rng(C.seed_of('Q', store_seed, d, q, ns, j0)), t)
            scores[key] += ((FQ[:, :ww] * Sq) @ Cv[:, :ww].T).double()
    return {k: (v.argmax(1).cpu().numpy() if bool(v.abs().sum() > 0) else np.zeros(v.shape[0], int))
            for k, v in scores.items()}


def vsa_cfgs(st, bpf_list, acct_list, q_list, f_list):
    M = st['M']
    out = []
    for bpf, acct, q, f in itertools.product(bpf_list, acct_list, q_list, f_list):
        if acct == 'raw' and q == 1 and 'ideal' in acct_list:
            continue  # identical to ideal (a sign costs one bit either way); copied below
        ns = max(1, int(math.floor(f * M)))
        out.append(((bpf, acct, q, f), C.vsa_dim(bpf * M, q, acct), q, ns))
    return out


def store_order(st):
    return np.random.default_rng(C.seed_of('subset', st['split'], st['M'], st['V'], st['rep'])).permutation(st['M'])


def store_seed(enc, st):
    return C.seed_of('vsa', enc, st['split'], st['M'], st['V'], st['rep'])


def run_vsa(st, preps, prep, feat, bpf_list, acct_list, q_list, f_list, enc):
    cfgs = vsa_cfgs(st, bpf_list, acct_list, q_list, f_list)
    P = preps[prep]
    pred = vsa_grid(P(st['keys']), st['vals'], P(st['queries']), st['V'], feat, cfgs, store_seed(enc, st),
                    store_order(st))
    out = {}
    for (bpf, acct, q, f), p in pred.items():
        out[bpf, acct, q, f] = p
        if acct == 'ideal' and q == 1 and 'raw' in acct_list:
            out[bpf, 'raw', q, f] = p
    return out


# ----------------------------------------------------------------------------------------------- index

def run_idx(st, preps, fams, prep, family, bpf, acct, f):
    order = store_order(st)
    ns = max(1, int(math.floor(f * st['M'])))
    keep = order[:ns]
    P = preps[prep]
    K, vals, Q = P(st['keys'][keep]), st['vals'][keep], P(st['queries'])
    kb = C.key_bits(bpf * st['M'], ns, st['V'], acct)
    c = fams[prep, family].config(kb)
    if c is None:
        return np.full(len(Q), int(np.bincount(vals, minlength=st['V']).argmax())), 0
    return vals[fams[prep, family].nearest(K, Q, c[1])], c[0]


# ----------------------------------------------------------------------------------------------- modes

def setup(enc, need_idx=True):
    pool = C.Pool(ENCS[enc])
    Tr = pool.train_matrix()
    preps = {p: C.Prep(p, Tr) for p in GRID['prep']}
    fams = {(p, f): C.IndexFamily(f, preps[p](Tr), cache_tag=f'{enc}_{p}') for p in GRID['prep'] for f in GRID['family']} \
        if need_idx else {}
    return pool, preps, fams


def dev_stores(pool):
    return [C.make_store(pool, 'dev', M, V, rep) for M in GRID['M'] for V in GRID['V'] for rep in range(DEV_REPS[M])]


def dev(enc, out, part, device='cpu'):
    """part 'vsa': every VSA configuration (loops over (prep, fmap) outside the stores so that the fixed projections
    are drawn once); part 'idx': every index configuration and EXACT. One JSON line per (store, part)."""
    pool, preps, fams = setup(enc, need_idx=(part == 'idx'))
    done = set()
    if os.path.exists(out):
        for l in open(out):
            r = json.loads(l)
            done.add((r['M'], r['V'], r['rep'], r['part']))
    fo = open(out, 'a')
    stores = dev_stores(pool)
    if part == 'vsa':
        for prep in GRID['prep']:
            for fmap in GRID['fmap']:
                feat = Feat(enc, prep, fmap, pool.X.shape[1], device)
                for st in stores:
                    pn = f'vsa:{prep}:{fmap}'
                    if (st['M'], st['V'], st['rep'], pn) in done:
                        continue
                    t0 = time.time()
                    g = st['qgold']
                    res = run_vsa(st, preps, prep, feat, GRID['bpf'], GRID['acct'], GRID['q'], GRID['f'], enc)
                    rows = [{'arm': 'VSA', 'cfg': [prep, fmap, q, f], 'bpf': bpf, 'acct': acct,
                             'acc': float((p == g).mean())} for (bpf, acct, q, f), p in res.items()]
                    fo.write(json.dumps({'enc': enc, 'M': st['M'], 'V': st['V'], 'rep': st['rep'], 'nq': len(g),
                                         'part': pn, 'rows': rows}) + '\n')
                    fo.flush()
                    print(f'dev {enc} {pn} M={st["M"]} V={st["V"]} rep={st["rep"]} {time.time() - t0:.0f}s', flush=True)
                del feat
        return
    for st in stores:
        if (st['M'], st['V'], st['rep'], 'idx') in done:
            continue
        t0 = time.time()
        g = st['qgold']
        rows = []
        for prep, family, bpf, acct, f in itertools.product(GRID['prep'], GRID['family'], GRID['bpf'], GRID['acct'],
                                                            GRID['f']):
            p, kb = run_idx(st, preps, fams, prep, family, bpf, acct, f)
            rows.append({'arm': 'IDX', 'cfg': [prep, family, f], 'bpf': bpf, 'acct': acct, 'keybits': kb,
                         'acc': float((p == g).mean())})
        for prep in GRID['prep']:
            Pp = preps[prep]
            p = st['vals'][C.exact_nearest(Pp(st['keys']), Pp(st['queries']))]
            rows.append({'arm': 'EXACT', 'cfg': [prep], 'acc': float((p == g).mean())})
        fo.write(json.dumps({'enc': enc, 'M': st['M'], 'V': st['V'], 'rep': st['rep'], 'nq': len(g), 'part': 'idx',
                             'rows': rows}) + '\n')
        fo.flush()
        print(f'dev {enc} idx M={st["M"]} V={st["V"]} rep={st["rep"]} {time.time() - t0:.0f}s', flush=True)


def vsa_key(c):
    return (GRID['prep'].index(c[0]), GRID['fmap'].index(c[1]), GRID['q'].index(c[2]), GRID['f'].index(c[3]))


def idx_key(c):
    return (GRID['prep'].index(c[0]), GRID['family'].index(c[1]), GRID['f'].index(c[2]))


def choose(enc, devfile, out):
    """Per (M, V, bpf, acct) and memory: the configuration with the highest mean dev accuracy over the dev stores
    (mean over stores of per-store accuracy). Any configuration within 0.1 points of the best counts as tied; ties go
    to the first in grid order (prep, then fmap or family, then q, then stored share, each in GRID order)."""
    acc = {}
    reps = {}
    lines = [l for fn in devfile.split(',') for l in open(fn)]
    for l in lines:
        r = json.loads(l)
        for x in r['rows']:
            if x['arm'] == 'EXACT':
                k = (r['M'], r['V'], None, None, 'EXACT', tuple(x['cfg']))
            else:
                k = (r['M'], r['V'], x['bpf'], x['acct'], x['arm'], tuple(x['cfg']))
            acc.setdefault(k, {})[r['rep']] = x['acc']
            reps.setdefault((r['M'], r['V']), set()).add(r['rep'])
    best = {}
    for k, d in acc.items():
        M, V, bpf, acct, arm, cfg = k
        if len(d) != len(reps[M, V]):
            raise SystemExit(f'incomplete dev results for {k}')
        best.setdefault((M, V, bpf, acct, arm), []).append((float(np.mean(list(d.values()))), cfg))
    out_d = {}
    for (M, V, bpf, acct, arm), lst in best.items():
        top = max(a for a, _ in lst)
        keyf = vsa_key if arm == 'VSA' else (idx_key if arm == 'IDX' else (lambda c: GRID['prep'].index(c[0])))
        tied = sorted([c for a, c in lst if a >= top - 0.001], key=keyf)
        out_d[f'{M}|{V}|{bpf}|{acct}|{arm}'] = {'cfg': list(tied[0]), 'dev_acc': [a for a, c in lst if c == tied[0]][0],
                                               'best_dev_acc': top, 'n_tied': len(tied)}
    json.dump({'enc': enc, 'devfile': devfile, 'choice': out_d}, open(out, 'w'), indent=1, sort_keys=True)
    print(f'{len(out_d)} choices written to {out}')


def test(enc, choicefile, outdir, part, device='cpu'):
    """part 'vsa': the chosen VSA configuration of every cell; part 'idx': the chosen index configurations, EXACT and
    (for the primary encoder) the string look-ups. One gzip JSONL file per (enc, part, M, V); merge joins them."""
    pool, preps, fams = setup(enc, need_idx=(part == 'idx'))
    ch = json.load(open(choicefile))
    assert ch['enc'] == enc
    ch = ch['choice']
    os.makedirs(outdir, exist_ok=True)
    feats = {}
    for M in GRID['M']:
        for V in GRID['V']:
            fn = f'{outdir}/{part}_{enc}_M{M}_V{V}.jsonl.gz'
            if os.path.exists(fn):
                continue
            tmp = fn + '.tmp'
            with gzip.open(tmp, 'wt') as fo:
                for rep in range(TEST_REPS[M]):
                    t0 = time.time()
                    st = C.make_store(pool, 'test', M, V, rep)
                    rec = {'enc': enc, 'M': st['M'], 'V': V, 'rep': rep, 'pred': {}, 'keybits': {}, 'cfg': {}}
                    if part == 'vsa':
                        groups = {}
                        for bpf in GRID['bpf']:
                            for acct in GRID['acct']:
                                c = ch[f'{M}|{V}|{bpf}|{acct}|VSA']['cfg']
                                groups.setdefault((c[0], c[1]), []).append((bpf, acct, c[2], c[3]))
                                rec['cfg'][f'VSA|{bpf}|{acct}'] = c
                        for (prep, fmap), lst in groups.items():
                            if (prep, fmap) not in feats:
                                feats[prep, fmap] = Feat(enc, prep, fmap, pool.X.shape[1], device)
                            cfgs = [((bpf, acct, q, f), C.vsa_dim(bpf * st['M'], q, acct), q,
                                     max(1, int(math.floor(f * st['M'])))) for bpf, acct, q, f in lst]
                            P = preps[prep]
                            pr = vsa_grid(P(st['keys']), st['vals'], P(st['queries']), V, feats[prep, fmap], cfgs,
                                          store_seed(enc, st), store_order(st))
                            for (bpf, acct, q, f), p in pr.items():
                                rec['pred'][f'VSA|{bpf}|{acct}'] = p.tolist()
                    else:
                        rec.update({'uris': st['uris'], 'vals': st['vals'].tolist(), 'qent': st['qent'].tolist(),
                                    'qalias': st['qalias'], 'qgold': st['qgold'].tolist()})
                        for bpf in GRID['bpf']:
                            for acct in GRID['acct']:
                                c = ch[f'{M}|{V}|{bpf}|{acct}|IDX']['cfg']
                                p, kb = run_idx(st, preps, fams, c[0], c[1], bpf, acct, c[2])
                                rec['pred'][f'IDX|{bpf}|{acct}'] = p.tolist()
                                rec['keybits'][f'IDX|{bpf}|{acct}'] = kb
                                rec['cfg'][f'IDX|{bpf}|{acct}'] = c
                        prep = ch[f'{M}|{V}|None|None|EXACT']['cfg'][0]
                        Pp = preps[prep]
                        nn = C.exact_nearest(Pp(st['keys']), Pp(st['queries']))
                        rec['pred']['EXACT'] = st['vals'][nn].tolist()
                        rec['pred']['EXACT_ENT'] = nn.tolist()
                        if enc == 'qwen8':  # string look-ups do not depend on the encoder: computed once
                            for k, p in C.string_answers(st).items():
                                rec['pred'][k] = p.tolist()
                    fo.write(json.dumps(rec) + '\n')
                    print(f'test {enc} {part} M={st["M"]} V={V} rep={rep} Q={len(st["qgold"])} {time.time() - t0:.0f}s',
                          flush=True)
            os.replace(tmp, fn)


def merge(outdir, encs=None):
    """Join the vsa_ and idx_ files into preds_{enc}_M{M}_V{V}.jsonl.gz (checks that stores line up)."""
    for enc in (encs or ENCS):
        for M in GRID['M']:
            for V in GRID['V']:
                a = [json.loads(l) for l in gzip.open(f'{outdir}/idx_{enc}_M{M}_V{V}.jsonl.gz', 'rt')]
                b = [json.loads(l) for l in gzip.open(f'{outdir}/vsa_{enc}_M{M}_V{V}.jsonl.gz', 'rt')]
                assert [(r['M'], r['V'], r['rep']) for r in a] == [(r['M'], r['V'], r['rep']) for r in b]
                with gzip.open(f'{outdir}/preds_{enc}_M{M}_V{V}.jsonl.gz', 'wt') as fo:
                    for x, y in zip(a, b):
                        for k, p in y['pred'].items():
                            assert len(p) == len(x['qgold'])
                        x['pred'].update(y['pred'])
                        x['cfg'].update(y['cfg'])
                        fo.write(json.dumps(x) + '\n')
    print('merged')


def selftest(device='cpu'):
    rng = np.random.default_rng(0)
    # 1. the shared pass equals running each configuration alone (same seeds), on synthetic codes
    M, V, n, Qn = 300, 16, 64, 50
    K = rng.standard_normal((M, n)).astype(np.float32)
    K /= np.linalg.norm(K, axis=1, keepdims=True)
    vals = rng.integers(0, V, M)
    Qy = K[rng.integers(0, M, Qn)] + 0.3 * rng.standard_normal((Qn, n)).astype(np.float32)
    Qy /= np.linalg.norm(Qy, axis=1, keepdims=True)
    order = rng.permutation(M)
    for fmap in ('lin', 'sgn', 'sgn3'):
        feat = Feat('selftest', 'p', fmap, n, device)
        cfgs = [(('a', q, ns, d), d, q, ns) for q in (1, 2, 4) for ns in (M, M // 2) for d in (3000, 10000, 20000)]
        got = vsa_grid(K, vals, Qy, V, feat, cfgs, 7, order)
        for key, d, q, ns in cfgs:
            ref = vsa_grid(K, vals, Qy, V, Feat('selftest', 'p', fmap, n, device), [(key, d, q, ns)], 7, order)[key]
            assert np.array_equal(got[key], ref), (fmap, key)
    # 2. torch read equals a plain numpy read of the same definition (one configuration, lin, q = 2)
    feat = Feat('selftest', 'p', 'lin', n, device)
    d, q, ns = 12000, 2, M
    pt = vsa_grid(K, vals, Qy, V, feat, [('x', d, q, ns)], 7, order)['x']
    sc = np.zeros((Qn, V))
    for j0 in range(0, d, CH):
        w = min(CH, d - j0)
        G = np.random.default_rng(C.seed_of('G', 'selftest', 'p', 'lin', 0, j0, CH)).standard_normal((n, CH),
                                                                                                    dtype=np.float32)
        r = np.random.default_rng(C.seed_of('C', 'selftest', 'p', 'lin', j0, CH))
        Cv = np.where(r.random((V_MAX, CH), dtype=np.float32) < 0.5, -1.0, 1.0)[:V, :w]
        FK, FQ = K @ G[:, :w], Qy @ G[:, :w]
        S = (FK[order[:ns]] * Cv[vals[order[:ns]]]).sum(0)
        Sq = C.quantize(S, q, None)
        sc += (FQ * Sq) @ Cv.T
    agree = float((sc.argmax(1) == pt).mean())
    assert agree >= 0.98, agree
    # 3. with near-orthogonal random keys, many dimensions and exact queries, the VSA reads every fact
    M, V, n = 200, 1024, 2048
    K = rng.standard_normal((M, n)).astype(np.float32)
    K /= np.linalg.norm(K, axis=1, keepdims=True)
    vals = rng.integers(0, V, M)
    p = vsa_grid(K, vals, K, V, Feat('selftest', 'p', 'lin', n, device), [('x', 200 * 256, 1, M)], 3,
                 np.arange(M))['x']
    assert (p == vals).mean() == 1.0, (p == vals).mean()
    # 4. key_bits: ideal >= raw; budgets respected
    for M_, V_ in ((256, 2), (1024, 16), (1024, 1024)):
        for bpf in GRID['bpf']:
            assert C.key_bits(bpf * M_, M_, V_, 'ideal') >= C.key_bits(bpf * M_, M_, V_, 'raw')
            for q in GRID['q']:
                for acct in GRID['acct']:
                    assert C.vsa_bits(C.vsa_dim(bpf * M_, q, acct), q, acct) <= bpf * M_
    print(f'selftest ok (device {device}; torch vs numpy read agreement {agree:.3f})')


if __name__ == '__main__':
    a = sys.argv[1:]
    dv = a[a.index('--device') + 1] if '--device' in a else 'cpu'
    pt = a[a.index('--part') + 1] if '--part' in a else None
    if a[0] == 'selftest':
        selftest(dv)
    elif a[0] == 'dev':
        dev(a[1], a[2], pt, dv)
    elif a[0] == 'choose':
        choose(a[1], a[2], a[3])
    elif a[0] == 'test':
        test(a[1], a[2], a[3], pt, dv)
    elif a[0] == 'fit':  # compute note 2: fit one PQ / OPQ index into B32_FIT_CACHE (training split only)
        enc, prep, family, p = a[1], a[2], a[3], int(a[4])
        pool = C.Pool(ENCS[enc])
        Tr = pool.train_matrix()
        Pp = C.Prep(prep, Tr)
        C.IndexFamily(family, Pp(Tr), cache_tag=f'{enc}_{prep}')._fit(p)
        print('fitted', enc, prep, family, p)
    elif a[0] == 'merge':
        merge(a[1], a[2].split(',') if len(a) > 2 else None)
