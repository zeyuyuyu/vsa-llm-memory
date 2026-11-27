"""B53 (prereg/B53_value_codes_prereg.md): B49's panel re-run with ONE change, the VSA store's value codes.

B32 / B49's store binds each key to a value code drawn as independent random +-1 signs (b32_run.Feat: the first V rows
of a V_MAX x CH random sign block per (encoder, prep, fmap, block)). With V = 2 the two random codes agree on about half
of the coordinates, which then carry the same number for both values and cannot separate them. B53 replaces the codes:

  V = 2   antipodal:  c(0) = r, c(1) = -r
  V = 16  orthogonal: c_j(v) = r_j * H16[v, j mod 16]   (H16 the Sylvester-Hadamard matrix of order 16; the 16 codes
          are rows of the Sylvester-Hadamard matrix of order 16 * 2^k restricted to the store's d coordinates, exactly
          orthogonal on every aligned window of 16 coordinates)
  V >= 128 B49's random codes, unchanged (not re-run; B49's predictions are reused)

r = row 0 of B49's random sign block (same seed), so c(0) is B49's code of value 0 and every projection, store, query,
subset order and quantizer seed is B49's. Everything else (grids, choice rule, accounting, budgets) is B32 / B36's code,
called unchanged; the 1-2 bits-per-fact cells (B36's dev32 / test32) and 4-64 (B32's dev / test) run in one pass (a pass
of several configurations gives each configuration's own predictions: b32_run.selftest).

python3 b53_run.py selftest
python3 b53_run.py ENC md5
python3 b53_run.py ENC dev PREP:FMAP        # one (prep, fmap) part of the VSA dev grid, V in {2, 16}, 1-64 bpf
python3 b53_run.py ENC choose               # B32's rule on the six part files -> OUT/ENC/choice53_ENC.json
python3 b53_run.py ENC test CHOICE.json     # chosen configurations on the test stores -> OUT/ENC/vsa53_ENC_M_V.jsonl.gz
python3 b53_run.py ENC merge                # (local) B49's merged files with VSA replaced -> b53_out/fam/ENC/all53_*
environment: B53_DATA (root holding the encoders' b32_data directories; default this repository), B53_OUT (default
b53_out/ here).
"""
import gzip
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import b32_common as C  # noqa: E402
import b32_run as R  # noqa: E402

DATA = os.environ.get('B53_DATA', ROOT)
OUT = os.environ.get('B53_OUT', os.path.join(ROOT, 'b53_out'))
ENC_TAG = {'qwen8': 'qwen8_mean_L9_f16', 'e5': 'e5_base', 'b46': 'b46_granite33i_mean_L30_f16',
           'llama': 'llama31i_mean_L24_f16', 'mistral': 'mistral7i_eol_L24_f16', 'bge': 'bge_small', 'minilm': 'minilm'}
PANEL = ['qwen8', 'b46', 'llama', 'mistral', 'e5', 'bge', 'minilm']
V_NEW = [2, 16]
M_GRID = [256, 1024]
BPF = [1, 2, 4, 8, 16, 32, 64]
ACCT32 = ['ideal', 'raw']
PARTS = [(p, f) for p in R.GRID['prep'] for f in R.GRID['fmap']]
CH = R.CH
assert CH % 16 == 0


def sylvester(n):
    H = np.ones((1, 1))
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return H


H16 = sylvester(16).astype(np.float32)


def code_block(V, r):
    """B53's value codes on one block of CH coordinates from B49's row-0 signs r (length CH)."""
    if V == 2:
        return np.stack([r, -r]).astype(np.float32)
    if V == 16:
        return (np.tile(H16, (1, len(r) // 16)) * r[None, :]).astype(np.float32)
    raise ValueError(V)


class Feat53(R.Feat):
    """b32_run.Feat (same projections, same seeds) whose value codes are B53's for the store's V (set .V before a
    pass)."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.V = None
        self.ccache = {}

    def block(self, j0):
        Gs, Cfull = super().block(j0)
        if (j0, self.V) not in self.ccache:
            r = Cfull[0].cpu().numpy()
            self.ccache[j0, self.V] = self.torch.from_numpy(code_block(self.V, r)).to(self.dev)
        return Gs, self.ccache[j0, self.V]


def enc_dir(enc):
    if enc in ('qwen8', 'e5'):
        return DATA
    if enc == 'b46':
        return os.path.join(DATA, 'b46_out')
    return os.path.join(DATA, 'b49_out', f'enc_{enc}')


def setup(enc):
    os.chdir(enc_dir(enc))
    pool = C.Pool(ENC_TAG[enc])
    Tr = pool.train_matrix()
    preps = {p: C.Prep(p, Tr) for p in R.GRID['prep']}
    return pool, preps


def odir(enc):
    d = os.path.join(OUT, enc)
    os.makedirs(d, exist_ok=True)
    return d


def dev(enc, part):
    prep, fmap = part.split(':')
    pool, preps = setup(enc)
    out = os.path.join(odir(enc), f'dev_{enc}_{prep}_{fmap}.jsonl')
    done = set()
    if os.path.exists(out):
        done = {(r['M'], r['V'], r['rep']) for r in map(json.loads, open(out))}
    fo = open(out, 'a')
    feat = Feat53(enc, prep, fmap, pool.X.shape[1], 'cpu')
    for M in M_GRID:
        for V in V_NEW:
            for rep in range(R.DEV_REPS[M]):
                if (M, V, rep) in done:
                    continue
                t0 = time.time()
                st = C.make_store(pool, 'dev', M, V, rep)
                feat.V = V
                g = st['qgold']
                res = R.run_vsa(st, preps, prep, feat, BPF, ACCT32, R.GRID['q'], R.GRID['f'], enc)
                rows = [{'arm': 'VSA', 'cfg': [prep, fmap, q, f], 'bpf': bpf, 'acct': acct,
                         'acc': float((p == g).mean())} for (bpf, acct, q, f), p in res.items()]
                fo.write(json.dumps({'enc': enc, 'M': st['M'], 'V': V, 'rep': rep, 'nq': len(g),
                                     'part': f'vsa:{prep}:{fmap}', 'codes': 'b53', 'rows': rows}) + '\n')
                fo.flush()
                print(f'dev53 {enc} {prep}:{fmap} M={M} V={V} rep={rep} {time.time() - t0:.0f}s', flush=True)


def choose(enc):
    R.GRID['bpf'] = list(BPF)
    d = odir(enc)
    files = ','.join(os.path.join(d, f'dev_{enc}_{p}_{f}.jsonl') for p, f in PARTS)
    R.choose(enc, files, os.path.join(d, f'choice53_{enc}.json'))


def test(enc, choicefile):
    ch = json.load(open(os.path.abspath(choicefile)))
    assert ch['enc'] == enc
    ch = ch['choice']
    pool, preps = setup(enc)
    feats = {}
    for M in M_GRID:
        for V in V_NEW:
            fn = os.path.join(odir(enc), f'vsa53_{enc}_M{M}_V{V}.jsonl.gz')
            if os.path.exists(fn):
                continue
            with gzip.open(fn + '.tmp', 'wt') as fo:
                for rep in range(R.TEST_REPS[M]):
                    t0 = time.time()
                    st = C.make_store(pool, 'test', M, V, rep)
                    rec = {'enc': enc, 'M': st['M'], 'V': V, 'rep': rep, 'qgold': st['qgold'].tolist(), 'pred': {},
                           'cfg': {}, 'd': {}}
                    groups = {}
                    for bpf in BPF:
                        for acct in ACCT32:
                            c = ch[f'{M}|{V}|{bpf}|{acct}|VSA']['cfg']
                            groups.setdefault((c[0], c[1]), []).append((bpf, acct, c[2], c[3]))
                            rec['cfg'][f'VSA|{bpf}|{acct}'] = c
                    for (prep, fmap), lst in groups.items():
                        if (prep, fmap) not in feats:
                            feats[prep, fmap] = Feat53(enc, prep, fmap, pool.X.shape[1], 'cpu')
                        feats[prep, fmap].V = V
                        cfgs = [((bpf, acct, q, f), C.vsa_dim(bpf * st['M'], q, acct), q,
                                 max(1, int(math.floor(f * st['M'])))) for bpf, acct, q, f in lst]
                        P = preps[prep]
                        pr = R.vsa_grid(P(st['keys']), st['vals'], P(st['queries']), V, feats[prep, fmap], cfgs,
                                        R.store_seed(enc, st), R.store_order(st))
                        for (key, d, q, ns) in cfgs:
                            rec['pred'][f'VSA|{key[0]}|{key[1]}'] = pr[key].tolist()
                            rec['d'][f'VSA|{key[0]}|{key[1]}'] = [d, q, ns]
                    fo.write(json.dumps(rec) + '\n')
                    print(f'test53 {enc} M={M} V={V} rep={rep} {time.time() - t0:.0f}s', flush=True)
            os.replace(fn + '.tmp', fn)


def merge(enc):
    """B49's merged test records (b49_out/fam/ENC/test/all49_*) with the VSA's predictions and configurations replaced
    by B53's for V in {2, 16}; B49's store kept as VSAR (random codes). V >= 128: B49's files are used as they are."""
    amap = {'ideal': 'ideal', 'real': 'raw'}
    od = os.path.join(ROOT, 'b53_out', 'fam', enc)
    os.makedirs(od, exist_ok=True)
    for M in M_GRID:
        for V in V_NEW:
            old = [json.loads(l) for l in gzip.open(
                f'{ROOT}/b49_out/fam/{enc}/test/all49_{enc}_M{M}_V{V}.jsonl.gz', 'rt')]
            new = [json.loads(l) for l in gzip.open(f'{OUT}/{enc}/vsa53_{enc}_M{M}_V{V}.jsonl.gz', 'rt')]
            assert len(old) == len(new) == R.TEST_REPS[M]
            with gzip.open(f'{od}/all53_{enc}_M{M}_V{V}.jsonl.gz', 'wt') as fo:
                for o, n in zip(old, new):
                    assert o['rep'] == n['rep'] and o['qgold'] == n['qgold'] and o['M'] == n['M'] and o['V'] == V
                    for b in BPF:
                        for a in ('ideal', 'real'):
                            o['pred'][f'VSAR|{b}|{a}'] = o['pred'][f'VSA|{b}|{a}']
                            o['cfg'][f'VSAR|{b}|{a}'] = o['cfg'][f'VSA|{b}|{a}']
                            o['pred'][f'VSA|{b}|{a}'] = n['pred'][f'VSA|{b}|{amap[a]}']
                            o['cfg'][f'VSA|{b}|{a}'] = n['cfg'][f'VSA|{b}|{amap[a]}']
                            o.setdefault('vsa_d', {})[f'VSA|{b}|{a}'] = n['d'][f'VSA|{b}|{amap[a]}']
                    fo.write(json.dumps(o) + '\n')
    print('merged', enc)


def selftest():
    import torch
    rng = np.random.default_rng(0)
    r = np.where(rng.random(CH) < 0.5, -1.0, 1.0).astype(np.float32)
    c2 = code_block(2, r)
    assert np.all(c2[0] == -c2[1]) and np.all(np.abs(c2) == 1)
    c16 = code_block(16, r)
    assert np.all(np.abs(c16) == 1)
    for j0 in (0, 16, 4096, CH - 16):
        G = c16[:, j0:j0 + 16] @ c16[:, j0:j0 + 16].T
        assert np.allclose(G, 16 * np.eye(16)), j0
    G = c16 @ c16.T
    assert np.allclose(G, CH * np.eye(16))
    # a pass of several configurations equals each configuration alone, with B53's codes; the codes are used
    n, M, V = 48, 64, 2
    K = rng.standard_normal((M, n)).astype(np.float32)
    K /= np.linalg.norm(K, axis=1, keepdims=True)
    Qy = K + 0.3 * rng.standard_normal((M, n)).astype(np.float32)
    Qy /= np.linalg.norm(Qy, axis=1, keepdims=True)
    for V in (2, 16):
        vals = rng.integers(0, V, size=M)
        order = rng.permutation(M)
        for fmap in ('lin', 'sgn', 'sgn3'):
            feat = Feat53('selftest', 'p', fmap, n, 'cpu')
            feat.V = V
            cfgs = [(('a', d, q, ns)[0] + f'{d}{q}{ns}', d, q, ns) for d in (100, 9000) for q in (1, 2) for ns in (M, 32)]
            got = R.vsa_grid(K, vals, Qy, V, feat, cfgs, 7, order)
            for c in cfgs:
                one = Feat53('selftest', 'p', fmap, n, 'cpu')
                one.V = V
                ref = R.vsa_grid(K, vals, Qy, V, one, [c], 7, order)[c[0]]
                assert np.array_equal(got[c[0]], ref), (V, fmap, c)
            assert torch.equal(feat.block(0)[1][:, :16], torch.from_numpy(code_block(V, feat.block(0)[1][0].numpy()))[:, :16])
    # with many dimensions and exact queries the antipodal store reads every fact
    K2 = rng.standard_normal((M, 1024)).astype(np.float32)
    K2 /= np.linalg.norm(K2, axis=1, keepdims=True)
    vals = rng.integers(0, 2, size=M)
    feat = Feat53('selftest', 'p', 'lin', 1024, 'cpu')
    feat.V = 2
    p = R.vsa_grid(K2, vals, K2, 2, feat, [('x', 20000, 1, M)], 3, np.arange(M))['x']
    assert (p == vals).mean() > 0.95, (p == vals).mean()
    print('selftest ok')


def main():
    a = sys.argv[1:]
    if a[0] == 'selftest':
        selftest()
        return
    enc, mode = a[0], a[1]
    assert enc in ENC_TAG, enc
    if mode == 'md5':
        fn = os.path.join(enc_dir(enc), 'b32_data', f'emb_{ENC_TAG[enc]}.npy')
        print(enc, fn, hashlib.md5(open(fn, 'rb').read()).hexdigest())
    elif mode == 'dev':
        dev(enc, a[2])
    elif mode == 'choose':
        choose(enc)
    elif mode == 'test':
        test(enc, a[2])
    elif mode == 'merge':
        merge(enc)
    else:
        raise SystemExit(mode)


if __name__ == '__main__':
    main()
