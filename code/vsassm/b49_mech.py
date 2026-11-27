"""B49 mechanism runs (prereg/B49_panel_mechanism_competitors_prereg.md, section 6): the codes of one panel encoder are
transformed (T1 isotropise: B32's ZCA whitening applied to the codes; T2 anisotropise: centred spectrum raised to a power
so that PR / n = qwen8's; T3 dimension cap: a fixed 4096 x 768 Gaussian matrix), then B32's VSA and index, B36's
key-free families and B49's new families run unchanged on R1 only (V in {2, 16}, 4-16 bits per fact), dev choice, test
once. Fixed random components are seeded by the base encoder's tag. Working directory b49_out/mech_{base}_{T}/; the
transformed codes are b49_out/b32_data/emb_{base}_{T}.npy.

python3 b49_mech.py BASE T make            # write the transformed codes + their geometry (no memory)
python3 b49_mech.py BASE T fits            # B32 index fits (R1 budgets) and B36 PQ codebooks (training split)
python3 b49_mech.py BASE T dev32vsa|dev32idx|dev36 [--workers N]
python3 b49_mech.py BASE T choose          # B32 + B36 choices (R1) -> results/b32_choice_*.json, results/b36_choice_*.json
python3 b49_mech.py BASE T test32vsa|test32idx|test36 [--workers N]
python3 b49_mech.py BASE T merge           # B36-format merged R1 test files (b36_out/test/all_{base}_M{M}_V{V}.jsonl.gz)
(new families: b49_fam.py {BASE}_{T} prefit / dev / choose / test / merge with B49_GRID_V=2,16 B49_GRID_BPF=4,8,16)
"""
import gzip
import json
import os
import shutil
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
R1_V, R1_BPF = [2, 16], [4, 8, 16]
QWEN8_PR_FRAC = None  # read from results/b49_geom_existing.json (centred PR / n of qwen8)


def wd(base, T):
    return f'{ROOT}/b49_out/mech_{base}_{T}'


def make(base, T):
    import b32_common as C
    import b49_common as B
    import b49_fam as F
    os.chdir(f'{ROOT}/b49_out')
    pool = C.Pool(F.ENC_TAG[base])
    X = pool.X.astype(np.float32)
    info = {'base': base, 'T': T}
    if T == 'T1':
        Y = C.Prep('whiten', pool.train_matrix())(X)
    elif T == 'T2':
        g = json.load(open(f'{ROOT}/results/b49_geom_existing.json'))
        target = g['qwen8']['center']['pr_frac'] * X.shape[1]
        lo, hi = 1.0, 8.0
        for _ in range(30):  # bisection on gamma: PR of the centred training codes falls as gamma grows
            mid = (lo + hi) / 2
            pr = B.pr_of(B.spectrum_power(X, pool, mid), pool)
            lo, hi = (mid, hi) if pr > target else (lo, mid)
        gamma = (lo + hi) / 2
        Y = B.spectrum_power(X, pool, gamma)
        info.update({'gamma': gamma, 'target_pr': target, 'pr': B.pr_of(Y, pool)})
    elif T == 'T3':
        G = np.random.default_rng(C.seed_of('B49JL', 768)).standard_normal((X.shape[1], 768)).astype(np.float32)
        Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
        Y = Xn @ G
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    else:
        raise SystemExit(T)
    tag = f'{base}_{T}'
    np.save(f'b32_data/emb_{tag}.npy', Y.astype(np.float32))
    import b49_geom
    info['geom'] = b49_geom.measure(tag)
    json.dump(info, open(f'{ROOT}/results/b49_mech_codes_{tag}.json', 'w'), indent=1)
    W = wd(base, T)
    for d in ('b32_data', 'b32_out/fitcache', 'b32_out/test', 'b36_out/test', 'b36_out/pqcache', 'results', 'logs'):
        os.makedirs(f'{W}/{d}', exist_ok=True)
    for f in ('entities.json', 'emb_strings.json'):
        shutil.copyfile(f'{ROOT}/b32_data/{f}', f'{W}/b32_data/{f}')
    dst = f'{W}/b32_data/emb_{tag}.npy'
    if not os.path.exists(dst):
        os.symlink(f'{ROOT}/b49_out/b32_data/emb_{tag}.npy', dst)
    print(json.dumps({k: v for k, v in info.items() if k != 'geom'}), json.dumps(info['geom']['center'])[:300],
          json.dumps(info['geom']['whiten'])[:300])


def register(base, T):
    os.chdir(wd(base, T))
    os.environ.setdefault('B32_FIT_CACHE', 'b32_out/fitcache')
    import b32_run as R32
    import b36_run as R36
    tag = f'{base}_{T}'
    R32.ENCS[base] = tag
    R36.ENCS[base] = tag
    R32.GRID['V'] = list(R1_V)
    R32.GRID['bpf'] = list(R1_BPF)
    R36.V_GRID = list(R1_V)
    R36.BPF = list(R1_BPF)
    R36.BPF_NEW = []
    return R32, R36


def main():
    base, T, mode = sys.argv[1], sys.argv[2], sys.argv[3]
    a = sys.argv[4:]
    wk = int(a[a.index('--workers') + 1]) if '--workers' in a else 1
    if mode == 'make':
        make(base, T)
        return
    R32, R36 = register(base, T)
    import b32_common as C
    if mode == 'fits':  # fits [PREP FAMILY]: one (prep, family) per process, or all; 'pqbank': B36 codebooks only
        pool = C.Pool(R32.ENCS[base])
        Tr = pool.train_matrix()
        n = pool.X.shape[1]
        for prep in ([a[0]] if a else ('center', 'whiten')):
            Pp = C.Prep(prep, Tr)
            for fam in ([a[1]] if a else ('pq4', 'pq8', 'opq8')):
                f = C.IndexFamily(fam, Pp(Tr), cache_tag=f'{base}_{prep}')
                nb = 4 if fam == 'pq4' else 8
                only = [int(a[a.index('--only') + 1])] if '--only' in a else None
                for m in C.divisors(n):
                    if only is not None and m not in only:
                        continue
                    if m * nb <= 16 * 8 + 16:  # every m a key at <= 16 bits per fact and a stored share >= 1/8 reaches
                        f._fit(m)
                        print('fitted', prep, fam, m, flush=True)
        if a:
            return
    elif mode == 'pqbank':
        S = R36.Setup(base)
        for p in R36.PREPS:
            if S.banks[p].cent is None:
                S.banks[p].fit()
            print('pq bank', p, flush=True)
    elif mode == 'dev32vsa':
        R32.dev(base, f'b32_out/dev_{base}_vsa.jsonl', 'vsa', 'cpu')
    elif mode == 'dev32idx':
        R32.dev(base, f'b32_out/dev_{base}_idx.jsonl', 'idx', 'cpu')
    elif mode == 'dev36':
        R36.dev(base, f'b36_out/dev_{base}.jsonl', R36.FAMS, wk)
    elif mode == 'choose':
        R32.choose(base, f'b32_out/dev_{base}_vsa.jsonl,b32_out/dev_{base}_idx.jsonl', f'results/b32_choice_{base}.json')
        for p in ('vsa', 'idx'):
            open(f'b36_out/dev32_{base}_{p}.jsonl', 'a').close()  # no 1-2 bits per fact cells in R1
        R36.choose(base)
    elif mode == 'test32vsa':
        R32.test(base, f'results/b32_choice_{base}.json', 'b32_out/test', 'vsa', 'cpu')
    elif mode == 'test32idx':
        R32.test(base, f'results/b32_choice_{base}.json', 'b32_out/test', 'idx', 'cpu')
    elif mode == 'test36':
        R36.test(base, f'results/b36_choice_{base}.json', 'b36_out/test', wk)
    elif mode == 'merge':
        R32.merge('b32_out/test', [base])
        ch = json.load(open(f'results/b36_choice_{base}.json'))['choice']
        for M in (256, 1024):
            for V in R1_V:
                old = [json.loads(l) for l in gzip.open(f'b32_out/test/preds_{base}_M{M}_V{V}.jsonl.gz', 'rt')]
                new = [json.loads(l) for l in gzip.open(f'b36_out/test/new_{base}_M{M}_V{V}.jsonl.gz', 'rt')]
                with gzip.open(f'b36_out/test/all_{base}_M{M}_V{V}.jsonl.gz', 'wt') as fo:
                    for o, nw in zip(old, new):
                        assert o['rep'] == nw['rep'] and o['qgold'] == nw['qgold']
                        rec = {k: o[k] for k in ('enc', 'M', 'V', 'rep', 'uris', 'vals', 'qent', 'qalias', 'qgold')}
                        pred, cfg, bits = {'EXACT': o['pred']['EXACT'], 'EXACT_ENT': o['pred']['EXACT_ENT']}, {}, {}
                        for b in R1_BPF:
                            for acct, a32 in (('ideal', 'ideal'), ('real', 'raw')):
                                for arm in ('VSA', 'IDX'):
                                    pred[f'{arm}|{b}|{acct}'] = o['pred'][f'{arm}|{b}|{a32}']
                                    cfg[f'{arm}|{b}|{acct}'] = o['cfg'][f'{arm}|{b}|{a32}']
                                for f in R36.FAMS:
                                    pred[f'{f}|{b}|{acct}'] = nw['pred'][f'{f}|{b}|{acct}']
                                    cfg[f'{f}|{b}|{acct}'] = nw['cfg'][f'{f}|{b}|{acct}']
                                    bits[f'{f}|{b}|{acct}'] = nw['bits'][f'{f}|{b}|{acct}']
                                bf = ch[f'{M}|{V}|{b}|{acct}|BEST']['fam']
                                pred[f'BEST|{b}|{acct}'] = pred[f'{bf}|{b}|{acct}']
                                cfg[f'BEST|{b}|{acct}'] = [bf] + list(cfg[f'{bf}|{b}|{acct}'])
                        rec.update({'pred': pred, 'cfg': cfg, 'bits': bits, 'keybits': o.get('keybits', {})})
                        fo.write(json.dumps(rec) + '\n')
        print('merged', base, T)
    else:
        raise SystemExit(mode)


if __name__ == '__main__':
    main()
