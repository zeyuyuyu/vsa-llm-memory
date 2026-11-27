"""B59 part 1 (prereg/B59_maps_all_encoders_protected_full_prereg.md): three new random maps on Granite (b46), Llama,
Mistral and bge with B57's per-map procedure (b57_run's functions, unchanged), and the matched-filter (VSA) read of
every map's MF choice on the test stores for all seven encoders.

python3 b59_run.py ENC devm MAP PREP:FMAP      # B53's 72-configuration MF grid part with the map's seeds (b57_run.mf_dev)
python3 b59_run.py ENC choosem MAP             # B32's rule -> results/b59_choicem_ENC_mMAP.json
python3 b59_run.py ENC decm MAP SPLIT M V      # B56 engine (LM / RT / LR) on the map's MF choice (b57_run.engine)
python3 b59_run.py ENC lamm MAP SPLIT          # LAM's rp configurations with the map's projection (b57_run.lamm's code)
python3 b59_run.py ENC choose                  # per-map LM / RT / LR, STAR, LAMm, BESTm -> results/b59_choice_ENC.json
python3 b59_run.py ENC mft MAP                 # matched-filter read of the map's MF choice, test stores
python3 b59_run.py ENC merge                   # b57_out/fam/ENC/all57_* + the B59 map arms -> b59_out/fam/ENC/all59_*
environment: B59_OUT (default b59_out/ here); B53_DATA as b53_run.
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
import b32_run as R  # noqa: E402
import b49_common as B  # noqa: E402
import b53_run as R53  # noqa: E402
import b57_run as R57  # noqa: E402

OUT = os.environ.get('B59_OUT', os.path.join(ROOT, 'b59_out'))
R57.OUT = OUT                                   # b57_run.engine / odir write under B59's directory
NEW_ENCS = ['b46', 'llama', 'mistral', 'bge']
B57_ENCS = ['qwen8', 'e5', 'minilm']
MAPS = [1, 2, 3]
M_GRID, VN, BPF = R57.M_GRID, R57.VN, R57.BPF
ACCT32 = R57.ACCT32
PREPS = R57.PREPS
LAM_ALPHAS = R57.LAM_ALPHAS
E_ALL10 = R57.E_ALL10
tag = R57.tag


def odir(enc):
    d = os.path.join(OUT, enc)
    os.makedirs(d, exist_ok=True)
    return d


def chm_path(enc, m):
    if enc in B57_ENCS:
        return f'{ROOT}/results/b57_choicem_{enc}_m{m}.json'
    return f'{ROOT}/results/b59_choicem_{enc}_m{m}.json'


def ch_path(enc):
    return f'{ROOT}/results/b59_choice_{enc}.json'


# ----------------------------------------------------------------------------------------------- LAM on new maps
def lamm(enc, m, split):
    """b57_run.lamm with B59's choice file at test (code otherwise identical)."""
    pool, preps, Tr, _ = R57.setup(enc)
    tg = tag(enc, m)
    fn = os.path.join(odir(enc), f'lamm_{split}_{enc}_m{m}.jsonl.gz')
    if os.path.exists(fn):
        return
    ch = json.load(open(ch_path(enc)))['choice'] if split == 'test' else None
    lfs = {p: B.LamFeat(tg, p, preps[p](Tr)) for p in PREPS}
    with gzip.open(fn + '.tmp', 'wt') as fo:
        for M in M_GRID:
            for V in VN:
                for rep in range((R.DEV_REPS if split == 'dev' else R.TEST_REPS)[M]):
                    t0 = time.time()
                    st = C.make_store(pool, split, M, V, rep)
                    rec = {'enc': enc, 'map': m, 'M': M, 'V': V, 'rep': rep, 'qgold': st['qgold'].tolist(), 'pred': {},
                           'cfg': {}}
                    for prep in PREPS:
                        if split == 'dev':
                            cfgs = R57.lam_cfgs(['rp'])
                            want = None
                        else:
                            want = {}
                            for b in BPF:
                                for a in ('ideal', 'real'):
                                    c = ch[f'{M}|{V}|{b}|{a}|LAMm{m}']['cfg']
                                    if c[0] == prep:
                                        want.setdefault(tuple(c[1:]), []).append((b, a))
                            cfgs = list(want)
                        if not cfgs:
                            continue
                        P = preps[prep]
                        o = B.run_lam(st, P(st['keys']), P(st['queries']), lfs[prep], cfgs, BPF, ['ideal', 'real'],
                                      C.seed_of('b49', tg, split, M, V, rep))
                        for (c1, b, a), (p, bits) in o.items():
                            if want is not None and (b, a) not in want[tuple(c1)]:
                                continue
                            k = f'{prep}|{"|".join(map(str, c1))}|{b}|{a}'
                            rec['pred'][k] = np.asarray(p).tolist()
                            rec['cfg'][k] = bits
                    fo.write(json.dumps(rec) + '\n')
                    print(f'lamm {enc} m{m} {split} M={M} V={V} rep={rep} {time.time() - t0:.0f}s', flush=True)
    os.replace(fn + '.tmp', fn)


# ----------------------------------------------------------------------------------------------- MF test read
def mft(enc, m):
    """The map's MF-chosen configuration (B32's rule over B53's 72) read by the matched filter on the test stores:
    b53_run.test's code with the map's seed tag (the path b57_run.mf_dev evaluates on dev)."""
    ch = json.load(open(chm_path(enc, m)))['choice']
    pool, preps, Tr, lfs = R57.setup(enc)
    n = pool.X.shape[1]
    tg = tag(enc, m)
    fn = os.path.join(odir(enc), f'mft_test_{enc}_m{m}.jsonl.gz')
    if os.path.exists(fn):
        return
    feats = {}
    with gzip.open(fn + '.tmp', 'wt') as fo:
        for M in M_GRID:
            for V in VN:
                for rep in range(R.TEST_REPS[M]):
                    t0 = time.time()
                    st = C.make_store(pool, 'test', M, V, rep)
                    rec = {'enc': enc, 'map': m, 'M': M, 'V': V, 'rep': rep, 'qgold': st['qgold'].tolist(), 'pred': {},
                           'cfg': {}, 'd': {}}
                    groups = {}
                    for bpf in BPF:
                        for acct in ACCT32:
                            c = ch[f'{M}|{V}|{bpf}|{acct}|VSA']['cfg']
                            groups.setdefault((c[0], c[1]), []).append((bpf, acct, c[2], c[3]))
                            rec['cfg'][f'VSA|{bpf}|{acct}'] = c
                    for (prep, fmap), lst in groups.items():
                        if (prep, fmap) not in feats:
                            feats[prep, fmap] = R57.make_feat(tg, prep, fmap, n, V, lfs)
                        feats[prep, fmap].V = V
                        cfgs = [((bpf, acct, q, f), C.vsa_dim(bpf * st['M'], q, acct), q,
                                 max(1, int(math.floor(f * st['M'])))) for bpf, acct, q, f in lst]
                        P = preps[prep]
                        pr = R.vsa_grid(P(st['keys']), st['vals'], P(st['queries']), V, feats[prep, fmap], cfgs,
                                        R.store_seed(tg, st), R.store_order(st))
                        for (key, d, q, ns) in cfgs:
                            rec['pred'][f'VSA|{key[0]}|{key[1]}'] = pr[key].tolist()
                            rec['d'][f'VSA|{key[0]}|{key[1]}'] = [d, q, ns]
                    fo.write(json.dumps(rec) + '\n')
                    print(f'mft {enc} m{m} M={M} V={V} rep={rep} {time.time() - t0:.0f}s', flush=True)
    os.replace(fn + '.tmp', fn)


# ----------------------------------------------------------------------------------------------- choices (maps)
def choose(enc):
    """The map part of b57_run.choose, unchanged rules: MF (results/b59_choicem), LM / RT / LR (dec_choose), STAR,
    per-map LAM (rp re-run + B49's pca dev rows; B49's rule), per-map BEST (B49's ten-family choice with LAM's dev
    accuracy replaced by the map's)."""
    ch = {}
    b49 = {}
    for l in open(f'{ROOT}/b49_out/fam/{enc}/dev_{enc}.jsonl'):
        r = json.loads(l)
        if r['V'] not in VN:
            continue
        for x in r['rows']:
            if x['fam'] == 'LAM':
                b49.setdefault((r['M'], r['V'], x['bpf'], x['acct'], tuple(x['cfg'])), {})[r['rep']] = x['acc']
    ch49 = json.load(open(f'{ROOT}/results/b49_choice_{enc}.json'))['choice']
    for m in MAPS:
        chm = json.load(open(chm_path(enc, m)))['choice']
        for k, v in chm.items():
            ch[k.replace('|VSA', f'|MFm{m}')] = v
        Rl = [json.loads(l) for l in gzip.open(os.path.join(odir(enc), f'lamm_dev_{enc}_m{m}.jsonl.gz'), 'rt')]
        rows = {k: dict(v) for k, v in b49.items() if k[4][1] == 'pca'}
        for r in Rl:
            g = np.array(r['qgold'])
            for k, p in r['pred'].items():
                prep, ft, a, q, t, b, ac = k.split('|')
                c = (prep, ft, float(a), int(q), t)
                rows.setdefault((r['M'], r['V'], int(b), ac, c), {})[r['rep']] = float(np.mean(np.array(p) == g))
        for M in M_GRID:
            for V in VN:
                R_ = [json.loads(l) for l in gzip.open(os.path.join(odir(enc), f'decm_dev_{enc}_m{m}_M{M}_V{V}.jsonl.gz'),
                                                       'rt')]
                tmp = {}
                R57.dec_choose(R_, ['LM', 'RT', 'LR'], tmp, M, V)
                for k, v in tmp.items():
                    ch[k + f'm{m}'] = v
                for bpf in BPF:
                    for acct in ACCT32:
                        lm, rt = tmp[f'{M}|{V}|{bpf}|{acct}|LM'], tmp[f'{M}|{V}|{bpf}|{acct}|RT']
                        ch[f'{M}|{V}|{bpf}|{acct}|STARm{m}'] = {
                            'fam': 'RT' if rt['dev_acc'] > lm['dev_acc'] + 0.001 else 'LM'}
                for bpf in BPF:
                    for a in ('ideal', 'real'):
                        c, acc, ncfg = R57.lam_choice(rows, M, V, bpf, a)
                        ch[f'{M}|{V}|{bpf}|{a}|LAMm{m}'] = {'cfg': c, 'dev_acc': acc, 'n_cfg': ncfg}
                        dbf = dict(ch49[f'{M}|{V}|{bpf}|{a}|BEST']['dev_by_fam'])
                        dbf['LAM'] = acc
                        cands = [(dbf[f], -E_ALL10.index(f), f) for f in E_ALL10]
                        ch[f'{M}|{V}|{bpf}|{a}|BESTm{m}'] = {'fam': max(cands)[2], 'dev_by_fam': dbf}
    json.dump({'enc': enc, 'choice': ch}, open(ch_path(enc), 'w'), indent=1, sort_keys=True)
    print(len(ch), 'choices ->', ch_path(enc))


# ----------------------------------------------------------------------------------------------- merge
def merge(enc):
    """b57_out/fam/ENC/all57_* (B56's merged records + B57 arms; for B57's three map encoders already the map arms
    VSASm / LAMRFm / LAMm / BESTm) + B59: for the four new encoders the map arms (b57_run.merge's code), for all seven
    the matched-filter arm VSAm{m} (and VSAm0 := B53's VSA)."""
    amap = {'ideal': 'ideal', 'real': 'raw'}
    od = os.path.join(OUT, 'fam', enc)
    os.makedirs(od, exist_ok=True)
    ch = json.load(open(ch_path(enc)))['choice'] if enc in NEW_ENCS else None
    mf = {m: [json.loads(l) for l in gzip.open(os.path.join(odir(enc), f'mft_test_{enc}_m{m}.jsonl.gz'), 'rt')]
          for m in MAPS}
    for M in M_GRID:
        for V in VN:
            old = [json.loads(l) for l in gzip.open(f'{ROOT}/b57_out/fam/{enc}/all57_{enc}_M{M}_V{V}.jsonl.gz', 'rt')]
            mp = {}
            if enc in NEW_ENCS:
                for m in MAPS:
                    mp[m] = ([json.loads(l) for l in gzip.open(os.path.join(odir(enc), f'decm_test_{enc}_m{m}_M{M}_V{V}.jsonl.gz'), 'rt')],
                             [r for r in map(json.loads, gzip.open(os.path.join(odir(enc), f'lamm_test_{enc}_m{m}.jsonl.gz'), 'rt'))
                              if r['M'] == M and r['V'] == V])
            mfs = {m: [r for r in mf[m] if r['M'] == M and r['V'] == V] for m in MAPS}
            with gzip.open(f'{od}/all59_{enc}_M{M}_V{V}.jsonl.gz', 'wt') as fo:
                for i, o in enumerate(old):
                    o.setdefault('b59', {})
                    for m in MAPS:
                        assert mfs[m][i]['rep'] == o['rep'] and mfs[m][i]['qgold'] == o['qgold']
                    for b in BPF:
                        for a in ('ideal', 'real'):
                            a3 = amap[a]
                            o['pred'][f'VSAm0|{b}|{a}'] = o['pred'][f'VSA|{b}|{a}']
                            for m in MAPS:
                                o['pred'][f'VSAm{m}|{b}|{a}'] = mfs[m][i]['pred'][f'VSA|{b}|{a3}']
                                o['b59'][f'VSAm{m}|{b}|{a}'] = mfs[m][i]['cfg'][f'VSA|{b}|{a3}'] + mfs[m][i]['d'][f'VSA|{b}|{a3}']
                            for m, (dm, lm) in mp.items():
                                assert dm[i]['rep'] == o['rep'] and lm[i]['rep'] == o['rep']
                                assert dm[i]['qgold'] == o['qgold'] and lm[i]['qgold'] == o['qgold']
                                sel = {}
                                for arm in ('LM', 'RT', 'LR'):
                                    c = ch[f'{M}|{V}|{b}|{a3}|{arm}m{m}']['cfg']
                                    k = '|'.join(map(str, (arm, b, a3, c[0], c[1])))
                                    sel[arm] = dm[i]['pred'][k]
                                    o['b59'][f'{arm}m{m}|{b}|{a}'] = dm[i]['info'][k]
                                o['pred'][f'VSASm{m}|{b}|{a}'] = sel[ch[f'{M}|{V}|{b}|{a3}|STARm{m}']['fam']]
                                o['pred'][f'LAMRFm{m}|{b}|{a}'] = sel['LR']
                                lc = ch[f'{M}|{V}|{b}|{a}|LAMm{m}']['cfg']
                                o['pred'][f'LAMm{m}|{b}|{a}'] = lm[i]['pred'][f'{lc[0]}|{"|".join(map(str, lc[1:]))}|{b}|{a}']
                                o['b59'][f'LAMm{m}|{b}|{a}'] = lm[i]['cfg'][f'{lc[0]}|{"|".join(map(str, lc[1:]))}|{b}|{a}']
                                bf = ch[f'{M}|{V}|{b}|{a}|BESTm{m}']['fam']
                                o['pred'][f'BESTm{m}|{b}|{a}'] = o['pred'][f'LAMm{m}|{b}|{a}'] if bf == 'LAM' else \
                                    o['pred'][f'{bf}|{b}|{a}']
                    fo.write(json.dumps(o) + '\n')
    print('merged', enc)


def main():
    a = sys.argv[1:]
    enc, mode = a[0], a[1]
    if mode == 'devm':
        assert enc in NEW_ENCS
        m = int(a[2])
        R57.mf_dev(enc, m, a[3], os.path.join(odir(enc), f'devm_{enc}_m{m}_{a[3].replace(":", "_")}.jsonl'))
    elif mode == 'choosem':
        assert enc in NEW_ENCS
        m = int(a[2])
        R57.b32_choose([os.path.join(odir(enc), f'devm_{enc}_m{m}_{p}_{f}.jsonl') for p, f in R53.PARTS],
                       ['lin', 'sgn', 'sgn3'], chm_path(enc, m))
    elif mode == 'decm':
        assert enc in NEW_ENCS
        m, split, M, V = int(a[2]), a[3], int(a[4]), int(a[5])
        chm = json.load(open(chm_path(enc, m)))['choice']
        sel = None
        if split == 'test':
            ch = json.load(open(ch_path(enc)))['choice']
            ch2 = {k[:-len(f'm{m}')]: v for k, v in ch.items() if k.endswith(f'm{m}') and k.split('|')[-1][:2] in ('LM', 'RT', 'LR')}
            sel = R57.select_from(ch2, M, V, ['LM', 'RT', 'LR'])
        R57.engine(enc, m, split, M, V, chm, ['LM', 'RT', 'LR'], sel,
                   os.path.join(odir(enc), f'decm_{split}_{enc}_m{m}_M{M}_V{V}.jsonl.gz'))
    elif mode == 'lamm':
        assert enc in NEW_ENCS
        lamm(enc, int(a[2]), a[3])
    elif mode == 'choose':
        choose(enc)
    elif mode == 'mft':
        mft(enc, int(a[2]))
    elif mode == 'merge':
        merge(enc)
    else:
        raise SystemExit(mode)


if __name__ == '__main__':
    main()
