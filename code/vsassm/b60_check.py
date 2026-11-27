"""B60 checks (prereg/B60_corrected_lam_prereg.md, 'Checks'); a second angle on the verdict.

python3 b60_check.py scorer      # plain-loop cell accuracies and R1 point statistics vs results/b60_verdict.json
python3 b60_check.py bits        # every dev / test job row: bits <= budget and p maximal (own formula)
python3 b60_check.py overlap     # results/b60_overlap.json: LAMc >= 0.95 everywhere (and B49's LamFeat beside)
python3 b60_check.py step5       # agreement with the diagnosis's step5 implementation (Qwen3-8B)
python3 b60_check.py rederive    # float64 re-derivation, one test store per (encoder, M, V), 4 / 16 / 64 bpf, ideal
python3 b60_check.py sample      # sample reading material (Qwen3-8B, e5): R1-V16 cell with the lowest STORE - LAMc
All write to results/b60_check_*.json and append a summary line to results/b60_check.txt.
"""
import gzip
import json
import math
import os
import pickle
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
RES = f'{ROOT}/results'
JOBS = f'{ROOT}/b60_out/jobs'
PANEL = ['qwen8', 'b46', 'llama', 'mistral', 'e5', 'bge', 'minilm']
M_GRID, VN, BPF = [256, 1024], [2, 16], [1, 2, 4, 8, 16, 32, 64]
LD = f'{ROOT}/b60_out/step5'  # b60_step5.py's output (the diagnosis's step5 code, reconstructed; compute note 1)
HQ = {1: 1.0, 2: 1.9037, 3: 2.7606, 4: 3.6024}


def log(line):
    print(line, flush=True)
    with open(f'{RES}/b60_check.txt', 'a') as fo:
        fo.write(line + '\n')


# ----------------------------------------------------------------------------------------------- scorer

def scorer():
    import b60_verdict as VV
    J = json.load(open(f'{RES}/b60_verdict.json'))
    worst, n = 0.0, 0
    for enc in PANEL:
        data, status, ch = VV.load(enc)
        acc = {}
        for (M, V), recs in data.items():
            for b in BPF:
                for arm in ('STORE', 'LAMC', 'BESTC', 'LAM', 'BEST', 'VSAR', 'VSAS'):
                    k = f'{arm}|{b}|ideal'
                    if k not in recs[0]['pred']:
                        continue
                    hit, tot = 0, 0
                    for r in recs:
                        for p, g in zip(r['pred'][k], r['qgold']):
                            hit += int(p == g)
                            tot += 1
                    acc[M, V, b, arm] = hit / tot
                    jv = J['cells'][f'{enc}|ideal|{M}|{V}|{b}'].get(arm)
                    if jv is not None:
                        worst = max(worst, abs(jv - acc[M, V, b, arm]))
                        n += 1
        for rn, cells in (('R1', [(M, V, b) for M in M_GRID for V in VN for b in (4, 8, 16)]),
                          ('R1-V2', [(M, 2, b) for M in M_GRID for b in (4, 8, 16)]),
                          ('R1-V16', [(M, 16, b) for M in M_GRID for b in (4, 8, 16)])):
            for x, y in (('STORE', 'LAMC'), ('STORE', 'BESTC'), ('LAMC', 'LAM'), ('VSAR', 'LAMC'), ('VSAS', 'LAMC')):
                s = 0.0
                for c in cells:
                    s += acc[c + (x,)] - acc[c + (y,)]
                s /= len(cells)
                jv = J['regions'][f'{enc}|ideal|{rn}|{x}-{y}']['diff']
                worst = max(worst, abs(s - jv))
                n += 1
    json.dump({'max_abs_diff': worst, 'n_compared': n}, open(f'{RES}/b60_check_scorer.json', 'w'))
    log(f'scorer: {n} cell accuracies and region statistics recomputed by plain loops; max |difference| {worst:.3e} '
        f'({"PASS" if worst < 1e-9 else "FAIL"} at 1e-9)')


# ----------------------------------------------------------------------------------------------- bits

def own_bits(p, c, q, acct, scale):
    per = HQ[q] if acct == 'ideal' else q
    if p == 0:
        return 0.0
    groups = c * ((p + 8191) // 8192) if scale == 'col' else (c * p + 8191) // 8192
    return c * p * per + 32 * groups


def ncol(V, t):
    return V - 1 if t == 'V-1' else (V if t == 'V' else int(t[4:]))


def nfeat(enc):
    return {'qwen8': 4096, 'b46': 4096, 'llama': 4096, 'mistral': 4096, 'e5': 768, 'bge': 384, 'minilm': 384}[enc]


def bits():
    over = notmax = n = 0
    bad = []
    for fn in sorted(os.listdir(JOBS)):
        r = json.load(gzip.open(os.path.join(JOBS, fn), 'rt'))
        s = r['spec']
        for x in r['rows']:
            scale, t, prep, feat, a, q = x['cfg']
            c = ncol(s['V'], t)
            B = x['bpf'] * s['M']
            p = x['p']
            cap = 65536 if feat == 'rp' else nfeat(s['enc'])
            b1 = own_bits(p, c, q, x['acct'], scale)
            n += 1
            if b1 > B or abs(b1 - x['bits']) > 1e-6:
                over += 1
                bad.append([fn, x['cfg'], x['bpf'], x['acct'], p, b1, x['bits']])
            if p < cap and own_bits(p + 1, c, q, x['acct'], scale) <= B:
                notmax += 1
                bad.append([fn, x['cfg'], x['bpf'], x['acct'], p, 'not maximal'])
    json.dump({'n': n, 'over_or_mismatch': over, 'not_maximal': notmax, 'examples': bad[:20]},
              open(f'{RES}/b60_check_bits.json', 'w'))
    log(f'bits: {n} job rows (dev + test, all maps); over budget or logged bits mismatching {over}; not maximal {notmax} '
        f'({"PASS" if over == 0 and notmax == 0 else "FAIL"})')


# ----------------------------------------------------------------------------------------------- overlap

def overlap():
    o = json.load(open(f'{RES}/b60_overlap.json'))
    mn = min(v['lamc'] for v in o.values())
    w49 = {k: v['b49'] for k, v in o.items() if '|whiten|' in k}
    c49 = {k: v['b49'] for k, v in o.items() if '|center|' in k}
    log(f'overlap: LAMc pca map vs independent float64 top-p eigenvectors, min over {len(o)} (encoder, prep, p) = '
        f'{mn:.4f} ({"PASS" if mn >= 0.95 else "FAIL"} at 0.95); B49 LamFeat whiten min {min(w49.values()):.3f} '
        f'max {max(w49.values()):.3f}; B49 LamFeat center min {min(c49.values()):.3f}')


# ----------------------------------------------------------------------------------------------- step5 agreement

def step5():
    out = {'dev': {}, 'test': {}}
    ch = json.load(open(f'{RES}/b60_choice_qwen8.json'))['choice']
    for M in M_GRID:
        for V in VN:
            fn = f'{LD}/step5_M{M}_V{V}.pkl'
            if not os.path.exists(fn):
                continue
            S5 = pickle.load(open(fn, 'rb'))
            for split in ('dev', 'test'):
                mine = {}
                for rep in S5[split]:
                    for prep in ('center', 'whiten'):
                        jf = f'{JOBS}/qwen8_m0_{split}_M{M}_V{V}_r{rep}_{prep}.json.gz'
                        if not os.path.exists(jf):
                            continue
                        r = json.load(gzip.open(jf, 'rt'))
                        assert r['qgold'] == S5[split][rep]['qgold'].tolist()
                        for x in r['rows']:
                            if 'pred' in x and x['acct'] == 'ideal' and x['bpf'] in (4, 8, 16):
                                mine[rep, tuple(x['cfg']), x['bpf']] = np.array(x['pred'])
                for k5 in S5[split][0]['pred']:
                    if k5[0] != 'FX':
                        continue
                    _, pr, ft, a, q, cols, rule = k5
                    cfg = (rule, cols, pr, 'pca' if ft in ('pca', 'pcafix') else ft, float(a), int(q))
                    ident = not (pr == 'center' and ft == 'pca')
                    for b in (4, 8, 16):
                        agree, tot, acc5, acc6 = 0, 0, [], []
                        for rep in S5[split]:
                            m_ = mine.get((rep, cfg, b))
                            if m_ is None:
                                continue
                            p5 = np.asarray(S5[split][rep]['pred'][k5][b])
                            g = S5[split][rep]['qgold']
                            agree += int((p5 == m_).sum())
                            tot += len(m_)
                            acc5.append((p5 == g).mean())
                            acc6.append((m_ == g).mean())
                        if tot:
                            out[split][f'{M}|{V}|{b}|{"|".join(map(str, cfg))}'] = {
                                'agree': agree / tot, 'identical_maps': ident, 'n': tot,
                                'acc_step5': float(np.mean(acc5)), 'acc_b60': float(np.mean(acc6)),
                                'chosen': ch[f'{M}|{V}|{b}|ideal|LAMC']['cfg'] == list(cfg)}
    json.dump(out, open(f'{RES}/b60_check_step5.json', 'w'), indent=0)
    for split in ('dev', 'test'):
        idv = [v['agree'] for v in out[split].values() if v['identical_maps']]
        cpv = [v['agree'] for v in out[split].values() if not v['identical_maps']]
        da = [abs(v['acc_step5'] - v['acc_b60']) for v in out[split].values() if v['identical_maps']]
        if idv:
            log(f'step5 agreement ({split} stores, Qwen3-8B, R1 bpf, ideal): identical-map configurations {len(idv)} cells, '
                f'agreement min {min(idv):.4f} mean {np.mean(idv):.4f}, cells >= 0.99: {sum(x >= 0.99 for x in idv)} '
                f'({"PASS" if min(idv) >= 0.99 else "FAIL"} at 0.99); max |accuracy difference| {100 * max(da):.2f} pt; '
                f'center pca (different basis by design) {len(cpv)} cells, agreement min {min(cpv) if cpv else float("nan"):.4f} '
                f'mean {np.mean(cpv) if cpv else float("nan"):.4f}')


# ----------------------------------------------------------------------------------------------- float64 re-derivation

def rederive():
    """Independent float64 code: own preps (float64 eigh), own feature maps (rp blocks regenerated from B49's seeds),
    own ridge solve (numpy.linalg.solve, dual form), own quantizer and read."""
    import hashlib
    import b32_common as C

    def seed_of(*parts):
        return int(hashlib.sha256('|'.join(map(str, parts)).encode()).hexdigest()[:15], 16)

    MAXS = {2: 0.9957, 4: 0.3352}

    def quant(x, q, rng):
        x = np.asarray(x, np.float64)
        if q == 1:
            s = np.where(x > 0, 1.0, np.where(x < 0, -1.0, 0.0))
            z = s == 0
            if z.any():
                s[z] = rng.choice([-1.0, 1.0], size=int(z.sum()))
            return s * (np.abs(x).mean() or 1.0)
        sd = float(np.float32(x.astype(np.float32).std())) or 1.0
        st = MAXS[q] * sd
        L = 2 ** q
        k = np.clip(np.floor(x / st), -L // 2, L // 2 - 1)
        return (k + 0.5) * st

    out = {}
    os.chdir(f'{ROOT}/b49_out')
    for enc in PANEL:
        pool = C.Pool({'qwen8': 'qwen8_mean_L9_f16', 'e5': 'e5_base', 'b46': 'b46_granite33i_mean_L30_f16',
                       'llama': 'llama31i_mean_L24_f16', 'mistral': 'mistral7i_eol_L24_f16', 'bge': 'bge_small',
                       'minilm': 'minilm'}[enc])
        Tr = pool.train_matrix().astype(np.float64)
        mu = Tr.mean(0)
        w, U = np.linalg.eigh(np.cov((Tr - mu).T))
        w = np.maximum(w, 0)
        o = np.argsort(-w)
        U0 = U[:, o]
        Wz = (U / np.sqrt(w + 0.1 * w.mean())) @ U.T
        n = Tr.shape[1]
        ch = json.load(open(f'{RES}/b60_choice_{enc}.json'))['choice']

        def prep(X, kind):
            Y = np.asarray(X, np.float64) - mu
            if kind == 'whiten':
                Y = Y @ Wz
            return Y / np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)

        for M in M_GRID:
            for V in VN:
                st = C.make_store(pool, 'test', M, V, 0)
                for b in (4, 16, 64):
                    cfg = ch[f'{M}|{V}|{b}|ideal|LAMC']['cfg']
                    scale, t, pr, feat, alpha, q = cfg
                    jf = f'{JOBS}/{enc}_m0_test_M{M}_V{V}_r0_{pr}.json.gz'
                    row = [x for x in json.load(gzip.open(jf, 'rt'))['rows'] if x['cfg'] == cfg and x['bpf'] == b
                           and x['acct'] == 'ideal'][0]
                    p = row['p']
                    Yk, Yq = prep(st['keys'], pr), prep(st['queries'], pr)
                    if p == 0:
                        pred = np.full(len(Yq), np.bincount(st['vals'], minlength=V).argmax())
                    else:
                        if feat == 'pca':
                            G = U0[:, :p]
                        else:
                            G = np.hstack([np.random.default_rng(seed_of('LAMG', enc, pr, j0)).standard_normal(
                                (n, 8192), dtype=np.float32)[:, :min(8192, p - j0)].astype(np.float64) /
                                math.sqrt(n) for j0 in range(0, p, 8192)])
                        Fk, Fq = Yk @ G, Yq @ G
                        c = V - 1 if t == 'V-1' else V
                        Tm = np.eye(V)[st['vals']] - 1.0 / V
                        if t == 'V-1':
                            Tm = Tm[:, 1:]
                        K = Fk @ Fk.T
                        A = np.linalg.solve(K + alpha * np.trace(K) / M * np.eye(M), Tm)
                        W = Fk.T @ A
                        rng = np.random.default_rng(1)
                        if scale == 'col':
                            Wq = np.zeros_like(W)
                            for j0 in range(0, p, 8192):
                                for k in range(c):
                                    Wq[j0:j0 + 8192, k] = quant(W[j0:j0 + 8192, k], q, rng)
                        else:
                            fl = W.T.reshape(-1)
                            fq = np.concatenate([quant(fl[i:i + 8192], q, rng) for i in range(0, len(fl), 8192)])
                            Wq = fq.reshape(c, p).T
                        Sc = Fq @ Wq
                        if t == 'V-1':
                            Sc = np.concatenate([-Sc.sum(1, keepdims=True), Sc], 1)
                        pred = Sc.argmax(1)
                    logged = np.array(row['pred'])
                    out[f'{enc}|{M}|{V}|{b}'] = {'cfg': cfg, 'p': p, 'agree': float((pred == logged).mean()),
                                                 'acc_logged': float((logged == st['qgold']).mean()),
                                                 'acc_rederived': float((pred == st['qgold']).mean())}
                    print(enc, M, V, b, cfg, out[f'{enc}|{M}|{V}|{b}']['agree'], flush=True)
    json.dump(out, open(f'{RES}/b60_check_rederive.json', 'w'), indent=0)
    ag = [v['agree'] for v in out.values()]
    log(f'rederive: float64 independent code on {len(ag)} (encoder, M, V, bpf) test cells (rep 0, ideal, chosen LAMc): '
        f'agreement min {min(ag):.4f} mean {np.mean(ag):.4f} ({"PASS" if min(ag) >= 0.99 else "FAIL"} at 0.99)')


# ----------------------------------------------------------------------------------------------- sample reading material

def sample():
    import b60_verdict as VV
    J = json.load(open(f'{RES}/b60_verdict.json'))
    name = {e['uri']: e['name'] for e in json.load(open(f'{ROOT}/b32_data/entities.json'))['entities']}
    L = []
    for enc in ('qwen8', 'e5'):
        data, _, ch = VV.load(enc)
        cells = [(J['cells'][f'{enc}|ideal|{M}|16|{b}']['STORE'] - J['cells'][f'{enc}|ideal|{M}|16|{b}']['LAMC'], M, b)
                 for M in M_GRID for b in (4, 8, 16)]
        d, M, b = min(cells)
        recs = data[M, 16]
        only_l, only_s = [], []
        for r in recs:
            for i, g in enumerate(r['qgold']):
                s_ok = r['pred'][f'STORE|{b}|ideal'][i] == g
                l_ok = r['pred'][f'LAMC|{b}|ideal'][i] == g
                item = (r['rep'], r['qalias'][i], name.get(r['uris'][r['qent'][i]], r['uris'][r['qent'][i]]), g,
                        r['pred'][f'STORE|{b}|ideal'][i], r['pred'][f'LAMC|{b}|ideal'][i])
                if l_ok and not s_ok:
                    only_l.append(item)
                elif s_ok and not l_ok:
                    only_s.append(item)
        rng = np.random.default_rng(60)
        L.append(f'## {enc} M={M} V=16 bpf={b}: STORE - LAMc {100 * d:+.2f}; LAMc cfg {ch[f"{M}|16|{b}|ideal|LAMC"]["cfg"]}; '
                 f'only LAMc right {len(only_l)}, only store right {len(only_s)}')
        for nm, lst in (('only LAMc right', only_l), ('only store right', only_s)):
            L.append(f'  {nm}: (rep, alias, stored label, gold, store answer, LAMc answer)')
            for j in rng.choice(len(lst), size=min(10, len(lst)), replace=False):
                L.append(f'    {lst[j]}')
    open(f'{RES}/b60_sample_material.txt', 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    {'scorer': scorer, 'bits': bits, 'overlap': overlap, 'step5': step5, 'rederive': rederive,
     'sample': sample}[sys.argv[1]]()
