"""B62 checks (prereg/B62_online_ls_write_prereg.md, 'Checks'); a second angle on the verdict.

python3 b62_check.py scorer      # plain-loop cell accuracies and region point statistics vs results/b62_verdict.json
python3 b62_check.py bits        # every dev / test job row: bits <= budget and p maximal (own formula; RLScS with state)
python3 b62_check.py rederive    # float64 independent code with genuine primal NLMS / RLS recursions on a sample
python3 b62_check.py sample      # sample reading material (Qwen3-8B, e5): lowest / highest STORE - NLMSc R1 cell
All append a summary line to results/b62_check.txt.
"""
import gzip
import hashlib
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
RES = f'{ROOT}/results'
JOBS = f'{ROOT}/b62_out/jobs'
PANEL = ['qwen8', 'b46', 'llama', 'mistral', 'e5', 'bge', 'minilm']
M_GRID, VN, BPF = [256, 1024], [2, 16], [4, 8, 16]
HQ = {1: 1.0, 2: 1.9037, 3: 2.7606, 4: 3.6024}
NF = {'qwen8': 4096, 'b46': 4096, 'llama': 4096, 'mistral': 4096, 'e5': 768, 'bge': 384, 'minilm': 384}
TAG = {'qwen8': 'qwen8_mean_L9_f16', 'e5': 'e5_base', 'b46': 'b46_granite33i_mean_L30_f16',
       'llama': 'llama31i_mean_L24_f16', 'mistral': 'mistral7i_eol_L24_f16', 'bge': 'bge_small', 'minilm': 'minilm'}


def log(line):
    print(line, flush=True)
    with open(f'{RES}/b62_check.txt', 'a') as fo:
        fo.write(line + '\n')


# ----------------------------------------------------------------------------------------------- scorer

def scorer():
    import b62_verdict as VV
    J = json.load(open(f'{RES}/b62_verdict.json'))
    worst, n = 0.0, 0
    for enc in PANEL:
        data, ch = VV.load(enc)
        acc = {}
        for (M, V), recs in data.items():
            for b in BPF:
                for arm in ('STORE', 'LAMC', 'NLMSC', 'RLSC', 'RLSCS'):
                    k = f'{arm}|{b}|ideal'
                    hit, tot = 0, 0
                    for r in recs:
                        for p, g in zip(r['pred'][k], r['qgold']):
                            hit += int(p == g)
                            tot += 1
                    acc[M, V, b, arm] = hit / tot
                    worst = max(worst, abs(J['cells'][f'{enc}|{M}|{V}|{b}'][arm] - acc[M, V, b, arm]))
                    n += 1
        for rn, cells in (('R1', [(M, V, b) for M in M_GRID for V in VN for b in BPF]),
                          ('R1-V2', [(M, 2, b) for M in M_GRID for b in BPF]),
                          ('R1-V16', [(M, 16, b) for M in M_GRID for b in BPF])):
            for x, y in VV.PAIRS:
                s = 0.0
                for c in cells:
                    s += acc[c + (x,)] - acc[c + (y,)]
                s /= len(cells)
                worst = max(worst, abs(s - J['regions'][f'{enc}|{rn}|{x}-{y}']['diff']))
                n += 1
    json.dump({'max_abs_diff': worst, 'n_compared': n}, open(f'{RES}/b62_check_scorer.json', 'w'))
    log(f'scorer: {n} cell accuracies and region statistics recomputed by plain loops; max |difference| {worst:.3e} '
        f'({"PASS" if worst < 1e-9 else "FAIL"} at 1e-9)')


# ----------------------------------------------------------------------------------------------- bits

def own_bits(p, c, q, scale):
    if p == 0:
        return 0.0
    groups = c * ((p + 8191) // 8192) if scale == 'col' else (c * p + 8191) // 8192
    return c * p * HQ[q] + 32 * groups


def bits():
    over = notmax = n = 0
    bad = []
    for fn in sorted(os.listdir(JOBS)):
        r = json.load(gzip.open(os.path.join(JOBS, fn), 'rt'))
        s = r['spec']
        for x in r['rows']:
            scale, t, prep, feat, _, _, q = x['cfg']
            c = s['V'] - 1 if t == 'V-1' else s['V']
            B = x['bpf'] * s['M']
            p = x['p']
            cap = 65536 if feat == 'rp' else NF[s['enc']]
            st = (lambda k: 16 * k * (k + 1)) if x['arm'] == 'RLSCS' else (lambda k: 0)
            b1 = own_bits(p, c, q, scale)
            n += 1
            if st(p) + b1 > B or abs(b1 - x['bits']) > 1e-6:
                over += 1
                bad.append([fn, x['arm'], x['cfg'], x['bpf'], p, b1, x['bits']])
            if p < cap and st(p + 1) + own_bits(p + 1, c, q, scale) <= B:
                notmax += 1
                bad.append([fn, x['arm'], x['cfg'], x['bpf'], p, 'not maximal'])
    json.dump({'n': n, 'over_or_mismatch': over, 'not_maximal': notmax, 'examples': bad[:20]},
              open(f'{RES}/b62_check_bits.json', 'w'))
    log(f'bits: {n} job rows (dev + test); over budget or logged bits mismatching {over}; not maximal {notmax} '
        f'({"PASS" if over == 0 and notmax == 0 else "FAIL"})')


# ----------------------------------------------------------------------------------------------- re-derivation

def seed_of(*parts):
    return int(hashlib.sha256('|'.join(map(str, parts)).encode()).hexdigest()[:15], 16)


def quant(x, q, rng):
    x = np.asarray(x, np.float64)
    if q == 1:
        s = np.where(x > 0, 1.0, np.where(x < 0, -1.0, 0.0))
        z = s == 0
        if z.any():
            s[z] = rng.choice([-1.0, 1.0], size=int(z.sum()))
        return s * (np.abs(x).mean() or 1.0)
    sd = float(np.float32(x.astype(np.float32).std())) or 1.0
    st = {2: 0.9957, 4: 0.3352}[q] * sd
    L = 2 ** q
    return (np.clip(np.floor(x / st), -L // 2, L // 2 - 1) + 0.5) * st


def own_quantW(W, q, scale, rng):
    p, c = W.shape
    if scale == 'col':
        Wq = np.zeros_like(W)
        for j0 in range(0, p, 8192):
            for k in range(c):
                Wq[j0:j0 + 8192, k] = quant(W[j0:j0 + 8192, k], q, rng)
        return Wq
    fl = W.T.reshape(-1)
    return np.concatenate([quant(fl[i:i + 8192], q, rng) for i in range(0, len(fl), 8192)]).reshape(c, p).T


def own_nlms(F, Y, mu):
    W = np.zeros((F.shape[1], Y.shape[1]))
    S = np.zeros_like(W)
    for t in range(F.shape[0]):
        f = F[t]
        W += (mu / (f @ f + 1e-12)) * np.outer(f, Y[t] - f @ W)
        S += W
    return W, S / F.shape[0]


def own_rls(F, Y, delta, beta):
    p = F.shape[1]
    P = np.eye(p) / delta
    W = np.zeros((p, Y.shape[1]))
    for t in range(F.shape[0]):
        f = F[t]
        Pf = P @ f
        k = Pf / (beta + f @ Pf)
        W += np.outer(k, Y[t] - f @ W)
        P -= np.outer(k, Pf)
        P /= beta
    return W


def rederive():
    """Independent float64 code: own preps (float64 eigh), own feature maps (rp blocks regenerated from the registered
    seeds), own training norms, genuine primal NLMS loop and RLS recursion (p <= 4,096), own quantizer and read; and the
    primal / recursive W against b62_run's dual / closed forms on the same features."""
    import b32_common as C
    import b62_run as R62
    out = {}
    cwd = os.getcwd()
    sample = [(e, 256, V) for e in PANEL for V in VN] + [(e, 1024, V) for e in ('qwen8', 'e5') for V in VN]
    for enc in PANEL:
        os.chdir(f'{ROOT}/b49_out')
        pool = C.Pool(TAG[enc])
        os.chdir(cwd)
        Tr = pool.train_matrix().astype(np.float64)
        mu = Tr.mean(0)
        w, U = np.linalg.eigh(np.cov((Tr - mu).T))
        w = np.maximum(w, 0)
        U0 = U[:, np.argsort(-w)]
        Wz = (U / np.sqrt(w + 0.1 * w.mean())) @ U.T
        n = Tr.shape[1]
        ch = json.load(open(f'{RES}/b62_choice_{enc}.json'))['choice']
        idx = np.sort(np.random.default_rng(seed_of('B62TN', enc)).choice(len(Tr), size=min(2000, len(Tr)),
                                                                          replace=False))

        def prep(X, kind):
            Y = np.asarray(X, np.float64) - mu
            if kind == 'whiten':
                Y = Y @ Wz
            return Y / np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)

        def fmap(pr, feat, p):
            if feat == 'pca':
                return U0[:, :p]
            return np.hstack([np.random.default_rng(seed_of('LAMG', enc, pr, j0)).standard_normal(
                (n, 8192), dtype=np.float32)[:, :min(8192, p - j0)].astype(np.float64) / math.sqrt(n)
                for j0 in range(0, p, 8192)])
        for (e, M, V) in sample:
            if e != enc:
                continue
            st = C.make_store(pool, 'test', M, V, 0)
            order = np.random.default_rng(seed_of('subset', 'test', M, V, 0)).permutation(M)
            for b in (4, 16):
                for arm in ('NLMSC', 'RLSC'):
                    cfg = ch[f'{M}|{V}|{b}|{arm}']['cfg']
                    scale, t, pr, feat, x, y, q = cfg
                    jf = f'{JOBS}/{enc}_test_M{M}_V{V}_r0_{pr}.json.gz'
                    row = [r for r in json.load(gzip.open(jf, 'rt'))['rows']
                           if r['arm'] == arm and r['cfg'] == cfg and r['bpf'] == b][0]
                    p = row['p']
                    key = f'{enc}|{M}|{V}|{b}|{arm}'
                    if arm == 'RLSC' and p > 4096:
                        out[key] = {'cfg': cfg, 'p': p, 'skipped': 'p > 4096 (p x p recursion not run)'}
                        print(key, out[key], flush=True)
                        continue
                    G = fmap(pr, feat, p)
                    Fk, Fq = prep(st['keys'], pr) @ G, prep(st['queries'], pr) @ G
                    Fo = Fk[order]
                    Tm = np.eye(V)[st['vals'][order]] - 1.0 / V
                    if t == 'V-1':
                        Tm = Tm[:, 1:]
                    if arm == 'NLMSC':
                        Wl, Wa = own_nlms(Fo, Tm, x)
                        W = Wl if y == 'last' else Wa
                        D = R62.nlms_dual(Fo @ Fo.T, Tm, x)
                        om = (M - np.arange(M)) / M
                        Wd = Fo.T @ (D if y == 'last' else om[:, None] * D)
                    else:
                        s_p = float((((prep(Tr[idx], pr) @ G) ** 2).sum(1)).mean())
                        W = own_rls(Fo, Tm, x * s_p, y)
                        Wd = R62.rls_closed(Fo, Tm, x * s_p, y)
                    rel = float(np.abs(W - Wd).max() / max(np.abs(W).max(), 1e-300))
                    Wq = own_quantW(W, q, scale, np.random.default_rng(1))
                    Sc = Fq @ Wq
                    if t == 'V-1':
                        Sc = np.concatenate([-Sc.sum(1, keepdims=True), Sc], 1)
                    pred = Sc.argmax(1)
                    logged = np.array(row['pred'])
                    out[key] = {'cfg': cfg, 'p': p, 'agree': float((pred == logged).mean()), 'rel_dW': rel,
                                'acc_logged': float((logged == st['qgold']).mean()),
                                'acc_rederived': float((pred == st['qgold']).mean())}
                    print(key, out[key], flush=True)
    json.dump(out, open(f'{RES}/b62_check_rederive.json', 'w'), indent=0)
    ag = [v['agree'] for v in out.values() if 'agree' in v]
    rn = [v['rel_dW'] for k, v in out.items() if 'agree' in v and k.endswith('NLMSC')]
    rr = [v['rel_dW'] for k, v in out.items() if 'agree' in v and k.endswith('RLSC')]
    sk = sum('skipped' in v for v in out.values())
    log(f'rederive: float64 independent code (genuine primal loops) on {len(ag)} (encoder, M, V, bpf, arm) test cells '
        f'(rep 0; {sk} RLSc cells with p > 4,096 not recursed): agreement min {min(ag):.4f} mean {np.mean(ag):.4f} '
        f'({"PASS" if min(ag) >= 0.99 else "FAIL"} at 0.99); NLMS primal loop vs dual form max rel |dW| {max(rn):.2e} '
        f'({"PASS" if max(rn) < 1e-8 else "FAIL"} at 1e-8); RLS recursion vs closed form max rel |dW| {max(rr):.2e} '
        f'({"PASS" if max(rr) < 1e-6 else "FAIL"} at 1e-6)')


# ----------------------------------------------------------------------------------------------- sample reading material

def sample():
    import b62_verdict as VV
    J = json.load(open(f'{RES}/b62_verdict.json'))
    name = {e['uri']: e['name'] for e in json.load(open(f'{ROOT}/b49_out/b32_data/entities.json'))['entities']}
    L = []
    for enc in ('qwen8', 'e5'):
        data, ch = VV.load(enc)
        cells = sorted((J['cells'][f'{enc}|{M}|{V}|{b}']['STORE'] - J['cells'][f'{enc}|{M}|{V}|{b}']['NLMSC'], M, V, b)
                       for M in M_GRID for V in VN for b in BPF)
        for tag_, (d, M, V, b) in (('lowest', cells[0]), ('highest', cells[-1])):
            recs = data[M, V]
            only_n, only_s = [], []
            for r in recs:
                for i, g in enumerate(r['qgold']):
                    s_ok = r['pred'][f'STORE|{b}|ideal'][i] == g
                    n_ok = r['pred'][f'NLMSC|{b}|ideal'][i] == g
                    item = (r['rep'], r['qalias'][i], name.get(r['uris'][r['qent'][i]], r['uris'][r['qent'][i]]), g,
                            r['pred'][f'STORE|{b}|ideal'][i], r['pred'][f'NLMSC|{b}|ideal'][i],
                            r['pred'][f'LAMC|{b}|ideal'][i])
                    if n_ok and not s_ok:
                        only_n.append(item)
                    elif s_ok and not n_ok:
                        only_s.append(item)
            rng = np.random.default_rng(62)
            L.append(f'## {enc} ({tag_} STORE - NLMSc cell) M={M} V={V} bpf={b}: STORE - NLMSc {100 * d:+.2f}; NLMSc cfg '
                     f'{ch[f"{M}|{V}|{b}|NLMSC"]["cfg"]} p={ch[f"{M}|{V}|{b}|NLMSC"]["p"]}; only NLMSc right {len(only_n)}, '
                     f'only store right {len(only_s)}')
            for nm, lst in (('only NLMSc right', only_n), ('only store right', only_s)):
                L.append(f'  {nm}: (rep, alias, stored label, gold, store answer, NLMSc answer, LAMc answer)')
                for j in rng.choice(len(lst), size=min(10, len(lst)), replace=False):
                    L.append(f'    {lst[j]}')
    open(f'{RES}/b62_sample_material.txt', 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    {'scorer': scorer, 'bits': bits, 'rederive': rederive, 'sample': sample}[sys.argv[1]]()
