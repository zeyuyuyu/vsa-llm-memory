"""B56 (prereg/B56_decoders_retraining_prereg.md): the store's read and write rules inside the VSA framework, on B53's
panel, splits, stores, budgets and accounting.

Arms (each on B53's store, i.e. B49's projections and seeds with B53's antipodal / Hadamard value codes):
  LM   VSA-LMMSE: the bundle written by the store's MAP rule (B53's dev-chosen prep, feature map and stored share per
       cell; q in {1, 2, 4}), read by a linear minimum-mean-square-error decoder of the superposition whose statistics
       come from the TRAINING split only (T rows of the training matrix): per residue class c of the coordinates
       (V = 16: j = c mod 16, the classes that the Hadamard codes keep apart; V = 2: one class),
       m_c(q) = phi_c(q)^T (Sigma_c + lam_c I)^-1 z_c,  z = Q(S) * r,  Sigma_c = F_c^T F_c / T,
       lam_c = alpha * mean diag(Sigma_c), alpha in {inf (= the matched filter, B53), 10, 1, 0.1, 0.01};
       score(v) = sum_c H16[v, c] m_c (V = 16) or +-m (V = 2). 0 bits: nothing per store beyond the bundle.
  RT   VSA-retrained: the Hebbian bundle (same configuration) retrained on the stored facts by the HDC / perceptron
       rule (mini-batches of 32 in store order; for every misread fact S += phi(k) * (c(v) - c(v_hat))), E in
       {1, 3, 10} epochs, then quantized to q bits (same d, same accounting); read by the matched filter.
  LR   LAM-RF: B49's least-squares associative memory (ridge alphas, Max quantizer per column, onehot targets, its bit
       rule) on exactly the store's features (the cell's B53 prep and feature map, same projection matrices).
  W    widened store grid: B53's 72 configurations + 72 with a sparsifying ternary feature map
       phi_j = sign(p_j) [|p_j| >= theta_kappa] on the sign store's projections, kappa in {1/4, 1/16, 1/64}
       (theta = the two-sided Gaussian quantile; p_j ~ N(0, 1) for unit-norm codes); matched-filter read.

python3 b56_run.py selftest
python3 b56_run.py ENC devw PREP:FMAP            # one sparse part of the widened grid (dev stores, V 2 / 16, 1-64 bpf)
python3 b56_run.py ENC dec dev M V               # LM / RT / LR, every configuration, dev stores of (M, V)
python3 b56_run.py ENC choose                    # -> results/b56_choice_ENC.json
python3 b56_run.py ENC dec test M V              # chosen configurations only, test stores of (M, V)
python3 b56_run.py ENC testw                     # widened-grid choices that are sparse configurations, test stores
python3 b56_run.py ENC merge                     # B53's merged files + the B56 arms -> b56_out/fam/ENC/all56_*
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

OUT = os.environ.get('B56_OUT', os.path.join(ROOT, 'b56_out'))
PANEL = R53.PANEL
M_GRID, VN, BPF = [256, 1024], [2, 16], [1, 2, 4, 8, 16, 32, 64]
ACCT32 = ['ideal', 'raw']
QS = [1, 2, 4]
ALPHA_LM = [math.inf, 10.0, 1.0, 0.1, 0.01]
EPOCHS = [1, 3, 10]
ETAS = [1.0, 0.1]
BATCH = 32
T_V = {2: 1024, 16: 512}
LAM_ALPHAS = [1e-4, 1e-2, 1.0, 100.0]  # B49's
P_MAX = 65536
SPARSE = {'tern4': 0.25, 'tern16': 1 / 16, 'tern64': 1 / 64}
FMAPS_W = ['lin', 'sgn', 'sgn3'] + list(SPARSE)
PARTS_W = [(p, f) for p in ('center', 'whiten') for f in SPARSE]
H16 = R53.H16.astype(np.float64)
CH = R.CH


def odir(enc):
    d = os.path.join(OUT, enc)
    os.makedirs(d, exist_ok=True)
    return d


# ----------------------------------------------------------------------------------------------- W: sparse features

class FeatW(R53.Feat53):
    """The sign store's projections and value codes (seeds of (enc, prep, 'sgn')); sparsifying ternary map."""

    def __init__(self, enc, prep, fmap, n, device):
        super().__init__(enc, prep, 'sgn', n, device)
        from scipy.stats import norm
        self.fmap = fmap
        self.kappa = SPARSE[fmap]
        self.theta = float(norm.ppf(1 - self.kappa / 2))

    def phi(self, Y, Gs, w):
        t = self.torch
        P = Y @ Gs[0][:, :w]
        return t.where(P.abs() >= self.theta, t.sign(P), t.zeros_like(P))


def make_feat(enc, prep, fmap, n, V):
    f = FeatW(enc, prep, fmap, n, 'cpu') if fmap in SPARSE else R53.Feat53(enc, prep, fmap, n, 'cpu')
    f.V = V
    return f


def devw(enc, part):
    prep, fmap = part.split(':')
    assert fmap in SPARSE
    pool, preps = R53.setup(enc)
    out = os.path.join(odir(enc), f'devw_{enc}_{prep}_{fmap}.jsonl')
    done = set()
    if os.path.exists(out):
        done = {(r['M'], r['V'], r['rep']) for r in map(json.loads, open(out))}
    fo = open(out, 'a')
    feat = FeatW(enc, prep, fmap, pool.X.shape[1], 'cpu')
    for M in M_GRID:
        for V in VN:
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
                                     'part': f'vsaw:{prep}:{fmap}', 'rows': rows}) + '\n')
                fo.flush()
                print(f'devw {enc} {prep}:{fmap} M={M} V={V} rep={rep} {time.time() - t0:.0f}s', flush=True)


# ----------------------------------------------------------------------------------------------- engine (LM, RT, LR)

def feats_np(feat, Y, dmax):
    """phi(Y) on coordinates [0, dmax) as float32 numpy, blocks of CH (full-width projections, then cut)."""
    t = feat.torch
    Yt = t.from_numpy(np.ascontiguousarray(Y, dtype=np.float32))
    out = np.empty((Y.shape[0], dmax), np.float32)
    for j0 in range(0, dmax, CH):
        w = min(CH, dmax - j0)
        Gs, _ = feat.block(j0)
        out[:, j0:j0 + w] = feat.phi(Yt, Gs, CH)[:, :w].numpy()
    return out


def codes_np(feat, V, dmax):
    out = np.empty((V, dmax), np.float32)
    for j0 in range(0, dmax, CH):
        w = min(CH, dmax - j0)
        out[:, j0:j0 + w] = feat.block(j0)[1][:V, :w].numpy()
    return out


def bundle_q(feat, Fk, Cv, vals, order, d, q, ns, sseed):
    """B53's stored vector for (d, q, ns): the torch float32 sum and quantizer of b32_run.vsa_grid, block by block."""
    t = feat.torch
    idx = t.from_numpy(np.asarray(order[:ns], dtype=np.int64))
    vt = t.from_numpy(np.asarray(vals, dtype=np.int64))
    out = np.empty(d, np.float32)
    for j0 in range(0, d, CH):
        ww = min(CH, d - j0)
        Bd = t.from_numpy(Fk[:, j0:j0 + ww]) * t.from_numpy(Cv[:, j0:j0 + ww])[vt]
        cum = Bd[idx, :ww].sum(0)
        out[j0:j0 + ww] = R.quantize_t(cum.clone(), q, np.random.default_rng(C.seed_of('Q', sseed, d, q, ns, j0)),
                                       t).numpy()
    return out


def quant_np(S, q, seedparts):
    out = np.empty(len(S), np.float32)
    for j0 in range(0, len(S), CH):
        out[j0:j0 + CH] = C.quantize(S[j0:j0 + CH].astype(np.float32), q,
                                     np.random.default_rng(C.seed_of(*seedparts, j0)))
    return out


def classes(V, d):
    return [slice(c, d, 16) for c in range(16)] if V == 16 else [slice(0, d)]


def split_cls(X, V):
    """Coordinates grouped by residue class (V = 16: 16 contiguous arrays of the columns j = c mod 16; V = 2: X)."""
    return [np.ascontiguousarray(X[:, c::16]) for c in range(16)] if V == 16 else [X]


def ncls(V, c, d):
    return len(range(c, d, 16)) if V == 16 else d


def lm_decoders(F, V, ds):
    """Training-split statistics per needed d: for each class, eigen-decomposition of the T x T Gram F_c F_c^T and its
    trace (dual form of Sigma_c = F_c^T F_c / T). Cumulative over the sorted d's."""
    T = F.shape[0]
    Fc = split_cls(F, V)
    ncl = len(Fc)
    G = np.zeros((ncl, T, T))
    dec = {}
    prev = [0] * ncl
    for d in sorted(set(ds)):
        lst = []
        for c in range(ncl):
            n_c = ncls(V, c, d)
            X = Fc[c][:, prev[c]:n_c]
            G[c] += (X @ X.T).astype(np.float64)
            prev[c] = n_c
            lam, U = np.linalg.eigh(G[c])
            lst.append((np.maximum(lam, 0.0), U, float(np.trace(G[c])), n_c))
        dec[d] = lst
    return dec


def lm_read(Qc, Fc, z, dec_d, V, alphas, Acache):
    """Predictions of the LMMSE read for every alpha. Qc / Fc: query / training features split by class;
    Acache[c] = Q_c F_c^T at this d (nQ x T)."""
    d = len(z)
    T = Fc[0].shape[0]
    ncl = len(Qc)
    zc = [z[c::16] for c in range(16)] if V == 16 else [z]
    mf = np.stack([Qc[c][:, :len(zc[c])] @ zc[c].astype(np.float32) for c in range(ncl)], 1).astype(np.float64)
    bvec = [(Fc[c][:, :len(zc[c])] @ zc[c].astype(np.float32)).astype(np.float64) for c in range(ncl)]
    out = {}
    for a in alphas:
        if math.isinf(a):
            m = mf
        else:
            cols = []
            for c in range(ncl):
                lamv, U, tr, n_c = dec_d[c]
                lam_c = a * tr / (T * n_c)
                x = U @ ((U.T @ bvec[c]) / (T * lam_c + lamv))
                cols.append((mf[:, c] - Acache[c] @ x) / lam_c)
            m = np.stack(cols, 1)
        if V == 16:
            out[a] = (m @ H16.T).argmax(1)
        else:
            out[a] = np.where(m[:, 0] >= 0, 0, 1)
    return out


def retrain(Fk, Cv, vals, order, d, ns, epochs, eta=1.0):
    """HDC / perceptron retraining of the Hebbian bundle on the stored facts; snapshots after each epoch in epochs.
    Mini-batches of BATCH facts in store order, every fact of a batch read with the bundle at the batch start."""
    idx = np.asarray(order[:ns])
    X = np.ascontiguousarray(Fk[idx, :d])
    y = np.asarray(vals)[idx]
    Cd = np.ascontiguousarray(Cv[:, :d])
    S = (X * Cd[y]).astype(np.float64).sum(0)
    snaps = {}
    for ep in range(1, max(epochs) + 1):
        for b0 in range(0, ns, BATCH):
            Xb = X[b0:b0 + BATCH]
            yb = y[b0:b0 + BATCH]
            pr = (Xb @ (Cd * S.astype(np.float32)).T).argmax(1)
            w = np.nonzero(pr != yb)[0]
            if len(w):
                S += eta * (Xb[w].astype(np.float64) * (Cd[yb[w]] - Cd[pr[w]])).sum(0)
        if ep in epochs:
            snaps[ep] = S.copy()
    return snaps


def mf_read(Fq, Cv, Sq):
    d = len(Sq)
    return (Fq[:, :d] @ (Cv[:, :d] * Sq.astype(np.float32)).T).argmax(1)


def lam_rf(Fk, Fq, vals, V, cfgs, seed):
    """B49's LAM (b49_common.run_lam's algebra, float64) on given features. cfgs: list of (key, p, alpha, q, acct)."""
    M = Fk.shape[0]
    c = V - 1
    Tt, D = B.value_targets(V, 'onehot', seed)
    Y = Tt[vals]
    default = int(np.bincount(vals, minlength=V).argmax())
    out = {}
    ps = sorted({x[1] for x in cfgs if x[1] > 0})
    pm = ps[-1] if ps else 0
    Fk64 = Fk[:, :pm].astype(np.float64)
    Fq64 = Fq[:, :pm].astype(np.float64)
    K = np.zeros((M, M))
    prev = 0
    sols = {}
    for p in ps:
        X = Fk64[:, prev:p]
        K += X @ X.T
        prev = p
        for a in sorted({x[2] for x in cfgs if x[1] == p}):
            lam = a * float(np.trace(K)) / M
            sols[p, a] = np.linalg.solve(K + lam * np.eye(M), Y)
    for key, p, a, q, acct in cfgs:
        if p <= 0:
            out[key] = (np.full(Fq.shape[0], default), 0.0)
            continue
        s = np.zeros((Fq.shape[0], c))
        for j0 in range(0, p, CH):
            ww = min(CH, p - j0)
            Wb = Fk64[:, j0:j0 + ww].T @ sols[p, a]
            Wq = B.quant_cols(Wb, q, np.random.default_rng(C.seed_of('LAMQ56', seed, key, j0)))
            s += Fq64[:, j0:j0 + ww] @ Wq
        out[key] = ((s @ D).argmax(1), float(B.lam_bits(p, c, q, acct)))
    return out


def plan_cells(ch53, M, V, select=None):
    """Per (prep, fmap): the cells (bpf, acct) of B53's choice and their LM / RT / LR configurations."""
    groups = {}
    for bpf in BPF:
        for acct in ACCT32:
            prep, fmap, q53, f = ch53[f'{M}|{V}|{bpf}|{acct}|VSA']['cfg']
            ns = max(1, int(math.floor(f * M)))
            g = groups.setdefault((prep, fmap), {'LM': [], 'RT': [], 'LR': []})
            la = 'ideal' if acct == 'ideal' else 'real'
            for q in QS:
                d = C.vsa_dim(bpf * M, q, acct)
                for a in ALPHA_LM:
                    key = ('LM', bpf, acct, q, a)
                    if select is None or key in select:
                        g['LM'].append((key, d, q, ns, a))
                for e in EPOCHS:
                    for eta in ETAS:
                        key = ('RT', bpf, acct, q, f'{e}:{eta}')
                        if select is None or key in select:
                            g['RT'].append((key, d, q, ns, (e, eta)))
                for a in LAM_ALPHAS:
                    key = ('LR', bpf, acct, q, a)
                    if select is None or key in select:
                        g['LR'].append((key, B.lam_p(bpf * M, V - 1, q, la, P_MAX), a, q, la))
    return groups


def run_store_group(enc, feat, P, st, F, dec, g, V):
    """All LM / RT / LR configurations of one (prep, fmap) group on one store. Returns {key: (pred, info)}."""
    M = st['M']
    sseed = R.store_seed(enc, st)
    order = R.store_order(st)
    vals = st['vals']
    dmax = max([x[1] for x in g['LM'] + g['RT']] + [x[1] for x in g['LR']] + [1])
    tt = [time.time()]
    Fk = feats_np(feat, P(st['keys']), dmax)
    Fq = feats_np(feat, P(st['queries']), dmax)
    Cv = codes_np(feat, V, dmax)
    r = Cv[0].astype(np.float64)
    res = {}
    tt.append(time.time())
    # LM: cumulative Q_c F_c^T over the sorted d's (coordinates grouped by residue class)
    lm_by_d = {}
    for x in g['LM']:
        lm_by_d.setdefault(x[1], []).append(x)
    if lm_by_d:
        Qc, Fc = split_cls(Fq, V), split_cls(F, V)
        ncl = len(Qc)
        A = np.zeros((ncl, Fq.shape[0], F.shape[0]))
        prev = [0] * ncl
        for d in sorted(lm_by_d):
            for c in range(ncl):
                n_c = ncls(V, c, d)
                A[c] += Qc[c][:, prev[c]:n_c] @ Fc[c][:, prev[c]:n_c].T
                prev[c] = n_c
            byb = {}
            for (key, d_, q, ns, a) in lm_by_d[d]:
                byb.setdefault((q, ns), []).append((key, a))
            for (q, ns), lst in byb.items():
                Sq = bundle_q(feat, Fk, Cv, vals, order, d, q, ns, sseed)
                z = Sq.astype(np.float64) * r[:d]
                pr = lm_read(Qc, Fc, z, dec[d], V, sorted({a for _, a in lst}, reverse=True), A)
                for key, a in lst:
                    res[key] = (pr[a], [d, q, ns])
    tt.append(time.time())
    # RT
    rt = {}
    for x in g['RT']:
        rt.setdefault((x[1], x[3], x[4][1]), []).append(x)
    for (d, ns, eta), lst in rt.items():
        snaps = retrain(Fk, Cv, vals, order, d, ns, sorted({x[4][0] for x in lst}), eta)
        for (key, d_, q, ns_, (e, eta_)) in lst:
            Sq = quant_np(snaps[e], q, ('Q56', sseed, d, q, ns, e, eta))
            res[key] = (mf_read(Fq, Cv, Sq), [d, q, ns])
    tt.append(time.time())
    # LR (all M facts, as B49's LAM)
    if g['LR']:
        o = lam_rf(Fk, Fq, vals, V, g['LR'], C.seed_of('b56', enc, st['split'], M, V, st['rep']))
        for x in g['LR']:
            res[x[0]] = (o[x[0]][0], [x[1], x[3], o[x[0]][1]])
    tt.append(time.time())
    print('  stages feats %.0fs LM %.0fs RT %.0fs LR %.0fs' % tuple(np.diff(tt)), file=sys.stderr, flush=True)
    return res


def train_feats(feat, P, Tr, enc, prep, fmap, V, dmax):
    T = T_V[V]
    rows = np.sort(np.random.default_rng(C.seed_of('b56T', enc, prep, fmap, V)).choice(len(Tr), T, replace=False))
    return feats_np(feat, P(Tr[rows]), dmax)


def dec(enc, split, M, V):
    M, V = int(M), int(V)
    ch53 = json.load(open(f'{ROOT}/results/b53_choice_{enc}.json'))['choice']
    select = None
    if split == 'test':
        ch = json.load(open(f'{ROOT}/results/b56_choice_{enc}.json'))['choice']
        select = set()
        for bpf in BPF:
            for acct in ACCT32:
                for arm in ('LM', 'RT', 'LR'):
                    c = ch[f'{M}|{V}|{bpf}|{acct}|{arm}']['cfg']
                    select.add((arm, bpf, acct, c[0], c[1]))
    pool, preps = R53.setup(enc)
    Tr = pool.train_matrix()
    groups = plan_cells(ch53, M, V, select)
    reps = (R.DEV_REPS if split == 'dev' else R.TEST_REPS)[M]
    fn = os.path.join(odir(enc), f'dec_{split}_{enc}_M{M}_V{V}.jsonl.gz')
    if os.path.exists(fn):
        print('exists', fn)
        return
    recs = {rep: {'enc': enc, 'M': M, 'V': V, 'rep': rep, 'pred': {}, 'info': {}} for rep in range(reps)}
    for (prep, fmap), g in groups.items():
        t0 = time.time()
        feat = make_feat(enc, prep, fmap, pool.X.shape[1], V)
        P = preps[prep]
        dl = sorted({x[1] for x in g['LM']})
        dmax = max([x[1] for x in g['LM'] + g['RT']] + [x[1] for x in g['LR']] + [1])
        F = train_feats(feat, P, Tr, enc, prep, fmap, V, max(dl + [1]))
        decs = lm_decoders(F, V, dl)
        print(f'dec {enc} {split} M={M} V={V} {prep}:{fmap} decoders {len(dl)} d, {time.time() - t0:.0f}s', flush=True)
        for rep in range(reps):
            t1 = time.time()
            st = C.make_store(pool, split, M, V, rep)
            res = run_store_group(enc, feat, P, st, F, decs, g, V)
            rc = recs[rep]
            rc['qgold'] = st['qgold'].tolist()
            for key, (p, info) in res.items():
                k = '|'.join(map(str, key))
                rc['pred'][k] = np.asarray(p).tolist()
                rc['info'][k] = info + [prep, fmap]
            print(f'dec {enc} {split} M={M} V={V} {prep}:{fmap} rep={rep} {time.time() - t1:.0f}s', flush=True)
        del feat, F, decs
    with gzip.open(fn + '.tmp', 'wt') as fo:
        for rep in range(reps):
            fo.write(json.dumps(recs[rep]) + '\n')
    os.replace(fn + '.tmp', fn)


# ----------------------------------------------------------------------------------------------- choice

def wkey(c):
    return (('center', 'whiten').index(c[0]), FMAPS_W.index(c[1]), R.GRID['q'].index(c[2]), R.GRID['f'].index(c[3]))


def choose(enc):
    out = {}
    # W: B32's rule over B53's 72 + the 72 sparse configurations
    acc, reps = {}, {}
    files = [os.path.join(R53.OUT, enc, f'dev_{enc}_{p}_{f}.jsonl') for p, f in R53.PARTS] + \
            [os.path.join(odir(enc), f'devw_{enc}_{p}_{f}.jsonl') for p, f in PARTS_W]
    for fn in files:
        for l in open(fn):
            r = json.loads(l)
            reps.setdefault((r['M'], r['V']), set()).add(r['rep'])
            for x in r['rows']:
                acc.setdefault((r['M'], r['V'], x['bpf'], x['acct'], tuple(x['cfg'])), {})[r['rep']] = x['acc']
    best = {}
    for (M, V, bpf, acct, cfg), d in acc.items():
        assert len(d) == R.DEV_REPS[M], (M, V, bpf, acct, cfg, len(d))
        best.setdefault((M, V, bpf, acct), []).append((float(np.mean(list(d.values()))), cfg))
    for (M, V, bpf, acct), lst in best.items():
        assert len(lst) == 144, len(lst)
        top = max(a for a, _ in lst)
        tied = sorted([c for a, c in lst if a >= top - 0.001], key=wkey)
        out[f'{M}|{V}|{bpf}|{acct}|W'] = {'cfg': list(tied[0]), 'dev_acc': [a for a, c in lst if c == tied[0]][0],
                                          'n_tied': len(tied), 'n_cfg': len(lst)}
    # LM / RT / LR
    order = {'LM': [(q, a) for q in QS for a in ALPHA_LM], 'RT': [(q, f'{e}:{eta}') for q in QS for e in EPOCHS for eta in ETAS],
             'LR': [(q, a) for a in LAM_ALPHAS for q in QS]}
    for M in M_GRID:
        for V in VN:
            R_ = [json.loads(l) for l in gzip.open(os.path.join(odir(enc), f'dec_dev_{enc}_M{M}_V{V}.jsonl.gz'), 'rt')]
            assert len(R_) == R.DEV_REPS[M]
            for bpf in BPF:
                for acct in ACCT32:
                    for arm in ('LM', 'RT', 'LR'):
                        lst = []
                        for (q, x) in order[arm]:
                            k = '|'.join(map(str, (arm, bpf, acct, q, x)))
                            a = float(np.mean([np.mean(np.array(r['pred'][k]) == np.array(r['qgold'])) for r in R_]))
                            lst.append((a, (q, x)))
                        top = max(a for a, _ in lst)
                        tied = [c for a, c in lst if a >= top - 0.001]  # grid order kept
                        out[f'{M}|{V}|{bpf}|{acct}|{arm}'] = {
                            'cfg': list(tied[0]), 'dev_acc': [a for a, c in lst if c == tied[0]][0],
                            'n_tied': len(tied), 'n_cfg': len(lst),
                            'dev_all': {f'{c[0]}|{c[1]}': a for a, c in lst}}
                    lm, rt = out[f'{M}|{V}|{bpf}|{acct}|LM'], out[f'{M}|{V}|{bpf}|{acct}|RT']
                    star = 'RT' if rt['dev_acc'] > lm['dev_acc'] + 0.001 else 'LM'
                    out[f'{M}|{V}|{bpf}|{acct}|STAR'] = {'fam': star, 'dev': {'LM': lm['dev_acc'],
                                                                              'RT': rt['dev_acc']}}
                    w = out[f'{M}|{V}|{bpf}|{acct}|W']
                    cands = [(lm['dev_acc'], 0, 'LM'), (rt['dev_acc'], -1, 'RT'), (w['dev_acc'], -2, 'W')]
                    top = max(c[0] for c in cands)
                    anyf = [c[2] for c in cands if c[0] >= top - 0.001][0]
                    out[f'{M}|{V}|{bpf}|{acct}|ANY'] = {'fam': anyf, 'dev': {c[2]: c[0] for c in cands}}
    fn = f'{ROOT}/results/b56_choice_{enc}.json'
    json.dump({'enc': enc, 'choice': out}, open(fn, 'w'), indent=1, sort_keys=True)
    print(len(out), 'choices ->', fn)


# ----------------------------------------------------------------------------------------------- W test

def testw(enc):
    """Test predictions of every widened-grid choice that is not B53's own choice (sparse, or, when a sparse
    configuration moved the dev top, another of B53's 72)."""
    ch = json.load(open(f'{ROOT}/results/b56_choice_{enc}.json'))['choice']
    ch53 = json.load(open(f'{ROOT}/results/b53_choice_{enc}.json'))['choice']
    pool, preps = R53.setup(enc)
    feats = {}
    for M in M_GRID:
        for V in VN:
            fn = os.path.join(odir(enc), f'testw_{enc}_M{M}_V{V}.jsonl.gz')
            if os.path.exists(fn):
                continue
            with gzip.open(fn + '.tmp', 'wt') as fo:
                for rep in range(R.TEST_REPS[M]):
                    t0 = time.time()
                    st = C.make_store(pool, 'test', M, V, rep)
                    rec = {'enc': enc, 'M': M, 'V': V, 'rep': rep, 'qgold': st['qgold'].tolist(), 'pred': {},
                           'cfg': {}, 'd': {}}
                    groups = {}
                    for bpf in BPF:
                        for acct in ACCT32:
                            c = ch[f'{M}|{V}|{bpf}|{acct}|W']['cfg']
                            rec['cfg'][f'W|{bpf}|{acct}'] = c
                            if c != ch53[f'{M}|{V}|{bpf}|{acct}|VSA']['cfg']:
                                groups.setdefault((c[0], c[1]), []).append((bpf, acct, c[2], c[3]))
                    for (prep, fmap), lst in groups.items():
                        if (prep, fmap) not in feats:
                            feats[prep, fmap] = make_feat(enc, prep, fmap, pool.X.shape[1], V)
                        feats[prep, fmap].V = V
                        cfgs = [((bpf, acct, q, f), C.vsa_dim(bpf * M, q, acct), q,
                                 max(1, int(math.floor(f * M)))) for bpf, acct, q, f in lst]
                        P = preps[prep]
                        pr = R.vsa_grid(P(st['keys']), st['vals'], P(st['queries']), V, feats[prep, fmap], cfgs,
                                        R.store_seed(enc, st), R.store_order(st))
                        for (key, d, q, ns) in cfgs:
                            rec['pred'][f'W|{key[0]}|{key[1]}'] = pr[key].tolist()
                            rec['d'][f'W|{key[0]}|{key[1]}'] = [d, q, ns]
                    fo.write(json.dumps(rec) + '\n')
                    print(f'testw {enc} M={M} V={V} rep={rep} {time.time() - t0:.0f}s', flush=True)
            os.replace(fn + '.tmp', fn)


# ----------------------------------------------------------------------------------------------- merge

def merge(enc):
    """B53's merged records (VSA = B53's store, VSAR, LAM, BEST, ...) + VSAL (LM), VSAT (RT), VSAS (the dev-selected
    of the two), VSAW (widened grid), VSAX (dev-best of LM / RT / W), LAMRF."""
    ch = json.load(open(f'{ROOT}/results/b56_choice_{enc}.json'))['choice']
    amap = {'ideal': 'ideal', 'real': 'raw'}
    od = os.path.join(OUT, 'fam', enc)
    os.makedirs(od, exist_ok=True)
    for M in M_GRID:
        for V in VN:
            old = [json.loads(l) for l in gzip.open(f'{ROOT}/b53_out/fam/{enc}/all53_{enc}_M{M}_V{V}.jsonl.gz', 'rt')]
            dct = [json.loads(l) for l in gzip.open(os.path.join(odir(enc), f'dec_test_{enc}_M{M}_V{V}.jsonl.gz'), 'rt')]
            tw = [json.loads(l) for l in gzip.open(os.path.join(odir(enc), f'testw_{enc}_M{M}_V{V}.jsonl.gz'), 'rt')]
            assert len(old) == len(dct) == len(tw) == R.TEST_REPS[M]
            with gzip.open(f'{od}/all56_{enc}_M{M}_V{V}.jsonl.gz', 'wt') as fo:
                for o, n, w in zip(old, dct, tw):
                    assert o['rep'] == n['rep'] == w['rep'] and o['qgold'] == n['qgold'] == w['qgold']
                    o.setdefault('b56', {})
                    for b in BPF:
                        for a in ('ideal', 'real'):
                            a3 = amap[a]
                            sel = {}
                            for arm, name in (('LM', 'VSAL'), ('RT', 'VSAT'), ('LR', 'LAMRF')):
                                c = ch[f'{M}|{V}|{b}|{a3}|{arm}']['cfg']
                                k = '|'.join(map(str, (arm, b, a3, c[0], c[1])))
                                o['pred'][f'{name}|{b}|{a}'] = n['pred'][k]
                                o['cfg'][f'{name}|{b}|{a}'] = [c[0], c[1]] + n['info'][k][3:]
                                o['b56'][f'{name}|{b}|{a}'] = n['info'][k][:3]
                                sel[arm] = name
                            wc = ch[f'{M}|{V}|{b}|{a3}|W']['cfg']
                            if wc != o['cfg'][f'VSA|{b}|{a}']:
                                o['pred'][f'VSAW|{b}|{a}'] = w['pred'][f'W|{b}|{a3}']
                                o['b56'][f'VSAW|{b}|{a}'] = w['d'][f'W|{b}|{a3}']
                            else:
                                o['pred'][f'VSAW|{b}|{a}'] = o['pred'][f'VSA|{b}|{a}']
                                o['b56'][f'VSAW|{b}|{a}'] = o['vsa_d'][f'VSA|{b}|{a}']
                            o['cfg'][f'VSAW|{b}|{a}'] = wc
                            sf = ch[f'{M}|{V}|{b}|{a3}|STAR']['fam']
                            o['pred'][f'VSAS|{b}|{a}'] = o['pred'][f'{sel[sf]}|{b}|{a}']
                            o['cfg'][f'VSAS|{b}|{a}'] = [sf]
                            xf = ch[f'{M}|{V}|{b}|{a3}|ANY']['fam']
                            o['pred'][f'VSAX|{b}|{a}'] = o['pred'][f'{ {"LM": "VSAL", "RT": "VSAT", "W": "VSAW"}[xf]}|{b}|{a}']
                            o['cfg'][f'VSAX|{b}|{a}'] = [xf]
                    fo.write(json.dumps(o) + '\n')
    print('merged', enc)


# ----------------------------------------------------------------------------------------------- selftest

def selftest():
    import torch
    rng = np.random.default_rng(0)
    n, M = 48, 96
    K = rng.standard_normal((M, n)).astype(np.float32)
    K /= np.linalg.norm(K, axis=1, keepdims=True)
    Qy = K + 0.3 * rng.standard_normal((M, n)).astype(np.float32)
    Qy /= np.linalg.norm(Qy, axis=1, keepdims=True)
    Tr = rng.standard_normal((2000, n)).astype(np.float32)
    Tr /= np.linalg.norm(Tr, axis=1, keepdims=True)
    order = rng.permutation(M)
    worst = []
    for V in (2, 16):
        vals = rng.integers(0, V, size=M)
        for fmap in ('lin', 'sgn', 'tern16'):
            feat = make_feat('selftest', 'p', fmap, n, V)
            cfgs = [((d, q, ns), d, q, ns) for d in (500, 9000) for q in (1, 2) for ns in (M, 48)]
            ref = R.vsa_grid(K, vals, Qy, V, feat, cfgs, 7, order)
            Fk, Fq, Cv = feats_np(feat, K, 9000), feats_np(feat, Qy, 9000), codes_np(feat, V, 9000)
            F = feats_np(feat, Tr[:T_V[V]], 9000)
            dec_ = lm_decoders(F, V, [500, 9000])
            ncl = 16 if V == 16 else 1
            for (key, d, q, ns) in cfgs:
                Sq = bundle_q(feat, Fk, Cv, vals, order, d, q, ns, 7)
                # 1. the replicated bundle + matched filter equals b32_run.vsa_grid (B53's store)
                mine = mf_read(Fq, Cv, Sq.astype(np.float64))
                ag = float((mine == ref[key]).mean())
                worst.append(ag)
                sc = Fq[:, :d].astype(np.float64) @ (Cv[:, :d].astype(np.float64) * Sq).T
                ii = np.nonzero(mine != ref[key])[0]
                ii2 = np.nonzero(mine != sc.argmax(1))[0]  # float32 read vs float64 read: ties only
                gap2 = np.abs(sc[ii2, mine[ii2]] - sc[ii2].max(1))
                assert np.all(gap2 <= 1e-5 * np.abs(sc).max()), (V, fmap, key, gap2)
                gap = np.abs(sc[ii, mine[ii]] - sc[ii, ref[key][ii]])
                assert np.all(gap <= 1e-5 * np.abs(sc).max()), (V, fmap, key, ag, gap)  # only exact ties differ
                A = np.stack([Fq[:, s].astype(np.float64) @ F[:, s].astype(np.float64).T for s in classes(V, d)])
                pr = lm_read(split_cls(Fq, V), split_cls(F, V), Sq.astype(np.float64) * Cv[0, :d], dec_[d], V,
                             ALPHA_LM, A)
                i3 = np.nonzero(pr[math.inf] != mine)[0]
                gap3 = np.abs(sc[i3, mine[i3]] - sc[i3, pr[math.inf][i3]])
                assert np.all(gap3 <= 1e-5 * np.abs(sc).max()), (V, fmap, key, 'alpha=inf', gap3)
                # 2. LMMSE read equals a direct primal solve with numpy (small d only)
                if d == 500:
                    z = Sq.astype(np.float64) * Cv[0, :d]
                    for a in (1.0, 0.01):
                        m = []
                        for s in classes(V, d):
                            Fc = F[:, s].astype(np.float64)
                            Sig = Fc.T @ Fc / Fc.shape[0]
                            lam = a * np.trace(Sig) / Sig.shape[0]
                            m.append(Fq[:, s].astype(np.float64) @ np.linalg.solve(Sig + lam * np.eye(len(Sig)), z[s]))
                        m = np.stack(m, 1)
                        p2 = (m @ H16.T).argmax(1) if V == 16 else np.where(m[:, 0] >= 0, 0, 1)
                        assert (p2 == pr[a]).mean() > 0.99, (V, fmap, a, (p2 == pr[a]).mean())
            assert ncl in (1, 16)
            # 3. retraining fits the stored keys at least as well as the Hebbian bundle (generous d, sign features)
            sn = retrain(Fk, Cv, vals, order, 9000, M, [10])
            acc_h = (mf_read(Fk, Cv, (Fk[:, :9000] * Cv[vals, :9000]).sum(0)) == vals).mean()
            acc_r = (mf_read(Fk, Cv, sn[10]) == vals).mean()
            assert acc_r >= acc_h - 1e-9, (V, fmap, acc_h, acc_r)
    # 4. sparse map: the active fraction is kappa (unit-norm codes)
    feat = make_feat('selftest', 'p', 'tern16', n, 2)
    Fk = feats_np(feat, Tr, 8192)
    frac = float((Fk != 0).mean())
    assert abs(frac - 1 / 16) < 0.003, frac
    # 5. LAM-RF on sign features (full rank; M > n) reads the exact keys back; bits within the budget
    feat = make_feat('selftest', 'p', 'sgn', n, 2)
    vals = rng.integers(0, 2, size=M)
    Fk = feats_np(feat, K, 4000)
    o = lam_rf(Fk, Fk, vals, 2, [('x', 4000, 1e-4, 4, 'ideal')], 3)
    assert (o['x'][0] == vals).mean() > 0.95, (o['x'][0] == vals).mean()
    assert o['x'][1] <= 4000 * C.H_Q[4] + 32 + 1e-9
    print(f'selftest ok (sparse active fraction {frac:.4f}; replicated store vs vsa_grid agreement min {min(worst):.4f}, '
          f'exact in {sum(w == 1.0 for w in worst)} of {len(worst)} configurations)')


def main():
    a = sys.argv[1:]
    if a[0] == 'selftest':
        selftest()
        return
    enc, mode = a[0], a[1]
    assert enc in R53.ENC_TAG, enc
    if mode == 'devw':
        devw(enc, a[2])
    elif mode == 'dec':
        dec(enc, a[2], a[3], a[4])
    elif mode == 'choose':
        choose(enc)
    elif mode == 'testw':
        testw(enc)
    elif mode == 'merge':
        merge(enc)
    else:
        raise SystemExit(mode)


if __name__ == '__main__':
    main()
