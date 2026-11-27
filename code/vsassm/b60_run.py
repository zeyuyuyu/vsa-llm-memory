"""B60 (prereg/B60_corrected_lam_prereg.md): the corrected least-squares associative memory LAMc on B49's stores.

LAMc is B49's LAM (b49_common.run_lam: ridge W = F^T (F F^T + lam I)^-1 T, lam = alpha * mean diag(F F^T); read
score(v) = f(q)^T W_hat t(v), argmax) with three changes, each a grid option:
  pca map   'pca' features are the projections of the prep's code on the top-p eigenvectors of the centred training
            covariance of the RAW codes (computed before whitening; descending eigenvalue).  whiten: phi(x) =
            P_w(x) @ U0[:, :p], P_w = b32_common.Prep('whiten') (ZCA with eps = 0.1 x mean eigenvalue, then the row
            normalisation of the full code), which equals ((x - mu) U0 / sqrt(w + eps)) / norm, i.e. the whitened code's
            coordinates on the principal axes.  center: P_c(x) @ U0[:, :p] (the centred, row-normalised code on the
            same axes; B49's LamFeat basis of the centre-prepped codes failed the 0.95 overlap bar on Mistral at
            p = 32, so both preps use U0).  'rp': B49's, unchanged.
  targets   'V-1' (B49: centred one-hot, columns of values 1..V-1 stored, value 0 = minus their sum) or 'V' (all V
            columns stored and read); each charged for the columns it stores.  V >= 128 also B49's 'code32', 'code128'.
  scales    'col' (B49: one 32-bit scale per column and per block of 8,192 rows) or 'flat' (the store's rule: the
            p x c matrix flattened column by column, one 32-bit scale per block of 8,192 stored numbers).  Quantizer per
            scale group: q = 1 sign x mean |w| of the group; q >= 2 b32_common.quantize (Max's step on the group's std).
  bits      ideal: c p H_q + 32 x (number of scale groups); real: q instead of H_q.  p = the largest that fits
            (rp at most 65,536; pca at most n).
Grid (order = tie order): scale (col, flat) > target (V-1, V[, code32, code128]) > (prep, feat) (center rp, center
pca, whiten rp, whiten pca) > alpha (1e-4, 1e-2, 0.1, 1, 10, 100) > q (1, 2, 4): 288 configurations at V in {2, 16}.

python3 b60_run.py selftest
python3 b60_run.py overlap [ENC ...]                      # basis-overlap selftest (training split only)
python3 b60_run.py timing ENC M V                         # one TRAINING-split store, every configuration
python3 b60_run.py job SPEC.json                           # one (enc, map, split, M, V, rep, prep) job -> OUT/...
python3 b60_run.py plan dev|test|mapdev|maptest [--encs ..] [--vs ..]   # job specs -> OUT/plan_*.jsonl
python3 b60_run.py runplan PLAN.jsonl --workers N          # run a plan (skips finished jobs)
python3 b60_run.py choose ENC                              # dev rule -> results/b60_choice_{enc}.json
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
import b49_common as B  # noqa: E402

OUT = os.environ.get('B60_OUT', os.path.join(ROOT, 'b60_out'))
DATA = os.environ.get('B60_DATA', os.path.join(ROOT, 'b49_out'))  # cwd holding b32_data/ with every panel code
ENC_TAG = {'qwen8': 'qwen8_mean_L9_f16', 'e5': 'e5_base', 'b46': 'b46_granite33i_mean_L30_f16',
           'llama': 'llama31i_mean_L24_f16', 'mistral': 'mistral7i_eol_L24_f16', 'bge': 'bge_small', 'minilm': 'minilm'}
PANEL = ['qwen8', 'b46', 'llama', 'mistral', 'e5', 'bge', 'minilm']
M_GRID = [256, 1024]
VN = [2, 16]
VR2 = [1024]
BPF = [1, 2, 4, 8, 16, 32, 64]
ACCTS = ['ideal', 'real']
DEV_REPS = {256: 6, 1024: 3}
TEST_REPS = {256: 20, 1024: 5}
PREPS = ['center', 'whiten']
FEATS = ['rp', 'pca']
ALPHAS = [1e-4, 1e-2, 0.1, 1.0, 10.0, 100.0]
QS = [1, 2, 4]
SCALES = ['col', 'flat']
CH = 8192
P_MAX = 65536
EPS = 0.1  # b32_common.Prep's default: eps = 0.1 x mean eigenvalue of the training covariance
MAPS = [1, 2, 3]


def targets(V):
    return ['V-1', 'V'] + (['code32', 'code128'] if V >= 128 else [])


def grid(V):
    return [(s, t, p, f, a, q) for s in SCALES for t in targets(V) for p in PREPS for f in FEATS for a in ALPHAS
            for q in QS]


def ncols(V, t):
    return {'V-1': V - 1, 'V': V}.get(t) or int(t[4:])


def per_num(q, acct):
    return C.H_Q[q] if acct == 'ideal' else q


def bits(p, c, q, acct, scale):
    if p <= 0:
        return 0.0
    ng = c * math.ceil(p / CH) if scale == 'col' else math.ceil(c * p / CH)
    return c * p * per_num(q, acct) + 32 * ng


def p_of(Bud, c, q, acct, scale, pmax):
    per = per_num(q, acct)
    p = int(max(0, (Bud - 32) // (c * per)))
    while p > 0 and bits(p, c, q, acct, scale) > Bud:
        p -= max(1, int((bits(p, c, q, acct, scale) - Bud) // (c * per)))
    p = max(0, min(p, pmax))
    while p < pmax and bits(p + 1, c, q, acct, scale) <= Bud:  # maximal
        p += 1
    return p


def tag(enc, m):
    return enc if m == 0 else f'{enc}#b57m{m}'


# ----------------------------------------------------------------------------------------------- features

class FeatC:
    """LAMc's feature maps of one (encoder, prep, map). rp: B49's LamFeat blocks (seed ('LAMG', seedtag, prep, j0)).
    pca (both preps): the prepped code (Prep's scaling and row normalisation of the full code) on the raw codes'
    principal axes U0 (top eigenvectors of the centred training covariance of the raw codes; module doc)."""

    def __init__(self, seedtag, prep, P, Tr):
        self.prep = prep
        self.lf = B.LamFeat(seedtag, prep, P(Tr))
        self.n = self.lf.n
        mu = Tr.mean(0)
        Cv = np.cov((Tr - mu).T)
        w, U = np.linalg.eigh(Cv)
        o = np.argsort(-w)
        self.U0 = U[:, o].astype(np.float32)
        self.w0 = w[o]

    def pmax(self, feat):
        return P_MAX if feat == 'rp' else self.n

    def basis(self, p):
        """n x p directions of the pca map (for the overlap selftest)."""
        return self.U0[:, :p]

    def feats(self, Y, feat, j0, w):
        if feat == 'rp':
            return self.lf.feats(Y, 'rp', j0, w)
        return Y @ self.U0[:, j0:j0 + w]


def target_mats(V, t):
    """(T, D): T = V x c stored-column targets, D = c x V decoding to value scores."""
    if t == 'V-1':
        return B.value_targets(V, 'onehot', 0)
    if t == 'V':
        return np.eye(V) - 1.0 / V, np.eye(V)
    return B.value_targets(V, t, 0)


def qgroup(x, q, rng):
    if q == 1:
        s = np.sign(x)
        z = s == 0
        if z.any():
            s[z] = rng.choice([-1.0, 1.0], size=int(z.sum()))
        return s * (np.abs(x).mean() or 1.0)
    return C.quantize(x.astype(np.float32), q, None).astype(np.float64)


def quantW(W, q, scale, rng):
    """W: p x c (float64). col: per column and per block of CH rows; flat: column-major blocks of CH numbers."""
    p, c = W.shape
    if scale == 'col':
        out = np.empty_like(W)
        for j0 in range(0, p, CH):
            for k in range(c):
                out[j0:j0 + CH, k] = qgroup(W[j0:j0 + CH, k], q, rng)
        return out
    flat = W.T.ravel()
    out = np.empty_like(flat)
    for i0 in range(0, len(flat), CH):
        out[i0:i0 + CH] = qgroup(flat[i0:i0 + CH], q, rng)
    return out.reshape(c, p).T


def chol_solve(K, lam, Y):
    import scipy.linalg as sl
    A = K + lam * np.eye(K.shape[0])
    try:
        cf = sl.cho_factor(A, lower=True, check_finite=False)
        return sl.cho_solve(cf, Y, check_finite=False)
    except np.linalg.LinAlgError:
        return np.linalg.solve(A, Y)


def run_lamc(st, Yk, Yq, fc, cfgs, bpfs, accts, seed):
    """cfgs: list of (scale, target, feat, alpha, q) for ONE prep (fc). Returns {(cfg, bpf, acct): (pred, bits, p)}."""
    M, V = st['M'], st['V']
    vals = st['vals']
    default = int(np.bincount(vals, minlength=V).argmax())
    out = {}
    for feat in FEATS:
        plan = []
        for cfg in cfgs:
            if cfg[2] != feat:
                continue
            scale, t, _, alpha, q = cfg
            c = ncols(V, t)
            for b in bpfs:
                for a in accts:
                    p = p_of(b * M, c, q, a, scale, fc.pmax(feat))
                    plan.append((cfg, b, a, p, c))
        if not plan:
            continue
        for (cfg, b, a, p, c) in plan:
            if p <= 0:
                out[cfg, b, a] = (np.full(Yq.shape[0], default), 0.0, 0)
        live = [x for x in plan if x[3] > 0]
        if not live:
            continue
        tg = sorted({x[0][1] for x in live}, key=targets(V).index)
        TD = {t: target_mats(V, t) for t in tg}
        Ycat = np.hstack([TD[t][0][vals] for t in tg])  # M x (sum c)
        off, o = {}, 0
        for t in tg:
            off[t] = (o, o + TD[t][0].shape[1])
            o += TD[t][0].shape[1]
        ps = sorted({x[3] for x in live})
        pm = ps[-1]
        Fk = np.empty((M, pm), np.float32)
        Fq = np.empty((Yq.shape[0], pm), np.float32)
        for j0 in range(0, pm, CH):
            w = min(CH, pm - j0)
            Fk[:, j0:j0 + w] = fc.feats(Yk, feat, j0, w)
            Fq[:, j0:j0 + w] = fc.feats(Yq, feat, j0, w)
        by_p = {}
        for x in live:
            by_p.setdefault(x[3], []).append(x)
        K = np.zeros((M, M), np.float64)
        a0 = 0
        for p in ps:
            for s0 in range(a0, p, CH):  # accumulate the Gram matrix of columns [a0, p)
                s1 = min(p, s0 + CH)
                Fb = Fk[:, s0:s1].astype(np.float64)
                K += Fb @ Fb.T
            a0 = p
            lam0 = float(np.trace(K)) / M
            Fp = Fk[:, :p].astype(np.float64)
            Qp = Fq[:, :p].astype(np.float64)
            sols = {}
            for (cfg, b, a, _, c) in by_p[p]:
                alpha = cfg[3]
                if alpha not in sols:
                    A = chol_solve(K, alpha * lam0, Ycat)
                    sols[alpha] = Fp.T @ A  # p x sum c
                i0, i1 = off[cfg[1]]
                W = sols[alpha][:, i0:i1]
                rng = np.random.default_rng(C.seed_of('B60Q', seed, cfg, b, a))
                Wq = quantW(W, cfg[4], cfg[0], rng)
                S = Qp @ Wq
                if cfg[1] == 'V-1':  # D = [-1 | I]: value 0 = minus the sum of the stored columns
                    S = np.hstack([-S.sum(1, keepdims=True), S])
                elif cfg[1] != 'V':  # code targets: correlation with every value's code
                    S = S @ TD[cfg[1]][1]
                out[cfg, b, a] = (S.argmax(1), float(bits(p, c, cfg[4], a, cfg[0])), int(p))
            del Fp, Qp
        del Fk, Fq
    return out


# ----------------------------------------------------------------------------------------------- setup and jobs

_SET = {}


def setup(enc, m=0):
    key = (enc, m)
    if key not in _SET:
        cwd = os.getcwd()
        os.chdir(DATA)
        pool = C.Pool(ENC_TAG[enc])
        os.chdir(cwd)
        Tr = pool.train_matrix()
        preps = {p: C.Prep(p, Tr) for p in PREPS}
        fcs = {p: FeatC(tag(enc, m), p, preps[p], Tr) for p in PREPS}
        _SET.clear()
        _SET[key] = (pool, preps, fcs)
    return _SET[key]


def job_name(spec):
    return f'{spec["enc"]}_m{spec["map"]}_{spec["split"]}_M{spec["M"]}_V{spec["V"]}_r{spec["rep"]}_{spec["prep"]}'


def run_job(spec):
    """spec: enc, map, split, M, V, rep, prep, feats (list), cfgs (None = full grid restricted to prep / feats; else a
    list of [cfg, bpf, acct] to compute), save_pred (bool). Writes OUT/jobs/{name}.json.gz."""
    fn = os.path.join(OUT, 'jobs', job_name(spec) + '.json.gz')
    if os.path.exists(fn):
        return fn
    t0 = time.time()
    enc, m, split, M, V, rep, prep = (spec[k] for k in ('enc', 'map', 'split', 'M', 'V', 'rep', 'prep'))
    pool, preps, fcs = setup(enc, m)
    st = C.make_store(pool, split, M, V, rep)
    P = preps[prep]
    Yk, Yq = P(st['keys']), P(st['queries'])
    seed = C.seed_of('b60', tag(enc, m), split, M, V, rep)
    g = st['qgold']
    rec = {'spec': spec, 'qgold': g.tolist(), 'rows': []}
    if spec.get('cfgs') is None:
        cfgs = [(s, t, f, a, q) for (s, t, p, f, a, q) in grid(V) if p == prep and f in spec['feats']]
        o = run_lamc(st, Yk, Yq, fcs[prep], cfgs, BPF, ACCTS, seed)
        want = None
    else:
        want = {}
        for cfg, b, a in spec['cfgs']:
            want.setdefault(tuple(cfg[:1] + cfg[1:2] + cfg[3:]), []).append((b, a))
        o = {}
        for c5, ba in want.items():
            bs = sorted({b for b, _ in ba})
            o.update(run_lamc(st, Yk, Yq, fcs[prep], [c5], bs, ACCTS, seed))
    for (c5, b, a), (pred, bt, p) in o.items():
        if want is not None and (b, a) not in want[c5]:
            continue
        cfg = [c5[0], c5[1], prep, c5[2], c5[3], c5[4]]
        row = {'cfg': cfg, 'bpf': b, 'acct': a, 'acc': float((pred == g).mean()), 'bits': bt, 'p': p}
        if (spec.get('save_pred') and b in (4, 8, 16) and a == 'ideal') or want is not None:
            row['pred'] = np.asarray(pred).astype(int).tolist()
        rec['rows'].append(row)
    rec['sec'] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    with gzip.open(fn + '.tmp', 'wt') as fo:
        json.dump(rec, fo)
    os.replace(fn + '.tmp', fn)
    print(f'job {job_name(spec)} {rec["sec"]}s {len(rec["rows"])} rows', flush=True)
    return fn


def _run(spec):
    try:
        return run_job(spec)
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        return f'FAILED {job_name(spec)} {e}'


def runplan(planfile, workers):
    specs = [json.loads(l) for l in open(planfile)]
    specs = [s for s in specs if not os.path.exists(os.path.join(OUT, 'jobs', job_name(s) + '.json.gz'))]
    # largest first
    specs.sort(key=lambda s: -(s['M'] ** 2) * (2 if s['V'] == 2 else 1) * (4 if 'rp' in s.get('feats', ['rp']) else 1))
    print(f'{len(specs)} jobs to run', flush=True)
    import multiprocessing as mp
    if workers <= 1:
        for s in specs:
            print(_run(s), flush=True)
        return
    with mp.get_context('fork').Pool(workers, maxtasksperchild=8) as pl:
        for r in pl.imap_unordered(_run, specs):
            if str(r).startswith('FAILED'):
                print(r, flush=True)


def make_plan(kind, encs, vs):
    os.makedirs(OUT, exist_ok=True)
    fn = os.path.join(OUT, f'plan_{kind}.jsonl')
    n = 0
    with open(fn, 'w') as fo:
        for enc in encs:
            for V in vs:
                for M in M_GRID:
                    if kind in ('dev', 'mapdev'):
                        maps = [0] if kind == 'dev' else MAPS
                        for m in maps:
                            for rep in range(DEV_REPS[M]):
                                for prep in PREPS:
                                    feats = FEATS if kind == 'dev' else ['rp']
                                    spec = {'enc': enc, 'map': m, 'split': 'dev', 'M': M, 'V': V, 'rep': rep,
                                            'prep': prep, 'feats': feats, 'cfgs': None,
                                            'save_pred': enc == 'qwen8' and kind == 'dev' and V in VN}
                                    fo.write(json.dumps(spec) + '\n')
                                    n += 1
                    else:
                        maps = [0] if kind == 'test' else MAPS
                        for m in maps:
                            ch = json.load(open(f'{ROOT}/results/b60_choice_{enc}.json'))['choice']
                            arm = 'LAMC' if m == 0 else f'LAMCm{m}'
                            for prep in PREPS:
                                cfgs = []
                                for b in BPF:
                                    for a in ACCTS:
                                        k = f'{M}|{V}|{b}|{a}|{arm}'
                                        if k in ch and ch[k]['cfg'][2] == prep:
                                            cfgs.append([ch[k]['cfg'], b, a])
                                if not cfgs:
                                    continue
                                for rep in range(TEST_REPS[M]):
                                    spec = {'enc': enc, 'map': m, 'split': 'test', 'M': M, 'V': V, 'rep': rep,
                                            'prep': prep, 'cfgs': cfgs}
                                    fo.write(json.dumps(spec) + '\n')
                                    n += 1
    print(n, 'jobs ->', fn)


# ----------------------------------------------------------------------------------------------- dev choice

def load_dev(enc, m_list=(0,)):
    """{(m, M, V, bpf, acct, cfg): {rep: acc}} from the dev jobs (map m's rp rows; pca rows from map 0)."""
    acc = {}
    jd = os.path.join(OUT, 'jobs')
    for fn in sorted(os.listdir(jd)):
        if not fn.startswith(f'{enc}_m') or '_dev_' not in fn:
            continue
        r = json.load(gzip.open(os.path.join(jd, fn), 'rt'))
        s = r['spec']
        if s['map'] not in m_list:
            continue
        for x in r['rows']:
            acc.setdefault((s['map'], s['M'], s['V'], x['bpf'], x['acct'], tuple(x['cfg'])), {})[s['rep']] = x['acc']
    return acc


def choose_one(rows, M, V, b, a):
    """B49's rule: highest mean dev accuracy over the cell's dev stores; within 0.1 points tied, first in grid order."""
    lst = []
    for cfg in grid(V):
        d = rows.get((M, V, b, a, cfg))
        if d is None:
            raise SystemExit(f'missing dev rows {M} {V} {b} {a} {cfg}')
        if len(d) != DEV_REPS[M]:
            raise SystemExit(f'incomplete dev rows {M} {V} {b} {a} {cfg}: {len(d)}')
        lst.append((float(np.mean([d[k] for k in sorted(d)])), cfg))
    top = max(x for x, _ in lst)
    tied = [c for x, c in lst if x >= top - 0.001]
    return list(tied[0]), [x for x, c in lst if c == tied[0]][0], top, len(tied)


ALL10 = ['IDX', 'SF', 'CM', 'BF', 'FBF', 'SDM', 'LAM', 'LHI', 'LHC', 'LHS']


def choose(enc, vs, with_maps):
    ch = {}
    ch49 = json.load(open(f'{ROOT}/results/b49_choice_{enc}.json'))['choice']
    acc = load_dev(enc, [0] + (MAPS if with_maps else []))
    rows0 = {k[1:]: v for k, v in acc.items() if k[0] == 0}
    for M in M_GRID:
        for V in vs:
            for b in BPF:
                for a in ACCTS:
                    c, dacc, top, nt = choose_one(rows0, M, V, b, a)
                    ch[f'{M}|{V}|{b}|{a}|LAMC'] = {'cfg': c, 'dev_acc': dacc, 'best_dev_acc': top, 'n_tied': nt}
                    dbf = dict(ch49[f'{M}|{V}|{b}|{a}|BEST']['dev_by_fam'])
                    dbf['LAM'] = dacc
                    cands = [(dbf[f], -ALL10.index(f), f) for f in ALL10]
                    ch[f'{M}|{V}|{b}|{a}|BESTC'] = {'fam': max(cands)[2], 'dev_by_fam': dbf}
                    if with_maps and V in VN:
                        for m in MAPS:
                            rows = {k: v for k, v in rows0.items() if k[4][3] == 'pca'}
                            rows.update({k[1:]: v for k, v in acc.items() if k[0] == m and k[5][3] == "rp"})
                            c, dacc, top, nt = choose_one(rows, M, V, b, a)
                            ch[f'{M}|{V}|{b}|{a}|LAMCm{m}'] = {'cfg': c, 'dev_acc': dacc, 'n_tied': nt}
    fn = f'{ROOT}/results/b60_choice_{enc}.json'
    old = json.load(open(fn))['choice'] if os.path.exists(fn) else {}
    old.update(ch)
    json.dump({'enc': enc, 'choice': old}, open(fn, 'w'), indent=1, sort_keys=True)
    print(len(ch), 'choices ->', fn)


# ----------------------------------------------------------------------------------------------- selftests

def selftest():
    rng = np.random.default_rng(0)
    # bits / p rules: within budget and maximal
    for V in (2, 16, 1024):
        for t in targets(V):
            c = ncols(V, t)
            for s in SCALES:
                for q in QS:
                    for a in ACCTS:
                        for M in M_GRID:
                            for b in BPF:
                                for pm in (384, 4096, P_MAX):
                                    p = p_of(b * M, c, q, a, s, pm)
                                    assert bits(p, c, q, a, s) <= b * M
                                    assert p == pm or bits(p + 1, c, q, a, s) > b * M
    # toy store: near-orthogonal keys, exact queries, large budget -> every fact read, both scale rules / targets
    M, V, n = 200, 16, 512
    Kk = rng.standard_normal((M, n)).astype(np.float32)
    Kk /= np.linalg.norm(Kk, axis=1, keepdims=True)
    vals = rng.integers(0, V, M)
    st = {'M': M, 'V': V, 'vals': vals}

    class P0:
        def __call__(self, X):
            return X

    Tr = rng.standard_normal((2000, n)).astype(np.float32)
    for prep in PREPS:
        fc = FeatC('selftest', prep, P0(), Tr)
        cfgs = [(s, t, f, 1e-4, 4) for s in SCALES for t in ('V-1', 'V') for f in FEATS]
        o = run_lamc(st, Kk, Kk, fc, cfgs, [256], ['ideal'], 1)
        for k, (pr, bt, p) in o.items():
            assert bt <= 256 * M
            acc = float((pr == vals).mean())
            assert acc >= 0.95, (k, acc)
    # 'col' + 'V-1' reproduces b49_common.run_lam on rp features (same features, same algebra; quantizer seeds differ
    # only for q = 1 zero ties, which do not occur)
    lf = B.LamFeat('selftest', 'center', Kk)
    fc = FeatC('selftest', 'center', P0(), Kk)
    for q in QS:
        o49 = B.run_lam(st, Kk, Kk * 0.9 + 0.1 * Kk[::-1], lf, [('rp', 1.0, q, 'onehot')], [8, 32], ['ideal'], 1)
        o60 = run_lamc(st, Kk, Kk * 0.9 + 0.1 * Kk[::-1], fc, [('col', 'V-1', 'rp', 1.0, q)], [8, 32], ['ideal'], 1)
        for b in (8, 32):
            a49, a60 = o49[('rp', 1.0, q, 'onehot'), b, 'ideal'][0], o60[('col', 'V-1', 'rp', 1.0, q), b, 'ideal'][0]
            agree = float((a49 == a60).mean())
            assert agree >= 0.995, (q, b, agree)
    print('selftest ok')


def overlap(encs, ps=(8, 32, 128, 512, 2048)):
    """Basis-overlap selftest (training split only): ||U_ref[:, :p]^T D_p||_F^2 / p, U_ref = top-p eigenvectors of the
    raw codes' centred training covariance (independent float64 eigh), D_p = the pca map's directions. Also B49's
    LamFeat whiten basis (the defect)."""
    res = {}
    for enc in encs:
        cwd = os.getcwd()
        os.chdir(DATA)
        pool = C.Pool(ENC_TAG[enc])
        os.chdir(cwd)
        Tr = pool.train_matrix().astype(np.float64)
        mu = Tr.mean(0)
        w, U = np.linalg.eigh(np.cov((Tr - mu).T))
        Uref = U[:, np.argsort(-w)]
        for prep in PREPS:
            P = C.Prep(prep, pool.train_matrix())
            fc = FeatC(enc, prep, P, pool.train_matrix())
            for p in ps:
                if p > fc.n:
                    continue
                D = fc.basis(p).astype(np.float64)
                D, _ = np.linalg.qr(D)
                ov = float(np.linalg.norm(Uref[:, :p].T @ D) ** 2 / p)
                ov49 = float(np.linalg.norm(Uref[:, :p].T @ np.linalg.qr(fc.lf.U[:, :p].astype(np.float64))[0]) ** 2 / p)
                res[f'{enc}|{prep}|{p}'] = {'lamc': ov, 'b49': ov49}
                print(f'{enc:8s} {prep:7s} p={p:5d} overlap LAMc {ov:.4f}  B49 LamFeat {ov49:.4f}', flush=True)
    return res


def timing(enc, M, V):
    pool, preps, fcs = setup(enc)
    st = C.make_store(pool, 'train', M, V, 0)
    for prep in PREPS:
        t0 = time.time()
        cfgs = [(s, t, f, a, q) for (s, t, p, f, a, q) in grid(V) if p == prep]
        P = preps[prep]
        o = run_lamc(st, P(st['keys']), P(st['queries']), fcs[prep], cfgs, BPF, ACCTS, 0)
        print(f'timing {enc} train M={M} V={V} {prep}: {len(o)} (cfg, bpf, acct) in {time.time() - t0:.1f}s; '
              f'queries {len(st["qgold"])}', flush=True)


def main():
    a = sys.argv[1:]
    if a[0] == 'selftest':
        selftest()
    elif a[0] == 'overlap':
        r = overlap(a[1:] or PANEL)
        if len(a) == 1:
            json.dump(r, open(f'{ROOT}/results/b60_overlap.json', 'w'), indent=1)
    elif a[0] == 'timing':
        timing(a[1], int(a[2]), int(a[3]))
    elif a[0] == 'job':
        run_job(json.load(open(a[1])))
    elif a[0] == 'plan':
        encs = a[a.index('--encs') + 1].split(',') if '--encs' in a else PANEL
        vs = [int(x) for x in a[a.index('--vs') + 1].split(',')] if '--vs' in a else VN
        make_plan(a[1], encs, vs)
    elif a[0] == 'runplan':
        wk = int(a[a.index('--workers') + 1]) if '--workers' in a else 1
        runplan(a[1], wk)
    elif a[0] == 'choose':
        vs = [int(x) for x in a[a.index('--vs') + 1].split(',')] if '--vs' in a else VN
        choose(a[1], vs, '--maps' in a)
    else:
        raise SystemExit(a[0])


if __name__ == '__main__':
    main()
