"""B62 (prereg/B62_online_ls_write_prereg.md): online decorrelating writes with LAMc's corrected options (B60), on R1.

NLMSc  one pass of the normalised delta rule over the M facts in a fixed random order (b32_run.store_order):
       W_t = W_{t-1} + mu f_t (y_t - W_{t-1}^T f_t)^T / (f_t^T f_t + 1e-12); output W_M ('last') or the Polyak average of
       the iterates ('avg'). Computed exactly in the dual form: (L + diag(K + 1e-12) / mu) D = Y (L = strictly lower
       triangle of the ordered Gram K = F F^T; forward substitution = the online recursion), W_M = F^T D,
       W_bar = F^T (omega * D), omega_t = (M - t + 1) / M. Selftest / check: a genuine primal loop.
RLSc   exponentially weighted RLS (P_0 = I / delta, forgetting beta): after the M-th fact W = F^T (K + delta
       diag(beta^t))^-1 Y (dual) = (F^T Lam F + beta^M delta I)^-1 F^T Lam Y, Lam = diag(beta^(M - t)) (primal; used when
       p < M / 2). delta = alpha x s_p, s_p = mean ||f_p(x)||^2 over 2,000 training-split codes (fixed before any
       fact arrives). Selftest / check: a genuine recursion.
RLScS  RLSc with p cut so that 32 p (p + 1) / 2 (the p x p working matrix, 32-bit) + stored bits <= budget.
Features, targets, scales, quantizer, bits, read: b60_run's (LAMc).

python3 b62_run.py selftest
python3 b62_run.py trainnorm ENC                  # training-split feature norms -> OUT/trainnorm_{enc}.npz
python3 b62_run.py timing ENC M V                 # one TRAINING-split store, every configuration
python3 b62_run.py plan dev|test [--encs ..]      # job specs -> OUT/plan_*.jsonl
python3 b62_run.py runplan PLAN.jsonl --workers N
python3 b62_run.py choose ENC                     # dev rule -> results/b62_choice_{enc}.json
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
import b60_run as L60  # noqa: E402

OUT = os.environ.get('B62_OUT', os.path.join(ROOT, 'b62_out'))
PANEL = L60.PANEL
M_GRID = [256, 1024]
VN = [2, 16]
BPF = [4, 8, 16]  # R1
ACCT = 'ideal'
DEV_REPS = L60.DEV_REPS
TEST_REPS = L60.TEST_REPS
PREPS = L60.PREPS
FEATS = L60.FEATS
SCALES = L60.SCALES
TARGETS = ['V-1', 'V']
QS = L60.QS
ALPHAS = L60.ALPHAS
MUS = [1.0, 0.5, 0.25, 0.1, 0.03]
OUTS = ['last', 'avg']
BETAS = [1.0, 0.999, 0.99]
CH = L60.CH
ARMS = ['NLMSC', 'RLSC', 'RLSCS']
TN_P = {'rp': 16384}  # largest R1 p on rp features (p_of(16 * 1024, 1, 1, ideal, *) = 16,320)
TN_N = 2000


def grid(arm):
    """Configurations (scale, target, prep, feat, x, y, q) in tie order; x, y = (mu, out) or (alpha, beta)."""
    xy = [(m, o) for m in MUS for o in OUTS] if arm == 'NLMSC' else [(a, b) for a in ALPHAS for b in BETAS]
    return [(s, t, p, f, x, y, q) for s in SCALES for t in TARGETS for p in PREPS for f in FEATS for (x, y) in xy
            for q in QS]


def bits(p, c, q, scale):
    return L60.bits(p, c, q, ACCT, scale)


def state_bits(p):
    return 32 * p * (p + 1) // 2


def p_state(Bud, c, q, scale, pmax):
    """RLScS: the largest p with the 32-bit symmetric p x p state + stored bits <= budget (and maximal)."""
    p = 0
    while p < pmax and state_bits(p + 1) + bits(p + 1, c, q, scale) <= Bud:
        p += 1
    return p


def p_for(arm, Bud, c, q, scale, pmax):
    if arm == 'RLSCS':
        return p_state(Bud, c, q, scale, pmax)
    return L60.p_of(Bud, c, q, ACCT, scale, pmax)


def store_order(st):
    """= b32_run.store_order (the store's write order; seed ('subset', split, M, V, rep))."""
    return np.random.default_rng(C.seed_of('subset', st['split'], st['M'], st['V'], st['rep'])).permutation(st['M'])


# ----------------------------------------------------------------------------------------------- algebra

def nlms_dual(K, Y, mu):
    """Innovations D (M x c) of one NLMS pass in the order of K's rows: (L + diag(K + 1e-12) / mu) D = Y."""
    import scipy.linalg as sl
    A = np.tril(K, -1)
    A[np.diag_indices_from(A)] = (np.diag(K) + 1e-12) / mu
    return sl.solve_triangular(A, Y, lower=True, check_finite=False)


def nlms_primal(F, Y, mu):
    """Genuine one-fact-at-a-time loop (rows of F in write order). Returns (W_last, W_avg)."""
    W = np.zeros((F.shape[1], Y.shape[1]))
    Wsum = np.zeros_like(W)
    for t in range(F.shape[0]):
        f = F[t]
        e = Y[t] - f @ W
        W += (mu / (f @ f + 1e-12)) * np.outer(f, e)
        Wsum += W
    return W, Wsum / F.shape[0]


def rls_closed(F, Y, delta, beta, K=None):
    """W after M exponentially weighted RLS updates (rows of F in write order), exact closed form."""
    M, p = F.shape
    t = np.arange(1, M + 1)
    if p < M / 2:
        lw = beta ** (M - t)
        G = F.T @ (lw[:, None] * F)
        G[np.diag_indices_from(G)] += (beta ** M) * delta
        return L60.chol_solve(G, 0.0, F.T @ (lw[:, None] * Y))
    if K is None:
        K = F @ F.T
    A = K.copy()
    A[np.diag_indices_from(A)] += delta * beta ** t
    return F.T @ L60.chol_solve(A, 0.0, Y)


def rls_recursive(F, Y, delta, beta):
    """Genuine RLS recursion (P_0 = I / delta, forgetting beta)."""
    p = F.shape[1]
    P = np.eye(p) / delta
    W = np.zeros((p, Y.shape[1]))
    for t in range(F.shape[0]):
        f = F[t]
        Pf = P @ f
        k = Pf / (beta + f @ Pf)
        W += np.outer(k, Y[t] - f @ W)
        P = (P - np.outer(k, Pf)) / beta
    return W


# ----------------------------------------------------------------------------------------------- training norms

def trainnorm(enc):
    """s_p (cumulative mean ||f_p(x)||^2 over a fixed sample of training codes) per (prep, feat)."""
    pool, preps, fcs = L60.setup(enc)
    Tr = pool.train_matrix()
    idx = np.sort(np.random.default_rng(C.seed_of('B62TN', enc)).choice(len(Tr), size=min(TN_N, len(Tr)),
                                                                        replace=False))
    out = {}
    for prep in PREPS:
        Y = preps[prep](Tr[idx])
        fc = fcs[prep]
        for feat in FEATS:
            pm = min(TN_P.get(feat, fc.n), fc.pmax(feat))
            s = np.zeros(pm)
            for j0 in range(0, pm, CH):
                w = min(CH, pm - j0)
                F = fc.feats(Y, feat, j0, w).astype(np.float64)
                s[j0:j0 + w] = (F ** 2).mean(0)
            out[f'{prep}|{feat}'] = np.cumsum(s)
    os.makedirs(OUT, exist_ok=True)
    np.savez(os.path.join(OUT, f'trainnorm_{enc}.npz'), **{k.replace('|', '__'): v for k, v in out.items()})
    print('trainnorm', enc, {k: (len(v), float(v[min(len(v), 64) - 1]), float(v[-1])) for k, v in out.items()})


_TN = {}


def tn(enc, prep, feat, p):
    if enc not in _TN:
        z = np.load(os.path.join(OUT, f'trainnorm_{enc}.npz'))
        _TN[enc] = {k.replace('__', '|'): z[k] for k in z.files}
    s = _TN[enc][f'{prep}|{feat}']
    assert p <= len(s), (enc, prep, feat, p, len(s))
    return float(s[p - 1])


# ----------------------------------------------------------------------------------------------- engine

def run_online(st, Yk, Yq, fc, enc, prep, want, seed):
    """want: list of (arm, cfg7, bpf) for ONE prep (fc). Returns {(arm, cfg7, bpf): (pred, bits, p)}."""
    M, V = st['M'], st['V']
    vals = st['vals']
    order = store_order(st)
    default = int(np.bincount(vals, minlength=V).argmax())
    TD = {t: L60.target_mats(V, t) for t in TARGETS}
    Yo = np.hstack([TD[t][0][vals[order]] for t in TARGETS])  # M x (V - 1 + V), write order
    off = {'V-1': (0, V - 1), 'V': (V - 1, 2 * V - 1)}
    omega = (M - np.arange(M)) / M
    out = {}
    for feat in FEATS:
        plan = []
        for (arm, cfg, b) in want:
            if cfg[3] != feat:
                continue
            c = L60.ncols(V, cfg[1])
            p = p_for(arm, b * M, c, cfg[6], cfg[0], fc.pmax(feat))
            if p <= 0:
                out[arm, cfg, b] = (np.full(Yq.shape[0], default), 0.0, 0)
            else:
                plan.append((arm, cfg, b, p, c))
        if not plan:
            continue
        ps = sorted({x[3] for x in plan})
        pm = ps[-1]
        Fk = np.empty((M, pm), np.float32)
        Fq = np.empty((Yq.shape[0], pm), np.float32)
        for j0 in range(0, pm, CH):
            w = min(CH, pm - j0)
            Fk[:, j0:j0 + w] = fc.feats(Yk, feat, j0, w)
            Fq[:, j0:j0 + w] = fc.feats(Yq, feat, j0, w)
        Fk = Fk[order]
        by_p = {}
        for x in plan:
            by_p.setdefault(x[3], []).append(x)
        K = np.zeros((M, M), np.float64)
        a0 = 0
        for p in ps:
            for s0 in range(a0, p, CH):
                s1 = min(p, s0 + CH)
                Fb = Fk[:, s0:s1].astype(np.float64)
                K += Fb @ Fb.T
            a0 = p
            Fp = Fk[:, :p].astype(np.float64)
            Qp = Fq[:, :p].astype(np.float64)
            sols = {}
            for (arm, cfg, b, _, c) in by_p[p]:
                s, t, _, _, x, y, q = cfg
                if arm == 'NLMSC':
                    if ('N', x) not in sols:
                        D = nlms_dual(K, Yo, x)
                        sols['N', x] = {'last': Fp.T @ D, 'avg': Fp.T @ (omega[:, None] * D)}
                    W = sols['N', x][y]
                else:
                    if ('R', x, y) not in sols:
                        sols['R', x, y] = rls_closed(Fp, Yo, x * tn(enc, prep, feat, p), y, K)
                    W = sols['R', x, y]
                i0, i1 = off[t]
                rng = np.random.default_rng(C.seed_of('B62Q', seed, arm, cfg, b))
                Wq = L60.quantW(W[:, i0:i1], q, s, rng)
                S = Qp @ Wq
                if t == 'V-1':
                    S = np.hstack([-S.sum(1, keepdims=True), S])
                out[arm, cfg, b] = (S.argmax(1), float(bits(p, c, q, s)), int(p))
            del Fp, Qp
        del Fk, Fq
    return out


def job_name(spec):
    return f'{spec["enc"]}_{spec["split"]}_M{spec["M"]}_V{spec["V"]}_r{spec["rep"]}_{spec["prep"]}'


def run_job(spec):
    """spec: enc, split, M, V, rep, prep, want (None = every configuration of the prep for the three arms; else a list
    of [arm, cfg7, bpf]). Writes OUT/jobs/{name}.json.gz (test rows carry predictions)."""
    fn = os.path.join(OUT, 'jobs', job_name(spec) + '.json.gz')
    if os.path.exists(fn):
        return fn
    t0 = time.time()
    enc, split, M, V, rep, prep = (spec[k] for k in ('enc', 'split', 'M', 'V', 'rep', 'prep'))
    pool, preps, fcs = L60.setup(enc)
    st = C.make_store(pool, split, M, V, rep)
    P = preps[prep]
    Yk, Yq = P(st['keys']), P(st['queries'])
    seed = C.seed_of('b62', enc, split, M, V, rep)
    if spec.get('want') is None:
        want = [(a, c, b) for a in ARMS for c in grid(a) if c[2] == prep for b in BPF]
    else:
        want = [(a, tuple(c), b) for a, c, b in spec['want']]
    o = run_online(st, Yk, Yq, fcs[prep], enc, prep, want, seed)
    g = st['qgold']
    rec = {'spec': spec, 'qgold': g.tolist(), 'rows': []}
    for (arm, cfg, b) in want:
        pred, bt, p = o[arm, cfg, b]
        row = {'arm': arm, 'cfg': list(cfg), 'bpf': b, 'acc': float((pred == g).mean()), 'bits': bt, 'p': p}
        if spec.get('want') is not None:
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
    big = {'qwen8': 4, 'b46': 4, 'llama': 4, 'mistral': 4, 'e5': 2, 'bge': 1, 'minilm': 1}
    specs.sort(key=lambda s: -(s['M'] ** 2) * big[s['enc']] * (2 if s['V'] == 2 else 1))
    print(f'{len(specs)} jobs to run', flush=True)
    import multiprocessing as mp
    if workers <= 1:
        for s in specs:
            print(_run(s), flush=True)
        return
    with mp.get_context('fork').Pool(workers, maxtasksperchild=4) as pl:
        for r in pl.imap_unordered(_run, specs):
            if str(r).startswith('FAILED'):
                print(r, flush=True)


def make_plan(kind, encs):
    os.makedirs(OUT, exist_ok=True)
    fn = os.path.join(OUT, f'plan_{kind}.jsonl')
    n = 0
    with open(fn, 'w') as fo:
        for enc in encs:
            ch = json.load(open(f'{ROOT}/results/b62_choice_{enc}.json'))['choice'] if kind == 'test' else None
            for V in VN:
                for M in M_GRID:
                    for prep in PREPS:
                        if kind == 'dev':
                            for rep in range(DEV_REPS[M]):
                                fo.write(json.dumps({'enc': enc, 'split': 'dev', 'M': M, 'V': V, 'rep': rep,
                                                     'prep': prep, 'want': None}) + '\n')
                                n += 1
                            continue
                        want = [[a, ch[f'{M}|{V}|{b}|{a}']['cfg'], b] for a in ARMS for b in BPF
                                if ch[f'{M}|{V}|{b}|{a}']['cfg'][2] == prep]
                        if not want:
                            continue
                        for rep in range(TEST_REPS[M]):
                            fo.write(json.dumps({'enc': enc, 'split': 'test', 'M': M, 'V': V, 'rep': rep, 'prep': prep,
                                                 'want': want}) + '\n')
                            n += 1
    print(n, 'jobs ->', fn)


# ----------------------------------------------------------------------------------------------- dev choice

def choose(enc):
    """B60's rule per (M, V, bpf, arm): highest mean dev accuracy; within 0.1 points tied, first in grid order."""
    acc = {}
    jd = os.path.join(OUT, 'jobs')
    for fn in sorted(os.listdir(jd)):
        if not fn.startswith(f'{enc}_dev_'):
            continue
        r = json.load(gzip.open(os.path.join(jd, fn), 'rt'))
        s = r['spec']
        for x in r['rows']:
            acc.setdefault((s['M'], s['V'], x['bpf'], x['arm'], tuple(x['cfg'])), {})[s['rep']] = (x['acc'], x['p'],
                                                                                                    x['bits'])
    ch = {}
    for M in M_GRID:
        for V in VN:
            for b in BPF:
                for arm in ARMS:
                    lst = []
                    for cfg in grid(arm):
                        d = acc.get((M, V, b, arm, cfg))
                        if d is None or len(d) != DEV_REPS[M]:
                            raise SystemExit(f'missing / incomplete dev rows {enc} {M} {V} {b} {arm} {cfg}')
                        lst.append((float(np.mean([d[k][0] for k in sorted(d)])), cfg, d[0][1], d[0][2]))
                    top = max(x[0] for x in lst)
                    tied = [x for x in lst if x[0] >= top - 0.001]
                    ch[f'{M}|{V}|{b}|{arm}'] = {'cfg': list(tied[0][1]), 'dev_acc': tied[0][0], 'best_dev_acc': top,
                                                'n_tied': len(tied), 'p': tied[0][2], 'bits': tied[0][3]}
    fn = f'{ROOT}/results/b62_choice_{enc}.json'
    json.dump({'enc': enc, 'choice': ch}, open(fn, 'w'), indent=1, sort_keys=True)
    print(len(ch), 'choices ->', fn)


# ----------------------------------------------------------------------------------------------- costs

def costs(arm, M, V, n, cfg, p, c, store=None):
    """(write multiply-adds total, write per fact, feature multiply-adds of the write, read per query, working state in
    numbers) from the shapes (prereg 'Accounting')."""
    if arm == 'STORE':
        prep, fmap, q, share = store['cfg']
        d, _, ns = store['d']
        Kf = 3 if fmap == 'sgn3' else 1
        return {'write': ns * d, 'per_fact': ns * d / M, 'feat': ns * Kf * n * d, 'read': Kf * n * d + d + V * d,
                'state': 2 * d}
    rd = n * p + p * c + (V if cfg and cfg[1] == 'V-1' else 0)
    if arm == 'NLMSC':
        w = M * (2 * p * c + p) + (M * p * c if cfg[5] == 'avg' else 0)
        st = p * c * (2 if cfg[5] == 'avg' else 1) + p
    elif arm in ('RLSC', 'RLSCS'):
        w = M * (2 * p * p + 2 * p * c + 3 * p) + (M * p * p if cfg[5] < 1 else 0)
        st = p * (p + 1) // 2 + p * c + p
    elif arm == 'LAMC':
        dual = M * M * p + M ** 3 / 3 + M * M * c + M * p * c
        primal = M * p * p + p ** 3 / 3 + M * p * c + p * p * c
        w = min(dual, primal)
        st = M * p + M * M + p * c
    else:
        raise ValueError(arm)
    return {'write': w, 'per_fact': w / M, 'feat': M * n * p, 'read': rd, 'state': st}


# ----------------------------------------------------------------------------------------------- selftests

def selftest():
    rng = np.random.default_rng(0)
    # 1. bits rules (B60's p_of reused; RLScS's state rule): within budget and maximal
    for V in VN:
        for t in TARGETS:
            c = L60.ncols(V, t)
            for s in SCALES:
                for q in QS:
                    for M in M_GRID:
                        for b in BPF:
                            for pm in (384, 4096, L60.P_MAX):
                                for arm in ('NLMSC', 'RLSCS'):
                                    p = p_for(arm, b * M, c, q, s, pm)
                                    extra = state_bits if arm == 'RLSCS' else (lambda _: 0)
                                    assert extra(p) + bits(p, c, q, s) <= b * M
                                    assert p == pm or extra(p + 1) + bits(p + 1, c, q, s) > b * M
                                    if arm == 'RLSCS':
                                        assert p <= 31
    # 2. NLMS dual form = genuine primal loop (last and avg)
    for (M, p) in ((60, 200), (200, 40)):
        F = rng.standard_normal((M, p))
        Y = rng.standard_normal((M, 3))
        for mu in MUS:
            D = nlms_dual(F @ F.T, Y, mu)
            Wl, Wa = nlms_primal(F, Y, mu)
            om = (M - np.arange(M)) / M
            for A, B_ in ((F.T @ D, Wl), (F.T @ (om[:, None] * D), Wa)):
                r = np.abs(A - B_).max() / np.abs(B_).max()
                assert r < 1e-10, (M, p, mu, r)
    # 3. RLS closed form (primal and dual) = genuine recursion
    for (M, p) in ((60, 20), (40, 90), (120, 100)):
        F = rng.standard_normal((M, p))
        Y = rng.standard_normal((M, 2))
        for beta in BETAS:
            for delta in (0.01, 1.0):
                Wr = rls_recursive(F, Y, delta, beta)
                Wc = rls_closed(F, Y, delta, beta)
                r = np.abs(Wr - Wc).max() / np.abs(Wr).max()
                assert r < 1e-7, (M, p, beta, delta, r)
                # the other form
                t = np.arange(1, M + 1)
                A = F @ F.T
                A[np.diag_indices_from(A)] += delta * beta ** t
                Wd = F.T @ np.linalg.solve(A, Y)
                assert np.abs(Wd - Wr).max() / np.abs(Wr).max() < 1e-7
    # 4. toy store: near-orthogonal keys, exact queries, ample budget (256 bits per fact: p >> M; at 64 bits per fact
    #    p ~ M and one pass of NLMS read 0.75, a property of one pass, not of the code) -> NLMSc (mu 1) and RLSc
    M, V, n = 200, 16, 512
    Kk = rng.standard_normal((M, n)).astype(np.float32)
    Kk /= np.linalg.norm(Kk, axis=1, keepdims=True)
    vals = rng.integers(0, V, M)
    st = {'M': M, 'V': V, 'vals': vals, 'split': 'toy', 'rep': 0}

    class P0:
        def __call__(self, X):
            return X
    Tr = rng.standard_normal((2000, n)).astype(np.float32)
    for prep in PREPS:
        fc = L60.FeatC('selftest', prep, P0(), Tr)
        _TN['selftest'] = {f'{prep}|{f}': np.cumsum(np.full(fc.pmax(f), 1.0 / n)) for f in FEATS}
        want = [(a, (s, t, prep, f, x, y, 4), 256) for a in ('NLMSC', 'RLSC') for s in SCALES for t in TARGETS
                for f in FEATS for (x, y) in ((1.0, 'last'), (1e-4, 1.0))
                if (a == 'NLMSC') == (y == 'last')]
        o = run_online(st, Kk, Kk, fc, 'selftest', prep, want, 1)
        for k, (pr, bt, p) in o.items():
            assert bt <= 256 * M and p > 0
            acc = float((pr == vals).mean())
            assert acc >= 0.9, (k, acc)
    # 5. store order = b32_run.store_order
    try:
        import b32_run as R
        s2 = {'split': 'dev', 'M': 256, 'V': 16, 'rep': 3}
        assert np.array_equal(R.store_order(s2), store_order(s2))
    except ImportError:
        print('b32_run not importable; store order not compared')
    print('selftest ok')


def timing(enc, M, V):
    pool, preps, fcs = L60.setup(enc)
    st = C.make_store(pool, 'train', M, V, 0)
    for prep in PREPS:
        t0 = time.time()
        want = [(a, c, b) for a in ARMS for c in grid(a) if c[2] == prep for b in BPF]
        P = preps[prep]
        o = run_online(st, P(st['keys']), P(st['queries']), fcs[prep], enc, prep, want, 0)
        accs = {}
        for (a, c, b), (pr, bt, p) in o.items():
            accs.setdefault(a, []).append(float((pr == st['qgold']).mean()))
        print(f'timing {enc} train M={M} V={V} {prep}: {len(o)} (arm, cfg, bpf) in {time.time() - t0:.1f}s; '
              f'queries {len(st["qgold"])}; best train acc ' + ', '.join(f'{a} {max(v):.3f}' for a, v in accs.items()),
              flush=True)


def main():
    a = sys.argv[1:]
    if a[0] == 'selftest':
        selftest()
    elif a[0] == 'trainnorm':
        for enc in a[1:] or PANEL:
            trainnorm(enc)
    elif a[0] == 'timing':
        timing(a[1], int(a[2]), int(a[3]))
    elif a[0] == 'job':
        run_job(json.load(open(a[1])))
    elif a[0] == 'plan':
        encs = a[a.index('--encs') + 1].split(',') if '--encs' in a else PANEL
        make_plan(a[1], encs)
    elif a[0] == 'runplan':
        runplan(a[1], int(a[a.index('--workers') + 1]) if '--workers' in a else 1)
    elif a[0] == 'choose':
        for enc in a[1:]:
            choose(enc)
    else:
        raise SystemExit(a[0])


if __name__ == '__main__':
    main()
