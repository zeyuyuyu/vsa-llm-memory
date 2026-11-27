"""B64 (prereg/B64_sdm_training_locations_prereg.md): sparse distributed memory whose hard locations are the training
split's own label and alias codes (SDMt), on B49's stores, against the store's first and final designs.

SDMt (a B36 family; same stores, budgets, default value, decide rule, bit conventions and choice rule as B36's SDM):
  locations  a fixed set of training-split codes (no dev / test string and no code row equal to a dev / test code,
             checked), in a fixed seeded order
             (seed ('B64LOC', set)); 'LA' = every training label and alias (5,621), 'L' = labels (2,818), 'A' = aliases
             (2,803). The memory uses the first N locations of the order; N = the largest point of a fixed geometric
             grid (NGRID points from 1 to the set size) whose cost fits the budget.  Addresses are the prep's codes
             (center / whiten, b32_common.Prep fitted on the training split, rows unit-norm), so the address of
             location l is P(x_l).  Free (fixed, training-fitted), like B36's PQ codebooks or the store's whitening.
  write      each stored fact (the first round(share M) facts of b36_run.store_order) activates the a locations of
             the first N whose address has the largest cosine with P(key) (all N if N < a).  Counters per location:
             'onehot' = V counters, the count of activating facts with each value; 'bits' = ceil(log2 V) counters,
             the sum of the activating facts' bipolar value bits (2 bit_j(v) - 1).
  counters   q 'cnt' exact; 'clip3' saturating (onehot min(c, 3); bits clip to [-3, 3]); 'tern' (onehot min(c, 1),
             i.e. one presence bit per value; bits sign(c)).
  read       the query activates its a nearest of the first N locations; onehot: score(v) = sum over its active
             locations of Q(C[l, v]); bits: score(v) = sum_j (sum_l Q(C[l, j])) (2 bit_j(v) - 1); answer =
             b36_common.decide (ties to the store's plurality value, then the smallest value).
  bits       ideal (primary): N c H0 (c counters per location; H0 = zeroth-order empirical entropy of the stored
             counter symbols; B36's SDM rule) + ceil(log2 V) for the default value.  real (secondary): fixed width
             per counter (onehot: cnt ceil(log2(max + 1)) + 8, clip3 2, tern 1; bits: cnt ceil(log2(2 max|c| + 1))
             + 8, clip3 3, tern 2) + ceil(log2 V).  N = 0 (nothing fits): the default value, 0 bits (B36's SDM).
Grid (order = tie order): prep (center, whiten) > loc (LA, L, A) > a (1, 2, 4, 8, 16, 32, 64) > ctr (onehot, bits)
> q (cnt, clip3, tern) > share (1, 1/2): 504 configurations at V in {2, 16}.

python3 b64_run.py selftest
python3 b64_run.py disjoint                               # location sets vs every dev / test string and code row
python3 b64_run.py timing ENC M V                         # one TRAINING-split store (timing only)
python3 b64_run.py plan dev|test [--encs ..]
python3 b64_run.py runplan PLAN.jsonl --workers N
python3 b64_run.py choose ENC                             # -> results/b64_choice_{enc}.json
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
import b36_common as K  # noqa: E402

OUT = os.environ.get('B64_OUT', os.path.join(ROOT, 'b64_out'))
DATA = os.environ.get('B64_DATA', os.path.join(ROOT, 'b49_out'))  # cwd holding b32_data/ with every panel code
ENC_TAG = {'qwen8': 'qwen8_mean_L9_f16', 'e5': 'e5_base', 'b46': 'b46_granite33i_mean_L30_f16',
           'llama': 'llama31i_mean_L24_f16', 'mistral': 'mistral7i_eol_L24_f16', 'bge': 'bge_small', 'minilm': 'minilm'}
PANEL = ['qwen8', 'b46', 'llama', 'mistral', 'e5', 'bge', 'minilm']
M_GRID = [256, 1024]
VN = [2, 16]
BPF = [4, 8, 16]            # R1 budgets only (prereg)
ACCTS = ['ideal', 'real']
DEV_REPS = {256: 6, 1024: 3}
TEST_REPS = {256: 20, 1024: 5}
PREPS = ['center', 'whiten']
LOCS = ['LA', 'L', 'A']
AS = [1, 2, 4, 8, 16, 32, 64]
CTRS = ['onehot', 'bits']
QS = ['cnt', 'clip3', 'tern']
SHARES = [1.0, 0.5]
NGRID = 160
BLK = 2048
GRID = [(p, l, a, c, q, s) for p in PREPS for l in LOCS for a in AS for c in CTRS for q in QS for s in SHARES]


def store_order(st):  # b36_run.store_order (copied to avoid importing b36_run's module state)
    return np.random.default_rng(C.seed_of('subset', st['split'], st['M'], st['V'], st['rep'])).permutation(st['M'])


def ngrid(nmax):
    g = np.unique(np.round(np.geomspace(1, nmax, NGRID)).astype(int))
    return [int(x) for x in g if 1 <= x <= nmax]


def nbits(V):
    return max(1, math.ceil(math.log2(V)))


def vbip(V):
    nb = nbits(V)
    return (((np.arange(V)[:, None] >> np.arange(nb)) & 1) * 2 - 1).astype(np.int64)  # V x nb


def qtrans(Cn, ctr, q):
    if q == 'cnt':
        return Cn
    if q == 'clip3':
        return np.minimum(Cn, 3) if ctr == 'onehot' else np.clip(Cn, -3, 3)
    return np.minimum(Cn, 1) if ctr == 'onehot' else np.sign(Cn)


def ctr_bits(Qc, ctr, q, acct):
    """Bits of the stored counters (N x c); the default value's bits are added by the caller."""
    if Qc.size == 0:
        return 0.0
    if acct == 'ideal':
        return float(Qc.size * K.H0(np.bincount((Qc - Qc.min()).ravel())))
    if q == 'tern':
        w = 1 if ctr == 'onehot' else 2
        return float(Qc.size * w)
    if q == 'clip3':
        w = 2 if ctr == 'onehot' else 3
        return float(Qc.size * w)
    mx = int(np.abs(Qc).max())
    w = max(1, math.ceil(math.log2(mx + 1 + 1e-9))) if ctr == 'onehot' else \
        max(1, math.ceil(math.log2(2 * mx + 1 + 1e-9)))
    return float(Qc.size * w + 8)


def counters(act, vals, V, N, ctr):
    """act: rows x a location indices (-1 = none); returns N x c integer counters."""
    a = act.shape[1]
    idx = np.where(act < 0, N, act).ravel()
    if ctr == 'onehot':
        Cn = np.zeros((N + 1, V), np.int64)
        np.add.at(Cn, (idx, np.repeat(vals, a)), 1)
    else:
        vb = vbip(V)
        Cn = np.zeros((N + 1, vb.shape[1]), np.int64)
        np.add.at(Cn, idx, np.repeat(vb[vals], a, axis=0))
    return Cn[:N]


class TopA:
    """Incremental top-AMAX (largest cosine) among the first N locations, rows sorted by descending similarity."""

    def __init__(self, Y, L, amax):
        self.Y, self.L, self.amax = Y, L, amax
        self.S = np.full((Y.shape[0], 0), -np.inf, np.float32)
        self.I = np.full((Y.shape[0], 0), -1, np.int64)
        self.n = 0

    def advance(self, N):
        for j0 in range(self.n, N, BLK):
            j1 = min(N, j0 + BLK)
            S = np.hstack([self.S, self.Y @ self.L[j0:j1].T])
            I = np.hstack([self.I, np.broadcast_to(np.arange(j0, j1), (self.Y.shape[0], j1 - j0))])
            if S.shape[1] > self.amax:
                sel = np.argpartition(-S, self.amax - 1, axis=1)[:, :self.amax]
                S = np.take_along_axis(S, sel, 1)
                I = np.take_along_axis(I, sel, 1)
            o = np.argsort(-S, axis=1, kind='stable')
            self.S = np.take_along_axis(S, o, 1)
            self.I = np.take_along_axis(I, o, 1)
        self.n = max(self.n, N)
        return self.I

    def top(self, a):
        I = self.I[:, :a]
        if I.shape[1] < a:
            I = np.hstack([I, np.full((I.shape[0], a - I.shape[1]), -1, np.int64)])
        return I


def run_sdmt(st, Yk, Yq, Ls, cfgs, bpfs, accts, order):
    """One prep's codes: Yk (M x n stored keys), Yq (queries), Ls = {loc: location addresses (ordered, prepped)};
    cfgs: list of (loc, a, ctr, q, share). Returns {(cfg, bpf, acct): (pred, bits, N)}."""
    M, V = st['M'], st['V']
    vals = st['vals']
    default = K.default_value(vals, V)
    db = K.def_bits(V)
    vb = vbip(V)
    out = {}
    for loc in LOCS:
        cl = [c for c in cfgs if c[0] == loc]
        if not cl:
            continue
        L = Ls[loc]
        Ng = ngrid(L.shape[0])
        amax = max(c[1] for c in cl)
        # pass 1: stored keys' activations at every grid point; costs
        tk = TopA(Yk, L, amax)
        snaps, cost = {}, {}
        sub = {s: order[:max(1, int(round(s * M)))] for s in sorted({c[4] for c in cl})}
        groups = sorted({(c[1], c[2], c[4]) for c in cl})
        qof = {}
        for c in cl:
            qof.setdefault((c[1], c[2], c[4]), set()).add(c[3])
        for N in Ng:
            I = tk.advance(N)
            snaps[N] = I.astype(np.int32)
            for (a, ctr, s) in groups:
                rows = sub[s]
                Cn = counters(tk.top(a)[rows], vals[rows], V, N, ctr)
                for q in qof[a, ctr, s]:
                    Qc = qtrans(Cn, ctr, q)
                    for acct in accts:
                        cost[(a, ctr, q, s, acct), N] = ctr_bits(Qc, ctr, q, acct)
        # N per (cfg, bpf, acct): the largest grid point that fits
        need = {}
        for c in cl:
            a, ctr, q, s = c[1], c[2], c[3], c[4]
            for b in bpfs:
                Bav = b * M - db
                for acct in accts:
                    fit = [N for N in Ng if cost[(a, ctr, q, s, acct), N] <= Bav]
                    N = max(fit) if fit else 0
                    if N == 0:
                        out[c, b, acct] = (np.full(Yq.shape[0], default), 0.0, 0)
                    else:
                        need.setdefault(N, []).append((c, b, acct))
        # pass 2: queries' activations at the needed N, read
        tq = TopA(Yq, L, amax)
        cc = {}
        for N in sorted(need):
            tq.advance(N)
            for (c, b, acct) in need[N]:
                a, ctr, q, s = c[1], c[2], c[3], c[4]
                key = (a, ctr, s, N)
                if key not in cc:
                    rows = sub[s]
                    Ia = snaps[N][:, :a]
                    if Ia.shape[1] < a:
                        Ia = np.hstack([Ia, np.full((Ia.shape[0], a - Ia.shape[1]), -1, np.int64)])
                    cc[key] = counters(Ia[rows], vals[rows], V, N, ctr)
                Qc = qtrans(cc[key], ctr, q)
                Z = np.vstack([Qc, np.zeros((1, Qc.shape[1]), Qc.dtype)]).astype(np.float64)
                Iq = tq.top(a)
                Iq = np.where(Iq < 0, N, Iq)
                R = Z[Iq].sum(1)
                score = R if ctr == 'onehot' else R @ vb.T.astype(np.float64)
                bt = ctr_bits(Qc, ctr, q, acct) + db
                out[c, b, acct] = (K.decide(score, default), float(bt), int(N))
            # keep cache small: counters at N no longer needed
            cc = {k: v for k, v in cc.items() if k[3] == N}
    return out


# ----------------------------------------------------------------------------------------------- setup and jobs

def other_rows(pool):
    rows = set()
    for s in ('dev', 'test'):
        for e in pool.by_split[s]:
            rows.add(pool.idx[e['name']])
            rows.update(pool.idx[a] for a in e['aliases'])
    return rows


def loc_rows(pool):
    """{loc: row indices of pool.X, in the fixed seeded order}; training strings only, none a dev / test string."""
    lab, ali = [], []
    for e in pool.by_split['train']:
        lab.append(pool.idx[e['name']])
        ali += [pool.idx[a] for a in e['aliases']]
    bad = other_rows(pool)
    badc = {pool.X[r].tobytes() for r in bad}
    out = {}
    for loc, rows in (('LA', lab + ali), ('L', lab), ('A', ali)):
        u = []
        seen = set()
        for r in rows:
            if r not in seen and r not in bad:
                seen.add(r)
                u.append(r)
        perm = np.random.default_rng(C.seed_of('B64LOC', loc)).permutation(len(u))
        o = np.array(u)[perm]
        # also drop any location whose CODE equals a dev / test code (e.g. 'Rio' / 'Rio' with an accent under an
        # accent-stripping tokenizer); order of the remaining locations unchanged
        keep = np.array([pool.X[r].tobytes() not in badc for r in o], bool)
        out[loc] = o[keep]
    return out


_SET = {}


def setup(enc):
    if enc not in _SET:
        cwd = os.getcwd()
        os.chdir(DATA)
        pool = C.Pool(ENC_TAG[enc])
        os.chdir(cwd)
        Tr = pool.train_matrix()
        preps = {p: C.Prep(p, Tr) for p in PREPS}
        lr = loc_rows(pool)
        Ls = {p: {l: preps[p](pool.X[lr[l]]) for l in LOCS} for p in PREPS}
        _SET.clear()
        _SET[enc] = (pool, preps, Ls, lr)
    return _SET[enc]


def job_name(spec):
    return f'{spec["enc"]}_{spec["split"]}_M{spec["M"]}_V{spec["V"]}_r{spec["rep"]}'


def run_job(spec):
    """spec: enc, split, M, V, rep, cfgs (None = full grid; else list of [cfg, bpf, acct]). -> OUT/jobs/{name}.json.gz"""
    fn = os.path.join(OUT, 'jobs', job_name(spec) + '.json.gz')
    if os.path.exists(fn):
        return fn
    t0 = time.time()
    enc, split, M, V, rep = (spec[k] for k in ('enc', 'split', 'M', 'V', 'rep'))
    pool, preps, Ls, _ = setup(enc)
    st = C.make_store(pool, split, M, V, rep)
    order = store_order(st)
    g = st['qgold']
    rec = {'spec': spec, 'qgold': g.tolist(), 'rows': []}
    want = None
    if spec.get('cfgs') is not None:
        want = {}
        for cfg, b, a in spec['cfgs']:
            want.setdefault(tuple(cfg), set()).add((b, a))
    for prep in PREPS:
        cs = [c for c in GRID if c[0] == prep and (want is None or c in want)]
        if not cs:
            continue
        P = preps[prep]
        bs = BPF if want is None else sorted({b for c in cs for b, _ in want[c]})
        o = run_sdmt(st, P(st['keys']), P(st['queries']), Ls[prep], [c[1:] for c in cs], bs, ACCTS, order)
        for (c5, b, a), (pred, bt, N) in o.items():
            cfg = (prep,) + tuple(c5)
            if want is not None and (b, a) not in want[cfg]:
                continue
            row = {'cfg': list(cfg), 'bpf': b, 'acct': a, 'acc': float((pred == g).mean()), 'bits': bt, 'N': N}
            if want is not None:
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


def runplan(planfile, workers, byenc=False):
    """byenc (scheduling only): jobs sorted by encoder and handed out in chunks of 5, so a worker keeps its encoder's
    setup (results do not depend on the schedule: every job is computed from registered seeds)."""
    specs = [json.loads(l) for l in open(planfile)]
    specs = [s for s in specs if not os.path.exists(os.path.join(OUT, 'jobs', job_name(s) + '.json.gz'))]
    specs.sort(key=lambda s: (s['enc'], -s['M']) if byenc else (-s['M'], s['enc']))
    print(f'{len(specs)} jobs to run', flush=True)
    if workers <= 1:
        for s in specs:
            print(_run(s), flush=True)
        return
    import multiprocessing as mp
    with mp.get_context('fork').Pool(workers, maxtasksperchild=None if byenc else 4) as pl:
        for r in pl.imap_unordered(_run, specs, chunksize=5 if byenc else 1):
            if str(r).startswith('FAILED'):
                print(r, flush=True)


def make_plan(kind, encs):
    os.makedirs(OUT, exist_ok=True)
    fn = os.path.join(OUT, f'plan_{kind}.jsonl')
    n = 0
    with open(fn, 'w') as fo:
        for enc in encs:
            ch = json.load(open(f'{ROOT}/results/b64_choice_{enc}.json'))['choice'] if kind == 'test' else None
            for V in VN:
                for M in M_GRID:
                    reps = DEV_REPS[M] if kind == 'dev' else TEST_REPS[M]
                    cfgs = None
                    if kind == 'test':
                        cfgs = [[ch[f'{M}|{V}|{b}|{a}|SDMT']['cfg'], b, a] for b in BPF for a in ACCTS]
                    for rep in range(reps):
                        spec = {'enc': enc, 'split': kind, 'M': M, 'V': V, 'rep': rep, 'cfgs': cfgs}
                        fo.write(json.dumps(spec) + '\n')
                        n += 1
    print(n, 'jobs ->', fn)


# ----------------------------------------------------------------------------------------------- dev choice

SIX = ['IDX', 'SF', 'CM', 'BF', 'FBF', 'SDM']
SEVEN = SIX + ['SDMT']


def choose(enc):
    """B36's rule per (M, V, bpf, acct): highest mean dev accuracy over the cell's dev stores; within 0.1 points tied,
    first in grid order.  BEST7: B36's BEST rule over IDX, SF, CM, BF, FBF, SDM, SDMT (highest dev accuracy of each
    family's chosen configuration, exact ties in that order); the six families' dev accuracies are B49's
    (results/b49_choice_{enc}.json 'dev_by_fam', copied from B36 / B46 / B49's B36-protocol choices)."""
    acc = {}
    jd = os.path.join(OUT, 'jobs')
    for fn in sorted(os.listdir(jd)):
        if not fn.startswith(f'{enc}_dev_'):
            continue
        r = json.load(gzip.open(os.path.join(jd, fn), 'rt'))
        s = r['spec']
        for x in r['rows']:
            acc.setdefault((s['M'], s['V'], x['bpf'], x['acct'], tuple(x['cfg'])), {})[s['rep']] = x['acc']
    ch49 = json.load(open(f'{ROOT}/results/b49_choice_{enc}.json'))['choice']
    ch = {}
    for M in M_GRID:
        for V in VN:
            for b in BPF:
                for a in ACCTS:
                    lst = []
                    for cfg in GRID:
                        d = acc.get((M, V, b, a, cfg))
                        if d is None or len(d) != DEV_REPS[M]:
                            raise SystemExit(f'missing / incomplete dev rows {enc} {M} {V} {b} {a} {cfg}')
                        lst.append((float(np.mean([d[k] for k in sorted(d)])), cfg))
                    top = max(x for x, _ in lst)
                    tied = [c for x, c in lst if x >= top - 0.001]
                    dacc = [x for x, c in lst if c == tied[0]][0]
                    ch[f'{M}|{V}|{b}|{a}|SDMT'] = {'cfg': list(tied[0]), 'dev_acc': dacc, 'best_dev_acc': top,
                                                   'n_tied': len(tied)}
                    dbf = {f: ch49[f'{M}|{V}|{b}|{a}|BEST']['dev_by_fam'][f] for f in SIX}
                    c6 = [(dbf[f], -SIX.index(f), f) for f in SIX]
                    b36 = ch49[f'{M}|{V}|{b}|{a}|BEST36']['fam']
                    assert max(c6)[2] == b36, (enc, M, V, b, a, max(c6), b36)  # reproduces B36's BEST
                    dbf['SDMT'] = dacc
                    c7 = [(dbf[f], -SEVEN.index(f), f) for f in SEVEN]
                    ch[f'{M}|{V}|{b}|{a}|BEST7'] = {'fam': max(c7)[2], 'dev_acc': max(c7)[0], 'dev_by_fam': dbf,
                                                    'BEST36': b36}
    fn = f'{ROOT}/results/b64_choice_{enc}.json'
    json.dump({'enc': enc, 'choice': ch}, open(fn, 'w'), indent=1, sort_keys=True)
    print(len(ch), 'choices ->', fn)


# ----------------------------------------------------------------------------------------------- selftests

def brute(st, Yk, Yq, L, cfg, b, acct, order):
    """Independent slow path for the selftest: full sort, explicit N scan, explicit counters."""
    loc, a, ctr, q, s = cfg
    M, V = st['M'], st['V']
    vals = st['vals']
    default = K.default_value(vals, V)
    db = K.def_bits(V)
    rows = order[:max(1, int(round(s * M)))]
    Sk = Yk.astype(np.float64) @ L.astype(np.float64).T
    Sq = Yq.astype(np.float64) @ L.astype(np.float64).T
    best = (0, None)
    for N in ngrid(L.shape[0]):
        act = [np.argsort(-Sk[r, :N], kind='stable')[:a] for r in rows]
        nb = nbits(V)
        Cn = np.zeros((N, V if ctr == 'onehot' else nb), np.int64)
        vb = vbip(V)
        for r, ac in zip(rows, act):
            for l in ac:
                if ctr == 'onehot':
                    Cn[l, vals[r]] += 1
                else:
                    Cn[l] += vb[vals[r]]
        Qc = qtrans(Cn, ctr, q)
        if ctr_bits(Qc, ctr, q, acct) <= b * M - db:
            best = (N, Qc)
    N, Qc = best
    if N == 0:
        return np.full(Yq.shape[0], default), 0
    sc = np.zeros((Yq.shape[0], V))
    for i in range(Yq.shape[0]):
        for l in np.argsort(-Sq[i, :N], kind='stable')[:a]:
            sc[i] += Qc[l] if ctr == 'onehot' else vbip(V) @ Qc[l]
    return K.decide(sc, default), N


def selftest():
    rng = np.random.default_rng(0)
    # toy: clustered codes; queries = noisy copies of the keys; locations = other draws of the same clusters
    n, M, V = 64, 120, 16
    cent = rng.standard_normal((40, n))

    def draw(k, noise):
        z = cent[rng.integers(0, 40, k)] + noise * rng.standard_normal((k, n))
        return (z / np.linalg.norm(z, axis=1, keepdims=True)).astype(np.float32)

    Yk = draw(M, 0.8)
    Yq = (Yk + 0.15 * rng.standard_normal(Yk.shape)).astype(np.float32)
    Yq /= np.linalg.norm(Yq, axis=1, keepdims=True)
    vals = rng.integers(0, V, M)
    st = {'M': M, 'V': V, 'vals': vals, 'split': 'train', 'rep': 0}
    order = rng.permutation(M)
    Ls = {l: draw(300, 0.8) for l in LOCS}
    cfgs = [(l, a, c, q, s) for l in LOCS for a in (1, 4, 64) for c in CTRS for q in QS for s in SHARES]
    o = run_sdmt(st, Yk, Yq, Ls, cfgs, [4, 16, 256], ACCTS, order)
    for (cfg, b, a), (pr, bt, N) in o.items():
        assert bt <= b * M, (cfg, b, a, bt)
    # the fast path equals the slow independent path on a sample of configurations (both V)
    nchk = 0
    for V2 in (2, 16):
        st2 = dict(st, V=V2, vals=rng.integers(0, V2, M))
        o2 = run_sdmt(st2, Yk, Yq, Ls, cfgs, [4, 16], ['ideal', 'real'], order)
        for cfg in cfgs[::5]:
            for b in (4, 16):
                for acct in ('ideal', 'real'):
                    p1, _, N1 = o2[cfg, b, acct]
                    p2, N2 = brute(st2, Yk, Yq, Ls[cfg[0]], cfg, b, acct, order)
                    assert N1 == N2 and (p1 == p2).mean() >= 0.99, (V2, cfg, b, acct, N1, N2, (p1 == p2).mean())
                    nchk += 1
    # exact keys, large budget, a = 1, onehot counts, locations = the keys themselves: every fact read
    o3 = run_sdmt(st, Yk, Yk, {l: Yk[rng.permutation(M)] for l in LOCS}, [('LA', 1, 'onehot', 'cnt', 1.0)], [256],
                  ['ideal'], order)
    acc3 = float((o3[('LA', 1, 'onehot', 'cnt', 1.0), 256, 'ideal'][0] == vals).mean())
    assert acc3 >= 0.99, acc3
    print(f'selftest ok: budgets respected ({len(o)} rows); fast == slow path on {nchk} (cfg, bpf, acct); '
          f'exact-key read {acc3:.3f}')


def disjoint(encs=PANEL):
    """Location sets vs every dev / test label and alias: string overlap and code-row equality (each encoder)."""
    res = {}
    cwd = os.getcwd()
    os.chdir(DATA)
    for enc in encs:
        pool = C.Pool(ENC_TAG[enc])
        lr = loc_rows(pool)
        bad = sorted(other_rows(pool))
        Xo = pool.X[bad]
        So = {pool.strings[r] for r in bad}
        So_norm = {C.norm(s) for s in So} if hasattr(C, 'norm') else set()
        for l in LOCS:
            rows = lr[l]
            strs = [pool.strings[r] for r in rows]
            n_str = sum(s in So for s in strs)
            n_norm = sum(C.norm(s) in So_norm for s in strs) if So_norm else -1
            Xl = pool.X[rows]
            # exact code-row equality: hash rows
            ho = {hash(x.tobytes()) for x in Xo}
            n_code = sum(hash(x.tobytes()) in ho for x in Xl)
            res[f'{enc}|{l}'] = {'n_loc': int(len(rows)), 'string_overlap': int(n_str),
                                 'normalised_string_overlap': int(n_norm), 'code_row_equal': int(n_code)}
            print(enc, l, res[f'{enc}|{l}'], flush=True)
    os.chdir(cwd)
    return res


def timing(enc, M, V):
    pool, preps, Ls, _ = setup(enc)
    st = C.make_store(pool, 'train', M, V, 0)
    order = store_order(st)
    for prep in PREPS:
        t0 = time.time()
        P = preps[prep]
        o = run_sdmt(st, P(st['keys']), P(st['queries']), Ls[prep], [c[1:] for c in GRID if c[0] == prep], BPF,
                     ACCTS, order)
        print(f'timing {enc} train M={M} V={V} {prep}: {len(o)} rows in {time.time() - t0:.1f}s', flush=True)


def main():
    a = sys.argv[1:]
    if a[0] == 'selftest':
        selftest()
    elif a[0] == 'disjoint':
        r = disjoint(a[1:] or PANEL)
        if len(a) == 1:
            json.dump(r, open(f'{ROOT}/results/b64_disjoint.json', 'w'), indent=1)
    elif a[0] == 'timing':
        timing(a[1], int(a[2]), int(a[3]))
    elif a[0] == 'job':
        run_job(json.load(open(a[1])))
    elif a[0] == 'plan':
        encs = a[a.index('--encs') + 1].split(',') if '--encs' in a else PANEL
        make_plan(a[1], encs)
    elif a[0] == 'runplan':
        wk = int(a[a.index('--workers') + 1]) if '--workers' in a else 1
        runplan(a[1], wk, '--byenc' in a)
    elif a[0] == 'choose':
        choose(a[1])
    else:
        raise SystemExit(a[0])


if __name__ == '__main__':
    main()
