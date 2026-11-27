"""B71 (prereg/B71_hebbian_remedies_prereg.md): remedies for the Hebbian store at equal bits over R1, and the controls
that separate the write rule from the features.

Arms run here (R1 = V 2 / 16 x 4, 8, 16 bits per fact x M 256 / 1,024; ideal accounting; B49's codes, splits, stores):
  NLMSF  B62's NLMSc (one pass of the normalised delta rule, LAMc's targets / scales / quantizer / bits / read;
         b62_run.run_online UNCHANGED) on the STORE's own features: kind R = the final design's random map of the cell
         (B53's dev-chosen prep / fmap; b66_lsfc.FeatStoreR), kind P = VSAPc's principal map of the cell (b66_lsfc.
         FeatStoreP). Grid: scale x target x mu x output x q = 120 (B62's order). Arms NLMSFR, NLMSFP.
  VSAPR  (part B) Hebbian store on rotated principal features: phi(y) = fmap(((P(y) - mu) U_p) Rot), U_p = the store's
         principal basis (b66_lsfc.basis: centre = B57's LamFeat basis, whiten = the corrected U0, mu = 0), p in P(n);
         Rot = p x d, columns of independent Haar-random orthogonal p x p matrices stacked side by side (seeded), so
         d <= p is a rotation followed by truncation and d > p a union of rotations; fmap lin / sgn / sgn3 (product of
         three independently rotated sign maps); B53's value codes, quantizer, seeds, stored-subset order and read
         (b32_run.vsa_grid unchanged), d = B32's size rule (no cap). Grid = the store's 72 x |P(n)|.
  VSAD   (part C) the store written by an error-driven normalised delta rule on its OWN bundle and value codes: per fact
         in store order, m_c = <x_c, S_c> (x = phi(k) * r, coordinates grouped by residue class c = j mod 16 at V = 16,
         one class at V = 2; the classes the Hadamard / antipodal codes keep apart, so the matched-filter score of value
         u is sum_c H[u, c] m_c), e_c = y_c - m_c with y_c = H16[v, c] (V = 16) or +-1 (V = 2), S_c += mu e_c x_c /
         ||x_c||^2; rule 'nlms' = every fact, 'err' = only facts the current bundle misreads (perceptron-style).
         S_0 = 0, float64 accumulator, E passes, quantized once at the end by the store's quantizer to q bits (d by
         B32's rule), matched-filter read (b56_run.mf_read). Kind R = B53's chosen (prep, fmap, share) of the cell (the
         same bundle B56 retrained); kind P = VSAPc's chosen (prep, pfmap, share) (d <= n, as VSAPc).
         Grid: E = 1: rule x mu x output (last / Polyak average) x q = 60; E in {3, 10}: rule x mu x q (last) = 60.
  AUDIT  (dev only, kind R) B56's retraining re-run at its grid and an extended one (eta 1, 0.1, 0.01; E 1, 3, 10,
         30; q 1, 2, 4), quantized and unquantized reads, accuracy on the stored keys, update counts; the Hebbian bundle
         (B53's quantizer stream: b56_run.bundle_q) quantized and unquantized.

python3 b71_run.py selftest
python3 b71_run.py timing ENC M V
python3 b71_run.py plan dev|test [--encs ..] [--arms NLMSF,VSAPR,VSAD]
python3 b71_run.py runplan PLAN --workers N
python3 b71_run.py choose ENC                      # -> results/b71_choice_{enc}.json
"""
import os

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import gzip  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import b32_common as C  # noqa: E402
import b32_run as R  # noqa: E402
import b53_run as R53  # noqa: E402
import b56_run as E56  # noqa: E402
import b60_run as L60  # noqa: E402
import b62_run as O62  # noqa: E402
import b66_lsfc as LS  # noqa: E402

OUT = os.environ.get('B71_OUT', os.path.join(ROOT, 'b71_out'))
RES = os.environ.get('B71_RES', os.path.join(ROOT, 'results'))
INRES = os.path.join(ROOT, 'results')
PANEL = L60.PANEL
LLM, SENT = PANEL[:4], PANEL[4:]
M_GRID, VN, BPF = [256, 1024], [2, 16], [4, 8, 16]
ACCT = 'ideal'
DEV_REPS, TEST_REPS = L60.DEV_REPS, L60.TEST_REPS
CH = C.CH
H16 = R53.H16.astype(np.float64)
QS = [1, 2, 4]
# NLMSF: B62's grid without the (prep, feat) factor (fixed by the store's features)
NF_GRID = [(s, t, mu, o, q) for s in O62.SCALES for t in O62.TARGETS for mu in O62.MUS for o in O62.OUTS for q in QS]
# VSAPR: the store's 72 x p
FMAPS = ['lin', 'sgn', 'sgn3']
SHARES = [1.0, 0.5, 0.25, 0.125]


def p_grid(n):
    return sorted({p for p in (64, 256, 1024) if p < n} | {n})


PR_ORDER = lambda c: (('center', 'whiten').index(c[0]), FMAPS.index(c[1]), c[2], QS.index(c[3]), SHARES.index(c[4]))  # noqa: E731
# VSAD
RULES = ['nlms', 'err']
MUS_D = [1.0, 0.5, 0.25, 0.1, 0.03]
EPS_D = [1, 3, 10]
D_GRID = [(1, r, mu, o, q) for r in RULES for mu in MUS_D for o in ('last', 'avg') for q in QS] + \
         [(E, r, mu, 'last', q) for E in (3, 10) for r in RULES for mu in MUS_D for q in QS]  # 120, tie order
D1_GRID = [g for g in D_GRID if g[0] == 1]
# AUDIT (B56's retraining, extended)
RT_E, RT_ETA = [1, 3, 10, 30], [1.0, 0.1, 0.01]
RT_B56 = {(e, eta) for e in (1, 3, 10) for eta in (1.0, 0.1)}


def odir(*a):
    d = os.path.join(OUT, *a)
    os.makedirs(d, exist_ok=True)
    return d


def n_of(enc):
    return int(LS.setup(enc)['pool'].X.shape[1])


def b53_cfg(enc):
    """{(M, V, bpf): (prep, fmap, q, f)} B53's dev choice (ideal) = the final design's configuration."""
    ch = json.load(open(f'{INRES}/b53_choice_{enc}.json'))['choice']
    return {(M, V, b): tuple(ch[f'{M}|{V}|{b}|ideal|VSA']['cfg']) for M in M_GRID for V in VN for b in BPF}


def pc_cfg(enc):
    """{(M, V, bpf): (prep, pfmap, q, f)} VSAPc's matched-filter dev choice (ideal)."""
    ch = json.load(open(f'{INRES}/b60_choicepc_{enc}.json'))['choice']
    return {(M, V, b): tuple(ch[f'{M}|{V}|{b}|ideal|VSA']['cfg']) for M in M_GRID for V in VN for b in BPF}


# ----------------------------------------------------------------------------------------------- NLMSF

def nf_cfg7(g, prep, kind):
    s, t, mu, o, q = g
    return (s, t, prep, LS.KINDS[kind], mu, o, q)


def run_nlmsf(spec):
    enc, kind, split, M, V, rep, prep, fmap = (spec[k] for k in ('enc', 'kind', 'split', 'M', 'V', 'rep', 'prep',
                                                                 'fmap'))
    S = LS.setup(enc)
    st = C.make_store(S['pool'], split, M, V, rep)
    P = S['preps'][prep]
    Yk, Yq = P(st['keys']), P(st['queries'])
    fc = LS.make_fc(enc, kind, prep, fmap)
    seed = C.seed_of('b71nf', enc, kind, split, M, V, rep)
    if spec.get('cfgs') is None:
        want = [('NLMSC', nf_cfg7(g, prep, kind), b) for g in NF_GRID for b in spec['bs']]
    else:
        want = [('NLMSC', nf_cfg7(tuple(g), prep, kind), b) for g, b in spec['cfgs']]
    o = O62.run_online(st, Yk, Yq, fc, enc, prep, want, seed)
    g = st['qgold']
    rows = []
    for (arm, c7, b) in want:
        pred, bt, p = o[arm, c7, b]
        row = {'cfg': [c7[0], c7[1], c7[4], c7[5], c7[6]], 'bpf': b, 'acc': float((pred == g).mean()), 'bits': bt,
               'p': p}
        if split == 'test':
            row['pred'] = np.asarray(pred).astype(int).tolist()
        rows.append(row)
    return {'qgold': g.tolist(), 'pmax': fc.pmax(LS.KINDS[kind]), 'rows': rows}


# ----------------------------------------------------------------------------------------------- VSAPR

def haar(p, seed):
    A = np.random.default_rng(seed).standard_normal((p, p))
    Q, Rr = np.linalg.qr(A)
    return (Q * np.sign(np.diag(Rr))[None, :]).astype(np.float32)


class FeatPR(R53.Feat53):
    """Rotated principal features with B53's value codes (key (enc, prep, fmap), as the random store's)."""

    def __init__(self, enc, prep, fmap, n, p, U, mu):
        super().__init__(enc, prep, fmap, n, 'cpu')
        t = self.torch
        t.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '1')))
        self.p = p
        self.Upt = t.from_numpy(np.ascontiguousarray(U[:, :p].astype(np.float32)))
        self.mut = t.from_numpy(np.ascontiguousarray(mu.astype(np.float32)))
        self.hcache = {}

    def rot_cols(self, m, j0, w):
        """Columns j0 .. j0 + w - 1 of Rot_m (p x inf): column j = column (j mod p) of Haar matrix k = j // p."""
        cols = []
        for k in range(j0 // self.p, (j0 + w - 1) // self.p + 1):
            if (m, k) not in self.hcache:
                self.hcache[m, k] = haar(self.p, C.seed_of('B71ROT', *self.key, self.p, m, k))
            a, b = max(j0, k * self.p), min(j0 + w, (k + 1) * self.p)
            cols.append(self.hcache[m, k][:, a - k * self.p:b - k * self.p])
        return np.hstack(cols)

    def codes(self, j0):
        """B53's value codes of block j0 for self.V (= Feat53's, without drawing the Gaussian projections)."""
        if (j0, self.V) not in self.ccache:
            r = np.random.default_rng(C.seed_of('C', *self.key, j0, CH))
            row0 = np.where(r.random((1, CH), dtype=np.float32) < 0.5, -1.0, 1.0).astype(np.float32)[0]
            self.ccache[j0, self.V] = self.torch.from_numpy(R53.code_block(self.V, row0))
        return self.ccache[j0, self.V]

    def block(self, j0):
        t = self.torch
        return [t.from_numpy(np.ascontiguousarray(self.rot_cols(m, j0, CH))) for m in range(self.K)], self.codes(j0)

    def phi(self, Y, Gs, w):
        Z = (Y - self.mut) @ self.Upt
        out = None
        for G in Gs:
            Pm = Z @ G[:, :w]
            f = Pm if self.fmap == 'lin' else self.torch.where(Pm >= 0, 1.0, -1.0)
            out = f if out is None else out * f
        return out


def run_vsapr(spec):
    """All reps of one (enc, split, M, V, prep, fmap, p): rows per (q, f, bpf)."""
    enc, split, M, V, prep, fmap, p = (spec[k] for k in ('enc', 'split', 'M', 'V', 'prep', 'fmap', 'p'))
    S = LS.setup(enc)
    U, mu = LS.basis(enc, prep)
    n = U.shape[0]
    feat = FeatPR(enc, prep, fmap, n, p, U, mu)
    feat.V = V
    if spec.get('cfgs') is None:
        qfb = [(q, f, b) for q in QS for f in SHARES for b in spec['bs']]
    else:
        qfb = [tuple(x) for x in spec['cfgs']]
    out = {}
    for rep in spec['reps']:
        st = C.make_store(S['pool'], split, M, V, rep)
        P = S['preps'][prep]
        cfgs = [((q, f, b), C.vsa_dim(b * M, q, ACCT), q, max(1, int(math.floor(f * M)))) for (q, f, b) in qfb]
        pred = R.vsa_grid(P(st['keys']), st['vals'], P(st['queries']), V, feat, cfgs, R.store_seed(enc, st),
                          R.store_order(st))
        g = st['qgold']
        rows = []
        for (q, f, b), d, _, ns in cfgs:
            pr = pred[q, f, b]
            row = {'cfg': [prep, fmap, p, q, f], 'bpf': b, 'acc': float((pr == g).mean()), 'd': d,
                   'bits': C.vsa_bits(d, q, ACCT)}
            if split == 'test':
                row['pred'] = np.asarray(pr).astype(int).tolist()
            rows.append(row)
        out[str(rep)] = {'qgold': g.tolist(), 'rows': rows}
    return {'reps': out}


# ----------------------------------------------------------------------------------------------- VSAD (+ audit)

def delta_write(X, y, V, d, rule, mu, E_list, nrm_eps=1e-12):
    """The store's bundle written by the normalised delta rule. X: ns x dmax (phi(k) * r, write order, float64),
    y: ns values. Returns {(E, out): S (d,)} for E in E_list ('avg' only at E = 1) and per-pass update counts."""
    ncl = 16 if V == 16 else 1
    D16 = -(-d // ncl) * ncl
    ns = X.shape[0]
    Xp = np.zeros((ns, D16))
    Xp[:, :d] = X[:, :d]
    X3 = Xp.reshape(ns, D16 // ncl, ncl)
    nrm = (X3 ** 2).sum(1) + nrm_eps
    T = H16[y] if V == 16 else np.where(y == 0, 1.0, -1.0)[:, None]
    z = np.zeros((D16 // ncl, ncl))
    zs = np.zeros_like(z)
    snaps, nupd = {}, []
    Emax = max(E_list)
    for ep in range(1, Emax + 1):
        cnt = 0
        for t in range(ns):
            xt = X3[t]
            m = (xt * z).sum(0)
            upd = True
            if rule == 'err':
                sc = H16 @ m if V == 16 else np.array([m[0], -m[0]])
                upd = int(np.argmax(sc)) != int(y[t])
            if upd:
                z += xt * (mu * (T[t] - m) / nrm[t])[None, :]
                cnt += 1
            if ep == 1:
                zs += z
        nupd.append(cnt)
        if ep in E_list:
            snaps[ep, 'last'] = z.reshape(-1)[:d].copy()
            if ep == 1:
                snaps[1, 'avg'] = (zs / ns).reshape(-1)[:d].copy()
    return snaps, nupd


def retrain_count(Fk, Cv, vals, order, d, ns, epochs, eta=1.0):
    """b56_run.retrain (identical arithmetic) that also returns the number of corrected facts per epoch."""
    idx = np.asarray(order[:ns])
    X = np.ascontiguousarray(Fk[idx, :d])
    y = np.asarray(vals)[idx]
    Cd = np.ascontiguousarray(Cv[:, :d])
    S = (X * Cd[y]).astype(np.float64).sum(0)
    snaps, cnts = {}, []
    for ep in range(1, max(epochs) + 1):
        c = 0
        for b0 in range(0, ns, E56.BATCH):
            Xb = X[b0:b0 + E56.BATCH]
            yb = y[b0:b0 + E56.BATCH]
            pr = (Xb @ (Cd * S.astype(np.float32)).T).argmax(1)
            w = np.nonzero(pr != yb)[0]
            if len(w):
                S += eta * (Xb[w].astype(np.float64) * (Cd[yb[w]] - Cd[pr[w]])).sum(0)
                c += len(w)
        cnts.append(c)
        if ep in epochs:
            snaps[ep] = S.copy()
    return snaps, cnts


def mf_float(Fq, Cv, S):
    d = len(S)
    return (Fq[:, :d].astype(np.float64) @ (Cv[:, :d].astype(np.float64) * S).T).argmax(1)


def d_of(kind, b, M, q, n):
    d = C.vsa_dim(b * M, q, ACCT)
    return min(d, n) if kind == 'P' else d


def store_feats_d(enc, kind, prep, fmap, V, Yk, Yq, dmax, n):
    """(Fk, Fq, Cv) float32: kind R = B53's random map (Feat53), kind P = VSAPc's principal map (FeatP's values)."""
    if kind == 'R':
        feat = R53.Feat53(enc, prep, fmap, n, 'cpu')
        feat.V = V
        return E56.feats_np(feat, Yk, dmax), E56.feats_np(feat, Yq, dmax), E56.codes_np(feat, V, dmax), feat
    fc = LS.FeatStoreP(enc, prep, fmap)
    feat = R53.Feat53(enc, prep, 'lin' if fmap == 'plin' else 'sgn', n, 'cpu')
    feat.V = V
    return fc.feats(Yk, 'pca', 0, dmax), fc.feats(Yq, 'pca', 0, dmax), E56.codes_np(feat, V, dmax), feat


def run_vsad(spec):
    """One store, one (kind, prep, fmap, f) group of R1 budgets. Rows: VSAD grid (dev: all; test: chosen);
    dev + kind R + spec['audit']: AUDIT rows."""
    enc, kind, split, M, V, rep, prep, fmap, f = (spec[k] for k in ('enc', 'kind', 'split', 'M', 'V', 'rep', 'prep',
                                                                    'fmap', 'f'))
    S = LS.setup(enc)
    n = n_of(enc)
    st = C.make_store(S['pool'], split, M, V, rep)
    P = S['preps'][prep]
    order = R.store_order(st)
    sseed = R.store_seed(enc, st)
    vals = np.asarray(st['vals'])
    ns = max(1, int(math.floor(f * M)))
    if spec.get('cfgs') is None:
        cells = [(tuple(g), b) for g in D_GRID for b in spec['bs']]
    else:
        cells = [(tuple(g), b) for g, b in spec['cfgs']]
    ds = {(q, b): d_of(kind, b, M, q, n) for (_, b) in cells for q in QS}
    if spec.get('audit'):
        for b in spec['bs']:
            for q in QS:
                ds[q, b] = d_of(kind, b, M, q, n)
    dmax = max(ds.values())
    Fk, Fq, Cv, feat = store_feats_d(enc, kind, prep, fmap, V, P(st['keys']), P(st['queries']), dmax, n)
    r = Cv[0].astype(np.float64)
    idx = order[:ns]
    X = Fk[idx].astype(np.float64) * r[None, :]
    yv = vals[idx]
    g = st['qgold']
    rows = []
    # VSAD: one write per (d, rule, mu) with snapshots
    need = {}
    for (E, rule, mu, o, q), b in cells:
        need.setdefault((ds[q, b], rule, mu), set()).add(E)
    snapc = {}
    for (d, rule, mu), Es in need.items():
        snapc[d, rule, mu] = delta_write(X, yv, V, d, rule, mu, sorted(Es | {1}))
    for (E, rule, mu, o, q), b in cells:
        d = ds[q, b]
        snaps, nupd = snapc[d, rule, mu]
        Sf = snaps[E, o]
        Sq = E56.quant_np(Sf, q, ('B71Q', sseed, kind, d, q, ns, rule, mu, E, o))
        pr = E56.mf_read(Fq, Cv, Sq)
        row = {'arm': 'VSAD', 'cfg': [E, rule, mu, o, q], 'bpf': b, 'acc': float((pr == g).mean()), 'd': d,
               'bits': C.vsa_bits(d, q, ACCT), 'nupd': nupd[:E]}
        if split == 'test':
            row['pred'] = np.asarray(pr).astype(int).tolist()
        else:
            row['acc_float'] = float((mf_float(Fq, Cv, Sf) == g).mean())
            row['acc_keys'] = float((E56.mf_read(Fk[idx], Cv, Sq) == yv).mean())
        rows.append(row)
    # AUDIT (dev, kind R): Hebbian bundle and B56's retraining (B56's seeds), extended grid
    if spec.get('audit') and split == 'dev' and kind == 'R':
        for b in spec['bs']:
            for q in QS:
                d = ds[q, b]
                Sq = E56.bundle_q(feat, Fk, Cv, vals, order, d, q, ns, sseed)
                Sh = (Fk[idx, :d].astype(np.float64) * Cv[vals[idx], :d]).sum(0)
                rows.append({'arm': 'HEB', 'cfg': [q], 'bpf': b, 'd': d,
                             'acc': float((E56.mf_read(Fq, Cv, Sq) == g).mean()),
                             'acc_float': float((mf_float(Fq, Cv, Sh) == g).mean()),
                             'acc_keys': float((E56.mf_read(Fk[idx], Cv, Sq) == yv).mean()),
                             'acc_keys_float': float((mf_float(Fk[idx], Cv, Sh) == yv).mean())})
                for eta in RT_ETA:
                    snaps, cnts = retrain_count(Fk, Cv, vals, order, d, ns, RT_E, eta)
                    for e in RT_E:
                        Sq = E56.quant_np(snaps[e], q, ('Q56', sseed, d, q, ns, e, eta))
                        rows.append({'arm': 'RT', 'cfg': [q, e, eta], 'bpf': b, 'd': d, 'b56grid': (e, eta) in RT_B56,
                                     'acc': float((E56.mf_read(Fq, Cv, Sq) == g).mean()),
                                     'acc_float': float((mf_float(Fq, Cv, snaps[e]) == g).mean()),
                                     'acc_keys': float((E56.mf_read(Fk[idx], Cv, Sq) == yv).mean()),
                                     'nupd': cnts[:e]})
    return {'qgold': g.tolist(), 'rows': rows, 'n': n}


# ----------------------------------------------------------------------------------------------- jobs

def job_name(s):
    if s['arm'] == 'VSAPR':
        return f'{s["enc"]}_VSAPR_{s["split"]}_M{s["M"]}_V{s["V"]}_{s["prep"]}_{s["fmap"]}_p{s["p"]}'
    return (f'{s["enc"]}_{s["arm"]}{s["kind"]}_{s["split"]}_M{s["M"]}_V{s["V"]}_r{s["rep"]}_{s["prep"]}_{s["fmap"]}'
            + (f'_f{s["f"]}' if 'f' in s else '') + (f'_{s["tag"]}' if s.get('tag') else ''))


RUN = {'NLMSF': run_nlmsf, 'VSAPR': run_vsapr, 'VSAD': run_vsad}


def run_job(spec):
    fn = os.path.join(OUT, 'jobs', job_name(spec) + '.json.gz')
    if os.path.exists(fn):
        return fn
    t0 = time.time()
    rec = RUN[spec['arm']](spec)
    rec['spec'] = spec
    rec['sec'] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    with gzip.open(fn + '.tmp', 'wt') as fo:
        json.dump(rec, fo)
    os.replace(fn + '.tmp', fn)
    print(f'job {job_name(spec)} {rec["sec"]}s', flush=True)
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
    # long jobs first (scheduling only; results depend on registered seeds only)
    pri = {'VSAPR': 0, 'VSAD': 1, 'NLMSF': 2}
    specs.sort(key=lambda s: (pri[s['arm']], -s['M'], -s.get('p', 0), PANEL.index(s['enc'])))
    print(f'{len(specs)} jobs to run', flush=True)
    if workers <= 1:
        for s in specs:
            print(_run(s), flush=True)
        return
    import multiprocessing as mp
    with mp.get_context('fork').Pool(workers) as pl:
        for r in pl.imap_unordered(_run, specs, chunksize=1):
            if str(r).startswith('FAILED'):
                print(r, flush=True)


def groups_of(cfgmap, M, V):
    """{(prep, fmap, f): [bpf]} from a {(M, V, b): (prep, fmap, q, f)} map."""
    g = {}
    for b in BPF:
        prep, fmap, _, f = cfgmap[M, V, b]
        g.setdefault((prep, fmap, f), []).append(b)
    return g


def make_plan(split, encs, arms):
    odir()
    fn = os.path.join(OUT, f'plan_{split}.jsonl')
    n = 0
    with open(fn, 'w') as fo:
        for enc in encs:
            ch = choice(enc) if split == 'test' else None
            nn = n_of(enc)
            reps = (DEV_REPS if split == 'dev' else TEST_REPS)
            cfg = {'R': b53_cfg(enc), 'P': pc_cfg(enc)}
            for M in M_GRID:
                for V in VN:
                    if 'NLMSF' in arms:
                        for kd in ('R', 'P'):
                            gpf = {}
                            for (prep, fmap, _f), bs in sorted(groups_of(cfg[kd], M, V).items()):
                                gpf.setdefault((prep, fmap), []).extend(bs)
                            for (prep, fmap), bs in sorted(gpf.items()):
                                bs = sorted(bs)
                                cfgs = None
                                if split == 'test':
                                    cfgs = [[ch[f'{M}|{V}|{b}|NLMSF{kd}']['cfg'], b] for b in bs]
                                for rep in range(reps[M]):
                                    s = {'arm': 'NLMSF', 'enc': enc, 'kind': kd, 'split': split, 'M': M, 'V': V,
                                         'rep': rep, 'prep': prep, 'fmap': fmap, 'bs': bs, 'cfgs': cfgs}
                                    fo.write(json.dumps(s) + '\n')
                                    n += 1
                    if 'VSAD' in arms:
                        for kd in ('R', 'P'):
                            for (prep, fmap, f), bs in sorted(groups_of(cfg[kd], M, V).items()):
                                cfgs = None
                                if split == 'test':
                                    cfgs = []
                                    for b in bs:
                                        for a in (f'VSAD{kd}1', f'VSAD{kd}E'):
                                            c = [ch[f'{M}|{V}|{b}|{a}']['cfg'], b]
                                            if c not in cfgs:
                                                cfgs.append(c)
                                for rep in range(reps[M]):
                                    s = {'arm': 'VSAD', 'enc': enc, 'kind': kd, 'split': split, 'M': M, 'V': V,
                                         'rep': rep, 'prep': prep, 'fmap': fmap, 'f': f, 'bs': bs, 'cfgs': cfgs,
                                         'audit': split == 'dev' and kd == 'R'}
                                    fo.write(json.dumps(s) + '\n')
                                    n += 1
                    if 'VSAPR' in arms:
                        if split == 'dev':
                            for prep in ('center', 'whiten'):
                                for fmap in FMAPS:
                                    for p in p_grid(nn):
                                        s = {'arm': 'VSAPR', 'enc': enc, 'split': split, 'M': M, 'V': V,
                                             'prep': prep, 'fmap': fmap, 'p': p, 'bs': BPF, 'cfgs': None,
                                             'reps': list(range(reps[M]))}
                                        fo.write(json.dumps(s) + '\n')
                                        n += 1
                        else:
                            grp = {}
                            for b in BPF:
                                c = ch[f'{M}|{V}|{b}|VSAPR']['cfg']
                                grp.setdefault((c[0], c[1], c[2]), []).append([c[3], c[4], b])
                            for (prep, fmap, p), cf in sorted(grp.items()):
                                s = {'arm': 'VSAPR', 'enc': enc, 'split': split, 'M': M, 'V': V, 'prep': prep,
                                     'fmap': fmap, 'p': p, 'bs': BPF, 'cfgs': cf, 'reps': list(range(reps[M]))}
                                fo.write(json.dumps(s) + '\n')
                                n += 1
    print(n, 'jobs ->', fn)


# ----------------------------------------------------------------------------------------------- dev choice

def choice(enc):
    return json.load(open(f'{RES}/b71_choice_{enc}.json'))['choice']


def pick(lst, key):
    """B60's rule: highest mean dev accuracy; within 0.1 points tied, first in the given order."""
    top = max(a for a, _ in lst)
    tied = sorted([c for a, c in lst if a >= top - 0.001], key=key)
    return tied[0], [a for a, c in lst if c == tied[0]][0], len(tied)


def load_dev(enc):
    jd = os.path.join(OUT, 'jobs')
    acc = {}
    for fn in sorted(os.listdir(jd)):
        if not fn.startswith(f'{enc}_') or '_dev_' not in fn:
            continue
        r = json.load(gzip.open(os.path.join(jd, fn), 'rt'))
        s = r['spec']
        if s['enc'] != enc:
            continue
        if s['arm'] == 'VSAPR':
            for rep, rr in r['reps'].items():
                for x in rr['rows']:
                    k = ('VSAPR', s['M'], s['V'], x['bpf'], tuple(x['cfg']))
                    acc.setdefault(k, {})[int(rep)] = x['acc']
        else:
            for x in r['rows']:
                arm = s['arm'] + s['kind'] if s['arm'] != 'VSAD' or x['arm'] == 'VSAD' else x['arm']
                k = (arm, s['M'], s['V'], x['bpf'], tuple(x['cfg']), s['prep'], s['fmap'])
                if s['rep'] in acc.get(k, {}):  # a re-computed job (compute note 3): must be identical
                    assert abs(acc[k][s['rep']]['acc'] - x['acc']) < 1e-12, ('duplicate dev row differs', k, s['rep'])
                    continue
                acc.setdefault(k, {})[s['rep']] = x
    return acc


def load_devsum(enc):
    """b71_devsum.py's summaries (computed where the dev jobs ran; compute note 4): {key: {'n': n, 'm': means}}."""
    out = {}
    for k, n, m in json.load(gzip.open(os.path.join(OUT, f'devsum_{enc}.json.gz'), 'rt')):
        key = tuple(k[:4]) + (tuple(k[4]),) + tuple(k[5:])
        out[key] = {'n': n, 'm': m}
    return out


def dev_mean(acc, k, need, fld='acc'):
    """Mean dev value over the cell's dev stores (job rows or summaries); raises if missing or incomplete."""
    d = acc.get(k)
    if d is None:
        raise SystemExit(f'missing dev rows {k}')
    if 'n' in d and 'm' in d:
        if d['n'] != need:
            raise SystemExit(f'incomplete dev rows {k}: {d["n"]}')
        return d['m'][fld]
    if len(d) != need:
        raise SystemExit(f'incomplete dev rows {k}: {len(d)}')
    return float(np.mean([(d[i][fld] if isinstance(d[i], dict) else d[i]) for i in sorted(d)]))


def choose(enc):
    acc = load_devsum(enc) if os.environ.get('B71_DEVSUM') == '1' else load_dev(enc)
    nn = n_of(enc)
    cfg = {'R': b53_cfg(enc), 'P': pc_cfg(enc)}
    ch56 = json.load(open(f'{INRES}/b56_choice_{enc}.json'))['choice']
    chpc = json.load(open(f'{INRES}/b60_choicepc_{enc}.json'))['choice']
    out = {}
    for M in M_GRID:
        for V in VN:
            for b in BPF:
                cell = f'{M}|{V}|{b}'

                def mean_of(k, need=DEV_REPS[M]):
                    return dev_mean(acc, k, need)
                for kd in ('R', 'P'):
                    prep, fmap, _, f = cfg[kd][M, V, b]
                    lst = [(mean_of((f'NLMSF{kd}', M, V, b, g, prep, fmap)), g) for g in NF_GRID]
                    c, a, nt = pick(lst, NF_GRID.index)
                    out[f'{cell}|NLMSF{kd}'] = {'cfg': list(c), 'dev_acc': a, 'n_tied': nt, 'feat': [prep, fmap],
                                                'n_cfg': len(lst)}
                    for nm, grid in ((f'VSAD{kd}1', D1_GRID), (f'VSAD{kd}E', D_GRID)):
                        lst = [(mean_of(('VSAD' + kd, M, V, b, g, prep, fmap)), g) for g in grid]
                        c, a, nt = pick(lst, D_GRID.index)
                        out[f'{cell}|{nm}'] = {'cfg': list(c), 'dev_acc': a, 'n_tied': nt, 'feat': [prep, fmap, f],
                                               'n_cfg': len(lst)}
                dr, dp = out[f'{cell}|NLMSFR']['dev_acc'], out[f'{cell}|NLMSFP']['dev_acc']
                out[f'{cell}|NLMSFX'] = {'src': 'NLMSFP' if dp > dr + 0.001 else 'NLMSFR', 'dev': {'R': dr, 'P': dp}}
                dr, dp = out[f'{cell}|VSADR1']['dev_acc'], out[f'{cell}|VSADP1']['dev_acc']
                out[f'{cell}|VSADX'] = {'src': 'VSADP1' if dp > dr + 0.001 else 'VSADR1', 'dev': {'R': dr, 'P': dp}}
                # VSAPR: the store's 72 x p, B32's rule (prep > fmap > p > q > share)
                lst = []
                for prep in ('center', 'whiten'):
                    for fmap in FMAPS:
                        for p in p_grid(nn):
                            for q in QS:
                                for f in SHARES:
                                    c = (prep, fmap, p, q, f)
                                    lst.append((dev_mean(acc, ('VSAPR', M, V, b, c), DEV_REPS[M]), c))
                c, a, nt = pick(lst, PR_ORDER)
                out[f'{cell}|VSAPR'] = {'cfg': list(c), 'dev_acc': a, 'n_tied': nt, 'n_cfg': len(lst)}
                # VSAPcX's dev accuracy (B60's rule: VSAPC iff its dev > VSAS's + 0.1 points)
                s = ch56[f'{cell}|ideal|STAR']
                pc = chpc[f'{cell}|ideal|PSTAR']
                ds, dpc = s['dev'][s['fam']], pc['dev'][pc['fam']]
                dx = dpc if dpc > ds + 0.001 else ds
                out[f'{cell}|VSAPRX'] = {'src': 'VSAPR' if a > dx + 0.001 else 'VSAPCX',
                                         'dev': {'VSAPCX': dx, 'VSAPR': a}}
                # audit summaries (dev, kind R): B56's RT re-chosen on its own grid (reproduction) and extended
                prep, fmap, _, f = cfg['R'][M, V, b]
                rt = []
                for q in QS:
                    for e in RT_E:
                        for eta in RT_ETA:
                            k = ('RT', M, V, b, (q, e, eta), prep, fmap)
                            rt.append((mean_of(k), (q, e, eta)))
                b56o = lambda c: (QS.index(c[0]), [1, 3, 10, 30].index(c[1]), [1.0, 0.1, 0.01].index(c[2]))  # noqa: E731
                c1, a1, _ = pick([x for x in rt if (x[1][1], x[1][2]) in RT_B56], b56o)
                c2, a2, _ = pick(rt, b56o)
                heb = {q: mean_of(('HEB', M, V, b, (q,), prep, fmap)) for q in QS}
                out[f'{cell}|AUDIT'] = {'rt_b56grid': {'cfg': list(c1), 'dev_acc': a1},
                                        'rt_ext': {'cfg': list(c2), 'dev_acc': a2}, 'heb': heb,
                                        'b56_rt_cfg': ch56[f'{cell}|ideal|RT']['cfg'],
                                        'b56_rt_dev': ch56[f'{cell}|ideal|RT']['dev_acc']}
    fn = f'{RES}/b71_choice_{enc}.json'
    json.dump({'enc': enc, 'choice': out}, open(fn, 'w'), indent=1, sort_keys=True)
    print(len(out), 'choices ->', fn)


# ----------------------------------------------------------------------------------------------- selftest / timing

def training_store(enc, M, V, rep=0):
    """A store drawn from the TRAINING split (selftest / timing only)."""
    S = LS.setup(enc)
    return C.make_store(S['pool'], 'train', M, V, rep)


def selftest():
    rng = np.random.default_rng(0)
    # 1. delta_write: one pass of the 'nlms' rule = per-class dual NLMS (forward substitution), V = 16 and V = 2
    for V in (2, 16):
        ns, d = 60, 200
        X = rng.standard_normal((ns, d))
        y = rng.integers(0, V, ns)
        for mu in (1.0, 0.25):
            snaps, _ = delta_write(X, y, V, d, 'nlms', mu, [1])
            S = snaps[1, 'last']
            ncl = 16 if V == 16 else 1
            T = H16[y] if V == 16 else np.where(y == 0, 1.0, -1.0)[:, None]
            S2 = np.zeros(d)
            for c in range(ncl):
                cols = np.arange(c, d, ncl)
                F = X[:, cols]
                D = O62.nlms_dual(F @ F.T, T[:, [c]], mu)
                S2[cols] = (F.T @ D)[:, 0]
            err = np.abs(S - S2).max() / np.abs(S2).max()
            assert err < 1e-8, (V, mu, err)
            # avg output = Polyak average
            snaps, _ = delta_write(X, y, V, d, 'nlms', mu, [1])
        # 'err' rule makes no more updates than 'nlms' and reads every stored key right after enough passes (toy)
        sn, cnt = delta_write(X, y, V, d, 'err', 1.0, [10])
        assert cnt[-1] <= ns, cnt
    print('delta_write = dual NLMS per class: ok')
    # 2. retrain_count = b56_run.retrain
    ns, d, V = 64, 300, 16
    Fk = rng.standard_normal((ns, d)).astype(np.float32)
    Cv = np.where(rng.random((V, d)) < 0.5, -1.0, 1.0).astype(np.float32)
    vals = rng.integers(0, V, ns)
    order = rng.permutation(ns)
    a = E56.retrain(Fk, Cv, vals, order, d, ns, [1, 3], 0.1)
    b, _ = retrain_count(Fk, Cv, vals, order, d, ns, [1, 3], 0.1)
    assert all(np.array_equal(a[e], b[e]) for e in (1, 3))
    print('retrain_count = b56_run.retrain: ok')
    # 3. on TRAINING-split stores: FeatPR codes = Feat53's; p = n rotation features are orthonormal maps; VSAD with
    #    the store's features; NLMSF via b62's engine within budget; VSAPR within budget
    for enc in ('minilm', 'qwen8'):
        S = LS.setup(enc)
        n = n_of(enc)
        for V in (2, 16):
            f53 = R53.Feat53(enc, 'whiten', 'lin', n, 'cpu')
            f53.V = V
            U, mu = LS.basis(enc, 'whiten')
            fpr = FeatPR(enc, 'whiten', 'lin', n, 64, U, mu)
            fpr.V = V
            assert np.array_equal(f53.block(0)[1].numpy(), fpr.codes(0).numpy()), 'codes differ'
            Rm = fpr.rot_cols(0, 0, 200)
            for k in range(0, 200, 64):
                blk = Rm[:, k:k + 64]
                assert np.abs(blk.T @ blk - np.eye(blk.shape[1])).max() < 1e-4
        print(enc, 'FeatPR codes = Feat53 codes; rotation blocks orthonormal: ok')
        st = training_store(enc, 256, 16)
        P = S['preps']['whiten']
        t0 = time.time()
        feat = FeatPR(enc, 'whiten', 'sgn', n, 256, *LS.basis(enc, 'whiten'))
        feat.V = 16
        cfgs = [((q, 1.0, b), C.vsa_dim(b * 256, q, ACCT), q, 256) for q in QS for b in BPF]
        pred = R.vsa_grid(P(st['keys']), st['vals'], P(st['queries']), 16, feat, cfgs, R.store_seed(enc, st),
                          R.store_order(st))
        for (q, f, b), d, _, _ in cfgs:
            assert C.vsa_bits(d, q, ACCT) <= b * 256 and C.vsa_bits(d + 1, q, ACCT) > b * 256
        print(enc, 'VSAPR train store acc', {k: round(float((v == st['qgold']).mean()), 3) for k, v in pred.items()},
              f'{time.time() - t0:.1f}s')
    print('selftest passed')


def timing(enc, M, V):
    M, V = int(M), int(V)
    S = LS.setup(enc)
    n = n_of(enc)
    st = training_store(enc, M, V)
    P = S['preps']['whiten']
    for fmap in FMAPS:
        for p in p_grid(n):
            t0 = time.time()
            feat = FeatPR(enc, 'whiten', fmap, n, p, *LS.basis(enc, 'whiten'))
            feat.V = V
            cfgs = [((q, f, b), C.vsa_dim(b * M, q, ACCT), q, max(1, int(f * M))) for q in QS for f in SHARES
                    for b in BPF]
            pred = R.vsa_grid(P(st['keys']), st['vals'], P(st['queries']), V, feat, cfgs, R.store_seed(enc, st),
                              R.store_order(st))
            best = max(float((v == st['qgold']).mean()) for v in pred.values())
            print(f'VSAPR {enc} M{M} V{V} {fmap} p{p}: {time.time() - t0:.1f}s best train acc {best:.3f}', flush=True)
    # VSAD timing on the B53 config of the 16-bit cell (store-shaped training store)
    prep, fmap, q, f = b53_cfg(enc)[M, V, 16]
    t0 = time.time()
    Yk, Yq = S['preps'][prep](st['keys']), S['preps'][prep](st['queries'])
    dmax = C.vsa_dim(16 * M, 1, ACCT)
    Fk, Fq, Cv, feat = store_feats_d(enc, 'R', prep, fmap, V, Yk, Yq, dmax, n)
    X = Fk[R.store_order(st)].astype(np.float64) * Cv[0][None, :]
    t1 = time.time()
    sn, cnt = delta_write(X, np.asarray(st['vals'])[R.store_order(st)], V, dmax, 'nlms', 0.25, [1, 3, 10])
    print(f'VSAD feats {t1 - t0:.1f}s, one 10-pass write at d={dmax}: {time.time() - t1:.1f}s', flush=True)
    t1 = time.time()
    retrain_count(Fk, Cv, st['vals'], R.store_order(st), dmax, M, RT_E, 0.1)
    print(f'RT 30-epoch retrain at d={dmax}: {time.time() - t1:.1f}s', flush=True)


def main():
    a = sys.argv[1:]
    if a[0] == 'selftest':
        selftest()
    elif a[0] == 'timing':
        timing(*a[1:4])
    elif a[0] == 'plan':
        encs = a[a.index('--encs') + 1].split(',') if '--encs' in a else PANEL
        arms = a[a.index('--arms') + 1].split(',') if '--arms' in a else ['NLMSF', 'VSAPR', 'VSAD']
        make_plan(a[1], encs, arms)
    elif a[0] == 'runplan':
        runplan(a[1], int(a[a.index('--workers') + 1]) if '--workers' in a else 1)
    elif a[0] == 'choose':
        choose(a[1])
    else:
        raise SystemExit(__doc__)


if __name__ == '__main__':
    main()
