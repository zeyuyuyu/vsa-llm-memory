"""B72 (prereg/B72_format_side_info_prereg.md): the write rule in the store's own format, symmetric side-information
accounting, store-level inference with few clusters, and nonlinear random features for least squares.  R1 = V 2 / 16 x
4, 8, 16 bits per fact x M 256 / 1,024; ideal accounting; B49's codes, splits and stores.

Arms run here:
  OPT   (A) the store's EXACT format (features phi, value codes c, d by B32's size rule at the format's q, quantizer,
        per-block scales, matched-filter read) with the Hebbian bundle S = sum phi(k) * c(v) replaced by the S that
        minimises the ridge-regularised squared error of the matched-filter read scores
        score_i(v) = sum_j S_j phi_j(k_i) c_j(v) against centred one-hot targets e_{v_i} - 1/V over the written facts:
            min_S  sum_i sum_v (score_i(v) - t_i(v))^2 + lam ||S||^2.
        With x = phi * r (r = the codes' row-0 signs) and the coordinates in residue classes c = j mod 16 (V = 16; one
        class at V = 2), score_i = H16 m_i with m_ic = <x_ic, S_c>; H16 H16 = 16 I, so the problem splits exactly into
        one ridge regression per class with targets H16[v_i, c] (c >= 1; class 0's target is 0, so S_0 = 0) or +-1
        (V = 2):  S_c = X_c^T (X_c X_c^T + lam I)^-1 y_c  (closed form, float64, dual / eigendecomposition), one lam for
        every class: lam = alpha x mean_c mean_i ||x_ic||^2.  As alpha -> inf, S -> X^T y / lam = the Hebbian bundle on
        classes 1..15 (the store itself, up to the constant class 0).  Target scaling is immaterial (S is linear in
        the targets and the quantizer is scale-equivariant).  S quantized ONCE by the store's quantizer
        (b56_run.quant_np: b32_common.quantize per block of 8,192, seeds ('B72Q', ...)), read by b56_run.mf_read.
        Kinds: R = the final design's format (B53's chosen prep, fmap, q of the cell; b71_run.store_feats_d kind R);
        S = the same random features at VSAS's chosen q (B56's family of the cell); P = VSAPc's format (VSAPc's chosen
        prep, pfmap, q; d <= n; kind P); Q (secondary) = kind R with q re-chosen in {1, 2, 4} (bits per number freed).  Grid = share f (1, 1/2, 1/4, 1/8: the store's own share grid; facts written
        = the store order's first floor(f M)) x alpha (1e-4, 1e-2, 0.1, 1, 10, 100, 1e4) = 28, tie order f > alpha.
  LAMP  (D) least squares with sign (sgn) and product-of-three-signs (sgn3) random features: b66_lsfc.run_job
        UNCHANGED (b60_run.run_lamc on b32_run.Feat's sign maps, LAMc's scales, targets, ridge grid and q: 72 per
        (prep, fmap)), (prep, fmap) in {center, whiten} x {sgn, sgn3}, every R1 cell.
  B*    (B) symmetric side information: every training-split-derived component charged at 8 bits per number (seeds 64
        bits) amortised over K stores and deducted from the memory's budget before its size rule:
        BV (STORE = B53's VSA / VSAR = B49's first design; b32_run.vsa_grid with d = B32's rule at B - charge / K),
        BL (LAMC = b60_run.run_lamc / NLMSC = b62_run.run_online, with b60_run.p_of replaced by the charged rule),
        BS (SDMT = b64_run.run_sdmt with the per-location charge in its N rule).

python3 b72_run.py selftest
python3 b72_run.py plan dev|test [--encs ..] [--arms OPT,LAMP,BV,BL,BS] [--ks 1e3,1e4,1e5]
python3 b72_run.py runplan PLAN --workers N
python3 b72_run.py choose ENC                      # -> results/b72_choice_{enc}.json
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
import b64_run as S64  # noqa: E402
import b66_lsfc as LS  # noqa: E402
import b71_run as E71  # noqa: E402

OUT = os.environ.get('B72_OUT', os.path.join(ROOT, 'b72_out'))
RES = os.environ.get('B72_RES', os.path.join(ROOT, 'results'))
INRES = os.path.join(ROOT, 'results')
PANEL = L60.PANEL
LLM, SENT = PANEL[:4], PANEL[4:]
M_GRID, VN, BPF = [256, 1024], [2, 16], [4, 8, 16]
ACCT = 'ideal'
DEV_REPS, TEST_REPS = L60.DEV_REPS, L60.TEST_REPS
CH = C.CH
H16 = R53.H16.astype(np.float64)
QS = [1, 2, 4]
SHARES = [1.0, 0.5, 0.25, 0.125]
OPT_ALPHAS = [1e-4, 1e-2, 0.1, 1.0, 10.0, 100.0, 1e4]
OPT_GRID = [(f, a) for f in SHARES for a in OPT_ALPHAS]  # 28, tie order
LAMP_FEATS = [(p, f) for p in ('center', 'whiten') for f in ('sgn', 'sgn3')]
KS = [1e3, 1e4, 1e5]
FMAPS = ['lin', 'sgn', 'sgn3']

# LAMP jobs: b66_lsfc's engine with its outputs in b72_out/lamp and ideal accounting only
LS.OUT = os.path.join(OUT, 'lamp')
LS.ACCTS = ['ideal']


def odir(*a):
    d = os.path.join(OUT, *a)
    os.makedirs(d, exist_ok=True)
    return d


_N = {}


def n_of(enc):
    if enc not in _N:
        _N[enc] = E71.n_of(enc)
    return _N[enc]


# ----------------------------------------------------------------------------------------------- cell formats

def formats(enc):
    """{(M, V, b): {'R': (prep, fmap, q), 'S': (prep, fmap, q_VSAS), 'P': (prep, pfmap, q), 'src': VSAS | VSAPC}}."""
    b53 = json.load(open(f'{INRES}/b53_choice_{enc}.json'))['choice']
    ch56 = json.load(open(f'{INRES}/b56_choice_{enc}.json'))['choice']
    chpc = json.load(open(f'{INRES}/b60_choicepc_{enc}.json'))['choice']
    out = {}
    for M in M_GRID:
        for V in VN:
            for b in BPF:
                cell = f'{M}|{V}|{b}'
                prep, fmap, q, _f = b53[f'{cell}|ideal|VSA']['cfg']
                s = ch56[f'{cell}|ideal|STAR']
                qs = int(ch56[f'{cell}|ideal|{s["fam"]}']['cfg'][0])
                pp, pf, pq, _pf = chpc[f'{cell}|ideal|VSA']['cfg']
                pc = chpc[f'{cell}|ideal|PSTAR']
                src = 'VSAPC' if pc['dev'][pc['fam']] > s['dev'][s['fam']] + 0.001 else 'VSAS'
                out[M, V, b] = {'R': (prep, fmap, int(q)), 'S': (prep, fmap, qs), 'P': (pp, pf, int(pq)), 'src': src}
    return out


# ----------------------------------------------------------------------------------------------- OPT (A)

def opt_solve(X, yv, V, d, nsa):
    """Ridge solution of the matched-filter read scores against centred one-hot targets, per residue class.
    X: M x dmax float64 (x = phi(k) * r in store order), yv: values in store order, nsa: {ns: set(alpha)}.
    Returns {(ns, alpha): S (d,) float64}."""
    ncl = 16 if V == 16 else 1
    out = {}
    nsmax = max(nsa)
    Kc, Fc = {}, {}
    for c in range(ncl):
        if V == 16 and c == 0:
            continue
        F = np.ascontiguousarray(X[:nsmax, c:d:ncl]) if ncl > 1 else np.ascontiguousarray(X[:nsmax, :d])
        Fc[c] = F
        Kc[c] = F @ F.T
    for ns in sorted(nsa):
        T = H16[yv[:ns]] if V == 16 else np.where(yv[:ns] == 0, 1.0, -1.0)[:, None]
        tr = float((X[:ns, :d] ** 2).sum()) / ns / ncl  # mean over the ncl classes of mean_i ||x_ic||^2
        sols = {a: np.zeros(d) for a in nsa[ns]}
        for c in Kc:
            w, U = np.linalg.eigh(Kc[c][:ns, :ns])
            w = np.maximum(w, 0.0)
            Uy = U.T @ T[:, c if V == 16 else 0]
            cols = slice(c, d, ncl) if ncl > 1 else slice(0, d)
            for a in nsa[ns]:
                A = U @ (Uy / (w + a * tr))
                sols[a][cols] = Fc[c][:ns].T @ A
        for a in nsa[ns]:
            out[ns, a] = sols[a]
    return out


def run_opt(spec):
    enc, kind, split, M, V, rep, prep, fmap = (spec[k] for k in ('enc', 'kind', 'split', 'M', 'V', 'rep', 'prep',
                                                                 'fmap'))
    S = LS.setup(enc)
    n = n_of(enc)
    st = C.make_store(S['pool'], split, M, V, rep)
    P = S['preps'][prep]
    order = R.store_order(st)
    sseed = R.store_seed(enc, st)
    vals = np.asarray(st['vals'])
    if spec.get('cfgs') is None:
        cells = [(g, b, q) for (b, q) in spec['bq'] for g in OPT_GRID]
    else:
        cells = [(tuple(g), b, q) for g, b, q in spec['cfgs']]
    dk = 'P' if kind == 'P' else 'R'
    ds = {(b, q): E71.d_of(dk, b, M, q, n) for (_, b, q) in cells}
    dmax = max(ds.values())
    Fk, Fq, Cv, _feat = E71.store_feats_d(enc, dk, prep, fmap, V, P(st['keys']), P(st['queries']), dmax, n)
    r = Cv[0].astype(np.float64)
    X = Fk[order].astype(np.float64) * r[None, :]
    yv = vals[order]
    g = st['qgold']
    need = {}
    for (f, a), b, q in cells:
        need.setdefault(ds[b, q], {}).setdefault(max(1, int(math.floor(f * M))), set()).add(a)
    sol = {d: opt_solve(X, yv, V, d, nsa) for d, nsa in need.items()}
    rows = []
    for (f, a), b, q in cells:
        d = ds[b, q]
        ns = max(1, int(math.floor(f * M)))
        Sf = sol[d][ns, a]
        Sq = E56.quant_np(Sf, q, ('B72Q', sseed, kind, d, q, ns, a))
        pr = E56.mf_read(Fq, Cv, Sq)
        row = {'cfg': [f, a], 'bpf': b, 'q': q, 'd': d, 'bits': C.vsa_bits(d, q, ACCT), 'acc': float((pr == g).mean())}
        if split == 'test':
            row['pred'] = np.asarray(pr).astype(int).tolist()
        else:
            row['acc_float'] = float((E71.mf_float(Fq, Cv, Sf) == g).mean())
        rows.append(row)
    return {'qgold': g.tolist(), 'rows': rows, 'n': n}


# ----------------------------------------------------------------------------------------------- side information (B)

def prep_entries(n, prep):
    """Mean (both preps) + the symmetric whitening (ZCA) matrix n (n + 1) / 2 (whiten)."""
    return n + (n * (n + 1) // 2 if prep == 'whiten' else 0)


def charge_fixed(mem, n, prep, fmap=None, feat=None):
    """Side bits independent of the memory's size (8 bits per number, 64 per seed)."""
    bits = 8 * prep_entries(n, prep)
    if mem in ('STORE', 'VSAR'):
        bits += 64 * ((1 if fmap in ('lin', 'sgn') else 3) + 1)  # projection seed(s) + value-code seed
    elif mem in ('LAMC', 'NLMSC'):
        bits += 64 if feat == 'rp' else 0  # projection seed; the principal basis is per feature (below)
    return float(bits)


def charge_unit(mem, n, prep, feat=None):
    """Side bits per unit of the memory's size: principal basis n per feature (centre-prepped principal features; with
    whitening the basis is derivable from the whitening matrix, B71's minimal rule, for every memory); SDM hard
    locations n per location."""
    if mem in ('LAMC', 'NLMSC') and feat == 'pca' and prep == 'center':
        return 8.0 * n
    if mem == 'SDMT':
        return 8.0 * n
    return 0.0


_P_OF = L60.p_of


def charged_p_of(n, mem, prep, K):
    def p_of(Bud, c, q, acct, scale, pmax):
        feat = 'pca' if pmax == n else 'rp'
        B2 = Bud - charge_fixed(mem, n, prep, feat=feat) / K
        u = charge_unit(mem, n, prep, feat=feat) / K
        if B2 <= 0:
            return 0
        per = L60.per_num(q, acct)
        p = min(_P_OF(B2, c, q, acct, scale, pmax), int(B2 // (c * per + u)) + 2)
        tot = lambda x: L60.bits(x, c, q, acct, scale) + u * x  # noqa: E731
        while p > 0 and tot(p) > B2:
            p -= 1
        while p < pmax and tot(p + 1) <= B2:
            p += 1
        return max(0, p)
    return p_of


def run_bv(spec):
    """STORE (B53's VSA: Feat53 codes) or VSAR (B49's first design: b32_run.Feat random codes) at charged budgets."""
    import torch
    torch.set_num_threads(1)
    mem, enc, K, split, M, V, rep, prep, fmap = (spec[k] for k in ('mem', 'enc', 'K', 'split', 'M', 'V', 'rep',
                                                                   'prep', 'fmap'))
    S = LS.setup(enc)
    n = n_of(enc)
    st = C.make_store(S['pool'], split, M, V, rep)
    P = S['preps'][prep]
    if mem == 'STORE':
        feat = R53.Feat53(enc, prep, fmap, n, 'cpu')
        feat.V = V
    else:
        feat = R.Feat(enc, prep, fmap, n, 'cpu')
    fx = charge_fixed(mem, n, prep, fmap=fmap) / K
    cells = ([((q, f), b) for q in QS for f in SHARES for b in BPF] if spec.get('cfgs') is None
             else [(tuple(c), b) for c, b in spec['cfgs']])
    cfgs, rows = [], []
    g = st['qgold']
    for (q, f), b in cells:
        B2 = b * M - fx
        d = C.vsa_dim(B2, q, ACCT) if B2 > 0 else 0
        if d > 0:
            cfgs.append(((q, f, b), d, q, max(1, int(math.floor(f * M)))))
        else:
            rows.append({'cfg': [prep, fmap, q, f], 'bpf': b, 'd': 0, 'feasible': False})
    pred = R.vsa_grid(P(st['keys']), st['vals'], P(st['queries']), V, feat, cfgs, R.store_seed(enc, st),
                      R.store_order(st)) if cfgs else {}
    for (q, f, b), d, _, _ in cfgs:
        pr = pred[q, f, b]
        row = {'cfg': [prep, fmap, q, f], 'bpf': b, 'd': d, 'feasible': True, 'bits': C.vsa_bits(d, q, ACCT),
               'charge': fx, 'acc': float((pr == g).mean())}
        if split == 'test':
            row['pred'] = np.asarray(pr).astype(int).tolist()
        rows.append(row)
    return {'qgold': g.tolist(), 'rows': rows}


def run_bl(spec):
    """LAMC (b60_run.run_lamc) or NLMSC (b62_run.run_online) with the charged size rule, their own job seeds."""
    mem, enc, K, split, M, V, rep, prep = (spec[k] for k in ('mem', 'enc', 'K', 'split', 'M', 'V', 'rep', 'prep'))
    pool, preps, fcs = L60.setup(enc)
    n = int(pool.X.shape[1])
    st = C.make_store(pool, split, M, V, rep)
    P = preps[prep]
    Yk, Yq = P(st['keys']), P(st['queries'])
    g = st['qgold']
    L60.p_of = charged_p_of(n, mem, prep, K)
    try:
        if mem == 'LAMC':
            seed = C.seed_of('b60', L60.tag(enc, 0), split, M, V, rep)
            if spec.get('cfgs') is None:
                want = [(c, b) for c in L60.grid(V) if c[2] == prep for b in BPF]
            else:
                want = [(tuple(c), b) for c, b in spec['cfgs']]
            o = {}
            byc = {}
            for c, b in want:
                byc.setdefault((c[0], c[1], c[3], c[4], c[5]), set()).add(b)
            if spec.get('cfgs') is None:
                o = L60.run_lamc(st, Yk, Yq, fcs[prep], sorted(byc), BPF, [ACCT], seed)
            else:
                for c5, bs in byc.items():
                    o.update(L60.run_lamc(st, Yk, Yq, fcs[prep], [c5], sorted(bs), [ACCT], seed))
            res = {}
            for c, b in want:
                c5 = (c[0], c[1], c[3], c[4], c[5])
                res[c, b] = o[c5, b, ACCT]
        else:
            seed = C.seed_of('b62', enc, split, M, V, rep)
            if spec.get('cfgs') is None:
                want = [(c, b) for c in O62.grid('NLMSC') if c[2] == prep for b in BPF]
            else:
                want = [(tuple(c), b) for c, b in spec['cfgs']]
            o = O62.run_online(st, Yk, Yq, fcs[prep], enc, prep, [('NLMSC', c, b) for c, b in want], seed)
            res = {(c, b): o['NLMSC', c, b] for c, b in want}
    finally:
        L60.p_of = _P_OF
    rows = []
    for (c, b), (pred, bt, p) in res.items():
        feat = c[3]
        chg = charge_fixed(mem, n, prep, feat=feat) / K + charge_unit(mem, n, prep, feat=feat) / K * p
        row = {'cfg': list(c), 'bpf': b, 'p': int(p), 'feasible': p > 0, 'bits': bt, 'charge': chg,
               'acc': float((pred == g).mean())}
        if split == 'test':
            row['pred'] = np.asarray(pred).astype(int).tolist()
        rows.append(row)
    return {'qgold': g.tolist(), 'rows': rows, 'n': n}


def run_sdmt_charged(st, Yk, Yq, Ls, cfgs, bpfs, accts, order, fixed, unit):
    """b64_run.run_sdmt (verbatim) with one change: N = the largest grid point whose counter bits + the default value's
    bits + the side charge (fixed + unit x N) fit the budget."""
    K = S64.K
    M, V = st['M'], st['V']
    vals = st['vals']
    default = K.default_value(vals, V)
    db = K.def_bits(V)
    vb = S64.vbip(V)
    out = {}
    for loc in S64.LOCS:
        cl = [c for c in cfgs if c[0] == loc]
        if not cl:
            continue
        L = Ls[loc]
        Ng = S64.ngrid(L.shape[0])
        amax = max(c[1] for c in cl)
        tk = S64.TopA(Yk, L, amax)
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
                Cn = S64.counters(tk.top(a)[rows], vals[rows], V, N, ctr)
                for q in qof[a, ctr, s]:
                    Qc = S64.qtrans(Cn, ctr, q)
                    for acct in accts:
                        cost[(a, ctr, q, s, acct), N] = S64.ctr_bits(Qc, ctr, q, acct)
        need = {}
        for c in cl:
            a, ctr, q, s = c[1], c[2], c[3], c[4]
            for b in bpfs:
                Bav = b * M - db - fixed
                for acct in accts:
                    fit = [N for N in Ng if cost[(a, ctr, q, s, acct), N] + unit * N <= Bav]
                    N = max(fit) if fit else 0
                    if N == 0:
                        out[c, b, acct] = (np.full(Yq.shape[0], default), 0.0, 0)
                    else:
                        need.setdefault(N, []).append((c, b, acct))
        tq = S64.TopA(Yq, L, amax)
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
                    cc[key] = S64.counters(Ia[rows], vals[rows], V, N, ctr)
                Qc = S64.qtrans(cc[key], ctr, q)
                Z = np.vstack([Qc, np.zeros((1, Qc.shape[1]), Qc.dtype)]).astype(np.float64)
                Iq = tq.top(a)
                Iq = np.where(Iq < 0, N, Iq)
                Rr = Z[Iq].sum(1)
                score = Rr if ctr == 'onehot' else Rr @ vb.T.astype(np.float64)
                bt = S64.ctr_bits(Qc, ctr, q, acct) + db
                out[c, b, acct] = (K.decide(score, default), float(bt), int(N))
            cc = {k: v for k, v in cc.items() if k[3] == N}
    return out


def run_bs(spec):
    enc, K, split, M, V, rep, prep = (spec[k] for k in ('enc', 'K', 'split', 'M', 'V', 'rep', 'prep'))
    pool, preps, Ls, _ = S64.setup(enc)
    n = int(pool.X.shape[1])
    st = C.make_store(pool, split, M, V, rep)
    order = S64.store_order(st)
    g = st['qgold']
    P = preps[prep]
    if spec.get('cfgs') is None:
        want = [(c, b) for c in S64.GRID if c[0] == prep for b in BPF]
    else:
        want = [(tuple(c), b) for c, b in spec['cfgs']]
    byc = {}
    for c, b in want:
        byc.setdefault(c, set()).add(b)
    fx, un = charge_fixed('SDMT', n, prep) / K, charge_unit('SDMT', n, prep) / K
    o = {}
    if spec.get('cfgs') is None:
        o = run_sdmt_charged(st, P(st['keys']), P(st['queries']), Ls[prep], [c[1:] for c in byc], BPF, [ACCT], order,
                             fx, un)
    else:
        for c, bs in byc.items():
            o.update(run_sdmt_charged(st, P(st['keys']), P(st['queries']), Ls[prep], [c[1:]], sorted(bs), [ACCT],
                                      order, fx, un))
    rows = []
    for c, b in want:
        pred, bt, N = o[c[1:], b, ACCT]
        row = {'cfg': list(c), 'bpf': b, 'N': int(N), 'feasible': N > 0, 'bits': bt, 'charge': fx + un * N,
               'acc': float((pred == g).mean())}
        if split == 'test':
            row['pred'] = np.asarray(pred).astype(int).tolist()
        rows.append(row)
    return {'qgold': g.tolist(), 'rows': rows, 'n': n}


# ----------------------------------------------------------------------------------------------- jobs

def kstr(K):
    return f'{K:.0e}'.replace('+0', '').replace('+', '')


def job_name(s):
    if s['arm'] == 'LAMP':
        return LS.job_name(s)
    if s['arm'] == 'OPT':
        return f'{s["enc"]}_OPT{s["kind"]}_{s["split"]}_M{s["M"]}_V{s["V"]}_r{s["rep"]}_{s["prep"]}_{s["fmap"]}'
    base = f'{s["enc"]}_{s["mem"]}_K{kstr(s["K"])}_{s["split"]}_M{s["M"]}_V{s["V"]}_r{s["rep"]}_{s["prep"]}'
    return base + (f'_{s["fmap"]}' if s['arm'] == 'BV' else '')


RUN = {'OPT': run_opt, 'BV': run_bv, 'BL': run_bl, 'BS': run_bs}


def run_job(spec):
    if spec['arm'] == 'LAMP':
        return LS.run_job(spec)
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


def job_file(s):
    if s['arm'] == 'LAMP':
        return os.path.join(LS.OUT, 'jobs', LS.job_name(s) + '.json.gz')
    return os.path.join(OUT, 'jobs', job_name(s) + '.json.gz')


def _run(spec):
    try:
        return run_job(spec)
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        return f'FAILED {job_name(spec)} {e}'


def runplan(planfile, workers):
    specs = [json.loads(l) for l in open(planfile)]
    specs = [s for s in specs if not os.path.exists(job_file(s))]
    pri = {'BS': 0, 'BL': 1, 'LAMP': 2, 'BV': 3, 'OPT': 4}
    specs.sort(key=lambda s: (pri[s['arm']], -s['M'], PANEL.index(s['enc'])))
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


def whiten_fits(n, K, M):
    """Some R1 budget of this M can hold the whitening matrix + mean at this K."""
    return 8 * prep_entries(n, 'whiten') / K < max(BPF) * M


def make_plan(split, encs, arms, ks, fn=None):
    odir()
    fn = fn or os.path.join(OUT, f'plan_{split}.jsonl')
    nj = 0
    with open(fn, 'w') as fo:
        def put(s):
            nonlocal nj
            fo.write(json.dumps(s) + '\n')
            nj += 1
        for enc in encs:
            ch = choice(enc) if split == 'test' else None
            n = n_of(enc)
            fm = formats(enc)
            reps = DEV_REPS if split == 'dev' else TEST_REPS
            for M in M_GRID:
                for V in VN:
                    if 'OPT' in arms:
                        grp = {}
                        for b in BPF:
                            f = fm[M, V, b]
                            for kd in ('R', 'S', 'P', 'Q'):
                                prep, fmap, q = f['R' if kd == 'Q' else kd]
                                gk = ('P' if kd == 'P' else 'R', prep, fmap)
                                if split == 'dev':
                                    for qq in (QS if kd == 'Q' else [q]):
                                        if (b, qq) not in grp.setdefault(gk, []):
                                            grp[gk].append((b, qq))
                                else:
                                    cq = ch[f'{M}|{V}|{b}|OPT{kd}']
                                    c = [cq['cfg'], b, cq['fmt'][2]]
                                    if c not in grp.setdefault(gk, []):
                                        grp[gk].append(c)
                        for (kind, prep, fmap), lst in sorted(grp.items()):
                            for rep in range(reps[M]):
                                s = {'arm': 'OPT', 'enc': enc, 'kind': kind, 'split': split, 'M': M, 'V': V,
                                     'rep': rep, 'prep': prep, 'fmap': fmap}
                                if split == 'dev':
                                    s.update({'bq': [list(x) for x in lst], 'cfgs': None})
                                else:
                                    s['cfgs'] = lst
                                put(s)
                    if 'LAMP' in arms:
                        for (prep, fmap) in LAMP_FEATS:
                            if split == 'dev':
                                ba, cfgs = [[b, 'ideal'] for b in BPF], None
                            else:
                                cfgs = [[ch[f'{M}|{V}|{b}|LAMP']['cfg4'], b, 'ideal'] for b in BPF
                                        if ch[f'{M}|{V}|{b}|LAMP']['src'] == 'LAMP'
                                        and tuple(ch[f'{M}|{V}|{b}|LAMP']['feat']) == (prep, fmap)]
                                if not cfgs:
                                    continue
                                ba = [[c[1], 'ideal'] for c in cfgs]
                            for rep in range(reps[M]):
                                put({'arm': 'LAMP', 'enc': enc, 'kind': 'R', 'split': split, 'M': M, 'V': V,
                                     'rep': rep, 'prep': prep, 'fmap': fmap, 'ba': ba, 'cfgs': cfgs})
                    if enc not in LLM:
                        continue
                    for K in ks:
                        for mem, arm in (('STORE', 'BV'), ('VSAR', 'BV'), ('LAMC', 'BL'), ('NLMSC', 'BL'),
                                         ('SDMT', 'BS')):
                            if arm not in arms:
                                continue
                            for prep in ('center', 'whiten'):
                                if prep == 'whiten' and not whiten_fits(n, K, M):
                                    continue
                                for fmap in (FMAPS if arm == 'BV' else [None]):
                                    if split == 'dev':
                                        cfgs = None
                                    else:
                                        cfgs = []
                                        pi = 2 if arm == 'BL' else 0  # position of prep in the configuration
                                        for b in BPF:
                                            c = ch[f'{M}|{V}|{b}|{mem}|K{kstr(K)}']
                                            if c['cfg'] is None or c['cfg'][pi] != prep or (
                                                    arm == 'BV' and c['cfg'][1] != fmap):
                                                continue
                                            cfgs.append([c['cfg'][2:] if arm == 'BV' else c['cfg'], b])
                                        if not cfgs:
                                            continue
                                    for rep in range(reps[M]):
                                        s = {'arm': arm, 'mem': mem, 'enc': enc, 'K': K, 'split': split, 'M': M,
                                             'V': V, 'rep': rep, 'prep': prep, 'cfgs': cfgs}
                                        if arm == 'BV':
                                            s['fmap'] = fmap
                                        put(s)
    print(nj, 'jobs ->', fn)


# ----------------------------------------------------------------------------------------------- dev choice

def choice(enc):
    return json.load(open(f'{RES}/b72_choice_{enc}.json'))['choice']


def pick(lst, key):
    """B60's / B32's / B36's rule: highest mean dev accuracy; within 0.1 points tied, first in the given order."""
    top = max(a for a, _ in lst)
    tied = sorted([c for a, c in lst if a >= top - 0.001], key=key)
    return tied[0], [a for a, c in lst if c == tied[0]][0], len(tied)


def load_dev(enc):
    """{key: {rep: row}} from B72's dev jobs (OPT, B*) and the LAMP dev jobs."""
    acc = {}
    for jd, lamp in ((os.path.join(OUT, 'jobs'), False), (os.path.join(LS.OUT, 'jobs'), True)):
        if not os.path.isdir(jd):
            continue
        for fn in sorted(os.listdir(jd)):
            if not fn.startswith(f'{enc}_') or '_dev_' not in fn:
                continue
            r = json.load(gzip.open(os.path.join(jd, fn), 'rt'))
            s = r['spec']
            if s['enc'] != enc:
                continue
            for x in r['rows']:
                if lamp:
                    k = ('LAMP', s['M'], s['V'], x['bpf'], (s['prep'], s['fmap']) + tuple(x['cfg']))
                elif s['arm'] == 'OPT':
                    k = ('OPT' + s['kind'], s['M'], s['V'], x['bpf'], (s['prep'], s['fmap'], x['q']) + tuple(x['cfg']))
                else:
                    k = (s['mem'], s['K'], s['M'], s['V'], x['bpf'], tuple(x['cfg']))
                assert s['rep'] not in acc.get(k, {}), ('duplicate dev row', k, s['rep'])
                acc.setdefault(k, {})[s['rep']] = x
    return acc


def dmean(acc, k, need):
    d = acc.get(k)
    if d is None:
        raise SystemExit(f'missing dev rows {k}')
    if len(d) != need:
        raise SystemExit(f'incomplete dev rows {k}: {len(d)}')
    return float(np.mean([d[i]['acc'] for i in sorted(d)]))


def choose(enc, ks=KS):
    acc = load_dev(enc)
    fm = formats(enc)
    ch60 = json.load(open(f'{INRES}/b60_choice_{enc}.json'))['choice']
    n = n_of(enc)
    out = {}
    have_lamp = any(k[0] == 'LAMP' for k in acc)
    have_b = {m for k in acc for m in ('STORE', 'VSAR', 'LAMC', 'NLMSC', 'SDMT') if k[0] == m} if ks else set()
    for M in M_GRID:
        need = DEV_REPS[M]
        for V in VN:
            for b in BPF:
                cell = f'{M}|{V}|{b}'
                f = fm[M, V, b]
                # (A) OPT per kind: (f, alpha) at the format's (prep, fmap, q)
                for kd in ('R', 'S', 'P'):
                    prep, fmap, q = f[kd]
                    ak = 'OPTP' if kd == 'P' else 'OPTR'
                    lst = [(dmean(acc, (ak, M, V, b, (prep, fmap, q) + g), need), g) for g in OPT_GRID]
                    c, a, nt = pick(lst, OPT_GRID.index)
                    fl = float(np.mean([acc[ak, M, V, b, (prep, fmap, q) + c][i]['acc_float']
                                        for i in range(need)]))
                    out[f'{cell}|OPT{kd}'] = {'cfg': list(c), 'dev_acc': a, 'dev_acc_float': fl, 'n_tied': nt,
                                              'fmt': [prep, fmap, q]}
                out[f'{cell}|OPTX'] = {'src': 'OPTP' if f['src'] == 'VSAPC' else 'OPTS', 'vsapcx_src': f['src']}
                # OPTQ (secondary): the store's features and codes with q re-chosen (q > f > alpha)
                prep, fmap, _q = f['R']
                lst = [(dmean(acc, ('OPTR', M, V, b, (prep, fmap, qq) + g), need), (qq,) + g) for qq in QS
                       for g in OPT_GRID]
                c, a, nt = pick(lst, lambda c: (QS.index(c[0]), OPT_GRID.index(c[1:])))
                out[f'{cell}|OPTQ'] = {'cfg': list(c[1:]), 'dev_acc': a, 'n_tied': nt, 'fmt': [prep, fmap, c[0]]}
                # (D) LAMP: best of the 288 sign-feature configurations; LAMc+ = it iff > LAMc's dev + 0.1 points
                if have_lamp:
                    lst = []
                    for (prep, fmap) in LAMP_FEATS:
                        for g in LS.GRID:
                            lst.append((dmean(acc, ('LAMP', M, V, b, (prep, fmap) + g), need), (prep, fmap) + g))
                    order = lambda c: (LS.SCALES.index(c[2]), LS.TARGETS.index(c[3]), LAMP_FEATS.index(c[:2]),  # noqa: E731
                                       LS.ALPHAS.index(c[4]), LS.QS.index(c[5]))
                    c, a, nt = pick(lst, order)
                    la = ch60[f'{cell}|ideal|LAMC']['dev_acc']
                    out[f'{cell}|LAMP'] = {'src': 'LAMP' if a > la + 0.001 else 'LAMC', 'feat': list(c[:2]),
                                           'cfg4': list(c[2:]), 'dev_acc': a, 'lamc_dev': la, 'n_tied': nt}
                # (B) charged memories
                for K in ks:
                    for mem in ('STORE', 'VSAR', 'LAMC', 'NLMSC', 'SDMT'):
                        if mem not in have_b:
                            continue
                        if mem in ('STORE', 'VSAR'):
                            grid = [(p, fmp, q, fs) for p in ('center', 'whiten') for fmp in FMAPS for q in QS
                                    for fs in SHARES]
                            key = lambda c: (('center', 'whiten').index(c[0]), FMAPS.index(c[1]), QS.index(c[2]),  # noqa: E731
                                             SHARES.index(c[3]))
                        elif mem == 'LAMC':
                            grid = L60.grid(V)
                            key = grid.index
                        elif mem == 'NLMSC':
                            grid = O62.grid('NLMSC')
                            key = grid.index
                        else:
                            grid = S64.GRID
                            key = grid.index
                        pi = 2 if mem in ('LAMC', 'NLMSC') else 0  # position of prep in the memory's configuration
                        lst = []
                        for c in grid:
                            if c[pi] == 'whiten' and not whiten_fits(n, K, M):
                                continue
                            d = acc.get((mem, K, M, V, b, tuple(c)))
                            if d is None:
                                raise SystemExit(f'missing dev rows {mem} {K} {M} {V} {b} {c}')
                            if len(d) != need:
                                raise SystemExit(f'incomplete dev rows {mem} {K} {M} {V} {b} {c}')
                            if not all(d[i]['feasible'] for i in d):
                                continue
                            lst.append((float(np.mean([d[i]['acc'] for i in sorted(d)])), tuple(c)))
                        if not lst:
                            out[f'{cell}|{mem}|K{kstr(K)}'] = {'cfg': None, 'n_feasible': 0}
                            continue
                        c, a, nt = pick(lst, key)
                        out[f'{cell}|{mem}|K{kstr(K)}'] = {'cfg': list(c), 'dev_acc': a, 'n_tied': nt,
                                                           'n_feasible': len(lst)}
    fn = f'{RES}/b72_choice_{enc}.json'
    json.dump({'enc': enc, 'choice': out}, open(fn, 'w'), indent=1, sort_keys=True)
    print(len(out), 'choices ->', fn)


# ----------------------------------------------------------------------------------------------- selftest

def selftest():
    rng = np.random.default_rng(0)
    # 1. opt_solve = the brute-force ridge minimiser of the matched-filter score error against centred one-hot targets
    for V in (2, 16):
        ns, d = 40, 96
        X = rng.standard_normal((ns, d))
        yv = rng.integers(0, V, ns)
        a = 0.1
        sol = opt_solve(X, yv, V, d, {ns: {a}, 20: {a}})
        for nsx in (ns, 20):
            Sx = sol[nsx, a]
            # brute force: S minimises sum_i ||A_i S - t_i||^2 + lam' ||S||^2 with A_i = [x_i * h_v(j)]_v (V x d) and
            # the score scale of the reduced problem (per class: factor 16 at V = 16, 2 at V = 2)
            Hc = H16[:, np.arange(d) % 16] if V == 16 else np.stack([np.ones(d), -np.ones(d)])
            A = np.concatenate([Hc * X[i][None, :] for i in range(nsx)])
            t = np.concatenate([np.eye(V)[yv[i]] - 1.0 / V for i in range(nsx)])
            scale = 16.0 if V == 16 else 2.0
            ncl = 16 if V == 16 else 1
            tr = float((X[:nsx] ** 2).sum()) / nsx / ncl
            lam = a * tr * scale  # the reduced problem's lam, mapped to the full objective
            Sb = np.linalg.solve(A.T @ A + lam * np.eye(d), A.T @ t)
            # the reduced targets are scaled by 1 / 16 (V = 16) or 1 / 2 (V = 2) relative to H16[v, c] / +-1
            k = 16.0 if V == 16 else 2.0
            err = np.abs(Sx / k - Sb).max() / np.abs(Sb).max()
            assert err < 1e-8, (V, nsx, err)
    print('opt_solve = brute-force ridge minimiser of the matched-filter score error: ok')
    # 2. large alpha -> Hebbian bundle (classes 1..15), float argmax identical
    for V in (2, 16):
        ns, d = 50, 160
        X = rng.standard_normal((ns, d))
        yv = rng.integers(0, V, ns)
        Sx = opt_solve(X, yv, V, d, {ns: {1e9}})[ns, 1e9]
        T = H16[yv] if V == 16 else np.where(yv == 0, 1.0, -1.0)[:, None]
        Sh = np.zeros(d)
        for j in range(d):
            Sh[j] = (X[:, j] * T[:, j % 16 if V == 16 else 0]).sum()
        if V == 16:
            Sh[0::16] = 0.0
        cosv = float(Sx @ Sh / np.linalg.norm(Sx) / np.linalg.norm(Sh))
        assert cosv > 1 - 1e-6, cosv
    print('alpha -> inf gives the Hebbian bundle: ok')
    # 3. charged size rules within budget and maximal
    n = 4096
    for K in KS:
        for prep in ('center', 'whiten'):
            for feat in ('rp', 'pca'):
                po = charged_p_of(n, 'LAMC', prep, K)
                for (Bud, c, q, scale) in [(1024, 1, 1, 'col'), (16384, 16, 4, 'flat'), (8192, 15, 2, 'col')]:
                    pmax = n if feat == 'pca' else 65536
                    p = po(Bud, c, q, 'ideal', scale, pmax)
                    tot = lambda x: L60.bits(x, c, q, 'ideal', scale) + (charge_unit('LAMC', n, prep, feat) / K) * x  # noqa: E731
                    fx = charge_fixed('LAMC', n, prep, feat=feat) / K
                    if p > 0:
                        assert tot(p) + fx <= Bud, (K, prep, feat, Bud, p)
                    assert p == pmax or tot(p + 1) + fx > Bud, (K, prep, feat, Bud, p)
    print('charged p rule within budget and maximal: ok')
    # 4. OPT on TRAINING-split stores: within budget, d maximal, beats chance; alpha = 1e4 close to the Hebbian store
    for enc in ('minilm', 'qwen8'):
        S = LS.setup(enc)
        nn = n_of(enc)
        for V in (2, 16):
            st = E71.training_store(enc, 256, V)
            P = S['preps']['whiten']
            t0 = time.time()
            order = R.store_order(st)
            vals = np.asarray(st['vals'])
            d = C.vsa_dim(8 * 256, 1, ACCT)
            Fk, Fq, Cv, feat = E71.store_feats_d(enc, 'R', 'whiten', 'lin', V, P(st['keys']), P(st['queries']), d, nn)
            X = Fk[order].astype(np.float64) * Cv[0].astype(np.float64)[None, :]
            sol = opt_solve(X, vals[order], V, d, {256: set(OPT_ALPHAS)})
            g = st['qgold']
            accs = {a: float((E56.mf_read(Fq, Cv, E56.quant_np(sol[256, a], 1, ('B72Qt', a))) == g).mean())
                    for a in OPT_ALPHAS}
            Sh = E56.bundle_q(feat, Fk, Cv, vals, order, d, 1, 256, R.store_seed(enc, st))
            heb = float((E56.mf_read(Fq, Cv, Sh) == g).mean())
            assert C.vsa_bits(d, 1, ACCT) <= 8 * 256 < C.vsa_bits(d + 1, 1, ACCT)
            print(enc, 'V', V, 'train store: OPT acc by alpha', {a: round(x, 3) for a, x in accs.items()},
                  'Hebbian', round(heb, 3), f'{time.time() - t0:.1f}s', flush=True)
    print('selftest passed')


def main():
    a = sys.argv[1:]
    if a[0] == 'selftest':
        selftest()
    elif a[0] == 'plan':
        encs = a[a.index('--encs') + 1].split(',') if '--encs' in a else PANEL
        arms = a[a.index('--arms') + 1].split(',') if '--arms' in a else ['OPT', 'LAMP', 'BV', 'BL', 'BS']
        ks = [float(x) for x in a[a.index('--ks') + 1].split(',')] if '--ks' in a else KS
        fn = a[a.index('--out') + 1] if '--out' in a else None
        make_plan(a[1], encs, arms, ks, fn)
    elif a[0] == 'runplan':
        runplan(a[1], int(a[a.index('--workers') + 1]) if '--workers' in a else 1)
    elif a[0] == 'choose':
        ks = [] if ('--ks' in a and a[a.index('--ks') + 1] == 'none') else KS  # 'none': (A) / (D) only
        choose(a[1], ks)
    else:
        raise SystemExit(__doc__)


if __name__ == '__main__':
    main()
