"""B67 checks (prereg/B67_theory_vs_measurement_prereg.md, C1 and C2), written separately from b67_theory.py.

C1 (a): theory H re-derived with own code (own dev-matrix second moment, own kernel moments by Monte Carlo, own
        quantizer constant by Monte Carlo, Monte Carlo of the Gaussian race: 20,000 draws per cell on a 200-pair
        subsample) for every cell; theory L re-derived for every cell whose feature dimension is <= 1,500 (others:
        the eigen-decomposition is the main cost; reported as not checked) with the deterministic equivalent solved by
        fixed-point iteration and the race by Monte Carlo; agreement with the main numbers, both on the same
        subsample.
C1 (b): D-a / S-a / T-a by plain formulas, D-b by scipy.integrate.quad, S-b by explicit loops over scipy.stats.binom;
        grid-rounded N* compared.
C2:    every dev row's entity split; the make_store guard raises on 'test'.

python3 b67_check.py a ENC      # -> results/b67_check_a_ENC.json
python3 b67_check.py b          # -> results/b67_check_b.json
python3 b67_check.py c2         # -> results/b67_check_c2.json
"""
import json
import math
import os
import sys

import numpy as np
from scipy import integrate, stats

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import b32_common as C  # noqa: E402
import b67_theory as T  # noqa: E402

RES = T.RES
SUB = 200
ND = 20000


def mc_kappa(q, rng):
    x = rng.standard_normal(2_000_000).astype(np.float32)
    Qx = C.quantize(x, q, rng).astype(np.float64)
    sd = float(x.std())
    a = float((Qx * x).mean()) / sd ** 2
    return float(((Qx - a * x) ** 2).mean() / (a * a * sd ** 2))


def mc_race(mu, s_true, s_comp_list, rng):
    """s_comp_list: list of (std array, count)."""
    acc = []
    for i in range(len(mu)):
        t = mu[i] + s_true[i] * rng.standard_normal(ND)
        ok = np.ones(ND, bool)
        for sw, m in s_comp_list:
            w = sw[i] * rng.standard_normal((ND, m))
            ok &= (t[:, None] > w).all(1)
        acc.append(ok.mean())
    return np.array(acc)


def check_a(enc):
    rng = np.random.default_rng(6701)
    dd = T.DevData(enc)
    th = json.load(open(f'{RES}/b67_a_{enc}.json'))['cells']
    chS = json.load(open(f'{RES}/b53_choice_{enc}.json'))['choice']
    sub = np.sort(rng.choice(len(dd.pair_lab), SUB, replace=False))
    KQmc = {q: mc_kappa(q, rng) for q in (1, 2, 4)}
    out = {'enc': enc, 'kappa_main': T.KQ, 'kappa_mc': KQmc, 'cells': {}}
    for (M, V, b) in T.CELLS:
        key = f'{M}|{V}|{b}'
        prep, fmap, q, f = chS[key + '|ideal|VSA']['cfg']
        d = C.vsa_dim(b * M, q, 'ideal')
        L, A, _ = dd.prepped(prep)
        Y = np.vstack([L, A])
        Lp = L[dd.pair_lab[sub]]
        As = A[sub]
        u = np.einsum('ij,ij->i', As, Lp)
        # own kernel moments: Monte Carlo over the dev rows (k' = random dev rows), not the Gaussian quadrature
        idx = rng.integers(0, len(Y), 4000)
        U = As @ Y[idx].T
        kf = (lambda x: x) if fmap == 'lin' else (lambda x: ((2 / np.pi) * np.arcsin(np.clip(x, -1, 1))) **
                                                (1 if fmap == 'sgn' else 3))
        mu = kf(u)
        m2 = (kf(U) ** 2).mean(1)
        Ms = max(1, int(math.floor(f * M)))
        if V == 2:
            sN2 = (1 + KQmc[q]) * Ms / d
            sd = np.sqrt((Ms - 1) * m2 + sN2)
            P = (mu[:, None] + sd[:, None] * rng.standard_normal((len(mu), ND)) > 0).mean(1)
        else:
            Lr = L[rng.integers(0, len(L), 3000)]
            Lr2 = L[rng.integers(0, len(L), 3000)]
            mbar = float(kf(np.einsum('ij,ij->i', Lr, Lr2)).mean())
            sN2 = (1 + KQmc[q]) * max(Ms + Ms * (Ms - 1) * mbar / V, 1.0) / d
            sig = np.sqrt((Ms - 1) * m2 / V + sN2)
            P = mc_race(mu, sig, [(sig, V - 1)], rng)
        accH = f * float(P.mean()) + (1 - f) / V
        mainH, _ = T.theory_H(_Sub(dd, sub), prep, fmap, q, f, M, V, d)
        rec = {'H_check': accH, 'H_main_sub': mainH, 'H_main_full': th[key]['STORE'], 'H_diff': accH - mainH}
        out['cells'][key] = rec
        print(enc, key, rec, flush=True)
    json.dump(out, open(f'{RES}/b67_check_a_{enc}.json', 'w'), indent=1)


class _Sub:
    """DevData restricted to a pair subsample (the same C2: the full dev matrix)."""

    def __init__(self, dd, sub):
        self.dd, self.sub = dd, sub
        self.pair_lab = dd.pair_lab[sub]

    def prepped(self, prep):
        L, A, C2 = self.dd.prepped(prep)
        return L, A[self.sub], C2


def check_L(enc, pmax_check=1500):
    """theory L: deterministic equivalent by fixed point + Monte Carlo race, cells with p <= pmax_check."""
    rng = np.random.default_rng(6702)
    dd = T.DevData(enc)
    th = json.load(open(f'{RES}/b67_a_{enc}.json'))['cells']
    sub = np.sort(rng.choice(len(dd.pair_lab), SUB, replace=False))
    sd_ = _Sub(dd, sub)
    out = {}
    for (M, V, b) in T.CELLS:
        key = f'{M}|{V}|{b}'
        for arm in ('LSFR', 'LAMC'):
            cfg = th[key][arm + '_cfg']
            p = cfg[-1]
            if p > pmax_check:
                out[f'{key}|{arm}'] = {'checked': False, 'p': p}
                continue
            if arm == 'LSFR':
                scale, target, alpha, q, prep, fmap, p = cfg
                if fmap != 'lin':
                    out[f'{key}|{arm}'] = {'checked': False, 'p': p, 'why': fmap}
                    continue
                G, _ = T.store_G(enc, prep, fmap, dd.n, p)
            else:
                scale, target, prep, feat, alpha, q, p = cfg
                G = T.lam_G(enc, prep, dd.n, p) if feat == 'rp' else T.pca_U0(dd)[:, :p]
            L, A, C2 = sd_.prepped(prep)
            # own spectrum: eigen of the p x p feature second moment computed from the dev rows directly
            Y = np.vstack(dd.prepped(prep)[:2])
            Z = Y @ G
            S = Z.T @ Z / len(Z)
            lam, W = np.linalg.eigh(S)
            lam = np.maximum(lam, 0)
            fK = L[sd_.pair_lab] @ G @ W
            fD = (A - L[sd_.pair_lab]) @ G @ W
            trS = lam.sum()
            lam_e = alpha * trS / M
            k = lam_e + 1e-12  # fixed point kappa = lam_e / (1 - df1 / M), geometric damping; restart above if df1 >= M
            for _ in range(5000):
                df1 = (lam / (lam + k)).sum()
                k_new = lam_e / (1 - df1 / M) if df1 < M else 2 * k
                k = math.sqrt(k * k_new)
            df2 = (lam ** 2 / (lam + k) ** 2).sum()
            kp = 1 / (1 - df2 / M)
            A1KK = (fK ** 2 / (lam + k)).sum(1)
            A1DK = (fK * fD / (lam + k)).sum(1)
            A2DD = (fD ** 2 * lam / (lam + k) ** 2).sum(1)
            cc = lam_e * M / k
            h = A1KK / (cc + A1KK)
            mu = h + A1DK / (cc + A1KK)
            r = (1 - h) ** 2 * (kp - 1)
            g = kp / M * A2DD
            vW = (1 / V) * (1 - 1 / V) * (kp / M) * (lam / (lam + k) ** 2).sum() / p
            fq2 = ((A @ G) ** 2).sum(1)
            bq = T.KQ[q] * vW * fq2
            P1 = None
            for variant in ('L', "L'"):
                if variant == "L'":  # exploratory: residual v = f(q) - mu f(k) through the leave-one-out inverse
                    vv = fK * (1 - mu)[:, None] + fD
                    r = kp / M * (vv ** 2 * lam / (lam + k) ** 2).sum(1)
                    g = 0 * r
                    P1 = P
                if V == 2:
                    var = r + g + (4 if target == 'V-1' else 2) * bq
                    P = (mu[:, None] + np.sqrt(var)[:, None] * rng.standard_normal((len(mu), ND)) > 0).mean(1)
                else:
                    cx = (r + g) / V
                    s = np.sqrt(cx + bq)
                    if target == 'V':
                        P = mc_race(mu, s, [(s, V - 1)], rng)
                    else:
                        s0 = np.sqrt(cx + (V - 1) * bq)
                        P = (V - 1) / V * mc_race(mu, s, [(s, V - 2), (s0, 1)], rng) + 1 / V * mc_race(mu, s0, [(s, V - 1)], rng)
            P2, P = P, P1
            spec = T.spec_linear(L, A, C2, G, sd_.pair_lab)
            mainL, _ = T.theory_L(spec, M, V, alpha, q, target)
            mainL2, _ = T.theory_L(spec, M, V, alpha, q, target, corrected=True)
            out[f'{key}|{arm}'] = {'checked': True, 'p': p, 'L_check': float(P.mean()), 'L_main_sub': mainL,
                                   'L_main_full': th[key][arm], 'diff': float(P.mean()) - mainL, 'kappa': k,
                                   'L2_check': float(P2.mean()), 'L2_main_sub': mainL2,
                                   'diff2': float(P2.mean()) - mainL2}
            print(enc, key, arm, out[f'{key}|{arm}'], flush=True)
    json.dump(out, open(f'{RES}/b67_check_L_{enc}.json', 'w'), indent=1)


def check_b():
    S = T.store_structure()
    B = json.load(open(f'{RES}/b67_b.json'))
    slots = float(np.mean([np.mean(st['slots']) for st in S]))
    eps = 1 - 0.995 ** (1 / slots)
    Nmed = float(np.median([2 * st['Nv'] - 1 for st in S]))
    out = {}
    for kname, e in T.COND.items():
        r = {}
        r['D-a'] = math.pi * math.log(Nmed / eps) / (1 - 2 * e) ** 2
        r['S-a'] = math.log2(Nmed / eps) / math.log(2)
        H2 = (-e * math.log2(e) - (1 - e) * math.log2(1 - e)) if e > 0 else 0.0
        r['T-b'] = max(st['logical_bits'] / st['n'] for st in S) / (1 - H2)

        def dfail(Bc):
            fr = []
            for st in S:
                n = st['n']
                tot = int(Bc * n)
                Ssh = max(1, math.ceil(tot / 8192))
                d = min(8192, tot // Ssh)
                sh = np.array([int(x) % Ssh for x in st['shard_seed']], dtype=np.int64)
                kidx = {k: i for i, k in enumerate(st['keys_all'])}
                load = np.bincount(sh[[kidx[k] for k in st['pair_keys']]], minlength=Ssh)
                N = 2 * st['Nv'] - 1
                uq, cnt = np.unique(load, return_counts=True)
                pfu = {}
                for Lr in uq:
                    s = math.sqrt((2 / math.pi) * d / max(1, Lr) * (1 - 2 * e) ** 2)
                    val, _ = integrate.quad(lambda x: stats.norm.pdf(x) * math.exp(N * stats.norm.logcdf(x + s)),
                                            -10, 10, limit=200)
                    pfu[int(Lr)] = 1 - val
                pf = float(sum(c * Lr * pfu[int(Lr)] for Lr, c in zip(uq, cnt)) / load.sum())
                fr.append(np.mean([1 - (1 - pf) ** m for m in st['slots']]))
            return float(np.mean(fr))
        r['D-b'] = next((g for g in T.GRIDB if dfail(g) <= 0.005), None)
        best = None
        for lam in T.LAMS:
            for g in T.GRIDB:
                nb = max(1, int(round(lam * g)))
                rho = 1 - math.exp(-lam)
                rp = rho * (1 - e) + (1 - rho) * e
                fr = []
                for st in S:
                    N = 2 * st['Nv'] - 1
                    tot = 0.0
                    for kk in range(nb + 1):
                        pt = stats.binom.pmf(kk, nb, 1 - e)
                        below = stats.binom.cdf(kk - 1, nb, rp) if kk > 0 else 0.0
                        at = stats.binom.pmf(kk, nb, rp)
                        win = below ** N
                        tie = (below + at) ** N - win
                        tot += pt * (win + 0.5 * tie)
                    pf = 1 - tot
                    fr.append(np.mean([1 - (1 - pf) ** m for m in st['slots']]))
                if np.mean(fr) <= 0.005:
                    if best is None or g < best:
                        best = g
                    break
        r['S-b'] = best
        m = B['pred'][kname]
        r['agree'] = {x: (abs(r[x] - m[x]) < 1e-6 * max(1, abs(m[x])) if isinstance(r[x], float) and m[x] is not None
                          else r[x] == m[x]) for x in ('D-a', 'S-a', 'T-b', 'D-b', 'S-b')}
        out[kname] = r
        print(kname, r, flush=True)
    json.dump(out, open(f'{RES}/b67_check_b.json', 'w'), indent=1)


def check_c2(encs):
    out = {}
    for enc in encs:
        dd = T.DevData(enc)
        ent_of = {}
        for e in dd.pool.ents:
            ent_of[dd.pool.idx[e['name']]] = e['split']
            for a in e['aliases']:
                ent_of[dd.pool.idx[a]] = e['split']
        rows = list(dd.lab_rows) + list(dd.pair_ali)
        sp = {}
        for r in rows:
            sp[ent_of[r]] = sp.get(ent_of[r], 0) + 1
        try:
            C.make_store(dd.pool, 'test', 256, 2, 0)
            guard = False
        except RuntimeError:
            guard = True
        out[enc] = {'rows_by_split': sp, 'guard_raises_on_test': guard, 'n_label_rows': len(dd.lab_rows),
                    'n_pairs': len(dd.pair_ali)}
        print(enc, out[enc], flush=True)
    json.dump(out, open(f'{RES}/b67_check_c2_{"_".join(encs)}.json', 'w'), indent=1)


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'a':
        check_a(sys.argv[2])
    elif cmd == 'L':
        check_L(sys.argv[2])
    elif cmd == 'b':
        check_b()
    elif cmd == 'c2':
        check_c2(sys.argv[2:] or T.PANEL)
