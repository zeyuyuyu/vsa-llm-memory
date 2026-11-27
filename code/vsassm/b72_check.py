"""B72 checks (prereg/B72_format_side_info_prereg.md, 'Checks'): plain-loop scorer, bits audit, float64 re-derivation of
the optimal-format solve on a sample of test stores, Monte Carlo check of the exact permutation p-values, independent
t interval.

python3 b72_check.py scorer [--encs ..]
python3 b72_check.py bits [--encs ..]
python3 b72_check.py rederive ENC
python3 b72_check.py perm [--encs ..]
python3 b72_check.py report
"""
import gzip
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import b72_verdict as V72  # noqa: E402

RES = V72.RES
OUT = V72.OUT
PANEL = V72.PANEL
M_GRID, VN, BPF = V72.M_GRID, V72.VN, V72.BPF
HQ = {1: 1.0, 2: 1.9037, 3: 2.7606, 4: 3.6024}


def encs_arg(a):
    return a[a.index('--encs') + 1].split(',') if '--encs' in a else PANEL


# ----------------------------------------------------------------------------------------------- scorer

def scorer(encs):
    out = {'max_abs_diff_cell': 0.0, 'max_abs_diff_region': 0.0, 'max_abs_diff_C': 0.0, 'max_abs_diff_t': 0.0,
           'n_cell': 0, 'n_region': 0, 'n_C': 0}
    for enc in encs:
        J = json.load(open(f'{RES}/b72_part_{enc}.json'))
        data = V72.load(enc, with_b=False)
        for k, v in J['cell_acc'].items():
            M_, V_, b_, arm = k.split('|')
            right, n = 0, 0
            for r in data[int(M_), int(V_)]:
                p = r['pred'][f'{arm}|{b_}|ideal']
                for i in range(len(p)):
                    right += 1 if int(p[i]) == int(r['qgold'][i]) else 0
                    n += 1
            out['max_abs_diff_cell'] = max(out['max_abs_diff_cell'], abs(right / n - v))
            out['n_cell'] += 1
        for grp in ('primary', 'secondary', 'B'):
            for key, e in J[grp].items():
                _, rn, nm = key.split('|')
                x, y = nm.split('-')
                tot = 0.0
                for (M, V, b) in V72.REG[rn]:
                    s, n = 0.0, 0
                    for r in data[M, V]:
                        px, py, g = r['pred'][f'{x}|{b}|ideal'], r['pred'][f'{y}|{b}|ideal'], r['qgold']
                        for i in range(len(g)):
                            s += (1.0 if int(px[i]) == int(g[i]) else 0.0) - (1.0 if int(py[i]) == int(g[i]) else 0.0)
                            n += 1
                    tot += s / n
                tot /= len(V72.REG[rn])
                out['max_abs_diff_region'] = max(out['max_abs_diff_region'], abs(tot - e['diff']))
                out['n_region'] += 1
                if f'{rn}|{nm}' in J['C']:
                    c = J['C'][f'{rn}|{nm}']
                    out['max_abs_diff_C'] = max(out['max_abs_diff_C'], abs(tot - c['diff']))
                    # independent t interval: per-store differences by plain loops, unweighted-by-hand formula
                    strata = sorted({(M, V) for (M, V, b) in V72.REG[rn]})
                    G = len(strata)
                    var, Tt, ag, ng = 0.0, 0.0, [], []
                    for (M, V) in strata:
                        bs = [b for (M_, V_, b) in V72.REG[rn] if (M_, V_) == (M, V)]
                        D, Q = [], []
                        for r in data[M, V]:
                            g = r['qgold']
                            tot_u = 0.0
                            for b in bs:
                                px, py = r['pred'][f'{x}|{b}|ideal'], r['pred'][f'{y}|{b}|ideal']
                                su = 0.0
                                for i in range(len(g)):
                                    su += (px[i] == g[i]) - (py[i] == g[i])
                                tot_u += su / len(g)
                            D.append(tot_u / len(bs))
                            Q.append(len(g))
                        Tg = sum(q * d for q, d in zip(Q, D)) / sum(Q)
                        qb = sum(Q) / len(Q)
                        s2 = sum((q / qb) ** 2 * (d - Tg) ** 2 for q, d in zip(Q, D)) / (len(Q) * (len(Q) - 1))
                        Tt += Tg / G
                        ag.append(s2 / G ** 2)
                        ng.append(len(Q))
                    var = sum(ag)
                    from scipy import stats
                    if var > 0:
                        df = var ** 2 / sum(a ** 2 / (n - 1) for a, n in zip(ag, ng))
                        h = stats.t.ppf(0.975, df) * math.sqrt(var)
                    else:  # identical predictions in every store
                        h = 0.0
                    out['max_abs_diff_t'] = max(out['max_abs_diff_t'], abs(Tt - h - c['t_ci95'][0]),
                                                abs(Tt + h - c['t_ci95'][1]))
                    out['n_C'] += 1
        print('scorer', enc, out, flush=True)
    out['pass'] = all(out[k] < 1e-9 for k in ('max_abs_diff_cell', 'max_abs_diff_region', 'max_abs_diff_C',
                                              'max_abs_diff_t'))
    json.dump(out, open(f'{RES}/b72_check_scorer.json', 'w'), indent=1)
    return out


# ----------------------------------------------------------------------------------------------- bits

def own_vsa_bits(d, q):
    return d * HQ[q] + 32 * math.ceil(d / 8192) if d > 0 else 0.0


def own_lam_bits(p, c, q, scale):
    if p <= 0:
        return 0.0
    groups = c * math.ceil(p / 8192) if scale == 'col' else math.ceil(c * p / 8192)
    return c * p * HQ[q] + 32 * groups


def own_charge(mem, n, prep, fmap=None, feat=None, size=0):
    e = n + (n * (n + 1) // 2 if prep == 'whiten' else 0)
    bits = 8.0 * e
    if mem in ('STORE', 'VSAR'):
        bits += 64 * ((1 if fmap in ('lin', 'sgn') else 3) + 1)
    if mem in ('LAMC', 'NLMSC'):
        bits += 64 if feat == 'rp' else 0
        if feat == 'pca' and prep == 'center':
            bits += 8.0 * n * size
    if mem == 'SDMT':
        bits += 8.0 * n * size
    return bits


def bits(encs):
    res = {'rows': 0, 'over': 0, 'not_maximal': 0, 'formula_mismatch': 0, 'charge_mismatch': 0, 'examples': []}
    nmap = {'qwen8': 4096, 'b46': 4096, 'llama': 4096, 'mistral': 4096, 'e5': 768, 'bge': 384, 'minilm': 384}
    for jd, lamp in ((os.path.join(OUT, 'jobs'), False), (os.path.join(OUT, 'lamp', 'jobs'), True)):
        for fn in sorted(os.listdir(jd)):
            enc = fn.split('_')[0]
            if enc not in encs:
                continue
            r = json.load(gzip.open(os.path.join(jd, fn), 'rt'))
            s = r['spec']
            M, V = s['M'], s['V']
            n = nmap[enc]
            for x in r['rows']:
                B = x['bpf'] * M
                res['rows'] += 1
                bad = None
                if lamp:
                    scale, t, alpha, q = x['cfg']
                    c = V - 1 if t == 'V-1' else V
                    p = x['p']
                    ob = own_lam_bits(p, c, q, scale)
                    if abs(ob - x['bits']) > 1e-6:
                        res['formula_mismatch'] += 1
                    if ob > B:
                        bad = 'over'
                    elif p < r['pmax'] and own_lam_bits(p + 1, c, q, scale) <= B:
                        bad = 'not_maximal'
                elif s['arm'] == 'OPT':
                    q, d = x['q'], x['d']
                    ob = own_vsa_bits(d, q)
                    if abs(ob - x['bits']) > 1e-6:
                        res['formula_mismatch'] += 1
                    cap = n if s['kind'] == 'P' else None
                    if ob > B:
                        bad = 'over'
                    elif own_vsa_bits(d + 1, q) <= B and not (cap is not None and d == cap):
                        bad = 'not_maximal'
                elif s['arm'] == 'BV':
                    prep, fmap, q, f = x['cfg']
                    K = s['K']
                    ch = own_charge(s['mem'], n, prep, fmap=fmap) / K
                    d = x['d']
                    if not x['feasible']:
                        if B - ch > 0 and own_vsa_bits(1, q) + ch <= B:
                            bad = 'not_maximal'
                    else:
                        if abs(ch - x['charge']) > 1e-6:
                            res['charge_mismatch'] += 1
                        ob = own_vsa_bits(d, q)
                        if abs(ob - x['bits']) > 1e-6:
                            res['formula_mismatch'] += 1
                        if ob + ch > B + 1e-9:
                            bad = 'over'
                        elif own_vsa_bits(d + 1, q) + ch <= B:
                            bad = 'not_maximal'
                elif s['arm'] == 'BL':
                    cfg = x['cfg']
                    scale, t, prep, feat, q = cfg[0], cfg[1], cfg[2], cfg[3], cfg[-1]
                    c = V - 1 if t == 'V-1' else V
                    p = x['p']
                    K = s['K']
                    pmax = n if feat == 'pca' else 65536
                    chf = lambda pp: own_charge(s['mem'], n, prep, feat=feat, size=pp) / K  # noqa: E731
                    if abs(chf(p) - x['charge']) > 1e-6:
                        res['charge_mismatch'] += 1
                    if p > 0:
                        ob = own_lam_bits(p, c, q, scale)
                        if abs(ob - x['bits']) > 1e-6:
                            res['formula_mismatch'] += 1
                        if ob + chf(p) > B + 1e-9:
                            bad = 'over'
                    if bad is None and p < pmax and own_lam_bits(p + 1, c, q, scale) + chf(p + 1) <= B:
                        bad = 'not_maximal'
                elif s['arm'] == 'BS':
                    prep = x['cfg'][0]
                    K = s['K']
                    ch = own_charge('SDMT', n, prep, size=x['N']) / K
                    if abs(ch - x['charge']) > 1e-6:
                        res['charge_mismatch'] += 1
                    if x['N'] > 0 and x['bits'] + ch > B + 1e-9:
                        bad = 'over'
                if bad:
                    res[bad] += 1
                    if len(res['examples']) < 10:
                        res['examples'].append([fn, x.get('cfg'), x['bpf'], bad])
    res['pass'] = all(res[k] == 0 for k in ('over', 'not_maximal', 'formula_mismatch', 'charge_mismatch'))
    json.dump(res, open(f'{RES}/b72_check_bits.json', 'w'), indent=1)
    print('bits', res)
    return res


# ----------------------------------------------------------------------------------------------- float64 re-derivation

def own_ridge_class(F, y, lam):
    """Own float64 ridge per class: primal (F^T F + lam I) s = F^T y when the class has <= 4,096 coordinates, else the dual
    (F F^T + lam I) a = y, s = F^T a; numpy.linalg.solve (independent of b72_run's eigendecomposition path)."""
    if F.shape[1] <= 4096:
        return np.linalg.solve(F.T @ F + lam * np.eye(F.shape[1]), F.T @ y)
    return F.T @ np.linalg.solve(F @ F.T + lam * np.eye(F.shape[0]), y)


def rederive(enc):
    import b32_common as C
    import b32_run as R
    import b66_lsfc as LS
    import b71_check as K71
    ch = V72.choice(enc)
    S = LS.setup(enc)
    ti = V72.test_index(enc)
    out = []
    own_basis = {}
    for M in M_GRID:
        for V in VN:
            rep = 0
            st = C.make_store(S['pool'], 'test', M, V, rep)
            g = np.asarray(st['qgold'])
            vals = np.asarray(st['vals'])
            order = R.store_order(st)
            sseed = R.store_seed(enc, st)
            for b in (4, 16):
                cell = f'{M}|{V}|{b}'
                for kd in ('R', 'P'):
                    c = ch[f'{cell}|OPT{kd}']
                    prep, fmap, q = c['fmt']
                    f, alpha = c['cfg']
                    P = S['preps'][prep]
                    n = S['pool'].X.shape[1]
                    d = C.vsa_dim(b * M, q, 'ideal')
                    if kd == 'P':
                        d = min(d, n)
                        Up, mup = LS.basis(enc, prep)
                        if prep not in own_basis:  # own float64 eigh, computed once per prep
                            Tr = np.asarray(S['Tr'], np.float64)
                            if prep == 'whiten':
                                w, U = np.linalg.eigh(np.cov((Tr - Tr.mean(0)).T))
                                mu0 = np.zeros(Tr.shape[1])
                            else:
                                Pc = np.asarray(P(S['Tr']), np.float64)
                                mu0 = Pc.mean(0)
                                w, U = np.linalg.eigh(np.cov((Pc - mu0).T))
                            own_basis[prep] = (U[:, np.argsort(-w)], mu0)
                        U, mu0 = own_basis[prep]
                        U = U[:, :d].copy()
                        U *= np.sign(np.sum(U * Up[:, :d], 0))[None, :]
                        fk = (np.asarray(P(st['keys']), np.float64) - mu0) @ U
                        fq = (np.asarray(P(st['queries']), np.float64) - mu0) @ U
                        if fmap == 'psgn':
                            fk, fq = np.where(fk >= 0, 1.0, -1.0), np.where(fq >= 0, 1.0, -1.0)
                        cd = K71.own_codes(enc, prep, 'lin' if fmap == 'plin' else 'sgn', V, d)
                    else:
                        fk = LS.own_feats_R(enc, prep, fmap, P(st['keys']), d)
                        fq = LS.own_feats_R(enc, prep, fmap, P(st['queries']), d)
                        cd = K71.own_codes(enc, prep, fmap, V, d)
                    ns = max(1, int(math.floor(f * M)))
                    X = fk[order[:ns]] * cd[0][None, :]
                    yv = vals[order[:ns]]
                    ncl = 16 if V == 16 else 1
                    lam = alpha * float((X ** 2).sum()) / ns / ncl
                    Sd = np.zeros(d)
                    Hm = K71.sylv16()
                    for cl in range(ncl):
                        cols = np.arange(cl, d, ncl)
                        y = Hm[yv, cl] if V == 16 else np.where(yv == 0, 1.0, -1.0)
                        if V == 16 and cl == 0:
                            continue  # centred one-hot target of the constant class: 0
                        Sd[cols] = own_ridge_class(X[:, cols], y, lam)
                    Sq = K71.own_q_store(Sd, q, ('B72Q', sseed, kd if kd == 'P' else 'R', d, q, ns, alpha))
                    pr = (fq @ (cd * Sq).T).argmax(1)
                    jk = 'OPTP' if kd == 'P' else 'OPTR'
                    lg = np.asarray(ti[(jk, M, V, rep, prep, fmap, b, q, (f, alpha))])
                    out.append({'arm': f'OPT{kd}', 'cell': cell, 'cfg': c['cfg'], 'fmt': c['fmt'],
                                'agree': float((pr == lg).mean()), 'acc_own': float((pr == g).mean()),
                                'acc_log': float((lg == g).mean())})
                    print(enc, out[-1], flush=True)
    json.dump(out, open(f'{RES}/b72_rederive_{enc}.json', 'w'), indent=1)
    return out


# ----------------------------------------------------------------------------------------------- permutation check

def perm(encs):
    res = {'n': 0, 'max_z': 0.0, 'fail': [], 'pass': True}
    rng = np.random.default_rng(72072)
    for enc in encs:
        J = json.load(open(f'{RES}/b72_part_{enc}.json'))
        for k, c in J['C'].items():
            a = np.asarray(c['a'])
            T = abs(sum(a))
            B = 200000
            cnt = 0
            for c0 in range(0, B, 20000):
                sgn = rng.choice([-1.0, 1.0], size=(min(20000, B - c0), len(a)))
                cnt += int((np.abs(sgn @ a) >= T - 1e-12 * max(1.0, T)).sum())
            pm = cnt / B
            se = math.sqrt(max(c['perm_p'] * (1 - c['perm_p']), 1e-12) / B)
            z = abs(pm - c['perm_p']) / se if se > 0 else 0.0
            ok = abs(pm - c['perm_p']) <= 4 * se + 1e-5
            res['n'] += 1
            res['max_z'] = max(res['max_z'], z if ok else res['max_z'])
            if not ok:
                res['fail'].append([enc, k, c['perm_p'], pm])
                res['pass'] = False
        print('perm', enc, res['n'], len(res['fail']), flush=True)
    json.dump(res, open(f'{RES}/b72_check_perm.json', 'w'), indent=1)
    return res


def sample():
    import b66_lsfc as LS
    import b32_common as C
    L = ['# B72 sample-reading material (prereg: Qwen3-8B and e5-base-v2; the R1 cell with the largest |OPTR - STORE|)', '']
    for enc in ('qwen8', 'e5'):
        J = json.load(open(f'{RES}/b72_part_{enc}.json'))
        ca = J['cell_acc']
        data = V72.load(enc, with_b=False)
        pool = LS.setup(enc)['pool']
        x, y = 'OPTR', 'STORE'
        cells = [(M, V, b) for M in M_GRID for V in VN for b in BPF]
        M, V, b = max(cells, key=lambda c: abs(ca[f'{c[0]}|{c[1]}|{c[2]}|{x}'] - ca[f'{c[0]}|{c[1]}|{c[2]}|{y}']))
        ch = V72.choice(enc)[f'{M}|{V}|{b}|OPTR']
        L.append(f'## {enc}: cell M={M} V={V} {b} bits/fact; acc {x} {100 * ca[f"{M}|{V}|{b}|{x}"]:.2f} vs {y} '
                 f'{100 * ca[f"{M}|{V}|{b}|{y}"]:.2f}; OPTR format {ch["fmt"]} (f, alpha) {ch["cfg"]}')
        only = {x: [], y: []}
        for r in data[M, V]:
            st = C.make_store(pool, 'test', M, V, r['rep'])
            px, py = r['pred'][f'{x}|{b}|ideal'], r['pred'][f'{y}|{b}|ideal']
            for i, gd in enumerate(r['qgold']):
                ok_x, ok_y = px[i] == gd, py[i] == gd
                if ok_x != ok_y:
                    only[x if ok_x else y].append((st['qalias'][i], st['names'][int(st['qent'][i])], gd, px[i], py[i]))
        for k in (x, y):
            L.append(f'### only {k} right ({len(only[k])} queries in the cell); first 10: (alias | label | gold | {x} | {y})')
            for rec in only[k][:10]:
                L.append('  ' + ' | '.join(map(str, rec)))
        L.append('')
    open(f'{RES}/b72_sample_material.txt', 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))


def report():
    L = ['# B72 checks', '']
    for nm in ('scorer', 'bits', 'perm'):
        fn = f'{RES}/b72_check_{nm}.json'
        if os.path.exists(fn):
            L.append(f'{nm}: {json.dumps(json.load(open(fn)))[:1500]}')
    rd = []
    for enc in PANEL:
        fn = f'{RES}/b72_rederive_{enc}.json'
        if os.path.exists(fn):
            for x in json.load(open(fn)):
                rd.append((enc, x['arm'], x['cell'], x['agree']))
    if rd:
        by = {}
        for enc, arm, cell, a in rd:
            by.setdefault(arm, []).append(a)
        L.append('rederive (float64, independent code; bar >= 0.99 on every store): ' + ', '.join(
            f'{arm} n={len(v)} min {min(v):.4f} mean {np.mean(v):.4f}' for arm, v in by.items()))
        L.append('  below 0.99: ' + str([x for x in rd if x[3] < 0.99]))
    open(f'{RES}/b72_check.txt', 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))


def main():
    a = sys.argv[1:]
    if a[0] == 'scorer':
        scorer(encs_arg(a))
    elif a[0] == 'bits':
        bits(encs_arg(a))
    elif a[0] == 'rederive':
        rederive(a[1])
    elif a[0] == 'perm':
        perm(encs_arg(a))
    elif a[0] == 'report':
        report()
    elif a[0] == 'sample':
        sample()
    else:
        raise SystemExit(__doc__)


if __name__ == '__main__':
    main()
