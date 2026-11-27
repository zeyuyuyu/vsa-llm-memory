"""B71 checks (prereg/B71_hebbian_remedies_prereg.md, 'Checks').

python3 b71_check.py scorer [--encs ..]     # plain-loop cell accuracies and point statistics vs results/b71_part_*.json
python3 b71_check.py bits [--encs ..]       # every dev / test row of every new arm: within budget and maximal
python3 b71_check.py repro [--encs ..]      # audit: Hebbian bundle = B53's dev accuracy; retraining = B56's RT choice
python3 b71_check.py rederive ENC           # float64 re-derivation by independent code on a sample -> results/b71_rederive_{enc}.json
python3 b71_check.py sample                 # sample-reading material -> results/b71_sample_material.txt
python3 b71_check.py report                 # -> results/b71_check.txt
"""
import gzip
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import b71_verdict as V71  # noqa: E402

RES = V71.RES
JOBS = V71.JOBS
PANEL = V71.PANEL
M_GRID, VN, BPF = V71.M_GRID, V71.VN, V71.BPF
HQ = {1: 1.0, 2: 1.9037, 3: 2.7606, 4: 3.6024}


def encs_arg(a):
    return a[a.index('--encs') + 1].split(',') if '--encs' in a else PANEL


# ----------------------------------------------------------------------------------------------- scorer

def scorer(encs):
    out = {'max_abs_diff_cell': 0.0, 'max_abs_diff_region': 0.0, 'n_cell': 0, 'n_region': 0}
    for enc in encs:
        J = json.load(open(f'{RES}/b71_part_{enc}.json'))
        data = V71.load(enc)
        # plain loops: cell accuracies
        cellacc = {}
        for (M, V), recs in data.items():
            for b in BPF:
                for k in J['cell_acc']:
                    M_, V_, b_, arm = k.split('|')
                    if (int(M_), int(V_), int(b_)) != (M, V, b):
                        continue
                    right, n = 0, 0
                    for r in recs:
                        p = r['pred'][f'{arm}|{b}|ideal']
                        for i in range(len(p)):
                            right += 1 if int(p[i]) == int(r['qgold'][i]) else 0
                            n += 1
                    cellacc[k] = right / n
                    out['max_abs_diff_cell'] = max(out['max_abs_diff_cell'], abs(right / n - J['cell_acc'][k]))
                    out['n_cell'] += 1
        # plain loops: region point statistics of every registered contrast
        for grp in ('primary', 'secondary'):
            for key, e in J[grp].items():
                _, rn, nm = key.split('|')
                x, y = nm.split('-')
                tot = 0.0
                for (M, V, b) in V71.REG[rn]:
                    s, n = 0.0, 0
                    for r in data[M, V]:
                        px, py, g = r['pred'][f'{x}|{b}|ideal'], r['pred'][f'{y}|{b}|ideal'], r['qgold']
                        for i in range(len(g)):
                            s += (1.0 if int(px[i]) == int(g[i]) else 0.0) - (1.0 if int(py[i]) == int(g[i]) else 0.0)
                            n += 1
                    tot += s / n
                tot /= len(V71.REG[rn])
                out['max_abs_diff_region'] = max(out['max_abs_diff_region'], abs(tot - e['diff']))
                out['n_region'] += 1
        print('scorer', enc, out, flush=True)
    out['pass'] = out['max_abs_diff_cell'] < 1e-9 and out['max_abs_diff_region'] < 1e-9
    json.dump(out, open(f'{RES}/b71_check_scorer.json', 'w'), indent=1)
    return out


# ----------------------------------------------------------------------------------------------- bits

def own_vsa_bits(d, q):
    return d * HQ[q] + 32 * math.ceil(d / 8192) if d > 0 else 0.0


def own_lam_bits(p, c, q, scale):
    if p <= 0:
        return 0.0
    groups = c * math.ceil(p / 8192) if scale == 'col' else math.ceil(c * p / 8192)
    return c * p * HQ[q] + 32 * groups


def bits(encs):
    res = {'rows': 0, 'over': 0, 'not_maximal': 0, 'formula_mismatch': 0, 'examples': []}
    nmap = {}
    for fn in sorted(os.listdir(JOBS)):
        enc = fn.split('_')[0]
        if enc not in encs:
            continue
        r = json.load(gzip.open(os.path.join(JOBS, fn), 'rt'))
        s = r['spec']
        M = s['M']
        if s['arm'] == 'VSAPR':
            rows = [x for rr in r['reps'].values() for x in rr['rows']]
        else:
            rows = r['rows']
        for x in rows:
            B = x['bpf'] * M
            res['rows'] += 1
            bad = None
            if s['arm'] == 'NLMSF':
                scale, t, mu, o, q = x['cfg']
                c = s['V'] - 1 if t == 'V-1' else s['V']
                p = x['p']
                ob = own_lam_bits(p, c, q, scale)
                if abs(ob - x['bits']) > 1e-6:
                    res['formula_mismatch'] += 1
                if ob > B:
                    bad = 'over'
                elif p < r['pmax'] and own_lam_bits(p + 1, c, q, scale) <= B:
                    bad = 'not_maximal'
            else:
                q = x['cfg'][3] if s['arm'] == 'VSAPR' else (x['cfg'][0] if x['arm'] in ('HEB', 'RT') else x['cfg'][4])
                d = x['d']
                ob = own_vsa_bits(d, q)
                if 'bits' in x and abs(ob - x['bits']) > 1e-6:
                    res['formula_mismatch'] += 1
                cap = None
                if s['arm'] == 'VSAD' and s['kind'] == 'P':
                    if enc not in nmap:
                        import b71_run as E
                        nmap[enc] = E.n_of(enc)
                    cap = nmap[enc]
                if ob > B:
                    bad = 'over'
                elif own_vsa_bits(d + 1, q) <= B and not (cap is not None and d == cap):
                    bad = 'not_maximal'
            if bad:
                res[bad] += 1
                if len(res['examples']) < 10:
                    res['examples'].append([fn, x.get('cfg'), x['bpf'], bad])
    res['pass'] = res['over'] == 0 and res['not_maximal'] == 0 and res['formula_mismatch'] == 0
    json.dump(res, open(f'{RES}/b71_check_bits.json', 'w'), indent=1)
    print('bits', res)
    return res


# ----------------------------------------------------------------------------------------------- reproduction

def repro(encs):
    out = {'cells': 0, 'heb_max_diff': 0.0, 'rt_cfg_mismatch': 0, 'rt_max_diff': 0.0, 'examples': []}
    for enc in encs:
        A = json.load(open(f'{RES}/b71_audit_{enc}.json'))['cells']
        for cell, a in A.items():
            out['cells'] += 1
            out['heb_max_diff'] = max(out['heb_max_diff'], abs(a['heb_dev_repro'] - a['b53_dev']))
            c1 = [a['rt_cfg_repro'][0], f'{a["rt_cfg_repro"][1]}:{float(a["rt_cfg_repro"][2])}']
            if [a['b56_rt_cfg'][0], a['b56_rt_cfg'][1]] != c1:
                out['rt_cfg_mismatch'] += 1
                out['examples'].append([enc, cell, a['b56_rt_cfg'], a['rt_cfg_repro']])
            else:
                out['rt_max_diff'] = max(out['rt_max_diff'], abs(a['rt_dev_repro'] - a['b56_rt_dev']))
    out['pass'] = out['heb_max_diff'] <= 1e-12 and out['rt_cfg_mismatch'] == 0 and out['rt_max_diff'] <= 1e-12
    json.dump(out, open(f'{RES}/b71_check_repro.json', 'w'), indent=1)
    print('repro', out)
    return out


# ----------------------------------------------------------------------------------------------- float64 re-derivation

H = None


def sylv16():
    Hm = np.ones((1, 1))
    while Hm.shape[0] < 16:
        Hm = np.block([[Hm, Hm], [Hm, -Hm]])
    return Hm


def own_q(x, q, block=8192):
    """Own float64 quantizer per block: q = 1 sign (exact zeros -> +1, measure zero); q >= 2 Max's uniform step on the
    block's std, mid-rise, clipped to 2^q levels."""
    out = np.empty_like(x)
    for j0 in range(0, len(x), block):
        v = x[j0:j0 + block]
        if q == 1:
            out[j0:j0 + block] = np.where(v >= 0, 1.0, -1.0)
        else:
            step = {2: 0.9957, 4: 0.3352}[q] * (float(v.std()) or 1.0)
            L = 2 ** q
            out[j0:j0 + block] = (np.clip(np.floor(v / step), -L // 2, L // 2 - 1) + 0.5) * step
    return out


def own_q_store(x, q, seedparts, block=8192):
    """The store's quantizer as specified (B32): q = 1 sign with exact zeros (integer sums of sign features) set to +-1 at
    random from a generator seeded ('Q', store seed, d, q, n_stored, j0) per block; q >= 2 Max's uniform step (own_q)."""
    import b32_common as C
    if q != 1:
        return own_q(x, q, block)
    out = np.where(x > 0, 1.0, -1.0)
    for j0 in range(0, len(x), block):
        v = x[j0:j0 + block]
        z = np.nonzero(v == 0)[0]
        if len(z):
            rng = np.random.default_rng(C.seed_of(*seedparts, j0))
            out[j0 + z] = rng.choice([-1.0, 1.0], size=len(z))
    return out


def own_codes(enc, prep, fmap, V, d):
    """B53's value codes (B49's row-0 signs r per block of 8192, then antipodal / Sylvester-Hadamard), own code."""
    import b32_common as C
    r = np.empty(d)
    for j0 in range(0, d, 8192):
        w = min(8192, d - j0)
        g = np.random.default_rng(C.seed_of('C', enc, prep, fmap, j0, 8192))
        r[j0:j0 + w] = np.where(g.random((1, 8192), dtype=np.float32)[0] < 0.5, -1.0, 1.0)[:w]
    if V == 2:
        return np.stack([r, -r])
    Hm = sylv16()
    return Hm[:, np.arange(d) % 16] * r[None, :]


def own_nlms_dual(K, Y, mu):
    """Forward substitution by an explicit loop: d_t = (y_t - sum_{s<t} K_ts d_s) mu / K_tt."""
    M = K.shape[0]
    D = np.zeros_like(Y)
    for t in range(M):
        D[t] = (Y[t] - K[t, :t] @ D[:t]) * mu / (K[t, t] + 1e-12)
    return D


def own_delta_err(X, y, V, d, mu, E, avg=False):
    """Own float64 loop of the misread-only rule (written separately from b71_run.delta_write); avg = the Polyak
    average of the bundle after every fact of the single pass."""
    Hm = sylv16()
    cls = (np.arange(d) % 16) if V == 16 else np.zeros(d, int)
    ncl = 16 if V == 16 else 1
    S = np.zeros(d)
    Ss = np.zeros(d)
    for _ in range(E):
        for t in range(X.shape[0]):
            x = X[t, :d]
            m = np.bincount(cls, weights=x * S, minlength=ncl)
            sc = Hm @ m if V == 16 else np.array([m[0], -m[0]])
            if int(np.argmax(sc)) != int(y[t]):
                tgt = Hm[y[t]] if V == 16 else np.array([1.0 if y[t] == 0 else -1.0])
                nr = np.bincount(cls, weights=x * x, minlength=ncl) + 1e-12
                S = S + mu * x * ((tgt - m) / nr)[cls]
            Ss += S
    return Ss / X.shape[0] if avg else S


def rederive(enc, arms=('NLMSFR', 'VSAPR', 'VSADR1')):
    import b32_common as C
    import b32_run as R
    import b66_lsfc as LS
    import b71_run as E
    ch = V71.choice(enc)
    S = LS.setup(enc)
    ti = V71.test_index(enc)
    out = []
    for M in M_GRID:
        for V in VN:
            rep = 0
            st = C.make_store(S['pool'], 'test', M, V, rep)
            g = np.asarray(st['qgold'])
            vals = np.asarray(st['vals'])
            order = R.store_order(st)
            for b in (4, 16):
                cell = f'{M}|{V}|{b}'
                if 'NLMSFR' in arms:
                    # NLMSFR: own features, genuine one-fact-at-a-time loop, own quantizer, own read
                    c = ch[f'{cell}|NLMSFR']
                    prep, fmap = c['feat']
                    scale, t, mu, o, q = c['cfg']
                    P = S['preps'][prep]
                    cc = V - 1 if t == 'V-1' else V
                    import b60_run as L60
                    p = L60.p_of(b * M, cc, q, 'ideal', scale, 65536)
                    Fk = LS.own_feats_R(enc, prep, fmap, P(st['keys']), p)[order]
                    Fq = LS.own_feats_R(enc, prep, fmap, P(st['queries']), p)
                    Y = np.eye(V)[vals[order]] - 1.0 / V
                    if t == 'V-1':
                        Y = Y[:, 1:]
                    W = np.zeros((p, Y.shape[1]))
                    Ws = np.zeros_like(W)
                    for i in range(M):
                        f = Fk[i]
                        W += (mu / (f @ f + 1e-12)) * np.outer(f, Y[i] - f @ W)
                        Ws += W
                    W = W if o == 'last' else Ws / M
                    # least squares' quantizer: per group (column x block of 8192 rows, or 8192 flattened numbers); q = 1 is
                    # the sign times the group's mean |w| (b66_lsfc.own_quant, independent of b60_run.quantW)
                    if scale == 'col':
                        Wq = np.empty_like(W)
                        for j0 in range(0, p, 8192):
                            for k in range(W.shape[1]):
                                Wq[j0:j0 + 8192, k] = LS.own_quant(W[j0:j0 + 8192, k], q)
                    else:
                        fl = W.T.ravel()
                        oq = np.empty_like(fl)
                        for i0 in range(0, len(fl), 8192):
                            oq[i0:i0 + 8192] = LS.own_quant(fl[i0:i0 + 8192], q)
                        Wq = oq.reshape(W.shape[1], p).T
                    sc = Fq @ Wq
                    if t == 'V-1':
                        sc = np.hstack([-sc.sum(1, keepdims=True), sc])
                    pr = sc.argmax(1)
                    lg = np.asarray(ti['NLMSFR', M, V, rep, b, tuple(c['cfg']), prep, fmap])
                    out.append({'arm': 'NLMSFR', 'cell': cell, 'agree': float((pr == lg).mean()),
                                'acc_own': float((pr == g).mean()), 'acc_log': float((lg == g).mean())})
                if 'VSAPR' in arms:
                    # VSAPR: own basis (float64 eigh, signs aligned to the pipeline's), same Haar seeds, own features
                    c = ch[f'{cell}|VSAPR']
                    prep, fmap, pp, q, fsh = c['cfg']
                    P = S['preps'][prep]
                    Up, mup = LS.basis(enc, prep)
                    Tr = np.asarray(S['Tr'], np.float64)
                    if prep == 'whiten':
                        w, U = np.linalg.eigh(np.cov((Tr - Tr.mean(0)).T))
                        mu0 = np.zeros(Tr.shape[1])
                    else:
                        Pc = np.asarray(P(S['Tr']), np.float64)
                        mu0 = Pc.mean(0)
                        w, U = np.linalg.eigh(np.cov((Pc - mu0).T))
                    U = U[:, np.argsort(-w)][:, :pp]
                    U *= np.sign(np.sum(U * Up[:, :pp], 0))[None, :]
                    d = C.vsa_dim(b * M, q, 'ideal')
                    K = 1 if fmap in ('lin', 'sgn') else 3
                    Zk = (np.asarray(P(st['keys']), np.float64) - mu0) @ U
                    Zq = (np.asarray(P(st['queries']), np.float64) - mu0) @ U
                    fk, fq = None, None
                    for m in range(K):
                        cols = []
                        for k in range(0, (d - 1) // pp + 1):
                            A_ = np.random.default_rng(C.seed_of('B71ROT', enc, prep, fmap, pp, m, k)).standard_normal((pp, pp))
                            Qm, Rr = np.linalg.qr(A_)
                            cols.append(Qm * np.sign(np.diag(Rr))[None, :])
                        Rot = np.hstack(cols)[:, :d]
                        hk, hq = Zk @ Rot, Zq @ Rot
                        if fmap != 'lin':
                            hk, hq = np.where(hk >= 0, 1.0, -1.0), np.where(hq >= 0, 1.0, -1.0)
                        fk = hk if fk is None else fk * hk
                        fq = hq if fq is None else fq * hq
                    cd = own_codes(enc, prep, fmap, V, d)
                    ns = max(1, int(math.floor(fsh * M)))
                    Sb = (fk[order[:ns]] * cd[vals[order[:ns]]]).sum(0)
                    Sq = own_q_store(Sb, q, ('Q', R.store_seed(enc, st), d, q, ns))
                    pr = (fq @ (cd * Sq).T).argmax(1)
                    lg = np.asarray(ti['VSAPR', M, V, rep, b, tuple(c['cfg'])])
                    out.append({'arm': 'VSAPR', 'cell': cell, 'agree': float((pr == lg).mean()),
                                'acc_own': float((pr == g).mean()), 'acc_log': float((lg == g).mean())})
                if 'VSADR1' in arms:
                    # VSADR1: dual form (nlms) or own loop (err), own quantizer and read
                    c = ch[f'{cell}|VSADR1']
                    prep, fmap, fsh = c['feat']
                    Ep, rule, mu, o, q = c['cfg']
                    P = S['preps'][prep]
                    d = C.vsa_dim(b * M, q, 'ideal')
                    fk = LS.own_feats_R(enc, prep, fmap, P(st['keys']), d)
                    fq = LS.own_feats_R(enc, prep, fmap, P(st['queries']), d)
                    cd = own_codes(enc, prep, fmap, V, d)
                    r0 = cd[0] if V == 2 else cd[0]  # row 0 = r (H[0, :] = 1)
                    ns = max(1, int(math.floor(fsh * M)))
                    X = fk[order[:ns]] * r0[None, :]
                    yv = vals[order[:ns]]
                    if rule == 'nlms':
                        Sd = np.zeros(d)
                        ncl = 16 if V == 16 else 1
                        Hm = sylv16()
                        for cl in range(ncl):
                            cols = np.arange(cl, d, ncl)
                            F = X[:, cols]
                            T = (Hm[yv, cl] if V == 16 else np.where(yv == 0, 1.0, -1.0))[:, None]
                            D = own_nlms_dual(F @ F.T, T, mu)
                            if o == 'avg':
                                D = D * ((ns - np.arange(ns)) / ns)[:, None]
                            Sd[cols] = (F.T @ D)[:, 0]
                    else:
                        assert o == 'last' or Ep == 1
                        Sd = own_delta_err(X, yv, V, d, mu, Ep, avg=(o == 'avg'))
                    pr = (fq @ (cd * own_q(Sd, q)).T).argmax(1)
                    lg = np.asarray(ti['VSADR', M, V, rep, b, tuple(c['cfg']), prep, fmap])
                    out.append({'arm': 'VSADR1', 'cell': cell, 'cfg': c['cfg'], 'agree': float((pr == lg).mean()),
                                'acc_own': float((pr == g).mean()), 'acc_log': float((lg == g).mean())})
                print(enc, out[-3:], flush=True)
    sfx = '' if len(arms) == 3 else '_' + '_'.join(arms)
    json.dump(out, open(f'{RES}/b71_rederive_{enc}{sfx}.json', 'w'), indent=1)
    return out


# ----------------------------------------------------------------------------------------------- sample material

def sample():
    import b66_lsfc as LS
    import b32_common as C
    L = ['# B71 sample-reading material (prereg: Qwen3-8B and e5-base-v2; VSADR1 - STORE largest R1 cell; VSAPR - VSAPC '
         'largest R1-V16 cell)', '']
    for enc in ('qwen8', 'e5'):
        J = json.load(open(f'{RES}/b71_part_{enc}.json'))
        ca = J['cell_acc']
        data = V71.load(enc)
        S = LS.setup(enc)
        pool = S['pool']
        for (x, y, cells) in (('VSADR1', 'STORE', [(M, V, b) for M in M_GRID for V in VN for b in BPF]),
                              ('VSAPR', 'VSAPC', [(M, 16, b) for M in M_GRID for b in BPF])):
            M, V, b = max(cells, key=lambda c: ca[f'{c[0]}|{c[1]}|{c[2]}|{x}'] - ca[f'{c[0]}|{c[1]}|{c[2]}|{y}'])
            L.append(f'## {enc} {x} vs {y}: cell M={M} V={V} {b} bits/fact; acc {x} '
                     f'{100 * ca[f"{M}|{V}|{b}|{x}"]:.2f} vs {y} {100 * ca[f"{M}|{V}|{b}|{y}"]:.2f}')
            only = {x: [], y: []}
            for r in data[M, V]:
                st = C.make_store(pool, 'test', M, V, r['rep'])
                px, py = r['pred'][f'{x}|{b}|ideal'], r['pred'][f'{y}|{b}|ideal']
                for i, gd in enumerate(r['qgold']):
                    ok_x, ok_y = px[i] == gd, py[i] == gd
                    if ok_x != ok_y:
                        rec = (st['qalias'][i], st['names'][int(st['qent'][i])], gd, px[i], py[i])
                        only[x if ok_x else y].append(rec)
            for k in (x, y):
                L.append(f'### only {k} right ({len(only[k])} queries in the cell); first 10: (alias | label | gold | {x} | {y})')
                for rec in only[k][:10]:
                    L.append('  ' + ' | '.join(map(str, rec)))
            L.append('')
    open(f'{RES}/b71_sample_material.txt', 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L[:60]))


def report():
    L = ['# B71 checks', '']
    for nm in ('scorer', 'bits', 'repro'):
        fn = f'{RES}/b71_check_{nm}.json'
        if os.path.exists(fn):
            L.append(f'{nm}: {json.dumps(json.load(open(fn)))}')
    rd = []
    for enc in PANEL:
        fn = f'{RES}/b71_rederive_{enc}.json'
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
    open(f'{RES}/b71_check.txt', 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))


def main():
    a = sys.argv[1:]
    if a[0] == 'scorer':
        scorer(encs_arg(a))
    elif a[0] == 'bits':
        bits(encs_arg(a))
    elif a[0] == 'repro':
        repro(encs_arg(a))
    elif a[0] == 'rederive':
        rederive(a[1], tuple(a[a.index('--arms') + 1].split(',')) if '--arms' in a else ('NLMSFR', 'VSAPR', 'VSADR1'))
    elif a[0] == 'sample':
        sample()
    elif a[0] == 'report':
        report()
    else:
        raise SystemExit(__doc__)


if __name__ == '__main__':
    main()
