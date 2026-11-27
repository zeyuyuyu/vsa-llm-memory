"""B74 checks (prereg/B74_factorial_labels_matched_prereg.md, "Checks"): plain-loop scorer, bits audit, float64
re-derivation by independent code on a sample, determinism re-run comparison, and the reproduction checks of (A).

python3 b74_check.py scorer [--encs ..]        # -> results/b74_check_scorer.json
python3 b74_check.py bits [--encs ..]          # -> results/b74_check_bits.json
python3 b74_check.py rederive ENC              # -> results/b74_rederive_{enc}.json
python3 b74_check.py repro [--encs ..]         # HP dev rows vs B60's VSAPc dev rows; HT test vs B71's VSAPR test
python3 b74_check.py detplan PLAN1,PLAN2 OUT    # determinism sample plan
python3 b74_check.py determ DIR                # compare re-run test jobs in DIR with b74_out
python3 b74_check.py report                    # -> results/b74_check.txt
"""
import gzip
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
RES = os.environ.get('B74_RES', f'{ROOT}/results')
OUT = os.environ.get('B74_OUT', f'{ROOT}/b74_out')
PANEL = ['qwen8', 'b46', 'llama', 'mistral', 'e5', 'bge', 'minilm']
M_GRID, VN, BPF = [256, 1024], [2, 16], [4, 8, 16]
H_Q = {1: 1.0, 2: 1.9037, 3: 2.7606, 4: 3.6024}


def encs_arg(a):
    return a[a.index('--encs') + 1].split(',') if '--encs' in a else PANEL


# ----------------------------------------------------------------------------------------------- plain-loop scorer

def plain_region(data, combo, cells, b_of=None):
    """Mean over cells of the cell mean of sum_arm coef * [pred == gold], by plain Python loops."""
    tot = 0.0
    for (M, V, b) in cells:
        s, n = 0.0, 0
        for r in data[M, V]:
            g = r['qgold']
            for i in range(len(g)):
                v = 0.0
                for arm, coef in combo.items():
                    v += coef * (1.0 if r['pred'][f'{arm}|{b}|ideal'][i] == g[i] else 0.0)
                s += v
                n += 1
        tot += s / n
    return tot / len(cells)


def scorer(encs):
    import b74_verdict as VV
    out = {'max_diff': 0.0, 'n': 0, 'path_max': 0.0, 'fail': []}
    for enc in encs:
        J = json.load(open(f'{RES}/b74_part_{enc}.json'))
        data = VV.load_A(enc)
        regs = VV.REG
        if 'A' in J:
            for rn, R in regs.items():
                for nm, combo in {**VV.COMBOS_A1, **VV.COMBOS_A2}.items():
                    v = plain_region(data, combo, R)
                    d = abs(v - J['A'][f'{rn}|{nm}']['diff'])
                    out['max_diff'] = max(out['max_diff'], d)
                    out['n'] += 1
                    if d >= 1e-9:
                        out['fail'].append([enc, rn, nm, d])
                for X in VV.KINDS:
                    a = J['A_attr'][f'{rn}|{X}']
                    p1 = a['path1']['write'] + a['path1']['features']
                    p2 = a['path2']['features'] + a['path2']['write']
                    # both paths sum to the gap; their write terms differ by I_X and their feature terms by -I_X
                    dw = a['path2']['write'] - a['path1']['write']
                    df = a['path1']['features'] - a['path2']['features']
                    out['path_max'] = max(out['path_max'], abs(dw - a['interaction']), abs(df - a['interaction']),
                                          abs(p1 - a['gap']), abs(p2 - a['gap']))
            for (M, V), recs in data.items():
                for b in BPF:
                    for arm in ('HP', 'LP', 'HT', 'LT'):
                        s = n = 0
                        for r in recs:
                            for i, gg in enumerate(r['qgold']):
                                s += r['pred'][f'{arm}|{b}|ideal'][i] == gg
                                n += 1
                        d = abs(s / n - J['A_acc'][f'{M}|{V}|{b}|{arm}'])
                        out['max_diff'] = max(out['max_diff'], d)
                        out['n'] += 1
        if 'B_label' in J:
            ldata, _ = VV.lab_preds(enc, data)
            for rn, R in regs.items():
                for x, y in VV.PAIRS_B:
                    v = plain_region(ldata, {x: 1, y: -1}, R)
                    d = abs(v - J['B_label'][f'{rn}|{x}-{y}']['diff'])
                    out['max_diff'] = max(out['max_diff'], d)
                    out['n'] += 1
                    pr = J['B_paired'][f'{rn}|{x}-{y}']
                    d2 = abs(pr['label'] - v) + abs(pr['alias'] - plain_region(data, {x: 1, y: -1}, R))
                    out['max_diff'] = max(out['max_diff'], d2)
                    if max(d, d2) >= 1e-9:
                        out['fail'].append([enc, rn, x, y, d, d2])
        if 'C' in J:
            VV.nl72_add(enc, data)
            for rn, R in regs.items():
                for dd in range(VV.NDRAW):
                    for base in ('STORE', 'VSAPCX'):
                        v = plain_region(data, {base: 1, f'N72_{dd}': -1}, R)
                        d = abs(v - J['C'][f'{rn}|{base}-N72_{dd}']['diff'])
                        out['max_diff'] = max(out['max_diff'], d)
                        out['n'] += 1
                        if d >= 1e-9:
                            out['fail'].append([enc, rn, base, dd, d])
        print(enc, out['n'], out['max_diff'], flush=True)
    out['pass'] = out['max_diff'] < 1e-9 and out['path_max'] < 1e-12
    json.dump(out, open(f'{RES}/b74_check_scorer.json', 'w'), indent=1)
    print(out)


# ----------------------------------------------------------------------------------------------- bits audit

def own_vsa_dim(B, q):
    """Own B32 size rule (ideal): the largest d with d H_q + 32 ceil(d / 8192) <= B."""
    per = H_Q[q]
    d = int(B // per) + 2
    while d > 0 and d * per + 32 * math.ceil(d / 8192) > B:
        d -= 1
    return d


def own_lam_bits(p, c, q, scale):
    groups = c * math.ceil(p / 8192) if scale == 'col' else math.ceil(p * c / 8192)
    return p * c * H_Q[q] + 32 * groups


def bits(encs):
    out = {'fx_rows': 0, 'fx_over': 0, 'fx_notmax': 0, 'fx_rule': 0, 'nl_rows': 0, 'nl_over': 0, 'nl_notmax': 0,
           'nl_formula': 0}
    jd = os.path.join(OUT, 'jobs')
    for fn in sorted(os.listdir(jd)):
        if not any(fn.startswith(f'{e}_FX') for e in encs):
            continue
        r = json.load(gzip.open(os.path.join(jd, fn), 'rt'))
        s = r['spec']
        n = r['n']
        for rr in r['reps'].values():
            for x in rr['rows']:
                B = x['bpf'] * s['M']
                dr = own_vsa_dim(B, x['q'])
                want = min(dr, n) if s['kind'] == 'P' else dr
                out['fx_rows'] += 1
                bt = x['d'] * H_Q[x['q']] + 32 * math.ceil(x['d'] / 8192)
                out['fx_over'] += bt > B + 1e-9
                out['fx_rule'] += x['d'] != want
                out['fx_notmax'] += (s['kind'] == 'T' or x['d'] < n) and (
                    (x['d'] + 1) * H_Q[x['q']] + 32 * math.ceil((x['d'] + 1) / 8192) <= B)
    nd = os.path.join(OUT, 'nl72', 'jobs')
    if os.path.isdir(nd):
        for fn in sorted(os.listdir(nd)):
            if not any(fn.startswith(f'{e}_') for e in encs):
                continue
            r = json.load(gzip.open(os.path.join(nd, fn), 'rt'))
            s = r['spec']
            for x in r['rows']:
                if x['arm'] != 'NLMSC':
                    continue
                scale, t, prep, feat, mu, o, q = x['cfg']
                V = s['V']
                c = V - 1 if t == 'V-1' else V
                B = x['bpf'] * s['M']
                p = x['p']
                pmax = 65536 if feat == 'rp' else r.get('n', None)
                out['nl_rows'] += 1
                if p > 0:
                    out['nl_over'] += own_lam_bits(p, c, q, scale) > B + 1e-9
                    out['nl_formula'] += abs(own_lam_bits(p, c, q, scale) - x['bits']) > 1e-6
                cap = 65536 if feat == 'rp' else None
                if cap is None:
                    import b60_run as L60
                    cap = L60.setup(s['enc'])[0].X.shape[1]
                out['nl_notmax'] += p < cap and own_lam_bits(p + 1, c, q, scale) <= B
    out['pass'] = all(out[k] == 0 for k in ('fx_over', 'fx_notmax', 'fx_rule', 'nl_over', 'nl_notmax', 'nl_formula'))
    json.dump(out, open(f'{RES}/b74_check_bits.json', 'w'), indent=1)
    print(out)


# ----------------------------------------------------------------------------------------------- re-derivation

def own_basis(S, P, prep):
    Tr = np.asarray(S['Tr'], np.float64)
    if prep == 'whiten':
        w, U = np.linalg.eigh(np.cov((Tr - Tr.mean(0)).T))
        mu0 = np.zeros(Tr.shape[1])
    else:
        Pc = np.asarray(P(S['Tr']), np.float64)
        mu0 = Pc.mean(0)
        w, U = np.linalg.eigh(np.cov((Pc - mu0).T))
    return U[:, np.argsort(-w)], mu0


def rederive(enc):
    """FX's four new arms on one test store per (M, V) at 4 and 16 bits per fact; label predictions of STORE and LAMc on
    the same stores. Own float64 code throughout (features, codes, Hebbian sum / primal ridge, quantizer, read)."""
    import b32_common as C
    import b32_run as R
    import b66_lsfc as LS
    import b71_check as K71
    import b72_check as K72
    import b74_verdict as VV
    ch = VV.choice(enc)
    S = LS.setup(enc)
    ti = VV.fx_index(enc)
    n = S['pool'].X.shape[1]
    out = []
    bas = {}
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
                for kind in ('P', 'T'):
                    ch_h, ch_l = ch[f'{cell}|H{kind}'], ch[f'{cell}|L{kind}']
                    feat = ch_h['feat']
                    q = ch_h['q']
                    prep, fmap = feat[0], feat[1]
                    P = S['preps'][prep]
                    if prep not in bas:
                        bas[prep] = own_basis(S, P, prep)
                    U, mu0 = bas[prep]
                    Up, _ = LS.basis(enc, prep)
                    d = C.vsa_dim(b * M, q, 'ideal')
                    if kind == 'P':
                        d = min(d, n)
                        Ud = U[:, :d].copy()
                        Ud *= np.sign(np.sum(Ud * Up[:, :d], 0))[None, :]
                        fk = (np.asarray(P(st['keys']), np.float64) - mu0) @ Ud
                        fq = (np.asarray(P(st['queries']), np.float64) - mu0) @ Ud
                        if fmap == 'psgn':
                            fk, fq = np.where(fk >= 0, 1.0, -1.0), np.where(fq >= 0, 1.0, -1.0)
                        cd = K71.own_codes(enc, prep, 'lin' if fmap == 'plin' else 'sgn', V, d)
                    else:
                        pp = feat[2]
                        Ud = U[:, :pp].copy()
                        Ud *= np.sign(np.sum(Ud * Up[:, :pp], 0))[None, :]
                        Zk = (np.asarray(P(st['keys']), np.float64) - mu0) @ Ud
                        Zq = (np.asarray(P(st['queries']), np.float64) - mu0) @ Ud
                        K = 1 if fmap in ('lin', 'sgn') else 3
                        fk = fq = None
                        for m in range(K):
                            cols = []
                            for k in range(0, (d - 1) // pp + 1):
                                A_ = np.random.default_rng(C.seed_of('B71ROT', enc, prep, fmap, pp, m, k)).standard_normal(
                                    (pp, pp))
                                Qm, Rr = np.linalg.qr(A_)
                                cols.append(Qm * np.sign(np.diag(Rr))[None, :])
                            Rot = np.hstack(cols)[:, :d]
                            hk, hq = Zk @ Rot, Zq @ Rot
                            if fmap != 'lin':
                                hk, hq = np.where(hk >= 0, 1.0, -1.0), np.where(hq >= 0, 1.0, -1.0)
                            fk = hk if fk is None else fk * hk
                            fq = hq if fq is None else fq * hq
                        cd = K71.own_codes(enc, prep, fmap, V, d)
                    # H: own Hebbian sum, the store's quantizer stream as specified, own read
                    fsh = ch_h['cfg'][0]
                    ns = max(1, int(math.floor(fsh * M)))
                    Sb = (fk[order[:ns]] * cd[vals[order[:ns]]]).sum(0)
                    Sq = K71.own_q_store(Sb, q, ('Q', sseed, d, q, ns))
                    pr = (fq @ (cd * Sq).T).argmax(1)
                    lg = np.asarray(ti['H' + kind, M, V, rep, b, tuple(ch_h['cfg'])])
                    out.append({'arm': 'H' + kind, 'cell': cell, 'agree': float((pr == lg).mean()),
                                'acc_own': float((pr == g).mean()), 'acc_log': float((lg == g).mean())})
                    # L: own primal ridge per class (b72_check.own_ridge_class), own quantizer, own read
                    f, alpha = ch_l['cfg']
                    ns = max(1, int(math.floor(f * M)))
                    X = fk[order[:ns]] * cd[0][None, :]
                    yv = vals[order[:ns]]
                    ncl = 16 if V == 16 else 1
                    lam = alpha * float((X ** 2).sum()) / ns / ncl
                    Sd = np.zeros(d)
                    Hm = K71.sylv16()
                    for cl in range(ncl):
                        if V == 16 and cl == 0:
                            continue
                        cols = np.arange(cl, d, ncl)
                        y = Hm[yv, cl] if V == 16 else np.where(yv == 0, 1.0, -1.0)
                        Sd[cols] = K72.own_ridge_class(X[:, cols], y, lam)
                    Sq = K71.own_q_store(Sd, q, ('B74Q', sseed, kind, d, q, ns, alpha))
                    pr = (fq @ (cd * Sq).T).argmax(1)
                    lg = np.asarray(ti['L' + kind, M, V, rep, b, tuple(ch_l['cfg'])])
                    out.append({'arm': 'L' + kind, 'cell': cell, 'agree': float((pr == lg).mean()),
                                'acc_own': float((pr == g).mean()), 'acc_log': float((lg == g).mean())})
                    print(enc, out[-2:], flush=True)
                # (B) labels: STORE's and LAMc's label predictions by own code
                lab = json.load(gzip.open(f'{OUT}/jobs/{enc}_LABS_M{M}_V{V}_r{rep}.json.gz', 'rt'))
                nq = lab['nq_alias']
                b53 = json.load(open(f'{ROOT}/results/b53_choice_{enc}.json'))['choice'][f'{cell}|ideal|VSA']['cfg']
                prep, fmap, q, fsh = b53
                P = S['preps'][prep]
                d = C.vsa_dim(b * M, q, 'ideal')
                fk = LS.own_feats_R(enc, prep, fmap, P(st['keys']), d)
                cd = K71.own_codes(enc, prep, fmap, V, d)
                ns = max(1, int(math.floor(fsh * M)))
                Sb = (fk[order[:ns]] * cd[vals[order[:ns]]]).sum(0)
                Sq = K71.own_q_store(Sb, q, ('Q', sseed, d, q, ns))
                pr = (fk @ (cd * Sq).T).argmax(1)
                lg = np.asarray(lab['pred'][f'STORE|{b}'][nq:])
                out.append({'arm': 'STORE-label', 'cell': cell, 'agree': float((pr == lg).mean()),
                            'acc_own': float((pr == vals).mean()), 'acc_log': float((lg == vals).mean())})
                out.append(lamc_label(enc, S, st, M, V, b, rep))
                print(enc, out[-2:], flush=True)
    json.dump(out, open(f'{RES}/b74_rederive_{enc}.json', 'w'), indent=1)
    return out


_LAMC_PREP = {}


def lamc_label(enc, S, st, M, V, b, rep):
    """Own float64 LAMc (b60_check.rederive's algebra, own preps and features) read with the label codes."""
    import b60_run as L60
    ch = json.load(open(f'{ROOT}/results/b60_choice_{enc}.json'))['choice']
    cfg = ch[f'{M}|{V}|{b}|ideal|LAMC']['cfg']
    scale, t, pr_, feat, alpha, q = cfg
    jf = f'{OUT}/lab60/jobs/{enc}_m0_test_M{M}_V{V}_r{rep}_{pr_}.json.gz'
    J = json.load(gzip.open(jf, 'rt'))
    row = [x for x in J['rows'] if x['cfg'] == cfg and x['bpf'] == b and x['acct'] == 'ideal'][0]
    nq = len(J['qgold']) - st['M']
    p = row['p']
    if enc not in _LAMC_PREP:  # own float64 preps, computed once per encoder
        Tr = np.asarray(S['Tr'], np.float64)
        mu_ = Tr.mean(0)
        w, U = np.linalg.eigh(np.cov((Tr - mu_).T))
        w = np.maximum(w, 0)
        _LAMC_PREP.clear()
        _LAMC_PREP[enc] = (mu_, U[:, np.argsort(-w)], (U / np.sqrt(w + 0.1 * w.mean())) @ U.T, Tr.shape[1])
    mu, U0, Wz, n = _LAMC_PREP[enc]

    def prep(X):
        Y = np.asarray(X, np.float64) - mu
        if pr_ == 'whiten':
            Y = Y @ Wz
        return Y / np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
    vals = np.asarray(st['vals'])
    Yk = prep(st['keys'])
    if feat == 'pca':
        G = U0[:, :p]
    else:
        import b32_common as C
        G = np.hstack([np.random.default_rng(C.seed_of('LAMG', enc, pr_, j0)).standard_normal(
            (n, 8192), dtype=np.float32)[:, :min(8192, p - j0)].astype(np.float64) / math.sqrt(n)
            for j0 in range(0, p, 8192)])
    Fk = Yk @ G
    c = V - 1 if t == 'V-1' else V
    Tm = np.eye(V)[vals] - 1.0 / V
    if t == 'V-1':
        Tm = Tm[:, 1:]
    K = Fk @ Fk.T
    A = np.linalg.solve(K + alpha * np.trace(K) / st['M'] * np.eye(st['M']), Tm)
    W = Fk.T @ A
    import b66_lsfc as LS
    if scale == 'col':
        Wq = np.zeros_like(W)
        for j0 in range(0, p, 8192):
            for k in range(c):
                Wq[j0:j0 + 8192, k] = LS.own_quant(W[j0:j0 + 8192, k], q)
    else:
        fl = W.T.reshape(-1)
        fq = np.concatenate([LS.own_quant(fl[i:i + 8192], q) for i in range(0, len(fl), 8192)])
        Wq = fq.reshape(c, p).T
    Sc = Fk @ Wq
    if t == 'V-1':
        Sc = np.concatenate([-Sc.sum(1, keepdims=True), Sc], 1)
    pred = Sc.argmax(1)
    lg = np.asarray(row['pred'][nq:])
    del L60
    return {'arm': 'LAMC-label', 'cell': f'{M}|{V}|{b}', 'agree': float((pred == lg).mean()),
            'acc_own': float((pred == vals).mean()), 'acc_log': float((lg == vals).mean())}


# ----------------------------------------------------------------------------------------------- reproduction (A)

def repro(encs):
    """HP's dev rows against B60's VSAPc matched-filter dev rows (same configuration); HT's test predictions against
    B71's VSAPR test predictions where the configurations coincide."""
    import b71_verdict as V71
    out = {'hp_dev': [0, 0, 0.0], 'ht_test': [0, 0, 0]}
    for enc in encs:
        ch = json.load(open(f'{RES}/b74_choice_{enc}.json'))['choice']
        # B60 VSAPc dev rows: devp files {(M, V, rep, bpf, cfg): acc}
        old = {}
        for p in ('center', 'whiten'):
            for f in ('plin', 'psgn'):
                fn = f'{ROOT}/b60_out/vsapc/{enc}/devp_{enc}_{p}_{f}.jsonl'
                if not os.path.exists(fn):
                    continue
                for l in open(fn):
                    r = json.loads(l)
                    for x in r['rows']:
                        if x['acct'] == 'ideal':
                            old[r['M'], r['V'], r['rep'], x['bpf'], tuple(x['cfg'])] = x['acc']
        jd = os.path.join(OUT, 'jobs')
        for fn in sorted(os.listdir(jd)):
            if not fn.startswith(f'{enc}_FXP_dev_'):
                continue
            r = json.load(gzip.open(os.path.join(jd, fn), 'rt'))
            s = r['spec']
            for rep, rr in r['reps'].items():
                for x in rr['rows']:
                    if x['w'] != 'H':
                        continue
                    k = (s['M'], s['V'], int(rep), x['bpf'], (s['prep'], s['fmap'], x['q'], x['cfg'][0]))
                    if k in old:
                        out['hp_dev'][1] += 1
                        dd = abs(old[k] - x['acc'])
                        out['hp_dev'][0] += dd < 1e-12
                        out['hp_dev'][2] = max(out['hp_dev'][2], dd)
        ti = V71.test_index(enc)
        ch71 = V71.choice(enc)
        import b74_verdict as VV
        fx = VV.fx_index(enc)
        for M in M_GRID:
            for V in VN:
                for b in BPF:
                    cell = f'{M}|{V}|{b}'
                    c71 = ch71[f'{cell}|VSAPR']['cfg']
                    c = ch[f'{cell}|HT']
                    if c['q'] == c71[3] and c['cfg'][0] == c71[4]:
                        for rep in range({256: 20, 1024: 5}[M]):
                            a = ti['VSAPR', M, V, rep, b, tuple(c71)]
                            bb = fx['HT', M, V, rep, b, tuple(c['cfg'])]
                            out['ht_test'][1] += 1
                            out['ht_test'][0] += a == bb
                            out['ht_test'][2] += sum(x != y for x, y in zip(a, bb))
        print(enc, out, flush=True)
    json.dump(out, open(f'{RES}/b74_check_repro.json', 'w'), indent=1)


# ----------------------------------------------------------------------------------------------- determinism

def determ(d2):
    """Every job file in d2/jobs (and its referenced engine file) against b74_out's: predictions identical."""
    out = {'n': 0, 'same': 0, 'diff': []}

    def preds(o, base):
        if 'reps' in o:
            return {(rep, x['w'], x['bpf'], tuple(x['cfg'])): x['pred'] for rep, rr in o['reps'].items()
                    for x in rr['rows']}
        if 'file' in o:
            fn = os.path.join(base, o['file'])
            if fn.endswith('.jsonl.gz'):
                return {i: json.loads(l)['pred'] for i, l in enumerate(gzip.open(fn, 'rt'))}
            j = json.load(gzip.open(fn, 'rt'))
            return {(x.get('arm'), tuple(x['cfg']), x['bpf'], x.get('acct')): x['pred'] for x in j['rows']}
        return o['pred']
    for fn in sorted(os.listdir(os.path.join(d2, 'jobs'))):
        a = json.load(gzip.open(os.path.join(d2, 'jobs', fn), 'rt'))
        b = json.load(gzip.open(os.path.join(OUT, 'jobs', fn), 'rt'))
        out['n'] += 1
        pa, pb = preds(a, d2), preds(b, OUT)
        if pa == pb:
            out['same'] += 1
        else:
            out['diff'].append(fn)
    out['pass'] = out['same'] == out['n']
    json.dump(out, open(f'{RES}/b74_check_determ.json', 'w'), indent=1)
    print(out)


def detplan(plans, out):
    """The determinism sample: the first test job (sorted by job name) of every (encoder, job type) in the plans."""
    import b74_run as R74
    first = {}
    for pf in plans:
        for l in open(pf):
            s = json.loads(l)
            if s.get('split', 'test') != 'test' and s['arm'] == 'FX':
                continue
            t = s['arm'] + (s.get('kind', '') if s['arm'] == 'FX' else '')
            nm = R74.job_name(s)
            k = (s['enc'], t)
            if k not in first or nm < first[k][0]:
                first[k] = (nm, s)
    with open(out, 'w') as fo:
        for k in sorted(first):
            fo.write(json.dumps(first[k][1]) + '\n')
    print(len(first), 'jobs ->', out)


def report():
    L = ['B74 checks', '']
    for nm in ('scorer', 'bits', 'repro', 'determ'):
        fn = f'{RES}/b74_check_{nm}.json'
        if os.path.exists(fn):
            L.append(f'{nm}: {json.dumps(json.load(open(fn)))[:1500]}')
    rd = []
    for e in PANEL:
        fn = f'{RES}/b74_rederive_{e}.json'
        if os.path.exists(fn):
            rd += [(e, x) for x in json.load(open(fn))]
    if rd:
        by = {}
        for e, x in rd:
            by.setdefault(x['arm'], []).append(x['agree'])
        L.append('rederive (agreement with the logged predictions, bar 0.99): ' + '; '.join(
            f'{a} {sum(v >= 0.99 for v in vs)}/{len(vs)} (min {min(vs):.4f})' for a, vs in sorted(by.items())))
        L += [f'  below bar: {e} {x}' for e, x in rd if x['agree'] < 0.99]
    open(f'{RES}/b74_check.txt', 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    a = sys.argv[1:]
    if a[0] == 'scorer':
        scorer(encs_arg(a))
    elif a[0] == 'bits':
        bits(encs_arg(a))
    elif a[0] == 'rederive':
        rederive(a[1])
    elif a[0] == 'repro':
        repro(encs_arg(a))
    elif a[0] == 'determ':
        determ(a[1])
    elif a[0] == 'detplan':
        detplan(a[1].split(','), a[2])
    elif a[0] == 'report':
        report()
    else:
        raise SystemExit(__doc__)
