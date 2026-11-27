"""B57 (prereg/B57_pca_store_maps_online_prereg.md): the store on principal-component features (VSAP), map variance
(three new random maps on three encoders), online decorrelating writes (normalised delta rule NLMS; recursive least
squares RLS), on B53 / B56's panel, splits, stores, budgets and accounting.

python3 b57_run.py selftest
python3 b57_run.py ENC devp PREP:FMAP          # VSAP matched-filter grid part (fmap plin | psgn), dev stores
python3 b57_run.py ENC choosep                 # B32's rule over VSAP's 48 -> results/b57_choicep_ENC.json
python3 b57_run.py ENC decp SPLIT M V          # LMMSE / retrained reads on VSAP's chosen bundle (test: chosen only)
python3 b57_run.py ENC devm MAP PREP:FMAP      # map MAP (1..3): B53's 72-configuration grid with the map's seeds
python3 b57_run.py ENC choosem MAP             # B32's rule -> results/b57_choicem_ENC_mMAP.json
python3 b57_run.py ENC decm MAP SPLIT M V      # B56 engine (LM / RT / LR) on the map's choice
python3 b57_run.py ENC lamm MAP SPLIT          # LAM's rp configurations with the map's projection (test: chosen only)
python3 b57_run.py ENC onl SPLIT               # NLMS on LAM's features (rp, pca), every configuration / chosen only
python3 b57_run.py ENC choose                  # all B57 choices -> results/b57_choice_ENC.json
python3 b57_run.py ENC rlstest                 # RLS (= ridge on pca features) chosen configurations, test stores
python3 b57_run.py ENC merge                   # B56's merged records + B57 arms -> b57_out/fam/ENC/all57_*
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
import b56_run as E  # noqa: E402

OUT = os.environ.get('B57_OUT', os.path.join(ROOT, 'b57_out'))
PANEL = R53.PANEL
MAP_ENCS = ['qwen8', 'e5', 'minilm']
MAPS = [1, 2, 3]
M_GRID, VN, BPF = [256, 1024], [2, 16], [1, 2, 4, 8, 16, 32, 64]
ACCT32 = ['ideal', 'raw']
PREPS = ['center', 'whiten']
PFMAPS = ['plin', 'psgn']
PARTS_P = [(p, f) for p in PREPS for f in PFMAPS]
LAM_ALPHAS = [1e-4, 1e-2, 1.0, 100.0]  # B49's
MUS = [1.0, 0.5, 0.25]
CH = R.CH


def tag(enc, m):
    return enc if m == 0 else f'{enc}#b57m{m}'


def odir(enc):
    d = os.path.join(OUT, enc)
    os.makedirs(d, exist_ok=True)
    return d


def setup(enc):
    pool, preps = R53.setup(enc)
    Tr = pool.train_matrix()
    lfs = {p: B.LamFeat(enc, p, preps[p](Tr)) for p in PREPS}
    return pool, preps, Tr, lfs


# ----------------------------------------------------------------------------------------------- VSAP features

class FeatP(R53.Feat53):
    """The store on LAM's principal directions: phi(y) = U^T (y - mu) ('plin') or its signs ('psgn'), U, mu =
    b49_common.LamFeat's (the training split's principal directions of the prep's codes, descending variance); the
    first d <= n coordinates. Value codes: B53's, seeded as the lin / sgn store's."""

    def __init__(self, enc, prep, fmap, n, device, lf):
        super().__init__(enc, prep, 'lin' if fmap == 'plin' else 'sgn', n, device)
        t = self.torch
        self.fmap = fmap
        U = np.zeros((n, CH), np.float32)
        U[:, :n] = lf.U
        self.Ut = t.from_numpy(U)
        self.mut = t.from_numpy(lf.mu.astype(np.float32))
        self.nmax = n

    def block(self, j0):
        assert j0 == 0, j0
        _, codes = super().block(j0)
        return [self.Ut], codes

    def phi(self, Y, Gs, w):
        assert w <= self.nmax or w == CH
        P = (Y - self.mut) @ Gs[0][:, :w]
        return P if self.fmap == 'plin' else self.torch.where(P >= 0, 1.0, -1.0)


def dcap(fmap, n):
    if fmap in PFMAPS:
        return lambda B_, q, acct: min(C.vsa_dim(B_, q, acct), n)
    return lambda B_, q, acct: C.vsa_dim(B_, q, acct)


def make_feat(seedtag, prep, fmap, n, V, lfs):
    f = FeatP(seedtag, prep, fmap, n, 'cpu', lfs[prep]) if fmap in PFMAPS else E.make_feat(seedtag, prep, fmap, n, V)
    f.V = V
    return f


# ----------------------------------------------------------------------------------------------- MF dev grids

def mf_dev(enc, m, part, out):
    prep, fmap = part.split(':')
    pool, preps, Tr, lfs = setup(enc)
    n = pool.X.shape[1]
    tg = tag(enc, m)
    done = set()
    if os.path.exists(out):
        done = {(r['M'], r['V'], r['rep']) for r in map(json.loads, open(out))}
    fo = open(out, 'a')
    dfun = dcap(fmap, n)
    for M in M_GRID:
        for V in VN:
            feat = make_feat(tg, prep, fmap, n, V, lfs)
            for rep in range(R.DEV_REPS[M]):
                if (M, V, rep) in done:
                    continue
                t0 = time.time()
                st = C.make_store(pool, 'dev', M, V, rep)
                feat.V = V
                cfgs = [((bpf, acct, q, f), dfun(bpf * M, q, acct), q, max(1, int(math.floor(f * M))))
                        for bpf in BPF for acct in ACCT32 for q in R.GRID['q'] for f in R.GRID['f']]
                P = preps[prep]
                pr = R.vsa_grid(P(st['keys']), st['vals'], P(st['queries']), V, feat, cfgs, R.store_seed(tg, st),
                                R.store_order(st))
                g = st['qgold']
                rows = [{'arm': 'VSA', 'cfg': [prep, fmap, q, f], 'bpf': bpf, 'acct': acct,
                         'acc': float((p == g).mean())} for (bpf, acct, q, f), p in pr.items()]
                fo.write(json.dumps({'enc': enc, 'map': m, 'M': M, 'V': V, 'rep': rep, 'nq': len(g), 'rows': rows}) + '\n')
                fo.flush()
                print(f'mfdev {enc} m{m} {part} M={M} V={V} rep={rep} {time.time() - t0:.0f}s', flush=True)


def b32_choose(files, fmaps, out):
    acc = {}
    for fn in files:
        for l in open(fn):
            r = json.loads(l)
            for x in r['rows']:
                acc.setdefault((r['M'], r['V'], x['bpf'], x['acct'], tuple(x['cfg'])), {})[r['rep']] = x['acc']
    best = {}
    for (M, V, bpf, acct, cfg), d in acc.items():
        assert len(d) == R.DEV_REPS[M], (M, V, bpf, acct, cfg, len(d))
        best.setdefault((M, V, bpf, acct), []).append((float(np.mean(list(d.values()))), cfg))
    ch = {}
    key = (lambda c: (PREPS.index(c[0]), fmaps.index(c[1]), R.GRID['q'].index(c[2]), R.GRID['f'].index(c[3])))
    for (M, V, bpf, acct), lst in best.items():
        top = max(a for a, _ in lst)
        tied = sorted([c for a, c in lst if a >= top - 0.001], key=key)
        ch[f'{M}|{V}|{bpf}|{acct}|VSA'] = {'cfg': list(tied[0]), 'dev_acc': [a for a, c in lst if c == tied[0]][0],
                                           'n_tied': len(tied), 'n_cfg': len(lst)}
    json.dump({'choice': ch}, open(out, 'w'), indent=1, sort_keys=True)
    print(len(ch), 'choices ->', out)


# ----------------------------------------------------------------------------------------------- engine wrapper

def plan(chmf, M, V, n, arms, select=None):
    """B56's plan_cells with the d rule of the chosen feature map (VSAP: d <= n) and a subset of arms."""
    groups = {}
    for bpf in BPF:
        for acct in ACCT32:
            prep, fmap, q53, f = chmf[f'{M}|{V}|{bpf}|{acct}|VSA']['cfg']
            ns = max(1, int(math.floor(f * M)))
            dfun = dcap(fmap, n)
            g = groups.setdefault((prep, fmap), {'LM': [], 'RT': [], 'LR': []})
            la = 'ideal' if acct == 'ideal' else 'real'
            for q in E.QS:
                d = dfun(bpf * M, q, acct)
                if 'LM' in arms:
                    for a in E.ALPHA_LM:
                        k = ('LM', bpf, acct, q, a)
                        if select is None or k in select:
                            g['LM'].append((k, d, q, ns, a))
                if 'RT' in arms:
                    for e in E.EPOCHS:
                        for eta in E.ETAS:
                            k = ('RT', bpf, acct, q, f'{e}:{eta}')
                            if select is None or k in select:
                                g['RT'].append((k, d, q, ns, (e, eta)))
                if 'LR' in arms:
                    for a in LAM_ALPHAS:
                        k = ('LR', bpf, acct, q, a)
                        if select is None or k in select:
                            g['LR'].append((k, B.lam_p(bpf * M, V - 1, q, la, E.P_MAX), a, q, la))
    return groups


def engine(enc, m, split, M, V, chmf, arms, select, fn):
    M, V = int(M), int(V)
    if os.path.exists(fn):
        print('exists', fn)
        return
    pool, preps, Tr, lfs = setup(enc)
    n = pool.X.shape[1]
    tg = tag(enc, m)
    groups = plan(chmf, M, V, n, arms, select)
    reps = (R.DEV_REPS if split == 'dev' else R.TEST_REPS)[M]
    recs = {rep: {'enc': enc, 'map': m, 'M': M, 'V': V, 'rep': rep, 'pred': {}, 'info': {}} for rep in range(reps)}
    for (prep, fmap), g in groups.items():
        if not (g['LM'] or g['RT'] or g['LR']):
            continue
        t0 = time.time()
        feat = make_feat(tg, prep, fmap, n, V, lfs)
        P = preps[prep]
        dl = sorted({x[1] for x in g['LM']})
        F = E.train_feats(feat, P, Tr, tg, prep, fmap, V, max(dl + [1]))
        decs = E.lm_decoders(F, V, dl)
        print(f'engine {enc} m{m} {split} M={M} V={V} {prep}:{fmap} decoders {len(dl)} d {time.time() - t0:.0f}s',
              flush=True)
        for rep in range(reps):
            st = C.make_store(pool, split, M, V, rep)
            res = E.run_store_group(tg, feat, P, st, F, decs, g, V)
            rc = recs[rep]
            rc['qgold'] = st['qgold'].tolist()
            for key, (p, info) in res.items():
                k = '|'.join(map(str, key))
                rc['pred'][k] = np.asarray(p).tolist()
                rc['info'][k] = info + [prep, fmap]
        del feat, F, decs
    with gzip.open(fn + '.tmp', 'wt') as fo:
        for rep in range(reps):
            fo.write(json.dumps(recs[rep]) + '\n')
    os.replace(fn + '.tmp', fn)


def select_from(ch, M, V, arms):
    s = set()
    for bpf in BPF:
        for acct in ACCT32:
            for arm in arms:
                c = ch[f'{M}|{V}|{bpf}|{acct}|{arm}']['cfg']
                s.add((arm, bpf, acct, c[0], c[1]))
    return s


# ----------------------------------------------------------------------------------------------- LAM rp on new maps

def lam_cfgs(feats):
    return [(ft, a, q, 'onehot') for ft in feats for a in LAM_ALPHAS for q in (1, 2, 4)]


def lamm(enc, m, split):
    """LAM's rp configurations with the map's projection blocks (seed tag of map m); test: the per-map choice."""
    pool, preps, Tr, _ = setup(enc)
    tg = tag(enc, m)
    fn = os.path.join(odir(enc), f'lamm_{split}_{enc}_m{m}.jsonl.gz')
    if os.path.exists(fn):
        return
    ch = json.load(open(f'{ROOT}/results/b57_choice_{enc}.json'))['choice'] if split == 'test' else None
    lfs = {p: B.LamFeat(tg, p, preps[p](Tr)) for p in PREPS}
    with gzip.open(fn + '.tmp', 'wt') as fo:
        for M in M_GRID:
            for V in VN:
                for rep in range((R.DEV_REPS if split == 'dev' else R.TEST_REPS)[M]):
                    t0 = time.time()
                    st = C.make_store(pool, split, M, V, rep)
                    rec = {'enc': enc, 'map': m, 'M': M, 'V': V, 'rep': rep, 'qgold': st['qgold'].tolist(), 'pred': {},
                           'cfg': {}}
                    for prep in PREPS:
                        if split == 'dev':
                            cfgs = [c for c in lam_cfgs(['rp'])]
                            want = None
                        else:
                            want = {}
                            for b in BPF:
                                for a in ('ideal', 'real'):
                                    c = ch[f'{M}|{V}|{b}|{a}|LAMm{m}']['cfg']
                                    if c[0] == prep:
                                        want.setdefault(tuple(c[1:]), []).append((b, a))
                            cfgs = list(want)
                        if not cfgs:
                            continue
                        P = preps[prep]
                        o = B.run_lam(st, P(st['keys']), P(st['queries']), lfs[prep], cfgs, BPF, ['ideal', 'real'],
                                      C.seed_of('b49', tg, split, M, V, rep))
                        for (c1, b, a), (p, bits) in o.items():
                            if want is not None and (b, a) not in want[tuple(c1)]:
                                continue
                            k = f'{prep}|{"|".join(map(str, c1))}|{b}|{a}'
                            rec['pred'][k] = np.asarray(p).tolist()
                            rec['cfg'][k] = bits
                    fo.write(json.dumps(rec) + '\n')
                    print(f'lamm {enc} m{m} {split} M={M} V={V} rep={rep} {time.time() - t0:.0f}s', flush=True)
    os.replace(fn + '.tmp', fn)


# ----------------------------------------------------------------------------------------------- NLMS / RLS

def nlms(Fk, Fq, vals, order, V, cfgs, seed):
    """One pass of the normalised delta rule over the M facts in store order on LAM's features and targets.
    cfgs: list of (key, p, mu, q, acct). W (p x c) <- W + mu f (y - f^T W)^T / (f^T f); then B49's quantizer."""
    M = Fk.shape[0]
    c = V - 1
    Tt, D = B.value_targets(V, 'onehot', seed)
    Y = Tt[vals]
    default = int(np.bincount(vals, minlength=V).argmax())
    out = {}
    Ws = {}
    pm = max([x[1] for x in cfgs] + [1])
    X64 = Fk[:, :pm].astype(np.float64)
    for key, p, mu, q, acct in cfgs:
        if p <= 0:
            out[key] = (np.full(Fq.shape[0], default), 0.0)
            continue
        if (p, mu) not in Ws:
            W = np.zeros((p, c))
            X = X64[:, :p]
            nn = np.einsum('ij,ij->i', X, X)
            for i in order:
                e = Y[i] - X[i] @ W
                W += (mu / max(nn[i], 1e-12)) * np.outer(X[i], e)
            Ws[p, mu] = W
        W = Ws[p, mu]
        s = np.zeros((Fq.shape[0], c))
        for j0 in range(0, p, CH):
            ww = min(CH, p - j0)
            Wq = B.quant_cols(W[j0:j0 + ww], q, np.random.default_rng(C.seed_of('NLMSQ', seed, key, j0)))
            s += Fq[:, j0:j0 + ww].astype(np.float64) @ Wq
        out[key] = ((s @ D).argmax(1), float(B.lam_bits(p, c, q, acct)))
    return out


def lam_feats(lf, Y, ft, pmax):
    out = np.empty((Y.shape[0], pmax), np.float32)
    for j0 in range(0, pmax, CH):
        w = min(CH, pmax - j0)
        out[:, j0:j0 + w] = lf.feats(Y, ft, j0, w)
    return out


def onl(enc, split):
    pool, preps, Tr, lfs = setup(enc)
    fn = os.path.join(odir(enc), f'onl_{split}_{enc}.jsonl.gz')
    if os.path.exists(fn):
        return
    ch = json.load(open(f'{ROOT}/results/b57_choice_{enc}.json'))['choice'] if split == 'test' else None
    with gzip.open(fn + '.tmp', 'wt') as fo:
        for M in M_GRID:
            for V in VN:
                for rep in range((R.DEV_REPS if split == 'dev' else R.TEST_REPS)[M]):
                    t0 = time.time()
                    st = C.make_store(pool, split, M, V, rep)
                    order = R.store_order(st)
                    rec = {'enc': enc, 'M': M, 'V': V, 'rep': rep, 'qgold': st['qgold'].tolist(), 'pred': {}, 'bits': {}}
                    for prep in PREPS:
                        P = preps[prep]
                        Yk, Yq = P(st['keys']), P(st['queries'])
                        for ft in ('rp', 'pca'):
                            cfgs = []
                            for b in BPF:
                                for a in ('ideal', 'real'):
                                    for mu in MUS:
                                        for q in (1, 2, 4):
                                            key = (prep, ft, mu, q, b, a)
                                            if split == 'test' and ch[f'{M}|{V}|{b}|{a}|NLMS']['cfg'] != [prep, ft, mu, q]:
                                                continue
                                            cfgs.append((key, B.lam_p(b * M, V - 1, q, a, lfs[prep].pmax(ft)), mu, q, a))
                            if not cfgs:
                                continue
                            pm = max([x[1] for x in cfgs] + [1])
                            Fk, Fq = lam_feats(lfs[prep], Yk, ft, pm), lam_feats(lfs[prep], Yq, ft, pm)
                            o = nlms(Fk, Fq, st['vals'], order, V, cfgs, C.seed_of('b57n', enc, st['split'], M, V, rep))
                            for key, (p, bits) in o.items():
                                k = '|'.join(map(str, key))
                                rec['pred'][k] = np.asarray(p).tolist()
                                rec['bits'][k] = bits
                    fo.write(json.dumps(rec) + '\n')
                    print(f'onl {enc} {split} M={M} V={V} rep={rep} {time.time() - t0:.0f}s', flush=True)
    os.replace(fn + '.tmp', fn)


def rls_recursive(F, Y, lam):
    """Genuine recursive least squares (P0 = I / lam), used by the check: returns W after M updates."""
    p = F.shape[1]
    Pm = np.eye(p) / lam
    W = np.zeros((p, Y.shape[1]))
    for i in range(F.shape[0]):
        f = F[i]
        Pf = Pm @ f
        k = Pf / (1.0 + f @ Pf)
        W += np.outer(k, Y[i] - f @ W)
        Pm -= np.outer(k, Pf)
    return W


def rlstest(enc):
    """RLS on LAM's pca features = the ridge solution with lam = alpha * tr(K) / M (computed by b49's batch solve);
    the chosen configuration of every cell on the test stores."""
    pool, preps, Tr, lfs = setup(enc)
    ch = json.load(open(f'{ROOT}/results/b57_choice_{enc}.json'))['choice']
    fn = os.path.join(odir(enc), f'rls_test_{enc}.jsonl.gz')
    if os.path.exists(fn):
        return
    with gzip.open(fn + '.tmp', 'wt') as fo:
        for M in M_GRID:
            for V in VN:
                for rep in range(R.TEST_REPS[M]):
                    st = C.make_store(pool, 'test', M, V, rep)
                    rec = {'enc': enc, 'M': M, 'V': V, 'rep': rep, 'qgold': st['qgold'].tolist(), 'pred': {}, 'cfg': {}}
                    for prep in PREPS:
                        want = {}
                        for b in BPF:
                            for a in ('ideal', 'real'):
                                c = ch[f'{M}|{V}|{b}|{a}|RLS']['cfg']
                                if c[0] == prep:
                                    want.setdefault(tuple(c[1:]), []).append((b, a))
                        if not want:
                            continue
                        P = preps[prep]
                        o = B.run_lam(st, P(st['keys']), P(st['queries']), lfs[prep], list(want), BPF, ['ideal', 'real'],
                                      C.seed_of('b49', enc, 'test', M, V, rep))
                        for (c1, b, a), (p, bits) in o.items():
                            if (b, a) in want[tuple(c1)]:
                                rec['pred'][f'RLS|{b}|{a}'] = np.asarray(p).tolist()
                                rec['cfg'][f'RLS|{b}|{a}'] = [prep] + list(c1) + [bits]
                    fo.write(json.dumps(rec) + '\n')
    os.replace(fn + '.tmp', fn)


# ----------------------------------------------------------------------------------------------- choices

def dec_choose(R_, arms, out, M, V):
    order = {'LM': [(q, a) for q in E.QS for a in E.ALPHA_LM],
             'RT': [(q, f'{e}:{eta}') for q in E.QS for e in E.EPOCHS for eta in E.ETAS],
             'LR': [(q, a) for a in LAM_ALPHAS for q in E.QS]}
    for bpf in BPF:
        for acct in ACCT32:
            for arm in arms:
                lst = []
                for (q, x) in order[arm]:
                    k = '|'.join(map(str, (arm, bpf, acct, q, x)))
                    a = float(np.mean([np.mean(np.array(r['pred'][k]) == np.array(r['qgold'])) for r in R_]))
                    lst.append((a, (q, x)))
                top = max(a for a, _ in lst)
                tied = [c for a, c in lst if a >= top - 0.001]
                out[f'{M}|{V}|{bpf}|{acct}|{arm}'] = {'cfg': list(tied[0]), 'dev_acc': [a for a, c in lst if c == tied[0]][0]}


def lam_choice(rows_by, M, V, bpf, acct):
    """B49's LAM rule over (prep, ft, alpha, q, onehot) in lam_grid order. rows_by[(cfg)] = list of dev accuracies."""
    grid = [(p, ft, a, q, 'onehot') for p in PREPS for ft in ('rp', 'pca') for a in LAM_ALPHAS for q in (1, 2, 4)]
    lst = []
    for c in grid:
        accs = rows_by.get((M, V, bpf, acct, c))
        if accs is None:
            continue
        assert len(accs) == R.DEV_REPS[M], (c, len(accs))
        lst.append((float(np.mean(list(accs.values()))), c))
    top = max(a for a, _ in lst)
    tied = [c for a, c in lst if a >= top - 0.001]
    return list(tied[0]), [a for a, c in lst if c == tied[0]][0], len(lst)


def choose(enc):
    ch = {}
    acct3 = {'ideal': 'ideal', 'real': 'raw'}
    # VSAP: MF choice (results/b57_choicep), LM / RT on it, STAR rule
    chp = json.load(open(f'{ROOT}/results/b57_choicep_{enc}.json'))['choice']
    for k, v in chp.items():
        ch[k.replace('|VSA', '|PMF')] = v
    for M in M_GRID:
        for V in VN:
            R_ = [json.loads(l) for l in gzip.open(os.path.join(odir(enc), f'decp_dev_{enc}_M{M}_V{V}.jsonl.gz'), 'rt')]
            tmp = {}
            dec_choose(R_, ['LM', 'RT'], tmp, M, V)
            for k, v in tmp.items():
                ch[k.replace('|LM', '|PLM').replace('|RT', '|PRT')] = v
            for bpf in BPF:
                for acct in ACCT32:
                    lm, rt = tmp[f'{M}|{V}|{bpf}|{acct}|LM'], tmp[f'{M}|{V}|{bpf}|{acct}|RT']
                    ch[f'{M}|{V}|{bpf}|{acct}|PSTAR'] = {'fam': 'PRT' if rt['dev_acc'] > lm['dev_acc'] + 0.001 else 'PLM',
                                                         'dev': {'PLM': lm['dev_acc'], 'PRT': rt['dev_acc']}}
    # maps: MF, LM / RT / LR, STAR, per-map LAM (rp re-run + B49's pca dev rows), per-map BEST
    b49 = {}
    for l in open(f'{ROOT}/b49_out/fam/{enc}/dev_{enc}.jsonl'):
        r = json.loads(l)
        if r['V'] not in VN:
            continue
        for x in r['rows']:
            if x['fam'] == 'LAM':
                b49.setdefault((r['M'], r['V'], x['bpf'], x['acct'], tuple(x['cfg'])), {})[r['rep']] = x['acc']
    ch49 = json.load(open(f'{ROOT}/results/b49_choice_{enc}.json'))['choice']
    # NLMS dev choice; RLS = LAM restricted to pca rows
    if os.path.exists(os.path.join(odir(enc), f'onl_dev_{enc}.jsonl.gz')):
        Ro = [json.loads(l) for l in gzip.open(os.path.join(odir(enc), f'onl_dev_{enc}.jsonl.gz'), 'rt')]
        for M in M_GRID:
            for V in VN:
                RR = [r for r in Ro if r['M'] == M and r['V'] == V]
                assert len(RR) == R.DEV_REPS[M]
                for b in BPF:
                    for a in ('ideal', 'real'):
                        lst = []
                        for prep in PREPS:
                            for ft in ('rp', 'pca'):
                                for mu in MUS:
                                    for q in (1, 2, 4):
                                        k = '|'.join(map(str, (prep, ft, mu, q, b, a)))
                                        lst.append((float(np.mean([np.mean(np.array(r['pred'][k]) == np.array(r['qgold']))
                                                                   for r in RR])), [prep, ft, mu, q]))
                        top = max(x for x, _ in lst)
                        c = [cc for x, cc in lst if x >= top - 0.001][0]
                        ch[f'{M}|{V}|{b}|{a}|NLMS'] = {'cfg': c, 'dev_acc': [x for x, cc in lst if cc == c][0]}
                        pca_rows = {k: v for k, v in b49.items() if k[4][1] == 'pca'}
                        c, acc, ncfg = lam_choice(pca_rows, M, V, b, a)
                        ch[f'{M}|{V}|{b}|{a}|RLS'] = {'cfg': [c[0], c[1], c[2], c[3], c[4]], 'dev_acc': acc, 'n_cfg': ncfg}
    if enc in MAP_ENCS:
        for m in MAPS:
            chm = json.load(open(f'{ROOT}/results/b57_choicem_{enc}_m{m}.json'))['choice']
            for k, v in chm.items():
                ch[k.replace('|VSA', f'|MFm{m}')] = v
            Rl = [json.loads(l) for l in gzip.open(os.path.join(odir(enc), f'lamm_dev_{enc}_m{m}.jsonl.gz'), 'rt')]
            rows = {k: dict(v) for k, v in b49.items() if k[4][1] == 'pca'}
            for r in Rl:
                g = np.array(r['qgold'])
                for k, p in r['pred'].items():
                    prep, ft, a, q, t, b, ac = k.split('|')
                    c = (prep, ft, float(a), int(q), t)
                    rows.setdefault((r['M'], r['V'], int(b), ac, c), {})[r['rep']] = float(np.mean(np.array(p) == g))
            for M in M_GRID:
                for V in VN:
                    R_ = [json.loads(l) for l in gzip.open(os.path.join(odir(enc), f'decm_dev_{enc}_m{m}_M{M}_V{V}.jsonl.gz'),
                                                           'rt')]
                    tmp = {}
                    dec_choose(R_, ['LM', 'RT', 'LR'], tmp, M, V)
                    for k, v in tmp.items():
                        ch[k + f'm{m}'] = v
                    for bpf in BPF:
                        for acct in ACCT32:
                            lm, rt = tmp[f'{M}|{V}|{bpf}|{acct}|LM'], tmp[f'{M}|{V}|{bpf}|{acct}|RT']
                            ch[f'{M}|{V}|{bpf}|{acct}|STARm{m}'] = {
                                'fam': 'RT' if rt['dev_acc'] > lm['dev_acc'] + 0.001 else 'LM'}
                    for bpf in BPF:
                        for a in ('ideal', 'real'):
                            c, acc, ncfg = lam_choice(rows, M, V, bpf, a)
                            ch[f'{M}|{V}|{bpf}|{a}|LAMm{m}'] = {'cfg': c, 'dev_acc': acc, 'n_cfg': ncfg}
                            dbf = dict(ch49[f'{M}|{V}|{bpf}|{a}|BEST']['dev_by_fam'])
                            dbf['LAM'] = acc
                            cands = [(dbf[f], -E_ALL10.index(f), f) for f in E_ALL10]
                            ch[f'{M}|{V}|{bpf}|{a}|BESTm{m}'] = {'fam': max(cands)[2], 'dev_by_fam': dbf}
    fn = f'{ROOT}/results/b57_choice_{enc}.json'
    json.dump({'enc': enc, 'choice': ch}, open(fn, 'w'), indent=1, sort_keys=True)
    print(len(ch), 'choices ->', fn)


E_ALL10 = ['IDX', 'SF', 'CM', 'BF', 'FBF', 'SDM', 'LAM', 'LHI', 'LHC', 'LHS']


# ----------------------------------------------------------------------------------------------- merge

def merge(enc):
    ch = json.load(open(f'{ROOT}/results/b57_choice_{enc}.json'))['choice']
    od = os.path.join(OUT, 'fam', enc)
    os.makedirs(od, exist_ok=True)
    amap = {'ideal': 'ideal', 'real': 'raw'}
    on = [json.loads(l) for l in gzip.open(os.path.join(odir(enc), f'onl_test_{enc}.jsonl.gz'), 'rt')]
    rl = [json.loads(l) for l in gzip.open(os.path.join(odir(enc), f'rls_test_{enc}.jsonl.gz'), 'rt')]
    for M in M_GRID:
        for V in VN:
            old = [json.loads(l) for l in gzip.open(f'{ROOT}/b56_out/fam/{enc}/all56_{enc}_M{M}_V{V}.jsonl.gz', 'rt')]
            dp = [json.loads(l) for l in gzip.open(os.path.join(odir(enc), f'decp_test_{enc}_M{M}_V{V}.jsonl.gz'), 'rt')]
            ons = [r for r in on if r['M'] == M and r['V'] == V]
            rls = [r for r in rl if r['M'] == M and r['V'] == V]
            mp = {}
            if enc in MAP_ENCS:
                for m in MAPS:
                    mp[m] = ([json.loads(l) for l in gzip.open(os.path.join(odir(enc), f'decm_test_{enc}_m{m}_M{M}_V{V}.jsonl.gz'), 'rt')],
                             [r for r in map(json.loads, gzip.open(os.path.join(odir(enc), f'lamm_test_{enc}_m{m}.jsonl.gz'), 'rt'))
                              if r['M'] == M and r['V'] == V])
            with gzip.open(f'{od}/all57_{enc}_M{M}_V{V}.jsonl.gz', 'wt') as fo:
                for i, o in enumerate(old):
                    assert dp[i]['rep'] == o['rep'] and dp[i]['qgold'] == o['qgold']
                    assert ons[i]['rep'] == o['rep'] and rls[i]['rep'] == o['rep']
                    o.setdefault('b57', {})
                    for b in BPF:
                        for a in ('ideal', 'real'):
                            a3 = amap[a]
                            for arm, name in (('PLM', 'VSAPL'), ('PRT', 'VSAPT')):
                                c = ch[f'{M}|{V}|{b}|{a3}|{arm}']['cfg']
                                k = '|'.join(map(str, (arm[1:], b, a3, c[0], c[1])))
                                o['pred'][f'{name}|{b}|{a}'] = dp[i]['pred'][k]
                                o['b57'][f'{name}|{b}|{a}'] = dp[i]['info'][k]
                            sf = ch[f'{M}|{V}|{b}|{a3}|PSTAR']['fam']
                            o['pred'][f'VSAP|{b}|{a}'] = o['pred'][f'{"VSAPL" if sf == "PLM" else "VSAPT"}|{b}|{a}']
                            o['b57'][f'VSAP|{b}|{a}'] = o['b57'][f'{"VSAPL" if sf == "PLM" else "VSAPT"}|{b}|{a}'] + [sf]
                            nc = ch[f'{M}|{V}|{b}|{a}|NLMS']['cfg']
                            k = '|'.join(map(str, nc + [b, a]))
                            o['pred'][f'NLMS|{b}|{a}'] = ons[i]['pred'][k]
                            o['b57'][f'NLMS|{b}|{a}'] = nc + [ons[i]['bits'][k]]
                            o['pred'][f'RLS|{b}|{a}'] = rls[i]['pred'][f'RLS|{b}|{a}']
                            o['b57'][f'RLS|{b}|{a}'] = rls[i]['cfg'][f'RLS|{b}|{a}']
                            for m, (dm, lm) in mp.items():
                                assert dm[i]['rep'] == o['rep'] and lm[i]['rep'] == o['rep']
                                sel = {}
                                for arm in ('LM', 'RT', 'LR'):
                                    c = ch[f'{M}|{V}|{b}|{a3}|{arm}m{m}']['cfg']
                                    k = '|'.join(map(str, (arm, b, a3, c[0], c[1])))
                                    sel[arm] = dm[i]['pred'][k]
                                    o['b57'][f'{arm}m{m}|{b}|{a}'] = dm[i]['info'][k]
                                o['pred'][f'VSASm{m}|{b}|{a}'] = sel[ch[f'{M}|{V}|{b}|{a3}|STARm{m}']['fam']]
                                o['pred'][f'LAMRFm{m}|{b}|{a}'] = sel['LR']
                                lc = ch[f'{M}|{V}|{b}|{a}|LAMm{m}']['cfg']
                                o['pred'][f'LAMm{m}|{b}|{a}'] = lm[i]['pred'][f'{lc[0]}|{"|".join(map(str, lc[1:]))}|{b}|{a}']
                                bf = ch[f'{M}|{V}|{b}|{a}|BESTm{m}']['fam']
                                o['pred'][f'BESTm{m}|{b}|{a}'] = o['pred'][f'LAMm{m}|{b}|{a}'] if bf == 'LAM' else \
                                    o['pred'][f'{bf}|{b}|{a}']
                    fo.write(json.dumps(o) + '\n')
    print('merged', enc)


# ----------------------------------------------------------------------------------------------- selftest

def selftest():
    rng = np.random.default_rng(0)
    n, M = 48, 96
    Tr = rng.standard_normal((2000, n)).astype(np.float32)
    Tr /= np.linalg.norm(Tr, axis=1, keepdims=True)
    K = Tr[:M].copy()
    Qy = K + 0.2 * rng.standard_normal((M, n)).astype(np.float32)
    Qy /= np.linalg.norm(Qy, axis=1, keepdims=True)
    lf = B.LamFeat('selftest', 'p', Tr)
    order = rng.permutation(M)
    for V in (2, 16):
        vals = rng.integers(0, V, size=M)
        for fmap in PFMAPS:
            feat = make_feat('selftest', 'p', fmap, n, V, {'p': lf})
            # 1. plin features are LAM's pca features
            Fk = E.feats_np(feat, K, n)
            if fmap == 'plin':
                assert np.allclose(Fk, lf.feats(K, 'pca', 0, n), atol=1e-5)
            cfgs = [((d, q, ns), d, q, ns) for d in (16, n) for q in (1, 2) for ns in (M, 48)]
            ref = R.vsa_grid(K, vals, Qy, V, feat, cfgs, 7, order)
            Fq, Cv = E.feats_np(feat, Qy, n), E.codes_np(feat, V, n)
            for (key, d, q, ns) in cfgs:
                Sq = E.bundle_q(feat, Fk, Cv, vals, order, d, q, ns, 7)
                mine = E.mf_read(Fq, Cv, Sq)
                sc = Fq[:, :d].astype(np.float64) @ (Cv[:, :d].astype(np.float64) * Sq).T
                ii = np.nonzero(mine != ref[key])[0]
                assert np.all(np.abs(sc[ii, mine[ii]] - sc[ii, ref[key][ii]]) <= 1e-5 * np.abs(sc).max()), (V, fmap, key)
    # 2. NLMS with mu = 1 recalls the last-written fact exactly; RLS equals the ridge solution
    X = rng.standard_normal((40, 12))
    Y = rng.standard_normal((40, 3))
    W = rls_recursive(X, Y, 0.5)
    Wr = np.linalg.solve(X.T @ X + 0.5 * np.eye(12), X.T @ Y)
    assert np.allclose(W, Wr, atol=1e-8), np.abs(W - Wr).max()
    vals = rng.integers(0, 2, size=40)
    o = nlms(X.astype(np.float32), X.astype(np.float32), vals, np.arange(40), 2, [('k', 12, 1.0, 4, 'ideal')], 3)
    assert o['k'][0].shape == (40,)
    print('selftest ok')


def main():
    a = sys.argv[1:]
    if a[0] == 'selftest':
        selftest()
        return
    enc, mode = a[0], a[1]
    if mode == 'devp':
        mf_dev(enc, 0, a[2], os.path.join(odir(enc), f'devp_{enc}_{a[2].replace(":", "_")}.jsonl'))
    elif mode == 'choosep':
        b32_choose([os.path.join(odir(enc), f'devp_{enc}_{p}_{f}.jsonl') for p, f in PARTS_P], PFMAPS,
                   f'{ROOT}/results/b57_choicep_{enc}.json')
    elif mode == 'decp':
        split, M, V = a[2], int(a[3]), int(a[4])
        chp = json.load(open(f'{ROOT}/results/b57_choicep_{enc}.json'))['choice']
        sel = None
        if split == 'test':
            ch = json.load(open(f'{ROOT}/results/b57_choice_{enc}.json'))['choice']
            ch2 = {k.replace('|PLM', '|LM').replace('|PRT', '|RT'): v for k, v in ch.items() if '|P' in k}
            sel = select_from(ch2, M, V, ['LM', 'RT'])
        engine(enc, 0, split, M, V, chp, ['LM', 'RT'], sel,
               os.path.join(odir(enc), f'decp_{split}_{enc}_M{M}_V{V}.jsonl.gz'))
    elif mode == 'devm':
        m = int(a[2])
        mf_dev(enc, m, a[3], os.path.join(odir(enc), f'devm_{enc}_m{m}_{a[3].replace(":", "_")}.jsonl'))
    elif mode == 'choosem':
        m = int(a[2])
        b32_choose([os.path.join(odir(enc), f'devm_{enc}_m{m}_{p}_{f}.jsonl') for p, f in R53.PARTS],
                   ['lin', 'sgn', 'sgn3'], f'{ROOT}/results/b57_choicem_{enc}_m{m}.json')
    elif mode == 'decm':
        m, split, M, V = int(a[2]), a[3], int(a[4]), int(a[5])
        chm = json.load(open(f'{ROOT}/results/b57_choicem_{enc}_m{m}.json'))['choice']
        sel = None
        if split == 'test':
            ch = json.load(open(f'{ROOT}/results/b57_choice_{enc}.json'))['choice']
            ch2 = {k[:-len(f'm{m}')]: v for k, v in ch.items() if k.endswith(f'm{m}') and k.split('|')[-1][:2] in ('LM', 'RT', 'LR')}
            sel = select_from(ch2, M, V, ['LM', 'RT', 'LR'])
        engine(enc, m, split, M, V, chm, ['LM', 'RT', 'LR'], sel,
               os.path.join(odir(enc), f'decm_{split}_{enc}_m{m}_M{M}_V{V}.jsonl.gz'))
    elif mode == 'lamm':
        lamm(enc, int(a[2]), a[3])
    elif mode == 'onl':
        onl(enc, a[2])
    elif mode == 'choose':
        choose(enc)
    elif mode == 'rlstest':
        rlstest(enc)
    elif mode == 'merge':
        merge(enc)
    else:
        raise SystemExit(mode)


if __name__ == '__main__':
    main()
