"""B64 checks (prereg/B64_sdm_training_locations_prereg.md, 'Checks'): a second angle on the verdict.

python3 b64_check.py scorer          # plain-loop re-scoring of every R1 cell and region point statistic
python3 b64_check.py bits            # every dev / test row within budget
python3 b64_check.py disjoint        # location sets vs dev / test strings, normalised strings, code rows, max cosine
python3 b64_check.py rederive [ENC]  # independent float64 re-derivation of the chosen SDMt on one store per (M, V)
python3 b64_check.py all             # everything -> results/b64_check.txt (+ json files)
Independent of b64_run.py's implementation: own preprocessing (float64), own location list and order, own
nearest-location search (full stable sort), own counters (loops over facts), own entropy, own N scan, own read and
tie rule. Shared with the pipeline only: the data files, the registered seeds' hash (sha256 of the joined parts) and
the registered grid formula.
"""
import gzip
import hashlib
import json
import math
import os
import sys
import unicodedata
from collections import Counter

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.environ.get('B64_RES', f'{ROOT}/results')
JOBS = os.environ.get('B64_JOBS', f'{ROOT}/b64_out/jobs')
DATA = f'{ROOT}/b49_out/b32_data'
PANEL = ['qwen8', 'b46', 'llama', 'mistral', 'e5', 'bge', 'minilm']
TAG = {'qwen8': 'qwen8_mean_L9_f16', 'e5': 'e5_base', 'b46': 'b46_granite33i_mean_L30_f16',
       'llama': 'llama31i_mean_L24_f16', 'mistral': 'mistral7i_eol_L24_f16', 'bge': 'bge_small', 'minilm': 'minilm'}
M_GRID, VN, BPF, ACCTS = [256, 1024], [2, 16], [4, 8, 16], ['ideal', 'real']
TEST_REPS, DEV_REPS = {256: 20, 1024: 5}, {256: 6, 1024: 3}
SIX = ['IDX', 'SF', 'CM', 'BF', 'FBF', 'SDM']
SEVEN = SIX + ['SDMT']
LINES = []


def say(*a):
    s = ' '.join(str(x) for x in a)
    LINES.append(s)
    print(s, flush=True)


def seed(*parts):
    return int(hashlib.sha256('|'.join(map(str, parts)).encode()).hexdigest()[:15], 16)


def jl(fn):
    return [json.loads(l) for l in gzip.open(fn, 'rt')]


# ----------------------------------------------------------------------------------------------- scorer

def scorer():
    V = json.load(open(f'{RES}/b64_verdict.json'))
    worst = 0.0
    n = 0
    for enc in PANEL:
        if not any(k.startswith(enc + '|') for k in V['cells']):
            continue
        ch = json.load(open(f'{RES}/b64_choice_{enc}.json'))['choice']
        for acct in ACCTS:
            cell = {}
            for M in M_GRID:
                for Vv in VN:
                    recs = jl(f'{ROOT}/b59_out/fam/{enc}/all59_{enc}_M{M}_V{Vv}.jsonl.gz')
                    jobs = {}
                    for rep in range(TEST_REPS[M]):
                        j = json.load(gzip.open(f'{JOBS}/{enc}_test_M{M}_V{Vv}_r{rep}.json.gz', 'rt'))
                        jobs[rep] = j
                    for b in BPF:
                        c = ch[f'{M}|{Vv}|{b}|{acct}|SDMT']['cfg']
                        fam = ch[f'{M}|{Vv}|{b}|{acct}|BEST7']['fam']
                        tot = {x: 0 for x in ('VSAR', 'VSA', 'SDMT', 'BEST7', 'BEST36', 'SDM')}
                        nq = 0
                        for r in recs:
                            j = jobs[r['rep']]
                            sd = None
                            for x in j['rows']:
                                if x['cfg'] == c and x['bpf'] == b and x['acct'] == acct:
                                    sd = x['pred']
                            assert sd is not None
                            for i in range(len(r['qgold'])):
                                gold = r['qgold'][i]
                                nq += 1
                                tot['VSAR'] += r['pred'][f'VSAR|{b}|{acct}'][i] == gold
                                tot['VSA'] += r['pred'][f'VSA|{b}|{acct}'][i] == gold
                                tot['SDM'] += r['pred'][f'SDM|{b}|{acct}'][i] == gold
                                tot['BEST36'] += r['pred'][f'BEST36|{b}|{acct}'][i] == gold
                                tot['SDMT'] += sd[i] == gold
                                bp = sd[i] if fam == 'SDMT' else r['pred'][f'{fam}|{b}|{acct}'][i]
                                tot['BEST7'] += bp == gold
                        cell[M, Vv, b] = {x: tot[x] / nq for x in tot}
                        vc = V['cells'][f'{enc}|{acct}|{M}|{Vv}|{b}']
                        for x in tot:
                            worst = max(worst, abs(cell[M, Vv, b][x] - vc[x]))
                            n += 1
            for rn, cells in (('R1', [(M, Vv, b) for M in M_GRID for Vv in VN for b in BPF]),
                              ('R1-V2', [(M, 2, b) for M in M_GRID for b in BPF]),
                              ('R1-V16', [(M, 16, b) for M in M_GRID for b in BPF])):
                for x, y in (('VSAR', 'SDMT'), ('VSAR', 'BEST7'), ('VSA', 'BEST7'), ('VSAR', 'BEST36'), ('SDMT', 'SDM')):
                    s = sum(cell[k][x] - cell[k][y] for k in cells) / len(cells)
                    worst = max(worst, abs(s - V['regions'][f'{enc}|{acct}|{rn}|{x}-{y}']['diff']))
                    n += 1
    say(f'scorer: {n} cell accuracies and region statistics recomputed by plain loops; max |difference| {worst:.2e}',
        'PASS' if worst < 1e-9 else 'FAIL')
    return worst


# ----------------------------------------------------------------------------------------------- bits

def bits():
    n, over, worst = 0, 0, -1e18
    for fn in sorted(os.listdir(JOBS)):
        r = json.load(gzip.open(os.path.join(JOBS, fn), 'rt'))
        M = r['spec']['M']
        for x in r['rows']:
            n += 1
            B = x['bpf'] * M
            if x['bits'] > B + 1e-9:
                over += 1
            worst = max(worst, x['bits'] - B)
    say(f'bits: {n} logged dev / test rows; over budget {over}; max (bits - budget) {worst:.1f}',
        'PASS' if over == 0 else 'FAIL')
    return {'rows': n, 'over': over, 'max_excess': worst}


# ----------------------------------------------------------------------------------------------- own data and locations

def strip(s):
    return ''.join(ch for ch in unicodedata.normalize('NFKD', s) if not unicodedata.combining(ch)).casefold().strip()


class Data:
    def __init__(self, enc):
        ents = json.load(open(f'{DATA}/entities.json'))['entities']
        self.strings = json.load(open(f'{DATA}/emb_strings.json'))
        self.idx = {s: i for i, s in enumerate(self.strings)}
        self.Xraw = np.load(f'{DATA}/emb_{TAG[enc]}.npy')
        self.X = self.Xraw.astype(np.float64)
        self.split = {s: [e for e in ents if e['split'] == s] for s in ('train', 'dev', 'test')}
        tr = []
        for e in self.split['train']:
            tr.append(self.idx[e['name']])
            tr += [self.idx[a] for a in e['aliases']]
        self.Tr = self.X[tr]
        self.other = sorted({self.idx[e['name']] for s in ('dev', 'test') for e in self.split[s]} |
                            {self.idx[a] for s in ('dev', 'test') for e in self.split[s] for a in e['aliases']})

    def locs(self, loc):
        lab = [self.idx[e['name']] for e in self.split['train']]
        ali = [self.idx[a] for e in self.split['train'] for a in e['aliases']]
        rows = {'LA': lab + ali, 'L': lab, 'A': ali}[loc]
        bad = set(self.other)
        u = list(dict.fromkeys(r for r in rows if r not in bad))
        perm = np.random.default_rng(seed('B64LOC', loc)).permutation(len(u))
        o = [u[i] for i in perm]
        badc = {self.Xraw[r].tobytes() for r in self.other}
        return [r for r in o if self.Xraw[r].tobytes() not in badc]

    def prep(self, kind):
        mu = self.Tr.mean(0)
        if kind == 'center':
            return lambda Z: _rn(Z - mu)
        Cv = np.cov((self.Tr - mu).T)
        w, U = np.linalg.eigh(Cv)
        w = np.maximum(w, 0)
        Wz = (U / np.sqrt(w + 0.1 * w.mean())) @ U.T
        return lambda Z: _rn((Z - mu) @ Wz)


def _rn(Z):
    return Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12)


def disjoint():
    out = {}
    worst = 0
    for enc in PANEL:
        D = Data(enc)
        So = {D.strings[r] for r in D.other}
        Sn = {strip(s) for s in So}
        Xo = _rn(D.X[D.other])
        for loc in ('LA', 'L', 'A'):
            rows = D.locs(loc)
            ns = sum(D.strings[r] in So for r in rows)
            nn = sum(strip(D.strings[r]) in Sn for r in rows)
            Xl = _rn(D.X[rows])
            mx = 0.0
            for i0 in range(0, len(rows), 1024):
                mx = max(mx, float((Xl[i0:i0 + 1024] @ Xo.T).max()))
            hi = 0
            for i0 in range(0, len(rows), 1024):
                hi += int(((Xl[i0:i0 + 1024] @ Xo.T) >= 0.9999).any(1).sum())
            out[f'{enc}|{loc}'] = {'n': len(rows), 'string': ns, 'normalised_string': nn, 'cos_ge_0.9999': hi,
                                   'max_cos': mx}
            worst = max(worst, ns, hi)
            say(f'disjoint {enc:8s} {loc:2s}: {len(rows)} locations; string overlap {ns}; normalised (accents, case) '
                f'{nn}; raw cosine >= 0.9999 {hi}; max raw cosine {mx:.4f}')
    say('disjoint:', 'PASS' if worst == 0 else 'FAIL', '(string overlap and cos >= 0.9999 must be 0; normalised-string '
        'matches are reported, not barred: a different string with a different code is not the stored name)')
    json.dump(out, open(f'{RES}/b64_check_disjoint.json', 'w'), indent=1)
    return out


# ----------------------------------------------------------------------------------------------- re-derivation

def store(D, split, M, V, rep):
    ents = D.split[split]
    rng = np.random.default_rng(seed('store', split, M, V, rep))
    pick = np.sort(rng.permutation(len(ents))[:min(M, len(ents))])
    E = [ents[i] for i in pick]
    vals = rng.integers(0, V, size=len(E))
    keys = D.X[[D.idx[e['name']] for e in E]]
    qi, qg = [], []
    for j, e in enumerate(E):
        for a in e['aliases']:
            qi.append(D.idx[a])
            qg.append(int(vals[j]))
    order = np.random.default_rng(seed('subset', split, len(E), V, rep)).permutation(len(E))
    return keys, vals, D.X[qi], qg, order


def H0(sym):
    c = Counter(sym)
    n = sum(c.values())
    return -sum(k / n * math.log2(k / n) for k in c.values())


def own_bits(Qc, ctr, q, acct):
    if Qc.size == 0:
        return 0.0
    if acct == 'ideal':
        return Qc.size * H0(Qc.ravel().tolist())
    if q == 'tern':
        return Qc.size * (1 if ctr == 'onehot' else 2)
    if q == 'clip3':
        return Qc.size * (2 if ctr == 'onehot' else 3)
    mx = int(np.abs(Qc).max())
    w = max(1, math.ceil(math.log2(mx + 1 + 1e-9))) if ctr == 'onehot' else max(1, math.ceil(math.log2(2 * mx + 1 + 1e-9)))
    return Qc.size * w + 8


def own_q(Cn, ctr, q):
    if q == 'cnt':
        return Cn.copy()
    if q == 'clip3':
        return np.minimum(Cn, 3) if ctr == 'onehot' else np.clip(Cn, -3, 3)
    return np.minimum(Cn, 1) if ctr == 'onehot' else np.sign(Cn)


def rederive_one(D, preps, locs, enc, M, V, rep, cfg, b, acct, split='test'):
    prep, loc, a, ctr, q, sh = cfg
    keys, vals, Q, qg, order = store(D, split, M, V, rep)
    P = preps[prep]
    Lr = locs[loc]
    La = P(D.X[Lr])
    Kp, Qp = P(keys), P(Q)
    Sk = Kp @ La.T
    Sq = Qp @ La.T
    nb = max(1, math.ceil(math.log2(V)))
    bitsv = [[1 if (v >> j) & 1 else -1 for j in range(nb)] for v in range(V)]
    rows = order[:max(1, int(round(sh * M)))]
    default = int(np.bincount(vals, minlength=V).argmax())
    dbits = math.ceil(math.log2(V))
    grid = sorted({int(x) for x in np.round(np.geomspace(1, len(Lr), 160)).astype(int) if 1 <= x <= len(Lr)})
    rank_k = np.argsort(-Sk, axis=1, kind='stable')[rows]  # full sort over all locations; prefix N: ranks < N
    c = V if ctr == 'onehot' else nb
    best = (0, None)
    for N in sorted(grid, reverse=True):  # largest grid point that fits = first fit from the top
        mask = rank_k < N
        first = mask & (np.cumsum(mask, axis=1) <= a)
        rr, cc = np.nonzero(first)
        ls = rank_k[rr, cc]
        vv = vals[rows][rr]
        if ctr == 'onehot':
            Cn = np.bincount(ls * V + vv, minlength=N * V).reshape(N, V).astype(np.int64)
        else:
            Cn = np.zeros((N, nb), np.int64)
            for j in range(nb):
                sgn = np.array([bitsv[v][j] for v in range(V)])[vv]
                Cn[:, j] = np.bincount(ls, weights=sgn, minlength=N).astype(np.int64)
        Qc = own_q(Cn, ctr, q)
        if own_bits(Qc, ctr, q, acct) <= b * M - dbits:
            best = (N, Qc)
            break
    _ = c
    N, Qc = best
    if N == 0:
        return [default] * len(qg), 0, 0.0
    rank_q = np.argsort(-Sq[:, :N], axis=1, kind='stable')[:, :a]
    pred = []
    for i in range(len(qg)):
        sc = [0.0] * V
        for l in rank_q[i]:
            for v in range(V):
                sc[v] += Qc[l, v] if ctr == 'onehot' else sum(Qc[l, j] * bitsv[v][j] for j in range(nb))
        m = max(sc)
        cand = [v for v in range(V) if sc[v] == m]
        pred.append(default if default in cand else min(cand))
    return pred, N, own_bits(Qc, ctr, q, acct) + dbits


def rederive(encs):
    out = {}
    lo = 1.0
    for enc in encs:
        ch = json.load(open(f'{RES}/b64_choice_{enc}.json'))['choice']
        D = Data(enc)
        preps = {p: D.prep(p) for p in ('center', 'whiten')}
        locs = {l: D.locs(l) for l in ('LA', 'L', 'A')}
        for M in M_GRID:
            for V in VN:
                rep = int(np.random.default_rng(seed('B64RD', enc, M, V)).integers(0, TEST_REPS[M]))
                j = json.load(gzip.open(f'{JOBS}/{enc}_test_M{M}_V{V}_r{rep}.json.gz', 'rt'))
                for b in (4, 16):
                    cfg = ch[f'{M}|{V}|{b}|ideal|SDMT']['cfg']
                    row = [x for x in j['rows'] if x['cfg'] == cfg and x['bpf'] == b and x['acct'] == 'ideal'][0]
                    pred, N, bt = rederive_one(D, preps, locs, enc, M, V, rep, tuple(cfg), b, 'ideal')
                    ag = float(np.mean(np.array(pred) == np.array(row['pred'])))
                    lo = min(lo, ag)
                    out[f'{enc}|{M}|{V}|{b}|r{rep}'] = {'cfg': cfg, 'agree': ag, 'N_log': row['N'], 'N_own': N,
                                                        'bits_log': row['bits'], 'bits_own': bt,
                                                        'within_budget_own': bt <= b * M}
                    say(f'rederive {enc:8s} M={M:5d} V={V:3d} b={b:3d} rep={rep:2d} {"/".join(map(str, cfg))}: agreement '
                        f'{ag:.4f}; N log {row["N"]} own {N}; bits log {row["bits"]:.1f} own {bt:.1f} (budget {b * M})')
    say(f'rederive: min agreement {lo:.4f}', 'PASS' if lo >= 0.99 else 'FAIL')
    fn = f'{RES}/b64_check_rederive.json'
    old = json.load(open(fn)) if os.path.exists(fn) else {}
    old.update(out)
    json.dump(old, open(fn, 'w'), indent=1)
    return out


def main():
    a = sys.argv[1:]
    if a[0] == 'scorer':
        scorer()
    elif a[0] == 'bits':
        bits()
    elif a[0] == 'disjoint':
        disjoint()
    elif a[0] == 'rederive':
        rederive(a[1].split(',') if len(a) > 1 else PANEL)
    elif a[0] == 'all':
        scorer()
        bits()
        disjoint()
        rederive(PANEL)
    if a[0] in ('all', 'scorer', 'bits'):
        fn = f'{RES}/b64_check.txt'
        open(fn, 'a').write('\n'.join(LINES) + '\n')


if __name__ == '__main__':
    main()
