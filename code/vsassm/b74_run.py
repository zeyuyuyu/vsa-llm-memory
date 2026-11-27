"""B74 (prereg/B74_factorial_labels_matched_prereg.md): write rule x features at the store's fixed format, exact-label
queries, and the one-pass delta rule on a matched grid.  R1 = V 2 / 16 x 4, 8, 16 bits per fact x M 256 / 1,024; ideal
accounting; B49's codes, splits and stores.

Arms run here:
  FX    (A) the store's fixed format of the cell (B53's q_R; d_R = B32's size rule at q_R, kind P: min(d_R, n); B53's
        value codes, quantizer stream and matched-filter read) on two feature levels:
          P = VSAPc's matched-filter dev-chosen (prep, pfmap): b57_run.FeatP with b60_vsapc's corrected basis;
          T = B71's VSAPR dev-chosen (prep, fmap, p): b71_run.FeatPR (rotated top-p principal coordinates);
        and both write rules on the SAME features:
          H = the Hebbian sum, b32_run.vsa_grid unchanged (share f in 1, 1/2, 1/4, 1/8; B32's rule);
          L = B72's least-squares write in the store's format (b72_run.opt_solve unchanged; share x alpha = 28; B60's
              rule; quantized once by b56_run.quant_np with seeds ('B74Q', store seed, kind, d, q, n_written, alpha)).
        HR = STORE (B53) and LR = OPTR (B72) are existing arms (not run here).
  LAB   (B) label queries: each arm's ORIGINAL test code, unchanged, with b32_common.make_store wrapped so that a test
        store's queries are [aliases, label codes of the M stored entities] (qgold / qent extended) and the output
        directory redirected: LABS = b53_run.test's vsa_grid call (R1 ideal keys), LABV = b56_run.dec('test'),
        LABP = b57_run.engine through b60_vsapc (decp test), LABL = b60_run.run_job (B60's own test spec), LABN =
        b62_run.run_job (B62's own test spec).
  NL72  (C) NLMSc re-chosen on dev from 20 random 72-configuration sub-grids of its 480 (seeds ('B74C', 'NLMSC', d)),
        B62's rule; test = b62_run.run_job (unchanged engine) on the draw-chosen configurations not already saved.

python3 b74_run.py selftest
python3 b74_run.py plan dev|test [--encs ..] [--arms FX,LAB,NL72] [--out FILE]
python3 b74_run.py runplan PLAN --workers N
python3 b74_run.py choose ENC                     # (A) -> results/b74_choice_{enc}.json
python3 b74_run.py draws [--encs ..]              # (C) dev only -> results/b74_nl72_choices.json
"""
import os

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import contextlib  # noqa: E402
import gzip  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import b60_vsapc as PC  # noqa: E402,F401  (patches b57_run.setup with the corrected whiten basis; import first)
import b57_run as R57  # noqa: E402
import b32_common as C  # noqa: E402
import b32_run as R  # noqa: E402
import b53_run as R53  # noqa: E402
import b56_run as E56  # noqa: E402
import b60_run as L60  # noqa: E402
import b62_run as O62  # noqa: E402
import b66_lsfc as LS  # noqa: E402
import b71_run as E71  # noqa: E402
import b72_run as O72  # noqa: E402

OUT = os.environ.get('B74_OUT', os.path.join(ROOT, 'b74_out'))
RES = os.environ.get('B74_RES', os.path.join(ROOT, 'results'))
INRES = os.path.join(ROOT, 'results')
PANEL = L60.PANEL
LLM, SENT = PANEL[:4], PANEL[4:]
M_GRID, VN, BPF = [256, 1024], [2, 16], [4, 8, 16]
ACCT = 'ideal'
DEV_REPS, TEST_REPS = L60.DEV_REPS, L60.TEST_REPS
SHARES = [1.0, 0.5, 0.25, 0.125]
OPT_GRID = O72.OPT_GRID  # (f, alpha), 28, tie order f > alpha
KINDS = ['P', 'T']
NDRAW, KSUB = 20, 72
TIE = 0.001
_MS = C.make_store


def odir(*a):
    d = os.path.join(OUT, *a)
    os.makedirs(d, exist_ok=True)
    return d


# ----------------------------------------------------------------------------------------------- cell formats

def cellfmt(enc):
    """{(M, V, b): {'q': q_R, 'store': B53 cfg, 'P': (prep, pfmap), 'T': (prep, fmap, p)}}."""
    b53 = json.load(open(f'{INRES}/b53_choice_{enc}.json'))['choice']
    pc = json.load(open(f'{INRES}/b60_choicepc_{enc}.json'))['choice']
    pr = json.load(open(f'{INRES}/b71_choice_{enc}.json'))['choice']
    out = {}
    for M in M_GRID:
        for V in VN:
            for b in BPF:
                cell = f'{M}|{V}|{b}'
                prep, fmap, q, f = b53[f'{cell}|ideal|VSA']['cfg']
                pp, pf, _, _ = pc[f'{cell}|ideal|VSA']['cfg']
                tp, tf, p, _, _ = pr[f'{cell}|VSAPR']['cfg']
                out[M, V, b] = {'q': int(q), 'store': [prep, fmap, int(q), f], 'P': [pp, pf], 'T': [tp, tf, int(p)]}
    return out


def d_of(kind, b, M, q, n):
    d = C.vsa_dim(b * M, q, ACCT)
    return min(d, n) if kind == 'P' else d


# ----------------------------------------------------------------------------------------------- (A) FX

def fx_feat(enc, kind, prep, fmap, p, V):
    """(pool, prep function, feature object, seed tag) of the kind's source engine."""
    if kind == 'P':
        pool, preps, Tr, lfs = R57.setup(enc)
        n = pool.X.shape[1]
        tg = R57.tag(enc, 0)
        return pool, preps[prep], R57.make_feat(tg, prep, fmap, n, V, lfs), tg
    S = LS.setup(enc)
    U, mu = LS.basis(enc, prep)
    feat = E71.FeatPR(enc, prep, fmap, U.shape[0], p, U, mu)
    feat.V = V
    return S['pool'], S['preps'][prep], feat, enc


def run_fx(spec):
    """All reps of one (enc, kind, split, M, V, features). spec['cells']: dev [[b, q]], test [[b, q, fH, fL, aL]]."""
    enc, kind, split, M, V, prep, fmap = (spec[k] for k in ('enc', 'kind', 'split', 'M', 'V', 'prep', 'fmap'))
    pool, P, feat, tg = fx_feat(enc, kind, prep, fmap, spec.get('p'), V)
    n = pool.X.shape[1]
    test = split == 'test'
    cells = [tuple(c) for c in spec['cells']]
    ds = {c[0]: d_of(kind, c[0], M, c[1], n) for c in cells}
    dmax = max(ds.values())
    out = {}
    for rep in spec['reps']:
        st = C.make_store(pool, split, M, V, rep)
        order = R.store_order(st)
        sseed = R.store_seed(tg, st)
        Yk, Yq = P(st['keys']), P(st['queries'])
        vals = np.asarray(st['vals'])
        g = st['qgold']
        rows = []
        # H: the Hebbian sum through b32_run.vsa_grid (the store's own write, quantizer stream and read)
        hc = []
        for c in cells:
            b, q = c[0], c[1]
            for f in ([c[2]] if test else SHARES):
                hc.append(((b, f), ds[b], q, max(1, int(math.floor(f * M)))))
        hp = R.vsa_grid(Yk, st['vals'], Yq, V, feat, hc, sseed, order)
        for (b, f), d, q, ns in hc:
            row = {'w': 'H', 'bpf': b, 'q': q, 'd': d, 'cfg': [f], 'bits': C.vsa_bits(d, q, ACCT),
                   'acc': float((hp[b, f] == g).mean())}
            if test:
                row['pred'] = np.asarray(hp[b, f]).astype(int).tolist()
            rows.append(row)
        # L: b72_run.opt_solve on the same features and value codes, the store's quantizer, matched filter
        Fk = E56.feats_np(feat, Yk, dmax)
        Fq = E56.feats_np(feat, Yq, dmax)
        Cv = E56.codes_np(feat, V, dmax)
        X = Fk[order].astype(np.float64) * Cv[0].astype(np.float64)[None, :]
        yv = vals[order]
        lc = []
        for c in cells:
            b, q = c[0], c[1]
            for (f, a) in ([(c[3], c[4])] if test else OPT_GRID):
                lc.append((b, q, f, a))
        need = {}
        for (b, q, f, a) in lc:
            need.setdefault(ds[b], {}).setdefault(max(1, int(math.floor(f * M))), set()).add(a)
        sol = {d: O72.opt_solve(X, yv, V, d, nsa) for d, nsa in need.items()}
        for (b, q, f, a) in lc:
            d = ds[b]
            ns = max(1, int(math.floor(f * M)))
            Sf = sol[d][ns, a]
            Sq = E56.quant_np(Sf, q, ('B74Q', sseed, kind, d, q, ns, a))
            pr = E56.mf_read(Fq, Cv, Sq)
            row = {'w': 'L', 'bpf': b, 'q': q, 'd': d, 'cfg': [f, a], 'bits': C.vsa_bits(d, q, ACCT),
                   'acc': float((pr == g).mean())}
            if test:
                row['pred'] = np.asarray(pr).astype(int).tolist()
            else:
                row['acc_float'] = float((E71.mf_float(Fq, Cv, Sf) == g).mean())
            rows.append(row)
        out[str(rep)] = {'qgold': g.tolist(), 'rows': rows}
        del Fk, Fq, X, sol
    return {'reps': out, 'n': int(n)}


# ----------------------------------------------------------------------------------------------- (B) LAB

def lab_store(pool, split, M, V, rep):
    """make_store with the M stored entities' label codes appended to the queries (test stores only)."""
    st = _MS(pool, split, M, V, rep)
    if split != 'test' and not os.environ.get('B74_LAB_ANY'):
        return st
    st = dict(st)
    st['nq_alias'] = int(len(st['qgold']))
    st['queries'] = np.concatenate([st['queries'], st['keys']])
    st['qgold'] = np.concatenate([np.asarray(st['qgold']), np.asarray(st['vals'])])
    st['qent'] = np.concatenate([np.asarray(st['qent']), np.arange(st['M'])])
    st['qalias'] = list(st['qalias']) + list(st['names'])
    return st


@contextlib.contextmanager
def labeled():
    C.make_store = lab_store
    try:
        yield
    finally:
        C.make_store = _MS


def run_labs(spec):
    """STORE (B53's test: b53_run.test's vsa_grid call) on one labelled test store; R1 ideal keys."""
    enc, M, V, rep = (spec[k] for k in ('enc', 'M', 'V', 'rep'))
    ch = json.load(open(f'{INRES}/b53_choice_{enc}.json'))['choice']
    pool, preps = R53.setup(enc)
    os.chdir(ROOT)
    with labeled():
        st = C.make_store(pool, 'test', M, V, rep)
    groups = {}
    for b in BPF:
        c = ch[f'{M}|{V}|{b}|ideal|VSA']['cfg']
        groups.setdefault((c[0], c[1]), []).append((b, 'ideal', c[2], c[3]))
    rec = {'qgold': st['qgold'].tolist(), 'nq_alias': st['nq_alias'], 'pred': {}}
    for (prep, fmap), lst in groups.items():
        feat = R53.Feat53(enc, prep, fmap, pool.X.shape[1], 'cpu')
        feat.V = V
        cfgs = [((bpf, acct, q, f), C.vsa_dim(bpf * st['M'], q, acct), q, max(1, int(math.floor(f * st['M']))))
                for bpf, acct, q, f in lst]
        P = preps[prep]
        pr = R.vsa_grid(P(st['keys']), st['vals'], P(st['queries']), V, feat, cfgs, R.store_seed(enc, st),
                        R.store_order(st))
        for (key, d, q, ns) in cfgs:
            rec['pred'][f'STORE|{key[0]}'] = np.asarray(pr[key]).astype(int).tolist()
    return rec


def run_lab_engine(spec):
    """LABV / LABP / LABL / LABN: the original engine, unchanged, on labelled test stores; output redirected."""
    arm, enc = spec['arm'], spec['enc']
    cwd = os.getcwd()
    try:
        with labeled():
            if arm == 'LABV':
                E56.OUT = odir('lab56')
                E56.dec(enc, 'test', spec['M'], spec['V'])
                fn = os.path.join(E56.OUT, enc, f'dec_test_{enc}_M{spec["M"]}_V{spec["V"]}.jsonl.gz')
            elif arm == 'LABP':
                M, V = spec['M'], spec['V']
                ch = json.load(open(PC.choicefile(enc)))['choice']
                chp = {k: v for k, v in ch.items() if k.endswith('|VSA')}
                ch2 = {k.replace('|PLM', '|LM').replace('|PRT', '|RT'): v for k, v in ch.items()
                       if k.split('|')[-1] in ('PLM', 'PRT')}
                sel = R57.select_from(ch2, M, V, ['LM', 'RT'])
                fn = os.path.join(odir('lab57', enc), f'decp_test_{enc}_M{M}_V{V}.jsonl.gz')
                R57.engine(enc, 0, 'test', M, V, chp, ['LM', 'RT'], sel, fn)
            elif arm == 'LABL':
                L60.OUT = odir('lab60')
                L60.run_job(spec['spec0'])
                fn = os.path.join(L60.OUT, 'jobs', L60.job_name(spec['spec0']) + '.json.gz')
            elif arm == 'LABN':
                O62.OUT = odir('lab62')
                O62.run_job(spec['spec0'])
                fn = os.path.join(O62.OUT, 'jobs', O62.job_name(spec['spec0']) + '.json.gz')
            else:
                raise ValueError(arm)
    finally:
        os.chdir(cwd)
    return fn


# ----------------------------------------------------------------------------------------------- (C) NL72

def nl_grid():
    return [tuple(c) for c in O62.grid('NLMSC')]


def draw_idx(d, n=480):
    rng = np.random.default_rng(C.seed_of('B74C', 'NLMSC', d))
    return sorted(int(i) for i in rng.choice(n, KSUB, replace=False))


def nl_dev(enc, jd=None):
    """{(M, V, b): {cfg: mean dev accuracy}} from B62's dev jobs (NLMSC rows), means over sorted store indices."""
    jd = jd or os.path.join(ROOT, 'b62_out', 'jobs')
    acc = {}
    for fn in sorted(os.listdir(jd)):
        if not fn.startswith(f'{enc}_dev_'):
            continue
        r = json.load(gzip.open(os.path.join(jd, fn), 'rt'))
        s = r['spec']
        for x in r['rows']:
            if x['arm'] == 'NLMSC':
                acc.setdefault((s['M'], s['V'], x['bpf']), {}).setdefault(tuple(x['cfg']), {})[s['rep']] = x['acc']
    out = {}
    for cell, d in acc.items():
        out[cell] = {}
        for cfg, reps in d.items():
            if len(reps) != DEV_REPS[cell[0]]:
                raise SystemExit(f'incomplete dev rows {enc} {cell} {cfg}')
            out[cell][cfg] = float(np.mean([reps[k] for k in sorted(reps)]))
    return out


def nl_pick(devc, cfgs):
    """B62's rule: highest mean dev accuracy; within 0.1 points tied, the first in grid order."""
    top = max(devc[c] for c in cfgs)
    for c in cfgs:
        if devc[c] >= top - TIE:
            return c, devc[c]


def draws(encs):
    G = nl_grid()
    J = {'grid_size': len(G), 'k': KSUB, 'subgrids': [draw_idx(d, len(G)) for d in range(NDRAW)], 'enc': {}}
    for enc in encs:
        dev = nl_dev(enc)
        ch62 = json.load(open(f'{INRES}/b62_choice_{enc}.json'))['choice']
        E = {}
        for M in M_GRID:
            for V in VN:
                for b in BPF:
                    devc = dev[M, V, b]
                    full, fa = nl_pick(devc, G)
                    assert list(full) == ch62[f'{M}|{V}|{b}|NLMSC']['cfg'], (enc, M, V, b, full)
                    drs = []
                    for d in range(NDRAW):
                        c, a = nl_pick(devc, [G[i] for i in J['subgrids'][d]])
                        drs.append([list(c), a])
                    E[f'{M}|{V}|{b}'] = {'full': [list(full), fa], 'draws': drs}
        J['enc'][enc] = E
        print('draws', enc, flush=True)
    fn = f'{RES}/b74_nl72_choices.json'
    json.dump(J, open(fn, 'w'), indent=1)
    print('->', fn)


# ----------------------------------------------------------------------------------------------- jobs

def job_name(s):
    a = s['arm']
    if a == 'FX':
        tail = (f'_p{s["p"]}' if s['kind'] == 'T' else '') + '_r' + '-'.join(map(str, s['reps']))
        return f'{s["enc"]}_FX{s["kind"]}_{s["split"]}_M{s["M"]}_V{s["V"]}_{s["prep"]}_{s["fmap"]}{tail}'
    if a == 'LABS':
        return f'{s["enc"]}_LABS_M{s["M"]}_V{s["V"]}_r{s["rep"]}'
    if a in ('LABV', 'LABP'):
        return f'{s["enc"]}_{a}_M{s["M"]}_V{s["V"]}'
    if a in ('LABL', 'LABN'):
        s0 = s['spec0']
        return f'{s["enc"]}_{a}_M{s0["M"]}_V{s0["V"]}_r{s0["rep"]}_{s0["prep"]}'
    if a == 'NL72':
        s0 = s['spec0']
        return f'{s["enc"]}_NL72_M{s0["M"]}_V{s0["V"]}_r{s0["rep"]}_{s0["prep"]}'
    raise ValueError(a)


def job_file(s):
    return os.path.join(OUT, 'jobs', job_name(s) + '.json.gz')


def run_job(spec):
    fn = job_file(spec)
    if os.path.exists(fn):
        return fn
    t0 = time.time()
    a = spec['arm']
    if a == 'FX':
        rec = run_fx(spec)
    elif a == 'LABS':
        rec = run_labs(spec)
    elif a == 'NL72':
        O62.OUT = odir('nl72')
        O62.run_job(spec['spec0'])
        rec = {'file': os.path.relpath(os.path.join(O62.OUT, 'jobs', O62.job_name(spec['spec0']) + '.json.gz'), OUT)}
    else:
        rec = {'file': os.path.relpath(run_lab_engine(spec), OUT)}
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


def cost(s):
    big = 4 if s['enc'] in LLM else 1
    if s['arm'] == 'FX':
        return big * (s['M'] ** 2) * len(s['reps']) * (3 if s['kind'] == 'T' else 1)
    if s['arm'] in ('LABV', 'LABP'):
        return big * s['M'] ** 2 * 10
    return big * (s.get('M') or s['spec0']['M']) ** 2


def runplan(planfile, workers):
    specs = [json.loads(l) for l in open(planfile)]
    specs = [s for s in specs if not os.path.exists(job_file(s))]
    specs.sort(key=lambda s: -cost(s))
    print(f'{len(specs)} jobs to run', flush=True)
    if workers <= 1:
        for s in specs:
            print(_run(s), flush=True)
        return
    import multiprocessing as mp
    with mp.get_context('fork').Pool(workers, maxtasksperchild=4) as pl:
        for r in pl.imap_unordered(_run, specs, chunksize=1):
            if str(r).startswith('FAILED'):
                print(r, flush=True)


def fx_groups(fm, kind, M, V):
    """{features: [b]} for the cell group (M, V)."""
    g = {}
    for b in BPF:
        g.setdefault(tuple(fm[M, V, b][kind]), []).append(b)
    return g


def make_plan(split, encs, arms, fn=None):
    odir()
    fn = fn or os.path.join(OUT, f'plan_{split}.jsonl')
    nj = 0
    with open(fn, 'w') as fo:
        def put(s):
            nonlocal nj
            fo.write(json.dumps(s) + '\n')
            nj += 1
        for enc in encs:
            fm = cellfmt(enc)
            ch = choice(enc) if (split == 'test' and 'FX' in arms) else None
            reps = DEV_REPS if split == 'dev' else TEST_REPS
            for M in M_GRID:
                for V in VN:
                    if 'FX' in arms:
                        for kind in KINDS:
                            # M = 1,024: one job per store (scheduling only; results depend on registered seeds only)
                            chunks = [[r] for r in range(reps[M])] if M == 1024 else [list(range(reps[M]))]
                            for (feats, bs), rr in [(x, r) for x in sorted(fx_groups(fm, kind, M, V).items())
                                                    for r in chunks]:
                                s = {'arm': 'FX', 'enc': enc, 'kind': kind, 'split': split, 'M': M, 'V': V,
                                     'prep': feats[0], 'fmap': feats[1], 'reps': rr}
                                if kind == 'T':
                                    s['p'] = feats[2]
                                if split == 'dev':
                                    s['cells'] = [[b, fm[M, V, b]['q']] for b in bs]
                                else:
                                    s['cells'] = [[b, fm[M, V, b]['q'], ch[f'{M}|{V}|{b}|H{kind}']['cfg'][0]]
                                                  + ch[f'{M}|{V}|{b}|L{kind}']['cfg'] for b in bs]
                                put(s)
                    if split != 'test':
                        continue
                    if 'LAB' in arms:
                        put({'arm': 'LABV', 'enc': enc, 'M': M, 'V': V})
                        put({'arm': 'LABP', 'enc': enc, 'M': M, 'V': V})
                        for rep in range(TEST_REPS[M]):
                            put({'arm': 'LABS', 'enc': enc, 'M': M, 'V': V, 'rep': rep})
                            for prep in ('center', 'whiten'):
                                f60 = f'{ROOT}/b60_out/jobs/{enc}_m0_test_M{M}_V{V}_r{rep}_{prep}.json.gz'
                                if os.path.exists(f60):
                                    put({'arm': 'LABL', 'enc': enc,
                                         'spec0': json.load(gzip.open(f60, 'rt'))['spec']})
                                f62 = f'{ROOT}/b62_out/jobs/{enc}_test_M{M}_V{V}_r{rep}_{prep}.json.gz'
                                if os.path.exists(f62):
                                    put({'arm': 'LABN', 'enc': enc,
                                         'spec0': json.load(gzip.open(f62, 'rt'))['spec']})
                    if 'NL72' in arms:
                        J = json.load(open(f'{RES}/b74_nl72_choices.json'))
                        if enc not in J['enc']:
                            continue
                        E = J['enc'][enc]
                        want = {}
                        for b in BPF:
                            e = E[f'{M}|{V}|{b}']
                            for c, _ in e['draws']:
                                if c != e['full'][0]:
                                    w = ['NLMSC', c, b]
                                    if w not in want.setdefault(c[2], []):
                                        want[c[2]].append(w)
                        for prep, w in sorted(want.items()):
                            for rep in range(TEST_REPS[M]):
                                put({'arm': 'NL72', 'enc': enc,
                                     'spec0': {'enc': enc, 'split': 'test', 'M': M, 'V': V, 'rep': rep, 'prep': prep,
                                               'want': w}})
    print(nj, 'jobs ->', fn)


# ----------------------------------------------------------------------------------------------- dev choice (A)

def choice(enc):
    return json.load(open(f'{RES}/b74_choice_{enc}.json'))['choice']


def load_dev(enc):
    """{(w + kind, M, V, b, cfg tuple): {rep: row}} from the FX dev jobs."""
    acc = {}
    jd = os.path.join(OUT, 'jobs')
    for fn in sorted(os.listdir(jd)):
        if not fn.startswith(f'{enc}_FX') or '_dev_' not in fn:
            continue
        r = json.load(gzip.open(os.path.join(jd, fn), 'rt'))
        s = r['spec']
        if s['enc'] != enc:
            continue
        for rep, rr in r['reps'].items():
            for x in rr['rows']:
                k = (x['w'] + s['kind'], s['M'], s['V'], x['bpf'], tuple(x['cfg']))
                assert int(rep) not in acc.get(k, {}), ('duplicate dev row', k, rep)
                acc.setdefault(k, {})[int(rep)] = x
    return acc


def dmean(acc, k, need, fld='acc'):
    d = acc.get(k)
    if d is None:
        raise SystemExit(f'missing dev rows {k}')
    if len(d) != need:
        raise SystemExit(f'incomplete dev rows {k}: {len(d)}')
    return float(np.mean([d[i][fld] for i in sorted(d)]))


def pick(lst):
    """lst in tie order: highest mean dev accuracy; within 0.1 points tied, the first in order."""
    top = max(a for a, _ in lst)
    for a, c in lst:
        if a >= top - TIE:
            return c, a, sum(1 for a2, _ in lst if a2 >= top - TIE)


def choose(enc):
    acc = load_dev(enc)
    fm = cellfmt(enc)
    out = {}
    for M in M_GRID:
        need = DEV_REPS[M]
        for V in VN:
            for b in BPF:
                cell = f'{M}|{V}|{b}'
                for kind in KINDS:
                    lst = [(dmean(acc, ('H' + kind, M, V, b, (f,)), need), (f,)) for f in SHARES]
                    c, a, nt = pick(lst)
                    out[f'{cell}|H{kind}'] = {'cfg': list(c), 'dev_acc': a, 'n_tied': nt, 'q': fm[M, V, b]['q'],
                                              'feat': fm[M, V, b][kind]}
                    lst = [(dmean(acc, ('L' + kind, M, V, b, g), need), g) for g in OPT_GRID]
                    c, a, nt = pick(lst)
                    out[f'{cell}|L{kind}'] = {'cfg': list(c), 'dev_acc': a, 'n_tied': nt, 'q': fm[M, V, b]['q'],
                                              'feat': fm[M, V, b][kind],
                                              'dev_acc_float': dmean(acc, ('L' + kind, M, V, b, c), need, 'acc_float')}
    fn = f'{RES}/b74_choice_{enc}.json'
    json.dump({'enc': enc, 'choice': out}, open(fn, 'w'), indent=1, sort_keys=True)
    print(len(out), 'choices ->', fn)


# ----------------------------------------------------------------------------------------------- selftest

def selftest():
    # 1. draws: 20 distinct sub-grids of 72, sorted; the grid is B62's 480 in tie order
    G = nl_grid()
    assert len(G) == 480 and len(set(G)) == 480
    D = [draw_idx(d) for d in range(NDRAW)]
    assert all(len(x) == KSUB and len(set(x)) == KSUB and x == sorted(x) for x in D)
    assert len({tuple(x) for x in D}) == NDRAW and D == [draw_idx(d) for d in range(NDRAW)]
    print('NL72 draws: 20 distinct sub-grids of 72, deterministic: ok')
    # 2. B62's full-grid NLMSc choice recomputed from its existing dev rows (dev rows of B62 only, no new run)
    for enc in PANEL:
        dev = nl_dev(enc)
        ch62 = json.load(open(f'{INRES}/b62_choice_{enc}.json'))['choice']
        for M in M_GRID:
            for V in VN:
                for b in BPF:
                    c, _ = nl_pick(dev[M, V, b], G)
                    assert list(c) == ch62[f'{M}|{V}|{b}|NLMSC']['cfg'], (enc, M, V, b)
    print("B62's full-grid NLMSc choices reproduced from B62's dev rows in 84 / 84 cells x 7 encoders: ok")
    # 3. label wrapper on TRAINING stores: alias part of every engine's read identical with labels appended
    os.environ['B74_LAB_ANY'] = '1'
    for enc in ('minilm', 'qwen8'):
        t0 = time.time()
        pool, preps = R53.setup(enc)
        os.chdir(ROOT)
        n = pool.X.shape[1]
        for V in (2, 16):
            st0 = _MS(pool, 'train', 256, V, 0)
            st1 = lab_store(pool, 'train', 256, V, 0)
            nq = len(st0['qgold'])
            assert st1['nq_alias'] == nq and np.array_equal(st1['qgold'][nq:], st0['vals'])
            # STORE
            feat = R53.Feat53(enc, 'whiten', 'lin', n, 'cpu')
            feat.V = V
            cfgs = [((b,), C.vsa_dim(b * 256, 1, ACCT), 1, 256) for b in BPF]
            P = preps['whiten']
            p0 = R.vsa_grid(P(st0['keys']), st0['vals'], P(st0['queries']), V, feat, cfgs, R.store_seed(enc, st0),
                            R.store_order(st0))
            p1 = R.vsa_grid(P(st1['keys']), st1['vals'], P(st1['queries']), V, feat, cfgs, R.store_seed(enc, st1),
                            R.store_order(st1))
            for k in p0:
                assert np.array_equal(p0[k], p1[k][:nq]), ('STORE alias part differs', enc, V, k)
            lab = {k: float((p1[k][nq:] == st0['vals']).mean()) for k in p1}
            # LAMc and NLMSc engines
            pool6, preps6, fcs = L60.setup(enc)
            s0 = _MS(pool6, 'train', 256, V, 0)
            s1 = lab_store(pool6, 'train', 256, V, 0)
            P6 = preps6['whiten']
            cf = [('col', 'V-1', 'pca', 1.0, 1), ('flat', 'V', 'rp', 0.01, 2)]
            o0 = L60.run_lamc(s0, P6(s0['keys']), P6(s0['queries']), fcs['whiten'], cf, BPF, ['ideal'], 7)
            o1 = L60.run_lamc(s1, P6(s1['keys']), P6(s1['queries']), fcs['whiten'], cf, BPF, ['ideal'], 7)
            for k in o0:
                assert np.array_equal(o0[k][0], o1[k][0][:nq]), ('LAMc alias part differs', enc, V, k)
            want = [('NLMSC', ('col', 'V-1', 'whiten', 'pca', 0.25, 'last', 1), b) for b in BPF]
            n0 = O62.run_online(s0, P6(s0['keys']), P6(s0['queries']), fcs['whiten'], enc, 'whiten', want, 7)
            n1 = O62.run_online(s1, P6(s1['keys']), P6(s1['queries']), fcs['whiten'], enc, 'whiten', want, 7)
            for k in n0:
                assert np.array_equal(n0[k][0], n1[k][0][:nq]), ('NLMSc alias part differs', enc, V, k)
            labl = {k[1]: float((o1[k][0][nq:] == s0['vals']).mean()) for k in o1 if k[0][2] == 'pca'}
            print(enc, 'V', V, 'train store: STORE label recall', {k[0]: round(v, 3) for k, v in lab.items()},
                  'alias', {k[0]: round(float((p0[k] == st0['qgold']).mean()), 3) for k in p0},
                  'LAMc(pca) label', {k: round(v, 3) for k, v in labl.items()}, f'{time.time() - t0:.1f}s', flush=True)
    del os.environ['B74_LAB_ANY']
    print('label wrapper: alias predictions identical with labels appended (STORE, LAMc, NLMSc engines): ok')
    # 4. FX on TRAINING stores (kinds P and T): bits within budget and maximal, d rule, H and L rows
    for enc in ('minilm', 'qwen8'):
        fm = cellfmt(enc)
        for kind in KINDS:
            for V in (2, 16):
                t0 = time.time()
                feats = fm[256, V, 16][kind]
                q = fm[256, V, 16]['q']
                spec = {'enc': enc, 'kind': kind, 'split': 'train', 'M': 256, 'V': V, 'prep': feats[0],
                        'fmap': feats[1], 'reps': [0], 'cells': [[4, q], [16, q]]}
                if kind == 'T':
                    spec['p'] = feats[2]
                rec = run_fx(spec)
                n = rec['n']
                for x in rec['reps']['0']['rows']:
                    B = x['bpf'] * 256
                    dr = C.vsa_dim(B, x['q'], ACCT)
                    assert x['d'] == (min(dr, n) if kind == 'P' else dr) and x['bits'] <= B
                    assert kind == 'P' and x['d'] == n or C.vsa_bits(x['d'] + 1, x['q'], ACCT) > B
                best = {}
                for x in rec['reps']['0']['rows']:
                    k = (x['w'], x['bpf'])
                    best[k] = max(best.get(k, 0), x['acc'])
                print(enc, kind, 'V', V, feats, 'q', q, 'train store best acc', {f'{k[0]}{k[1]}': round(v, 3)
                                                                                    for k, v in sorted(best.items())},
                      f'{time.time() - t0:.1f}s', flush=True)
    print('selftest passed')


def main():
    a = sys.argv[1:]
    if a[0] == 'selftest':
        selftest()
    elif a[0] == 'plan':
        encs = a[a.index('--encs') + 1].split(',') if '--encs' in a else PANEL
        arms = a[a.index('--arms') + 1].split(',') if '--arms' in a else ['FX', 'LAB', 'NL72']
        fn = a[a.index('--out') + 1] if '--out' in a else None
        make_plan(a[1], encs, arms, fn)
    elif a[0] == 'runplan':
        runplan(a[1], int(a[a.index('--workers') + 1]) if '--workers' in a else 1)
    elif a[0] == 'choose':
        choose(a[1])
    elif a[0] == 'draws':
        draws(a[a.index('--encs') + 1].split(',') if '--encs' in a else PANEL)
    else:
        raise SystemExit(__doc__)


if __name__ == '__main__':
    main()
