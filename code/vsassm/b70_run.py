"""B70 (prereg/B70_useful_budgets_prereg.md): the equal-bit comparison at the budgets where the best memory is useful.

Region R2 (B70's; not B49's R2 = V 1,024): per setting s = (encoder, M, V), V in {2, 16}, M in {256, 1,024}:
  U_dev(s)      = 0.9 x the exact index's dev accuracy (B32's dev record, its dev-chosen prep);
  best_dev(s,b) = the highest dev accuracy at budget b (ideal) among the dev-chosen configurations of the 13 memories
                  with dev outputs at every budget 1-64 before B70: B49's ten families with LAM replaced by LAMc (B60's
                  BESTC 'dev_by_fam': IDX SF CM BF FBF SDM LAM(=LAMc) LHI LHC LHS), STORE (B53), VSAR (B49's VSA),
                  VSAPCX (max of B56's STAR and B60's PSTAR dev accuracies, B60's rule);
  b*(s)         = the smallest b in (1, 2, 4, 8, 16, 32, 64) with best_dev(s, b) >= U_dev(s);
  R2(s)         = {b in the grid: b >= b*(s)} (empty if no such b).
New runs: NLMSc (B62's engine and grid, B62's seeds) and SDMt (B64's engine and grid) at the R2 budgets 32 / 64
(dev also at 16: must reproduce B62's / B64's existing 16-bit dev rows exactly). R2 cells at 8 / 16 use B62's and B64's
committed choices and test predictions.

python3 b70_run.py region                          # -> results/b70_region.json (+ table on stdout)
python3 b70_run.py selftest                        # training-split stores only
python3 b70_run.py plan dev|test [--encs ..]       # -> OUT/plan_{kind}.jsonl (both arms)
python3 b70_run.py runplan PLAN.jsonl --workers N
python3 b70_run.py choose ENC                      # -> results/b70_choice_{enc}.json (+ reproduction check)
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
os.environ.setdefault('B60_DATA', os.path.join(ROOT, 'b49_out'))
os.environ.setdefault('B64_DATA', os.path.join(ROOT, 'b49_out'))
import b32_common as C  # noqa: E402
import b60_run as L60  # noqa: E402
import b62_run as N62  # noqa: E402
import b64_run as S64  # noqa: E402

OUT = os.environ.get('B70_OUT', os.path.join(ROOT, 'b70_out'))
RES = os.environ.get('B70_RES', os.path.join(ROOT, 'results'))
PANEL = ['qwen8', 'b46', 'llama', 'mistral', 'e5', 'bge', 'minilm']
M_GRID, VN, BPF = [256, 1024], [2, 16], [1, 2, 4, 8, 16, 32, 64]
NEW = (32, 64)        # budgets above R1: NLMSc / SDMt never run there before B70
REPRO = 16            # dev reproduction budget (B62 / B64 rows exist)
DEV_REPS, TEST_REPS = {256: 6, 1024: 3}, {256: 20, 1024: 5}
TEN = ['IDX', 'SF', 'CM', 'BF', 'FBF', 'SDM', 'LAM', 'LHI', 'LHC', 'LHS']
SIX = ['IDX', 'SF', 'CM', 'BF', 'FBF', 'SDM']
SEVEN = SIX + ['SDMT']
# compute note 3: SDMt (and BEST7) cut on the sentence encoders by the registered stop rule (its 16-bit dev rows do not
# reproduce B64's: near-tie location ranks of float32 codes depend on the evaluation path); kept on the language models
CUT_SDMT = {'e5', 'bge', 'minilm'}


def jl(fn):
    return json.load(open(fn))


def exact_choice(enc):
    for fn in (f'{ROOT}/results/b32_choice_{enc}.json', f'{ROOT}/results/b49_choice32_{enc}.json',
               f'{ROOT}/results/b46_choice_b32.json' if enc == 'b46' else ''):
        if fn and os.path.exists(fn):
            c = jl(fn)
            return c.get('choice', c)
    raise SystemExit(f'no exact-index dev record for {enc}')


def dev_table(enc):
    """{(M, V, b): {memory: dev accuracy}} and {(M, V): EXACT dev accuracy} from the committed dev choices."""
    ce = exact_choice(enc)
    c49 = jl(f'{ROOT}/results/b49_choice_{enc}.json')['choice']
    c60 = jl(f'{ROOT}/results/b60_choice_{enc}.json')['choice']
    c53 = jl(f'{ROOT}/results/b53_choice_{enc}.json')['choice']
    c56 = jl(f'{ROOT}/results/b56_choice_{enc}.json')['choice']
    cpc = jl(f'{ROOT}/results/b60_choicepc_{enc}.json')['choice']
    tab, ex = {}, {}
    for M in M_GRID:
        for V in VN:
            ex[M, V] = float(ce[f'{M}|{V}|None|None|EXACT']['dev_acc'])
            for b in BPF:
                dbf = c60[f'{M}|{V}|{b}|ideal|BESTC']['dev_by_fam']
                assert abs(dbf['LAM'] - c60[f'{M}|{V}|{b}|ideal|LAMC']['dev_acc']) < 1e-12
                d = {('LAMC' if f == 'LAM' else f): float(dbf[f]) for f in TEN}
                d['STORE'] = float(c53[f'{M}|{V}|{b}|ideal|VSA']['dev_acc'])
                d['VSAR'] = float(c49[f'{M}|{V}|{b}|ideal|VSA']['dev_acc'])
                s, p = c56[f'{M}|{V}|{b}|ideal|STAR'], cpc[f'{M}|{V}|{b}|ideal|PSTAR']
                d['VSAPCX'] = float(max(s['dev'][s['fam']], p['dev'][p['fam']]))
                tab[M, V, b] = d
    return tab, ex


def region():
    J = {'rule': 'b* = smallest b with best_dev >= 0.9 x EXACT dev; R2 = {b >= b*}', 'settings': {}, 'cells': []}
    L = ['# B70 region R2 from committed dev outputs (ideal). "*" = best_dev >= U_dev', '']
    for enc in PANEL:
        tab, ex = dev_table(enc)
        for M in M_GRID:
            for V in VN:
                U = 0.9 * ex[M, V]
                bstar = None
                row = []
                for b in BPF:
                    d = tab[M, V, b]
                    best = max(d, key=lambda k: (d[k], -sorted(d).index(k)))
                    ok = d[best] >= U
                    if ok and bstar is None:
                        bstar = b
                    row.append(f'{b}:{100 * d[best]:.1f}{best}{"*" if ok else ""}')
                cells = [b for b in BPF if bstar is not None and b >= bstar]
                J['settings'][f'{enc}|{M}|{V}'] = {
                    'EXACT_dev': ex[M, V], 'U_dev': U, 'bstar': bstar, 'R2': cells,
                    'best_dev': {str(b): max(tab[M, V, b].values()) for b in BPF},
                    'best_fam': {str(b): max(tab[M, V, b], key=tab[M, V, b].get) for b in BPF},
                    'dev': {str(b): tab[M, V, b] for b in BPF}}
                J['cells'] += [[enc, M, V, b] for b in cells]
                L.append(f'{enc:8s} M={M:5d} V={V:3d} EXACT {100 * ex[M, V]:.1f} U {100 * U:.1f} b* {bstar}  ' +
                         ' '.join(row))
    L.append('')
    L.append(f'R2 cells: {len(J["cells"])}; settings with a useful budget within 64: '
             f'{sum(v["bstar"] is not None for v in J["settings"].values())} of {len(J["settings"])}')
    for enc in PANEL:
        L.append(f'  {enc:8s} ' + ', '.join(f'(M {M}, V {V}, b {b})' for e, M, V, b in J['cells'] if e == enc))
    json.dump(J, open(f'{RES}/b70_region.json', 'w'), indent=1, sort_keys=True)
    print('\n'.join(L))


def r2_cells():
    return [tuple(x) for x in jl(f'{RES}/b70_region.json')['cells']]


def new_budgets(enc, M, V):
    return sorted({b for e, m, v, b in r2_cells() if (e, m, v) == (enc, M, V) and b in NEW})


# ----------------------------------------------------------------------------------------------- jobs

def job_name(spec):
    s = f'{spec["arm"]}_{spec["enc"]}_{spec["split"]}_M{spec["M"]}_V{spec["V"]}_r{spec["rep"]}'
    return s + (f'_{spec["prep"]}' if spec['arm'] == 'NLMSC' else '')


def run_job(spec):
    """spec: arm (NLMSC | SDMT), enc, split, M, V, rep, [prep], bpfs, cfgs (None = full grid; else [[cfg, bpf], ...]).
    Dev rows carry no predictions; test rows do."""
    fn = os.path.join(OUT, 'jobs', job_name(spec) + '.json.gz')
    if os.path.exists(fn):
        return fn
    t0 = time.time()
    arm, enc, split, M, V, rep = (spec[k] for k in ('arm', 'enc', 'split', 'M', 'V', 'rep'))
    test = spec.get('cfgs') is not None
    rows = []
    if arm == 'NLMSC':
        prep = spec['prep']
        pool, preps, fcs = L60.setup(enc)
        st = C.make_store(pool, split, M, V, rep)
        P = preps[prep]
        if test:
            want = [('NLMSC', tuple(c), b) for c, b in spec['cfgs']]
        else:
            want = [('NLMSC', c, b) for c in N62.grid('NLMSC') if c[2] == prep for b in spec['bpfs']]
        o = N62.run_online(st, P(st['keys']), P(st['queries']), fcs[prep], enc, prep, want,
                           C.seed_of('b62', enc, split, M, V, rep))  # B62's seed: 16-bit rows reproduce B62's
        g = st['qgold']
        for (a, cfg, b) in want:
            pred, bt, p = o[a, cfg, b]
            row = {'arm': a, 'cfg': list(cfg), 'bpf': b, 'acct': 'ideal', 'acc': float((pred == g).mean()),
                   'bits': bt, 'p': p}
            if test:
                row['pred'] = np.asarray(pred).astype(int).tolist()
            rows.append(row)
    else:
        pool, preps, Ls, _ = S64.setup(enc)
        st = C.make_store(pool, split, M, V, rep)
        order = S64.store_order(st)
        g = st['qgold']
        want = None
        if test:
            want = {}
            for cfg, b in spec['cfgs']:
                want.setdefault(tuple(cfg), set()).add(b)
        for prep in S64.PREPS:
            cs = [c for c in S64.GRID if c[0] == prep and (want is None or c in want)]
            if not cs:
                continue
            P = preps[prep]
            o = S64.run_sdmt(st, P(st['keys']), P(st['queries']), Ls[prep], [c[1:] for c in cs], spec['bpfs'],
                             ['ideal'], order)
            for (c5, b, a), (pred, bt, N) in o.items():
                cfg = (prep,) + tuple(c5)
                if want is not None and b not in want[cfg]:
                    continue
                row = {'arm': 'SDMT', 'cfg': list(cfg), 'bpf': b, 'acct': a, 'acc': float((pred == g).mean()),
                       'bits': bt, 'N': N}
                if test:
                    row['pred'] = np.asarray(pred).astype(int).tolist()
                rows.append(row)
    rec = {'spec': spec, 'qgold': st['qgold'].tolist(), 'rows': rows, 'sec': round(time.time() - t0, 1)}
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    with gzip.open(fn + '.tmp', 'wt') as fo:
        json.dump(rec, fo)
    os.replace(fn + '.tmp', fn)
    print(f'job {job_name(spec)} {rec["sec"]}s {len(rows)} rows', flush=True)
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
    big = {'qwen8': 4, 'b46': 4, 'llama': 4, 'mistral': 4, 'e5': 2, 'bge': 1, 'minilm': 1}
    specs.sort(key=lambda s: (-(s['M'] ** 2) * big[s['enc']] * (3 if s['arm'] == 'NLMSC' else 1), s['enc']))
    print(f'{len(specs)} jobs to run', flush=True)
    if workers <= 1:
        for s in specs:
            print(_run(s), flush=True)
        return
    import multiprocessing as mp
    with mp.get_context('fork').Pool(workers, maxtasksperchild=4) as pl:
        for r in pl.imap_unordered(_run, specs):
            if str(r).startswith('FAILED'):
                print(r, flush=True)


def make_plan(kind, encs):
    os.makedirs(OUT, exist_ok=True)
    fn = os.path.join(OUT, f'plan_{kind}.jsonl')
    n = 0
    with open(fn, 'w') as fo:
        for enc in encs:
            ch = jl(f'{RES}/b70_choice_{enc}.json')['choice'] if kind == 'test' else None
            for M in M_GRID:
                for V in VN:
                    nb = new_budgets(enc, M, V)
                    if not nb:
                        continue
                    reps = DEV_REPS[M] if kind == 'dev' else TEST_REPS[M]
                    bpfs = sorted({REPRO} | set(nb)) if kind == 'dev' else nb
                    for rep in range(reps):
                        if kind == 'dev':
                            for prep in N62.PREPS:
                                fo.write(json.dumps({'arm': 'NLMSC', 'enc': enc, 'split': 'dev', 'M': M, 'V': V,
                                                     'rep': rep, 'prep': prep, 'bpfs': bpfs, 'cfgs': None}) + '\n')
                                n += 1
                            fo.write(json.dumps({'arm': 'SDMT', 'enc': enc, 'split': 'dev', 'M': M, 'V': V, 'rep': rep,
                                                 'bpfs': bpfs, 'cfgs': None}) + '\n')
                            n += 1
                            continue
                        for prep in N62.PREPS:
                            cf = [[ch[f'{M}|{V}|{b}|NLMSC']['cfg'], b] for b in nb
                                  if ch[f'{M}|{V}|{b}|NLMSC']['cfg'][2] == prep]
                            if cf:
                                fo.write(json.dumps({'arm': 'NLMSC', 'enc': enc, 'split': 'test', 'M': M, 'V': V,
                                                     'rep': rep, 'prep': prep, 'bpfs': nb, 'cfgs': cf}) + '\n')
                                n += 1
                        if enc in CUT_SDMT:
                            continue
                        cf = [[ch[f'{M}|{V}|{b}|SDMT']['cfg'], b] for b in nb]
                        fo.write(json.dumps({'arm': 'SDMT', 'enc': enc, 'split': 'test', 'M': M, 'V': V, 'rep': rep,
                                             'bpfs': nb, 'cfgs': cf}) + '\n')
                        n += 1
    print(n, 'jobs ->', fn)


# ----------------------------------------------------------------------------------------------- dev choice

def choose(enc):
    """NLMSc: B62's (= B60's) rule; SDMt: B36's rule (as b64_run.choose); BEST7: B36's BEST rule over the six (B49's
    dev accuracies) + SDMt. Rule: highest mean dev accuracy over the cell's dev stores; within 0.1 points tied, first in
    grid order. Also: the 16-bit dev rows against B62's / B64's existing dev rows (reproduction check)."""
    acc = {}
    jd = os.path.join(OUT, 'jobs')
    for fn in sorted(os.listdir(jd)):
        if '_dev_' not in fn or not fn.split('_', 1)[1].startswith(f'{enc}_dev_'):
            continue
        r = json.load(gzip.open(os.path.join(jd, fn), 'rt'))
        s = r['spec']
        for x in r['rows']:
            acc.setdefault((s['M'], s['V'], x['bpf'], x['arm'], tuple(x['cfg'])), {})[s['rep']] = x
    grids = {'NLMSC': [tuple(c) for c in N62.grid('NLMSC')], 'SDMT': list(S64.GRID)}
    ch49 = jl(f'{ROOT}/results/b49_choice_{enc}.json')['choice']
    ch, repro = {}, {}
    for M in M_GRID:
        for V in VN:
            nb = new_budgets(enc, M, V)
            if not nb:
                continue
            for b in nb:
                for arm in ('NLMSC',) if enc in CUT_SDMT else ('NLMSC', 'SDMT'):
                    lst = []
                    for cfg in grids[arm]:
                        d = acc.get((M, V, b, arm, cfg))
                        if d is None or len(d) != DEV_REPS[M]:
                            raise SystemExit(f'missing / incomplete dev rows {enc} {M} {V} {b} {arm} {cfg}')
                        lst.append((float(np.mean([d[k]['acc'] for k in sorted(d)])), cfg, d[0]))
                    top = max(x[0] for x in lst)
                    tied = [x for x in lst if x[0] >= top - 0.001]
                    key = f'{M}|{V}|{b}|{arm}'
                    ch[key] = {'cfg': list(tied[0][1]), 'dev_acc': tied[0][0], 'best_dev_acc': top,
                               'n_tied': len(tied), 'bits_rep0': tied[0][2]['bits'],
                               'size_rep0': tied[0][2].get('p', tied[0][2].get('N'))}
                dbf = {f: ch49[f'{M}|{V}|{b}|ideal|BEST']['dev_by_fam'][f] for f in SIX}
                c6 = [(dbf[f], -SIX.index(f), f) for f in SIX]
                assert max(c6)[2] == ch49[f'{M}|{V}|{b}|ideal|BEST36']['fam'], (enc, M, V, b)
                if enc in CUT_SDMT:
                    continue
                dbf['SDMT'] = ch[f'{M}|{V}|{b}|SDMT']['dev_acc']
                c7 = [(dbf[f], -SEVEN.index(f), f) for f in SEVEN]
                ch[f'{M}|{V}|{b}|BEST7'] = {'fam': max(c7)[2], 'dev_acc': max(c7)[0], 'dev_by_fam': dbf,
                                            'BEST36': max(c6)[2]}
            # reproduction: 16-bit dev rows = B62's / B64's
            nmis, ncmp = 0, 0
            for rep in range(DEV_REPS[M]):
                for prep in N62.PREPS:
                    f62 = f'{ROOT}/b62_out/jobs/{enc}_dev_M{M}_V{V}_r{rep}_{prep}.json.gz'
                    old = {(tuple(x['cfg']), x['bpf']): x['acc'] for x in json.load(gzip.open(f62, 'rt'))['rows']
                           if x['arm'] == 'NLMSC'}
                    for cfg in grids['NLMSC']:
                        if cfg[2] != prep:
                            continue
                        ncmp += 1
                        nmis += abs(acc[M, V, REPRO, 'NLMSC', cfg][rep]['acc'] - old[cfg, REPRO]) > 1e-12
                f64 = f'{ROOT}/b64_out/jobs/{enc}_dev_M{M}_V{V}_r{rep}.json.gz'
                old = {tuple(x['cfg']): x['acc'] for x in json.load(gzip.open(f64, 'rt'))['rows']
                       if x['bpf'] == REPRO and x['acct'] == 'ideal'}
                for cfg in grids['SDMT']:
                    ncmp += 1
                    nmis += abs(acc[M, V, REPRO, 'SDMT', cfg][rep]['acc'] - old[cfg]) > 1e-12
            repro[f'{M}|{V}'] = {'compared': ncmp, 'mismatch': int(nmis)}
    fn = f'{RES}/b70_choice_{enc}.json'
    json.dump({'enc': enc, 'choice': ch, 'repro16': repro, 'cut': ['SDMT', 'BEST7'] if enc in CUT_SDMT else []}, open(fn, 'w'), indent=1, sort_keys=True)
    print(len(ch), 'choices ->', fn, 'repro', repro)


# ----------------------------------------------------------------------------------------------- selftest

def selftest():
    """Training-split stores only: budgets respected and p maximal for every NLMSc configuration at 32 / 64 bits per
    fact; the dual form = a genuine one-fact-at-a-time loop at 64 bits per fact; SDMt within budget at 32 / 64; the
    wrapper's rows equal a direct call of b62_run.run_online / b64_run.run_sdmt; timing."""
    enc = 'qwen8'
    pool, preps, fcs = L60.setup(enc)
    for M, V in ((256, 2), (256, 16)):
        st = C.make_store(pool, 'train', M, V, 0)
        P = preps['whiten']
        Yk, Yq = P(st['keys']), P(st['queries'])
        want = [('NLMSC', c, b) for c in N62.grid('NLMSC') if c[2] == 'whiten' for b in NEW]
        t0 = time.time()
        o = N62.run_online(st, Yk, Yq, fcs['whiten'], enc, 'whiten', want, 7)
        dt = time.time() - t0
        nover = nmax = 0
        for (a, c, b), (pred, bt, p) in o.items():
            cc = L60.ncols(V, c[1])
            nover += bt > b * M
            nmax += p < fcs['whiten'].pmax(c[3]) and L60.bits(p + 1, cc, c[6], 'ideal', c[0]) <= b * M
        best = max(float((pr == st['qgold']).mean()) for pr, _, _ in o.values())
        print(f'selftest NLMSc train M={M} V={V}: {len(o)} rows in {dt:.1f}s; over budget {nover}, not maximal {nmax}; '
              f'best train acc {best:.3f}', flush=True)
        assert nover == 0 and nmax == 0
        # dual = primal at 64 bits per fact (one rp configuration, V columns, mu 0.5, last)
        cfg = ('flat', 'V', 'whiten', 'rp', 0.5, 'last', 2)
        c = V
        p = L60.p_of(64 * M, c, 2, 'ideal', 'flat', fcs['whiten'].pmax('rp'))
        order = N62.store_order(st)
        F = np.hstack([fcs['whiten'].feats(Yk, 'rp', j0, min(L60.CH, p - j0)) for j0 in range(0, p, L60.CH)])
        F = F[order].astype(np.float64)
        Y = (np.eye(V) - 1.0 / V)[st['vals'][order]]
        Wl, _ = N62.nlms_primal(F, Y, 0.5)
        Wd = F.T @ N62.nlms_dual(F @ F.T, Y, 0.5)
        rel = float(np.abs(Wl - Wd).max() / np.abs(Wl).max())
        print(f'selftest NLMS primal vs dual at p = {p}: max rel |dW| {rel:.2e}', flush=True)
        assert rel < 1e-8
        # SDMt
        poolS, prepsS, Ls, _ = S64.setup(enc)
        stS = C.make_store(poolS, 'train', M, V, 0)
        PS = prepsS['whiten']
        t0 = time.time()
        oS = S64.run_sdmt(stS, PS(stS['keys']), PS(stS['queries']), Ls['whiten'],
                          [c[1:] for c in S64.GRID if c[0] == 'whiten'], list(NEW), ['ideal'], S64.store_order(stS))
        over = sum(bt > b * M for (c5, b, a), (pr, bt, N) in oS.items())
        bestS = max(float((pr == stS['qgold']).mean()) for pr, _, _ in oS.values())
        Ns = sorted({N for _, _, N in oS.values()})
        print(f'selftest SDMt train M={M} V={V}: {len(oS)} rows in {time.time() - t0:.1f}s; over budget {over}; '
              f'N range {Ns[0]}-{Ns[-1]}; best train acc {bestS:.3f}', flush=True)
        assert over == 0
    # wrapper = direct call (one training store, both arms; run_job writes into a temporary OUT)
    import tempfile
    global OUT
    keep = OUT
    OUT = tempfile.mkdtemp()
    spec = {'arm': 'NLMSC', 'enc': enc, 'split': 'train', 'M': 256, 'V': 16, 'rep': 0, 'prep': 'center',
            'bpfs': [16, 64], 'cfgs': None}
    r = json.load(gzip.open(run_job(spec), 'rt'))
    st = C.make_store(pool, 'train', 256, 16, 0)
    P = preps['center']
    want = [('NLMSC', c, b) for c in N62.grid('NLMSC') if c[2] == 'center' for b in (16, 64)]
    o = N62.run_online(st, P(st['keys']), P(st['queries']), fcs['center'], enc, 'center', want,
                       C.seed_of('b62', enc, 'train', 256, 16, 0))
    dif = sum(abs(x['acc'] - float((o['NLMSC', tuple(x['cfg']), x['bpf']][0] == st['qgold']).mean())) > 1e-12
              for x in r['rows'])
    spec = {'arm': 'SDMT', 'enc': enc, 'split': 'train', 'M': 256, 'V': 2, 'rep': 0, 'bpfs': [16, 64], 'cfgs': None}
    rS = json.load(gzip.open(run_job(spec), 'rt'))
    print(f'selftest wrapper: NLMSc {len(r["rows"])} rows, {dif} differ from a direct call; SDMt {len(rS["rows"])} rows',
          flush=True)
    assert dif == 0 and len(rS['rows']) == 504 * 2
    OUT = keep
    print('selftest PASS')


def main():
    a = sys.argv[1:]
    if a[0] == 'region':
        region()
    elif a[0] == 'selftest':
        selftest()
    elif a[0] == 'plan':
        encs = a[a.index('--encs') + 1:] if '--encs' in a else PANEL
        make_plan(a[1], encs)
    elif a[0] == 'runplan':
        runplan(a[1], int(a[a.index('--workers') + 1]) if '--workers' in a else 1)
    elif a[0] == 'choose':
        choose(a[1])
    else:
        raise SystemExit(__doc__)


if __name__ == '__main__':
    main()
