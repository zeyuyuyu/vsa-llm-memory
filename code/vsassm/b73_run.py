"""B73 (prereg/B73_confirmatory_prereg.md): the paper's equal-bit memories on FRESH data with every configuration frozen.

Every memory is run by its own module's code, called unchanged, with (i) the data root switched to the fresh entity set
(B53_DATA / B60_DATA / B64_DATA; b73_data.layout), (ii) the value seed changed (b32_common.make_store's seed tag 'store'
-> 'B73store'; nothing else in make_store changes), (iii) R1 only (V 2 / 16 x 4, 8, 16 bits per fact x M 256 / 1,024),
ideal accounting, and (iv) the configuration of each cell taken from the paper's PopQA dev choice files (results/*;
frozen table results/b73_frozen.json). No dev store of the fresh set is built or read; the fresh training split supplies
the fitted side information exactly as PopQA's did (centring / whitening statistics, principal bases, SDM locations,
LMMSE training statistics).

Arms (test stores: 20 at M = 256, 5 at M = 1,024 per (V), as the paper):
  STORE   final design = B53's VSA (b53_run.test's loop; Feat53 antipodal / Hadamard codes; results/b53_choice)
  VSAR    first design = B49's VSA (the same loop with b32_run.Feat's random codes; results/b49_choice '|VSA')
  VSAS    B56's dev-selected read / write (STAR: LM or RT) on B53's bundle (b56_run helpers; results/b56_choice)
  VSAPC   B60's VSAPc (b57_run.engine with b60_vsapc's corrected basis; results/b60_choicepc PSTAR)
  VSAPCX  per cell VSAPC if its PopQA dev accuracy > VSAS's + 0.001, else VSAS (b60_verdict.add_vsapx's rule; frozen)
  LAMC    B60's LAMc (b60_run.run_job; results/b60_choice '|LAMC')
  NLMSC   B62's NLMSc (b62_run.run_job, arm NLMSC only; results/b62_choice)
  SDMT    B64's SDMt (b64_run.run_job; results/b64_choice '|SDMT')

python3 b73_run.py frozen                      # -> results/b73_frozen.json
python3 b73_run.py plan [--encs ..] [--arms ..] > PLAN.txt   # one line per unit "ARM ENC M V [REP]"
python3 b73_run.py unit ARM ENC M V [REP]      # one unit (writes under B73_OUT)
python3 b73_run.py runplan PLAN.txt --workers N
python3 b73_run.py assemble ENC                # -> B73_OUT/asm/{enc}.json.gz (records with pred['ARM|b|ideal'])
environment: B73_ROOT (fresh data root, default b73_out/fresh), B73_OUT (default b73_out/run)
"""
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
FRESH = os.path.abspath(os.environ.get('B73_ROOT', os.path.join(ROOT, 'b73_out', 'fresh')))
OUT = os.path.abspath(os.environ.get('B73_OUT', os.path.join(ROOT, 'b73_out', 'run')))
for k, v in {'B53_DATA': FRESH, 'B60_DATA': os.path.join(FRESH, 'b49_out'), 'B64_DATA': os.path.join(FRESH, 'b49_out'),
             'B53_OUT': os.path.join(OUT, 'store'), 'B56_OUT': os.path.join(OUT, 'vsas'),
             'B57_OUT': os.path.join(OUT, 'vsapc'), 'B60_OUT': os.path.join(OUT, 'lamc'),
             'B62_OUT': os.path.join(OUT, 'nlmsc'), 'B64_OUT': os.path.join(OUT, 'sdmt'),
             'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}.items():
    os.environ[k] = v  # before any module is imported (each reads its paths at import)
import gzip  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, ROOT)
import b32_common as C  # noqa: E402

_make_store = C.make_store
SEED_TAG = 'B73store'


def make_store_b73(pool, split, M, V, rep):
    """b32_common.make_store with the seed tag 'store' replaced by 'B73store' (new entity draw and new values)."""
    orig = C.seed_of
    C.seed_of = lambda *parts: orig(*((SEED_TAG,) + parts[1:])) if parts and parts[0] == 'store' else orig(*parts)
    try:
        return _make_store(pool, split, M, V, rep)
    finally:
        C.seed_of = orig


C.make_store = make_store_b73

PANEL = ['qwen8', 'b46', 'llama', 'mistral', 'e5', 'bge', 'minilm']
LLM, SENT = PANEL[:4], PANEL[4:]
M_GRID, VN, BPF, ACCT = [256, 1024], [2, 16], [4, 8, 16], 'ideal'
TEST_REPS = {256: 20, 1024: 5}
ARMS = ['STORE', 'VSAR', 'VSAS', 'VSAPC', 'LAMC', 'NLMSC', 'SDMT']
CHOICE_FILES = {'STORE': 'b53_choice_{enc}.json', 'VSAR': 'b49_choice_{enc}.json', 'VSAS': 'b56_choice_{enc}.json',
                'VSAPC': 'b60_choicepc_{enc}.json', 'LAMC': 'b60_choice_{enc}.json', 'NLMSC': 'b62_choice_{enc}.json',
                'SDMT': 'b64_choice_{enc}.json'}


def ch(arm, enc):
    return json.load(open(os.path.join(ROOT, 'results', CHOICE_FILES[arm].format(enc=enc))))['choice']


def cells():
    return [(M, V, b) for M in M_GRID for V in VN for b in BPF]


# ----------------------------------------------------------------------------------------------- frozen table

def frozen():
    """Per encoder and R1 cell: each arm's PopQA dev-chosen configuration (and VSAPCX's source)."""
    J = {'rule': 'PopQA dev choices of the paper runs (B49 / B53 / B56 / B60 / B62 / B64), ideal accounting, R1 cells; '
                 'VSAPCX source = VSAPC iff its PopQA dev accuracy > VSAS dev accuracy + 0.001 (b60_verdict.add_vsapx)',
         'choice_files_md5': {}, 'cells': {}}
    for enc in PANEL:
        for arm, f in CHOICE_FILES.items():
            fn = os.path.join(ROOT, 'results', f.format(enc=enc))
            J['choice_files_md5'][os.path.basename(fn)] = hashlib.md5(open(fn, 'rb').read()).hexdigest()
        c53, c49, c56, cpc = ch('STORE', enc), ch('VSAR', enc), ch('VSAS', enc), ch('VSAPC', enc)
        c60, c62, c64 = ch('LAMC', enc), ch('NLMSC', enc), ch('SDMT', enc)
        for (M, V, b) in cells():
            k = f'{M}|{V}|{b}|ideal'
            star = c56[f'{k}|STAR']
            sf = star['fam']
            pst = cpc[f'{k}|PSTAR']
            pf = pst['fam']
            ds, dpc = star['dev'][sf], pst['dev'][pf]
            J['cells'][f'{enc}|{k}'] = {
                'STORE': c53[f'{k}|VSA']['cfg'], 'VSAR': c49[f'{k}|VSA']['cfg'],
                'VSAS': {'fam': sf, 'cfg': c56[f'{k}|{sf}']['cfg'], 'bundle': c53[f'{k}|VSA']['cfg'], 'dev': ds},
                'VSAPC': {'fam': pf, 'cfg': cpc[f'{k}|{pf}']['cfg'], 'bundle': cpc[f'{k}|VSA']['cfg'], 'dev': dpc},
                'VSAPCX': 'VSAPC' if dpc > ds + 0.001 else 'VSAS',
                'LAMC': c60[f'{k}|LAMC']['cfg'], 'NLMSC': c62[f'{M}|{V}|{b}|NLMSC']['cfg'], 'SDMT': c64[f'{k}|SDMT']['cfg']}
    fn = os.path.join(ROOT, 'results', 'b73_frozen.json')
    json.dump(J, open(fn, 'w'), indent=1, sort_keys=True)
    print(len(J['cells']), 'cells ->', fn)


# ----------------------------------------------------------------------------------------------- units

def odir(*a):
    d = os.path.join(OUT, *a)
    os.makedirs(d, exist_ok=True)
    return d


def unit_store(arm, enc, M, V):
    """STORE / VSAR: b53_run.test's loop for one (M, V), R1 budgets, ideal (VSAR: b32_run.Feat, B49's choice)."""
    import b32_run as R
    import b53_run as R53
    fn = os.path.join(odir(arm.lower(), enc), f'{arm.lower()}_{enc}_M{M}_V{V}.jsonl.gz')
    if os.path.exists(fn):
        return fn
    c = ch(arm, enc)
    pool, preps = R53.setup(enc)
    FeatCls = R53.Feat53 if arm == 'STORE' else R.Feat
    feats = {}
    with gzip.open(fn + '.tmp', 'wt') as fo:
        for rep in range(TEST_REPS[M]):
            t0 = time.time()
            st = C.make_store(pool, 'test', M, V, rep)
            rec = {'enc': enc, 'M': st['M'], 'V': V, 'rep': rep, 'qgold': st['qgold'].tolist(),
                   'qent': st['qent'].tolist(), 'pred': {}, 'cfg': {}, 'd': {}}
            groups = {}
            for bpf in BPF:
                cf = c[f'{M}|{V}|{bpf}|ideal|VSA']['cfg']
                groups.setdefault((cf[0], cf[1]), []).append((bpf, 'ideal', cf[2], cf[3]))
                rec['cfg'][f'{arm}|{bpf}|ideal'] = cf
            for (prep, fmap), lst in groups.items():
                if (prep, fmap) not in feats:
                    feats[prep, fmap] = FeatCls(enc, prep, fmap, pool.X.shape[1], 'cpu')
                feats[prep, fmap].V = V
                cfgs = [((bpf, acct, q, f), C.vsa_dim(bpf * st['M'], q, acct), q,
                         max(1, int(math.floor(f * st['M'])))) for bpf, acct, q, f in lst]
                P = preps[prep]
                pr = R.vsa_grid(P(st['keys']), st['vals'], P(st['queries']), V, feats[prep, fmap], cfgs,
                                R.store_seed(enc, st), R.store_order(st))
                for (key, d, q, ns) in cfgs:
                    rec['pred'][f'{arm}|{key[0]}|{key[1]}'] = pr[key].tolist()
                    rec['d'][f'{arm}|{key[0]}|{key[1]}'] = [d, q, ns]
            fo.write(json.dumps(rec) + '\n')
            print(f'{arm} {enc} M={M} V={V} rep={rep} {time.time() - t0:.0f}s', flush=True)
    os.replace(fn + '.tmp', fn)
    return fn


def unit_vsas(enc, M, V):
    """VSAS: b56_run.dec's test path for one (M, V), restricted to the STAR-chosen arm (LM or RT) of each R1 cell."""
    import b53_run as R53
    import b56_run as E56
    E56.BPF, E56.ACCT32 = list(BPF), ['ideal']
    fn = os.path.join(odir('vsas', enc), f'vsas_{enc}_M{M}_V{V}.jsonl.gz')
    if os.path.exists(fn):
        return fn
    ch53, ch56 = ch('STORE', enc), ch('VSAS', enc)
    select = set()
    for b in BPF:
        sf = ch56[f'{M}|{V}|{b}|ideal|STAR']['fam']
        c = ch56[f'{M}|{V}|{b}|ideal|{sf}']['cfg']
        select.add((sf, b, 'ideal', c[0], c[1]))
    pool, preps = R53.setup(enc)
    Tr = pool.train_matrix()
    groups = E56.plan_cells(ch53, M, V, select)
    recs = {rep: {'enc': enc, 'M': M, 'V': V, 'rep': rep, 'pred': {}, 'info': {}} for rep in range(TEST_REPS[M])}
    for (prep, fmap), g in groups.items():
        if not (g['LM'] or g['RT']):
            continue
        feat = E56.make_feat(enc, prep, fmap, pool.X.shape[1], V)
        P = preps[prep]
        dl = sorted({x[1] for x in g['LM']})
        F = E56.train_feats(feat, P, Tr, enc, prep, fmap, V, max(dl + [1]))
        decs = E56.lm_decoders(F, V, dl)
        for rep in range(TEST_REPS[M]):
            t1 = time.time()
            st = C.make_store(pool, 'test', M, V, rep)
            res = E56.run_store_group(enc, feat, P, st, F, decs, g, V)
            rc = recs[rep]
            rc['qgold'], rc['qent'] = st['qgold'].tolist(), st['qent'].tolist()
            for key, (p, info) in res.items():
                k = '|'.join(map(str, key))
                rc['pred'][k] = np.asarray(p).tolist()
                rc['info'][k] = info + [prep, fmap]
            print(f'VSAS {enc} M={M} V={V} {prep}:{fmap} rep={rep} {time.time() - t1:.0f}s', flush=True)
        del feat, F, decs
    with gzip.open(fn + '.tmp', 'wt') as fo:
        for rep in range(TEST_REPS[M]):
            fo.write(json.dumps(recs[rep]) + '\n')
    os.replace(fn + '.tmp', fn)
    return fn


def unit_vsapc(enc, M, V):
    """VSAPC: b60_vsapc's 'decp test' path (b57_run.engine on the corrected basis), restricted to the PSTAR arm."""
    import b57_run as R57
    import b60_vsapc  # noqa: F401  (patches b57_run.setup with the corrected whiten basis; B57_OUT already set)
    R57.BPF, R57.ACCT32 = list(BPF), ['ideal']
    import b56_run as E56
    E56.BPF, E56.ACCT32 = list(BPF), ['ideal']
    assert R57.OUT == os.path.join(OUT, 'vsapc'), R57.OUT
    fn = os.path.join(odir('vsapc', enc), f'vsapc_{enc}_M{M}_V{V}.jsonl.gz')
    if os.path.exists(fn):
        return fn
    cpc = ch('VSAPC', enc)
    chp = {k: v for k, v in cpc.items() if k.endswith('|VSA')}
    sel = set()
    for b in BPF:
        pf = cpc[f'{M}|{V}|{b}|ideal|PSTAR']['fam']
        c = cpc[f'{M}|{V}|{b}|ideal|{pf}']['cfg']
        sel.add((pf[1:], b, 'ideal', c[0], c[1]))
    R57.engine(enc, 0, 'test', M, V, chp, ['LM', 'RT'], sel, fn)
    return fn


def unit_rep(arm, enc, M, V, rep):
    """LAMC / NLMSC / SDMT: the module's run_job on test store rep with the frozen configurations of the R1 cells."""
    if arm == 'LAMC':
        import b60_run as L
        L.BPF, L.ACCTS = list(BPF), ['ideal']
        c = ch('LAMC', enc)
        for prep in L.PREPS:
            cfgs = [[c[f'{M}|{V}|{b}|ideal|LAMC']['cfg'], b, 'ideal'] for b in BPF
                    if c[f'{M}|{V}|{b}|ideal|LAMC']['cfg'][2] == prep]
            if cfgs:
                L.run_job({'enc': enc, 'map': 0, 'split': 'test', 'M': M, 'V': V, 'rep': rep, 'prep': prep,
                           'cfgs': cfgs})
    elif arm == 'NLMSC':
        import b62_run as O
        O.ARMS = ['NLMSC']
        c = ch('NLMSC', enc)
        for prep in O.PREPS:
            want = [['NLMSC', c[f'{M}|{V}|{b}|NLMSC']['cfg'], b] for b in BPF
                    if c[f'{M}|{V}|{b}|NLMSC']['cfg'][2] == prep]
            if want:
                O.run_job({'enc': enc, 'split': 'test', 'M': M, 'V': V, 'rep': rep, 'prep': prep, 'want': want})
    elif arm == 'SDMT':
        import b64_run as S
        S.BPF, S.ACCTS = list(BPF), ['ideal']
        c = ch('SDMT', enc)
        S.run_job({'enc': enc, 'split': 'test', 'M': M, 'V': V, 'rep': rep,
                   'cfgs': [[c[f'{M}|{V}|{b}|ideal|SDMT']['cfg'], b, 'ideal'] for b in BPF]})
    else:
        raise SystemExit(arm)


def unit(arm, enc, M, V, rep=None):
    M, V = int(M), int(V)
    if arm in ('STORE', 'VSAR'):
        return unit_store(arm, enc, M, V)
    if arm == 'VSAS':
        return unit_vsas(enc, M, V)
    if arm == 'VSAPC':
        return unit_vsapc(enc, M, V)
    lo, hi = (int(x) for x in str(rep).split('-')) if '-' in str(rep) else (int(rep), int(rep))
    for r in range(lo, hi + 1):  # one process per chunk of test stores: the side information is fitted once
        unit_rep(arm, enc, M, V, r)


def plan(encs, arms):
    """Units in the registered priority order: (1) LLM core STORE / VSAPCX (VSAS + VSAPC) / LAMC / NLMSC on Qwen3-8B,
    Llama, Mistral; (2) the same on the sentence encoders; (3) Granite; (4) VSAR; (5) SDMT (LLMs, then sentence)."""
    core = ['LAMC', 'NLMSC', 'STORE', 'VSAS', 'VSAPC']
    order = []
    for grp in (['qwen8', 'llama', 'mistral'], SENT, ['b46']):
        order += [(a, e) for e in grp for a in core]
    order += [('VSAR', e) for e in PANEL]
    order += [('SDMT', e) for e in LLM + SENT]
    lines = []
    for a, e in order:
        if e not in encs or a not in arms:
            continue
        for M in (1024, 256):
            for V in VN:
                if a in ('LAMC', 'NLMSC', 'SDMT'):
                    lines += [f'{a} {e} {M} {V} {r}-{min(r + 4, TEST_REPS[M] - 1)}' for r in range(0, TEST_REPS[M], 5)]
                else:
                    lines.append(f'{a} {e} {M} {V}')
    return lines


def runplan(planfile, workers):
    """Each unit in its own process (module state, caches and cwd are per process); units in plan order."""
    from concurrent.futures import ThreadPoolExecutor
    units = [l.split() for l in open(planfile) if l.strip()]
    py = sys.executable
    os.makedirs(odir('logs'), exist_ok=True)
    t0 = time.time()

    deadline = float(os.environ.get('B73_DEADLINE', 'inf'))  # epoch seconds; no unit is started after it (cut rule)

    def go(u):
        tag = '_'.join(u)
        if time.time() > deadline:
            print(f'{time.strftime("%H:%M:%S")} CUT {tag} (not started: deadline)', flush=True)
            return 'cut'
        lf = os.path.join(OUT, 'logs', f'{tag}.log')
        t1 = time.time()
        r = subprocess.run([py, os.path.abspath(__file__), 'unit'] + u, stdout=open(lf, 'w'), stderr=subprocess.STDOUT)
        msg = f'{time.strftime("%H:%M:%S")} done {tag} rc={r.returncode} {time.time() - t1:.0f}s'
        print(msg, flush=True)
        return r.returncode

    with ThreadPoolExecutor(workers) as ex:
        rcs = list(ex.map(go, units))
    print(f'finished {len(units)} units, failures {sum(r not in (0, "cut") for r in rcs)}, cut {sum(r == "cut" for r in rcs)}, '
          f'{time.time() - t0:.0f}s', flush=True)


# ----------------------------------------------------------------------------------------------- assemble

def assemble(enc):
    """Records per (M, V) and rep: qgold, qent, rep, pred['ARM|b|ideal'] for STORE, VSAR, VSAS, VSAPC, VSAPCX, LAMC,
    NLMSC, SDMT (absent arms omitted)."""
    fz = json.load(open(os.path.join(ROOT, 'results', 'b73_frozen.json')))['cells']
    data = {}
    for M in M_GRID:
        for V in VN:
            recs = None
            fn = os.path.join(OUT, 'store', 'store', enc, f'store_{enc}_M{M}_V{V}.jsonl.gz')
            fn = fn if os.path.exists(fn) else os.path.join(OUT, 'store', enc, f'store_{enc}_M{M}_V{V}.jsonl.gz')
            base = {}
            for arm in ('STORE', 'VSAR'):
                f2 = os.path.join(OUT, arm.lower(), enc, f'{arm.lower()}_{enc}_M{M}_V{V}.jsonl.gz')
                if os.path.exists(f2):
                    for r in map(json.loads, gzip.open(f2, 'rt')):
                        b0 = base.setdefault(r['rep'], {'rep': r['rep'], 'qgold': r['qgold'], 'qent': r['qent'],
                                                        'pred': {}})
                        assert b0['qgold'] == r['qgold'] and b0['qent'] == r['qent']
                        b0['pred'].update(r['pred'])
            for arm, sub in (('VSAS', 'vsas'), ('VSAPC', 'vsapc')):
                f2 = os.path.join(OUT, sub, enc, f'{sub}_{enc}_M{M}_V{V}.jsonl.gz')
                if not os.path.exists(f2):
                    continue
                for r in map(json.loads, gzip.open(f2, 'rt')):
                    b0 = base.setdefault(r['rep'], {'rep': r['rep'], 'qgold': r['qgold'], 'pred': {}})
                    assert b0['qgold'] == r['qgold']
                    for b in BPF:
                        z = fz[f'{enc}|{M}|{V}|{b}|ideal'][arm]
                        fam = z['fam'] if arm == 'VSAS' else z['fam'][1:]
                        kk = '|'.join(map(str, (fam, b, 'ideal', z['cfg'][0], z['cfg'][1])))
                        b0['pred'][f'{arm}|{b}|ideal'] = r['pred'][kk]
            for arm, sub in (('LAMC', 'lamc'), ('NLMSC', 'nlmsc'), ('SDMT', 'sdmt')):
                jd = os.path.join(OUT, sub, 'jobs')
                if not os.path.isdir(jd):
                    continue
                for rep in range(TEST_REPS[M]):
                    pref = {'LAMC': f'{enc}_m0_test_M{M}_V{V}_r{rep}_', 'NLMSC': f'{enc}_test_M{M}_V{V}_r{rep}_',
                            'SDMT': f'{enc}_test_M{M}_V{V}_r{rep}.json.gz'}[arm]
                    fs = [f for f in os.listdir(jd) if f.startswith(pref)] if arm != 'SDMT' else \
                        [f for f in os.listdir(jd) if f == pref]
                    for f in fs:
                        r = json.load(gzip.open(os.path.join(jd, f), 'rt'))
                        b0 = base.setdefault(rep, {'rep': rep, 'qgold': r['qgold'], 'pred': {}})
                        assert b0['qgold'] == r['qgold'], (arm, enc, M, V, rep)
                        for x in r['rows']:
                            b0['pred'][f'{arm}|{x["bpf"]}|ideal'] = x['pred']
                            b0.setdefault('bits', {})[f'{arm}|{x["bpf"]}|ideal'] = [x['bits'], x.get('p', x.get('N'))]
            for rep, b0 in base.items():
                for b in BPF:
                    src = fz[f'{enc}|{M}|{V}|{b}|ideal']['VSAPCX']
                    if f'{src}|{b}|ideal' in b0['pred']:
                        b0['pred'][f'VSAPCX|{b}|ideal'] = b0['pred'][f'{src}|{b}|ideal']
            recs = [base[k] for k in sorted(base)]
            if recs and 'qent' not in recs[0]:  # qent from a fresh make_store (no codes needed)
                pass
            data[f'{M}|{V}'] = recs
    fn = os.path.join(odir('asm'), f'{enc}.json.gz')
    with gzip.open(fn, 'wt') as fo:
        json.dump(data, fo)
    print('assembled', enc, {k: (len(v), sorted({a.split('|')[0] for r in v for a in r['pred']})) for k, v in data.items()})
    return fn


def main():
    a = sys.argv[1:]
    if a[0] == 'frozen':
        frozen()
    elif a[0] == 'plan':
        encs = a[a.index('--encs') + 1].split(',') if '--encs' in a else PANEL
        arms = a[a.index('--arms') + 1].split(',') if '--arms' in a else ARMS
        print('\n'.join(plan(encs, arms)))
    elif a[0] == 'unit':
        t0 = time.time()
        unit(*a[1:])
        print(f'unit {" ".join(a[1:])} {time.time() - t0:.0f}s', flush=True)
    elif a[0] == 'runplan':
        runplan(a[1], int(a[a.index('--workers') + 1]))
    elif a[0] == 'assemble':
        for e in a[1:]:
            assemble(e)
    else:
        raise SystemExit(a[0])


if __name__ == '__main__':
    main()
