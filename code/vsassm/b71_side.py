"""B71 part (D) (prereg/B71_hebbian_remedies_prereg.md): uncharged side information and operations per arm; charged
re-scoring D1 (full charge per store) and D2 (bounded to what the compared store design itself uses).

Entries (numbers) of the fixed components, at the arm's chosen configuration of each R1 cell (ideal):
  mean n; whitening (ZCA, symmetric) n (n + 1) / 2; principal basis n x p (dropped from the MINIMAL count when the arm also
  whitens: the ZCA matrix has the same eigenvectors); the LMMSE read's T training codes T x n; SDMt's N hard locations
  N x n; the learned map (LearnedPrep) n x r + n; random projections / value codes / rotations / location orders = 64-bit
  seeds. Bits at 8 bits per entry (and 32) + 64 per seed.

python3 b71_side.py table                   # -> results/b71_side.{json,txt}
python3 b71_side.py d2plan                  # D2: LAMc re-choice (dev rows exist) + SDMt capped dev plan
python3 b71_side.py d2lam_choose            # -> results/b71_d2_lam_choice.json
python3 b71_side.py d2sdm_run PLAN --workers N
python3 b71_side.py d2sdm_choose            # -> results/b71_d2_sdm_choice.json
python3 b71_side.py d2test_plan             # test jobs of the re-chosen configurations
python3 b71_side.py d2verdict               # -> results/b71_d2_verdict.{json,txt}
"""
import gzip
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import b32_common as C  # noqa: E402

RES = os.environ.get('B71_RES', f'{ROOT}/results')
OUT = os.environ.get('B71_OUT', f'{ROOT}/b71_out')
PANEL = ['qwen8', 'b46', 'llama', 'mistral', 'e5', 'bge', 'minilm']
NAMES = {'qwen8': 'Qwen3-8B', 'b46': 'Granite-3.3-8B', 'llama': 'Llama-3.1-8B', 'mistral': 'Mistral-7B',
         'e5': 'e5-base-v2', 'bge': 'bge-small', 'minilm': 'MiniLM-L6'}
NDIM = {'qwen8': 4096, 'b46': 4096, 'llama': 4096, 'mistral': 4096, 'e5': 768, 'bge': 384, 'minilm': 384}
M_GRID, VN, BPF = [256, 1024], [2, 16], [4, 8, 16]
CELLS = [(M, V, b) for M in M_GRID for V in VN for b in BPF]
T_V = {2: 1024, 16: 512}
ARMS = ['STORE', 'VSAR', 'VSAS', 'VSAPC', 'VSAPCX', 'VSAPR', 'VSADR1', 'LAMC', 'NLMSC', 'SDMT', 'LHI']


def jl(fn):
    return json.load(open(fn))


def objs_prep(prep, n):
    return {'mean': n} if prep == 'center' else {'mean': n, 'zca': n * (n + 1) // 2}


def minimal(o):
    o = dict(o)
    if 'zca' in o and 'basis' in o:
        o.pop('basis')  # the ZCA matrix's eigenvectors are the principal directions
    return o


def merged_cfg(enc, M, V, b, arm):
    fv = f'{ROOT}/results/b71_vsar_cfg.json'  # VSAR's configurations (copied from the merged records for the box)
    if arm == 'VSAR' and not os.path.exists(f'{ROOT}/b59_out') and os.path.exists(fv):
        return jl(fv)[f'{enc}|{M}|{V}|{b}']
    r = json.loads(gzip.open(f'{ROOT}/b59_out/fam/{enc}/all59_{enc}_M{M}_V{V}.jsonl.gz', 'rt').readline())
    return r['cfg'][f'{arm}|{b}|ideal']


_B64 = {}


def sdmt_N(enc, M, V, b, cfg):
    Ns = []
    for rep in range({256: 20, 1024: 5}[M]):
        fn = f'{ROOT}/b64_out/jobs/{enc}_test_M{M}_V{V}_r{rep}.json.gz'
        if fn not in _B64:
            _B64.clear() if len(_B64) > 30 else None
            _B64[fn] = json.load(gzip.open(fn, 'rt'))['rows']
        for x in _B64[fn]:
            if tuple(x['cfg']) == tuple(cfg) and x['bpf'] == b and x['acct'] == 'ideal':
                Ns.append(x['N'])
    return int(np.median(Ns)) if Ns else 0


def arm_side(enc):
    """{arm: {cell: (objects dict, seeds, write ops, read ops)}} at the chosen configurations."""
    import b60_run as L60
    n = NDIM[enc]
    ch53 = jl(f'{ROOT}/results/b53_choice_{enc}.json')['choice']
    ch56 = jl(f'{ROOT}/results/b56_choice_{enc}.json')['choice']
    chpc = jl(f'{ROOT}/results/b60_choicepc_{enc}.json')['choice']
    ch60 = jl(f'{ROOT}/results/b60_choice_{enc}.json')['choice']
    ch62 = jl(f'{ROOT}/results/b62_choice_{enc}.json')['choice']
    ch64 = jl(f'{ROOT}/results/b64_choice_{enc}.json')['choice']
    ch49 = jl(f'{ROOT}/results/b49_choice_{enc}.json')['choice']
    ch71 = jl(f'{RES}/b71_choice_{enc}.json')['choice'] if os.path.exists(f'{RES}/b71_choice_{enc}.json') else None
    lh = jl(f'{ROOT}/b49_out/fam/{enc}/lh_choice.json')
    out = {a: {} for a in ARMS}
    for (M, V, b) in CELLS:
        cell = f'{M}|{V}|{b}'
        Bud = b * M

        def store_like(prep, fmap, q, extra_w=0.0, extra_r=0.0, extra=None, seeds=1):
            d = C.vsa_dim(Bud, q, 'ideal')
            K = 3 if fmap == 'sgn3' else 1
            o = objs_prep(prep, n)
            o.update(extra or {})
            return (o, seeds, K * n * d + d + extra_w, K * n * d + V * d + extra_r)
        prep, fmap, q, f = ch53[f'{cell}|ideal|VSA']['cfg']
        out['STORE'][cell] = store_like(prep, fmap, q)
        out['VSADR1'][cell] = store_like(prep, fmap, q)
        if ch71:
            E = ch71[f'{cell}|VSADR1']['cfg'][0]
            d = C.vsa_dim(Bud, ch71[f'{cell}|VSADR1']['cfg'][4], 'ideal')
            out['VSADR1'][cell] = store_like(prep, fmap, ch71[f'{cell}|VSADR1']['cfg'][4], extra_w=E * 3 * d)
        fam = ch56[f'{cell}|ideal|STAR']['fam']
        c56 = ch56[f'{cell}|ideal|{fam}']['cfg']
        d = C.vsa_dim(Bud, c56[0], 'ideal')
        if fam == 'RT':
            E = int(str(c56[1]).split(':')[0])
            out['VSAS'][cell] = store_like(prep, fmap, c56[0], extra_w=E * (V * d + d))
        else:
            out['VSAS'][cell] = store_like(prep, fmap, c56[0], extra_r=T_V[V] * d,
                                           extra={'train_codes': T_V[V] * n})
        vp, vf, vq, _ = merged_cfg(enc, M, V, b, 'VSAR')
        out['VSAR'][cell] = store_like(vp, vf, vq)
        pp, pf, pq, _ = chpc[f'{cell}|ideal|VSA']['cfg']
        dp = min(C.vsa_dim(Bud, pq, 'ideal'), n)
        o = objs_prep(pp, n)
        o['basis'] = n * dp
        out['VSAPC'][cell] = (o, 1, n * dp + dp, n * dp + V * dp)
        s = ch56[f'{cell}|ideal|STAR']
        pc = chpc[f'{cell}|ideal|PSTAR']
        out['VSAPCX'][cell] = out['VSAPC'][cell] if pc['dev'][pc['fam']] > s['dev'][s['fam']] + 0.001 else out['VSAS'][cell]
        if ch71:
            rp, rf, p, rq, _ = ch71[f'{cell}|VSAPR']['cfg']
            d = C.vsa_dim(Bud, rq, 'ideal')
            K = 3 if rf == 'sgn3' else 1
            o = objs_prep(rp, n)
            o['basis'] = n * p
            out['VSAPR'][cell] = (o, 2, n * p + K * p * d + d, n * p + K * p * d + V * d)
        sc, t, lp, ft, al, lq = ch60[f'{cell}|ideal|LAMC']['cfg']
        c = V - 1 if t == 'V-1' else V
        pm = n if ft == 'pca' else 65536
        p = L60.p_of(Bud, c, lq, 'ideal', sc, pm)
        o = objs_prep(lp, n)
        if ft == 'pca':
            o['basis'] = n * p
        wb = min(M * M * p + M ** 3 / 3 + M * M * c + M * p * c, M * p * p + p ** 3 / 3 + M * p * c + p * p * c) / M
        out['LAMC'][cell] = (o, 0 if ft == 'pca' else 1, n * p + wb, n * p + p * c)
        x = ch62[f'{M}|{V}|{b}|NLMSC']
        sc, t, lp, ft, mu, oo, lq = x['cfg']
        c = V - 1 if t == 'V-1' else V
        p = x['p']
        o = objs_prep(lp, n)
        if ft == 'pca':
            o['basis'] = n * p
        out['NLMSC'][cell] = (o, 0 if ft == 'pca' else 1, n * p + 2 * p * c + p, n * p + p * c)
        cfg = ch64[f'{cell}|ideal|SDMT']['cfg']
        N = sdmt_N(enc, M, V, b, cfg)
        o = objs_prep(cfg[0], n)
        o['locations'] = N * n
        cc = V if cfg[3] == 'onehot' else max(1, math.ceil(math.log2(V)))
        out['SDMT'][cell] = (o, 1, N * n + cfg[2], N * n + cfg[2] * cc)
        r_ = lh['r']
        o = {'mean': n, 'lhmap': n * r_ + n}
        out['LHI'][cell] = (o, 1, n * r_, n * r_)
    return out


def table():
    J = {}
    L = ['# B71 (D): uncharged side information and operations at the chosen configurations (R1, ideal).',
         '# Entries = numbers of the fixed components (MINIMAL: principal basis not counted again when the arm whitens).',
         '# Bits at 8 bits per entry + 64 per seed; median [range] over the 12 R1 cells; ratio = side bits / budget.',
         '# D1 (full charge per store): cells (of 12) whose side information alone exceeds the budget; K* = stores over',
         '# which the side information must be amortised to fall to 10% of the budget (median).', '']
    for enc in PANEL:
        S = arm_side(enc)
        L += [f'## {NAMES[enc]} (n = {NDIM[enc]})', '',
              '| arm | objects (minimal) | entries | bits @8 | bits @32 | ratio to budget | cells over budget (D1) | K* | write MAC / fact | read MAC / query |',
              '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
        J[enc] = {}
        for arm in ARMS:
            if not S[arm]:
                continue
            ent, b8, b32, ratio, over, ks, wr, rd, objs = [], [], [], [], 0, [], [], [], set()
            for (M, V, b) in CELLS:
                o, seeds, w, r = S[arm][f'{M}|{V}|{b}']
                om = minimal(o)
                e = sum(om.values())
                ent.append(e)
                b8.append(8 * e + 64 * seeds)
                b32.append(32 * e + 64 * seeds)
                ratio.append(b8[-1] / (b * M))
                over += b8[-1] > b * M
                ks.append(b8[-1] / (0.1 * b * M))
                wr.append(w)
                rd.append(r)
                objs |= set(om)
            md = lambda v: f'{np.median(v):.3g} [{min(v):.3g}, {max(v):.3g}]'  # noqa: E731
            J[enc][arm] = {'entries': ent, 'bits8': b8, 'bits32': b32, 'ratio': ratio, 'cells_over': over,
                           'Kstar': ks, 'write': wr, 'read': rd, 'objects': sorted(objs)}
            L.append(f'| {arm} | {", ".join(sorted(objs))} | {md(ent)} | {md(b8)} | {md(b32)} | {md(ratio)} | {over} | '
                     f'{np.median(ks):.3g} | {np.median(wr):.3g} | {np.median(rd):.3g} |')
        L.append('')
    nst = {arm: sum(1 for enc in PANEL if arm in J[enc] and J[enc][arm]['cells_over'] < 12) for arm in ARMS}
    L.append('D1: encoders on which the arm is storable (side information within the budget) in at least one R1 cell: '
             + ', '.join(f'{a} {v}' for a, v in nst.items()))
    open(f'{RES}/b71_side.txt', 'w').write('\n'.join(L) + '\n')
    json.dump(J, open(f'{RES}/b71_side.json', 'w'))
    print('\n'.join(L))


# ----------------------------------------------------------------------------------------------- D2

def allowance(prep, n):
    """The compared store design's own total side information in entries (its seeds aside)."""
    return sum(objs_prep(prep, n).values())


def lam_excess(store_prep, lam_prep, feat, p, n):
    """Entries of LAMc's side information beyond what it shares with the store, minus the free allowance."""
    shared = objs_prep(store_prep, n)
    need = objs_prep(lam_prep, n)
    if feat == 'pca':
        need['basis'] = n * p
    if 'zca' in shared and 'basis' in need:
        need.pop('basis')  # derivable from the shared whitening matrix
    extra = sum(v for k, v in need.items() if k not in shared)
    return max(0, extra - allowance(store_prep, n))


def sdm_cap(vsar_prep, sdm_prep, n):
    """Largest N with SDMt's extra side information within the allowance (None = infeasible for any N)."""
    shared = objs_prep(vsar_prep, n)
    need = objs_prep(sdm_prep, n)
    extra = sum(v for k, v in need.items() if k not in shared)
    A = allowance(vsar_prep, n)
    if extra > A:
        return None
    return (A - extra) // n


def d2lam_choose():
    import b60_run as L60
    out = {}
    for enc in PANEL:
        n = NDIM[enc]
        ch53 = jl(f'{ROOT}/results/b53_choice_{enc}.json')['choice']
        ch60 = jl(f'{ROOT}/results/b60_choice_{enc}.json')['choice']
        dev = None
        for (M, V, b) in CELLS:
            cell = f'{M}|{V}|{b}'
            sp = ch53[f'{cell}|ideal|VSA']['cfg'][0]
            sc, t, lp, ft, al, lq = ch60[f'{cell}|ideal|LAMC']['cfg']
            c = V - 1 if t == 'V-1' else V
            p = L60.p_of(b * M, c, lq, 'ideal', sc, n if ft == 'pca' else 65536)
            ex = lam_excess(sp, lp, ft, p, n)
            rec = {'store_prep': sp, 'lamc_cfg': ch60[f'{cell}|ideal|LAMC']['cfg'], 'excess_entries': ex}
            if 8 * ex + L60.bits(p, c, lq, 'ideal', sc) > b * M:
                if dev is None:
                    rows = L60.load_dev(enc)
                    dev = {(k[1], k[2], k[3], k[4], k[5]): v for k, v in rows.items() if k[0] == 0}
                lst = []
                for cfg in L60.grid(V):
                    sc2, t2, lp2, ft2, al2, q2 = cfg
                    c2 = V - 1 if t2 == 'V-1' else V
                    p2 = L60.p_of(b * M, c2, q2, 'ideal', sc2, n if ft2 == 'pca' else 65536)
                    if 8 * lam_excess(sp, lp2, ft2, p2, n) + L60.bits(p2, c2, q2, 'ideal', sc2) > b * M:
                        continue
                    d = dev[(M, V, b, 'ideal', cfg)]
                    lst.append((float(np.mean([d[k] for k in sorted(d)])), cfg))
                top = max(x for x, _ in lst)
                tied = [cc for x, cc in lst if x >= top - 0.001]
                rec.update({'feasible': False, 'rechosen': list(tied[0]), 'n_feasible': len(lst),
                            'dev_acc': [x for x, cc in lst if cc == tied[0]][0]})
            else:
                rec['feasible'] = True
            out[f'{enc}|{cell}'] = rec
    json.dump(out, open(f'{RES}/b71_d2_lam_choice.json', 'w'), indent=1)
    print(sum(1 for v in out.values() if not v['feasible']), 'LAMc cells re-chosen of', len(out))


def d2plan():
    """SDMt capped dev plan: per (enc, M, V, rep) the caps needed by the cells' VSAR preps."""
    plan = []
    for enc in PANEL:
        n = NDIM[enc]
        for M in M_GRID:
            for V in VN:
                caps = {}
                for b in BPF:
                    vp = merged_cfg(enc, M, V, b, 'VSAR')[0]
                    for sp in ('center', 'whiten'):
                        cap = sdm_cap(vp, sp, n)
                        if cap is not None and cap >= 1:
                            caps.setdefault((sp, cap), []).append(b)
                for (sp, cap), bs in sorted(caps.items()):
                    for rep in range({256: 6, 1024: 3}[M]):
                        plan.append({'enc': enc, 'split': 'dev', 'M': M, 'V': V, 'rep': rep, 'prep': sp, 'cap': cap,
                                     'bs': bs})
    os.makedirs(f'{OUT}/d2sdm', exist_ok=True)
    with open(f'{OUT}/d2sdm/plan_dev.jsonl', 'w') as fo:
        for s in plan:
            fo.write(json.dumps(s) + '\n')
    print(len(plan), 'jobs ->', f'{OUT}/d2sdm/plan_dev.jsonl')


def d2sdm_job(spec):
    import b64_run as B64
    fn = (f'{OUT}/d2sdm/jobs/{spec["enc"]}_{spec["split"]}_M{spec["M"]}_V{spec["V"]}_r{spec["rep"]}_{spec["prep"]}'
          f'_cap{spec["cap"]}' + (f'_b{"-".join(map(str, spec["bs"]))}' if spec['split'] == 'test' else '') + '.json.gz')
    if os.path.exists(fn):
        return fn
    orig = B64.ngrid
    cap = spec['cap']
    B64.ngrid = lambda nmax: [x for x in orig(nmax) if x <= cap] or [1]
    pool, preps, Ls, _ = B64.setup(spec['enc'])
    st = C.make_store(pool, spec['split'], spec['M'], spec['V'], spec['rep'])
    order = B64.store_order(st)
    P = preps[spec['prep']]
    if spec.get('cfgs') is None:
        cs = [c[1:] for c in B64.GRID if c[0] == spec['prep']]
    else:
        cs = sorted({tuple(c[1:]) for c, b in spec['cfgs']})
    o = B64.run_sdmt(st, P(st['keys']), P(st['queries']), Ls[spec['prep']], cs, spec['bs'], ['ideal'], order)
    B64.ngrid = orig
    g = st['qgold']
    rows = []
    for (c5, b, a), (pred, bt, N) in o.items():
        row = {'cfg': [spec['prep']] + list(c5), 'bpf': b, 'acc': float((pred == g).mean()), 'bits': bt, 'N': N}
        if spec['split'] == 'test':
            row['pred'] = np.asarray(pred).astype(int).tolist()
        rows.append(row)
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    with gzip.open(fn + '.tmp', 'wt') as fo:
        json.dump({'spec': spec, 'qgold': g.tolist(), 'rows': rows}, fo)
    os.replace(fn + '.tmp', fn)
    print('job', fn, flush=True)
    return fn


def _sd(spec):
    try:
        return d2sdm_job(spec)
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        return f'FAILED {spec} {e}'


def d2sdm_run(planfile, workers):
    specs = [json.loads(l) for l in open(planfile)]
    import multiprocessing as mp
    with mp.get_context('fork').Pool(workers) as pl:
        for r in pl.imap_unordered(_sd, specs, chunksize=1):
            if str(r).startswith('FAILED'):
                print(r, flush=True)


def d2sdm_choose():
    import b64_run as B64
    out = {}
    for enc in PANEL:
        n = NDIM[enc]
        acc = {}
        jd = f'{OUT}/d2sdm/jobs'
        for fn in sorted(os.listdir(jd)):
            if not fn.startswith(f'{enc}_dev_'):
                continue
            r = json.load(gzip.open(os.path.join(jd, fn), 'rt'))
            s = r['spec']
            for x in r['rows']:
                acc.setdefault((s['M'], s['V'], x['bpf'], s['cap'], tuple(x['cfg'])), {})[s['rep']] = x['acc']
        for (M, V, b) in CELLS:
            vp = merged_cfg(enc, M, V, b, 'VSAR')[0]
            lst = []
            for cfg in B64.GRID:
                cap = sdm_cap(vp, cfg[0], n)
                if cap is None or cap < 1:
                    continue
                d = acc.get((M, V, b, cap, tuple(cfg)))
                assert d is not None and len(d) == {256: 6, 1024: 3}[M], (enc, M, V, b, cap, cfg)
                lst.append((float(np.mean([d[k] for k in sorted(d)])), cfg, cap))
            if not lst:
                out[f'{enc}|{M}|{V}|{b}'] = {'vsar_prep': vp, 'cfg': None}
                continue
            top = max(x for x, _, _ in lst)
            tied = [(c, cp) for x, c, cp in lst if x >= top - 0.001]
            c, cp = tied[0]
            out[f'{enc}|{M}|{V}|{b}'] = {'vsar_prep': vp, 'cfg': list(c), 'cap': cp,
                                         'dev_acc': [x for x, cc, _ in lst if cc == c][0]}
    json.dump(out, open(f'{RES}/b71_d2_sdm_choice.json', 'w'), indent=1)
    print(len(out), 'SDMt D2 choices')


def d2test_plan():
    lam = jl(f'{RES}/b71_d2_lam_choice.json')
    sdm = jl(f'{RES}/b71_d2_sdm_choice.json')
    pl, ps = [], []
    for key, v in lam.items():
        if v['feasible']:
            continue
        enc, M, V, b = key.split('|')
        M, V, b = int(M), int(V), int(b)
        cfg = v['rechosen']
        for rep in range({256: 20, 1024: 5}[M]):
            pl.append({'enc': enc, 'map': 0, 'split': 'test', 'M': M, 'V': V, 'rep': rep, 'prep': cfg[2],
                       'feats': [cfg[3]], 'cfgs': [[cfg, b, 'ideal']], 'tag': f'b{b}'})
    for key, v in sdm.items():
        if v['cfg'] is None:
            continue
        enc, M, V, b = key.split('|')
        M, V, b = int(M), int(V), int(b)
        for rep in range({256: 20, 1024: 5}[M]):
            ps.append({'enc': enc, 'split': 'test', 'M': M, 'V': V, 'rep': rep, 'prep': v['cfg'][0], 'cap': v['cap'],
                       'bs': [b], 'cfgs': [[v['cfg'], b]], 'b': b})
    os.makedirs(f'{OUT}/d2lam', exist_ok=True)
    with open(f'{OUT}/d2lam/plan_test.jsonl', 'w') as fo:
        for s in pl:
            fo.write(json.dumps(s) + '\n')
    with open(f'{OUT}/d2sdm/plan_test.jsonl', 'w') as fo:
        for s in ps:
            fo.write(json.dumps(s) + '\n')
    print(len(pl), 'LAMc test jobs;', len(ps), 'SDMt test jobs')


def d2lam_run(planfile, workers):
    os.environ['B60_OUT'] = f'{OUT}/d2lam'
    import b60_run as L60
    L60.OUT = f'{OUT}/d2lam'
    specs = [json.loads(l) for l in open(planfile)]
    # one job file per (store, prep) in b60's naming; cells of different budgets go to different files via the map tag
    import multiprocessing as mp

    with mp.get_context('fork').Pool(workers) as pl:
        for r in pl.imap_unordered(_lam_one, specs, chunksize=1):
            print(r, flush=True)


def _lam_one(s):
    import b60_run as L60
    L60.OUT = f'{OUT}/d2lam/{s["tag"]}'  # one directory per budget: b60's job names omit the budget
    s2 = {k: v for k, v in s.items() if k != 'tag'}
    try:
        return L60.run_job(s2)
    except Exception as e:  # noqa: BLE001
        return f'FAILED {s} {e}'


def d2verdict():
    import b36_verdict as V36
    import b60_verdict as V60
    import b64_verdict as V64
    import b66_storeboot as SB
    lam = jl(f'{RES}/b71_d2_lam_choice.json')
    sdm = jl(f'{RES}/b71_d2_sdm_choice.json')
    J, L = {}, ['# B71 (D2): labels with side information charged beyond what the compared store design uses (8 bits per entry).', '']
    for enc in PANEL:
        ch60, ch64 = V60.choice(enc), V64.choice(enc)
        data = {}
        for M in M_GRID:
            for V in VN:
                recs = [json.loads(l) for l in gzip.open(f'{ROOT}/b59_out/fam/{enc}/all59_{enc}_M{M}_V{V}.jsonl.gz', 'rt')]
                recs.sort(key=lambda r: r['rep'])
                for r in recs:
                    for b in BPF:
                        r['pred'][f'STORE|{b}|ideal'] = r['pred'][f'VSA|{b}|ideal']
                assert V60.add_lamc(recs, enc, M, V, ch60, 0, bpfs=tuple(BPF))
                rows = V64.job_rows(enc, M, V)
                for r in recs:
                    for b in BPF:
                        cfg = tuple(ch64[f'{M}|{V}|{b}|ideal|SDMT']['cfg'])
                        r['pred'][f'SDMT|{b}|ideal'] = rows[r['rep']][cfg, b, 'ideal'][0]
                        # D2 LAMc
                        v = lam[f'{enc}|{M}|{V}|{b}']
                        if v['feasible']:
                            r['pred'][f'LAMCD2|{b}|ideal'] = r['pred'][f'LAMC|{b}|ideal']
                        else:
                            c = v['rechosen']
                            fn = f'{OUT}/d2lam/b{b}/jobs/{enc}_m0_test_M{M}_V{V}_r{r["rep"]}_{c[2]}.json.gz'
                            jr = json.load(gzip.open(fn, 'rt'))
                            assert jr['qgold'] == r['qgold']
                            pr = [x['pred'] for x in jr['rows'] if x['cfg'] == c and x['bpf'] == b and x['acct'] == 'ideal']
                            assert len(pr) == 1
                            r['pred'][f'LAMCD2|{b}|ideal'] = pr[0]
                        # D2 SDMt
                        v = sdm[f'{enc}|{M}|{V}|{b}']
                        if v['cfg'] is None:
                            dv = int(np.bincount(r['vals'], minlength=V).argmax()) if 'vals' in r else 0
                            r['pred'][f'SDMTD2|{b}|ideal'] = [dv] * len(r['qgold'])
                        else:
                            c = v['cfg']
                            fn = (f'{OUT}/d2sdm/jobs/{enc}_test_M{M}_V{V}_r{r["rep"]}_{c[0]}_cap{v["cap"]}_b{b}.json.gz')
                            jr = json.load(gzip.open(fn, 'rt'))
                            assert jr['qgold'] == r['qgold']
                            pr = [x['pred'] for x in jr['rows'] if x['cfg'] == c and x['bpf'] == b]
                            assert len(pr) == 1
                            r['pred'][f'SDMTD2|{b}|ideal'] = pr[0]
                data[M, V] = recs
        CON = [('LAMC', 'STORE', 'ref'), ('LAMCD2', 'STORE', 'D2'), ('SDMT', 'VSAR', 'ref'), ('SDMTD2', 'VSAR', 'D2')]
        R = SB.analyse(data, PANEL.index(enc) + 200, CON, accts=['ideal'], nb={'ideal': 10000}, regions=SB.REG)
        J[enc] = R['regions']
    L += ['| encoder | LAMc - store (as run) | LAMc - store (D2) | SDMt - first design (as run) | SDMt - first design (D2) |',
          '| --- | --- | --- | --- | --- |']
    for enc in PANEL:
        row = []
        for nm in ('LAMC-STORE', 'LAMCD2-STORE', 'SDMT-VSAR', 'SDMTD2-VSAR'):
            e = J[enc][f'ideal|R1|{nm}']
            lo, hi = e['one']['ci95']
            row.append(f'{100 * e["diff"]:+.2f} [{100 * lo:+.2f}, {100 * hi:+.2f}] {e["one"]["label"]}')
        L.append(f'| {NAMES[enc]} | ' + ' | '.join(row) + ' |')
    nch = {k: sum(J[e][f'ideal|R1|{k}D2-{y}']['one']['label'].replace('D2', '') != J[e][f'ideal|R1|{k}-{y}']['one']['label']
                  for e in PANEL) for k, y in (('LAMC', 'STORE'), ('SDMT', 'VSAR'))}
    L.append(f'Labels changed by D2 (R1, one-level): LAMc - store {nch["LAMC"]} of 7; SDMt - first design {nch["SDMT"]} of 7.')
    json.dump(J, open(f'{RES}/b71_d2_verdict.json', 'w'))
    open(f'{RES}/b71_d2_verdict.txt', 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))


def main():
    a = sys.argv[1:]
    cmd = a[0]
    if cmd == 'table':
        table()
    elif cmd == 'd2plan':
        d2plan()
    elif cmd == 'd2lam_choose':
        d2lam_choose()
    elif cmd == 'd2sdm_run':
        d2sdm_run(a[1], int(a[a.index('--workers') + 1]))
    elif cmd == 'd2sdm_choose':
        d2sdm_choose()
    elif cmd == 'd2test_plan':
        d2test_plan()
    elif cmd == 'd2lam_run':
        d2lam_run(a[1], int(a[a.index('--workers') + 1]))
    elif cmd == 'd2verdict':
        d2verdict()
    else:
        raise SystemExit(__doc__)


if __name__ == '__main__':
    main()
