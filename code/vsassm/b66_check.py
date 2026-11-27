"""B66 checks for part (A) (prereg/B66_reconciliation_prereg.md, 'Checks'); B, C, D have their own check commands.

python3 b66_check.py scorer      # plain-loop re-scoring of the A1 / A2 cells and region point statistics
python3 b66_check.py bits        # SDMTCR / SDMTm rows within budget; VSARm d within budget and maximal
python3 b66_check.py rederive    # float64 re-derivation: SDMTCR, SDMTm (b64_check's independent code, map order seed),
                                 # VSARm (own numpy float64 store write / read)
python3 b66_check.py all         # -> results/b66_A_check.txt
"""
import gzip
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import b64_check as K  # noqa: E402

RES = os.environ.get('B66_RES', f'{ROOT}/results')
OUT = os.environ.get('B66_OUT', f'{ROOT}/b66_out')
PANEL = ['qwen8', 'b46', 'llama', 'mistral', 'e5', 'bge', 'minilm']
M_GRID, VN, BPF, ACCTS = [256, 1024], [2, 16], [4, 8, 16], ['ideal', 'real']
TEST_REPS = {256: 20, 1024: 5}
LINES = []


def say(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    LINES.append(s)


def jl(fn):
    return [json.loads(l) for l in gzip.open(fn, 'rt')]


def tag(enc, m):
    return enc if m == 0 else f'{enc}#b57m{m}'


# ----------------------------------------------------------------------------------------------- scorer

def acc_loop(pred, gold):
    n = c = 0
    for p, g in zip(pred, gold):
        n += 1
        c += int(p == g)
    return c / n


def scorer():
    worst, n = 0.0, 0
    A1 = json.load(open(f'{RES}/b66_A1.json'))
    for enc in PANEL:
        P = A1[enc]['pipeline']
        chcr = json.load(open(f'{RES}/b66_choice_crgrid_{enc}.json'))['choice']
        cellv = {}
        for M in M_GRID:
            for V in VN:
                recs = sorted(jl(f'{ROOT}/b59_out/fam/{enc}/all59_{enc}_M{M}_V{V}.jsonl.gz'), key=lambda r: r['rep'])
                jobs = {rep: json.load(gzip.open(f'{OUT}/b64cr/jobs/{enc}_test_M{M}_V{V}_r{rep}.json.gz', 'rt'))
                        for rep in range(TEST_REPS[M])}
                for b in BPF:
                    for a in ACCTS:
                        cfg = chcr[f'{M}|{V}|{b}|{a}|SDMTCR']['cfg']
                        fam = chcr[f'{M}|{V}|{b}|{a}|BEST7CR']['fam']
                        accs = {'VSAR': [], 'SDMTCR': [], 'BEST7CR': []}
                        for r in recs:
                            row = [x for x in jobs[r['rep']]['rows'] if x['cfg'] == cfg and x['bpf'] == b and x['acct'] == a][0]
                            accs['VSAR'] += [int(p == g) for p, g in zip(r['pred'][f'VSAR|{b}|{a}'], r['qgold'])]
                            accs['SDMTCR'] += [int(p == g) for p, g in zip(row['pred'], r['qgold'])]
                            bp = row['pred'] if fam == 'SDMTCR' else r['pred'][f'{fam}|{b}|{a}']
                            accs['BEST7CR'] += [int(p == g) for p, g in zip(bp, r['qgold'])]
                        for k, v in accs.items():
                            x = sum(v) / len(v)
                            cellv[a, M, V, b, k] = x
                            d = abs(x - P['cells'][f'{a}|{M}|{V}|{b}'][k])
                            worst, n = max(worst, d), n + 1
        for a in ACCTS:
            for rn in ('R1', 'R1-V2', 'R1-V16'):
                cs = [(M, V, b) for M in M_GRID for V in VN for b in BPF if rn == 'R1' or rn == f'R1-V{V}']
                for y in ('SDMTCR', 'BEST7CR'):
                    s = 0.0
                    for c in cs:
                        s += cellv[(a,) + c + ('VSAR',)] - cellv[(a,) + c + (y,)]
                    d = abs(s / len(cs) - P['regions'][f'{a}|{rn}|VSAR-{y}']['diff'])
                    worst, n = max(worst, d), n + 1
    # clean-room side: cell accuracies from the saved per-query correctness vs the logged test accuracies
    for enc in ('qwen8', 'llama'):
        Cc = A1[enc]['clean']
        for c, v in Cc['cells'].items():
            M, V, b = map(int, c.split('|'))
            z = np.load(f'{OUT}/cr/out_{enc}/corr_M{M}_V{V}_b{b}.npz')
            for arm, x in v.items():
                cc = z[arm].tolist()
                d = abs(100.0 * sum(int(t) for t in cc) / len(cc) - x)
                worst, n = max(worst, d / 100), n + 1
    # A2 per-map cells
    for enc in PANEL:
        fn = f'{RES}/b66_A2_part_{enc}.json'
        if not os.path.exists(fn):
            continue
        A2 = json.load(open(fn))
        for m in (1, 2, 3):
            chm = json.load(open(f'{RES}/b66_choice_sdmtm{m}_{enc}.json'))['choice']
            for M in M_GRID:
                for V in VN:
                    recs = sorted(jl(f'{ROOT}/b59_out/fam/{enc}/all59_{enc}_M{M}_V{V}.jsonl.gz'), key=lambda r: r['rep'])
                    vt = {r['rep']: r for r in jl(f'{OUT}/vsarm/test_{enc}_m{m}_M{M}_V{V}.jsonl.gz')}
                    jobs = {rep: json.load(gzip.open(f'{OUT}/sdmtm{m}/jobs/{enc}_test_M{M}_V{V}_r{rep}.json.gz', 'rt'))
                            for rep in range(TEST_REPS[M])}
                    for b in BPF:
                        a = 'ideal'
                        cfg = chm[f'{M}|{V}|{b}|{a}|SDMT']['cfg']
                        sv, ss, n_ = 0, 0, 0
                        for r in recs:
                            row = [x for x in jobs[r['rep']]['rows'] if x['cfg'] == cfg and x['bpf'] == b and x['acct'] == a][0]
                            for p, q_, g in zip(vt[r['rep']]['pred'][f'VSARm{m}|{b}|{a}'], row['pred'], r['qgold']):
                                sv += int(p == g)
                                ss += int(q_ == g)
                                n_ += 1
                        c = A2['cells'][f'{a}|{M}|{V}|{b}']
                        for x, k in ((sv / n_, f'VSARm{m}'), (ss / n_, f'SDMTm{m}')):
                            d = abs(x - c[k])
                            worst, n = max(worst, d), n + 1
    say(f'scorer: {n} cell accuracies / region statistics recomputed with plain loops; max |difference| {worst:.2e}',
        'PASS' if worst < 1e-9 else 'FAIL')
    return worst


# ----------------------------------------------------------------------------------------------- bits

def bits():
    n = over = 0
    dirs = [f'{OUT}/b64cr/jobs'] + [f'{OUT}/sdmtm{m}/jobs' for m in (1, 2, 3)]
    for d in dirs:
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            r = json.load(gzip.open(os.path.join(d, fn), 'rt'))
            M = r['spec']['M']
            for x in r['rows']:
                n += 1
                over += x['bits'] > x['bpf'] * M + 1e-9
    say(f'bits SDMTCR / SDMTm: {n} logged dev and test rows; over budget {over}', 'PASS' if over == 0 else 'FAIL')
    nv = bad = notmax = 0
    import b32_common as C
    for fn in sorted(os.listdir(f'{OUT}/vsarm')):
        if not fn.startswith('test_'):
            continue
        for r in jl(f'{OUT}/vsarm/{fn}'):
            for k, (d, q, ns) in r['d'].items():
                b = int(k.split('|')[1])
                acct = 'raw' if k.endswith('|real') else 'ideal'
                B = b * r['M']
                nv += 1
                bad += C.vsa_bits(d, q, acct) > B
                notmax += C.vsa_bits(d + 1, q, acct) <= B
    say(f'bits VSARm: {nv} test (store, cell) sizes; over budget {bad}; not maximal {notmax}',
        'PASS' if bad == 0 and notmax == 0 else 'FAIL')
    return over, bad, notmax


# ----------------------------------------------------------------------------------------------- re-derivation

def locs_m(D, loc, enc, m):
    lab = [D.idx[e['name']] for e in D.split['train']]
    ali = [D.idx[a] for e in D.split['train'] for a in e['aliases']]
    rows = {'LA': lab + ali, 'L': lab, 'A': ali}[loc]
    bad = set(D.other)
    u = list(dict.fromkeys(r for r in rows if r not in bad))
    sd = K.seed('B64LOC', loc) if m == 0 else K.seed('B64LOC', loc, tag(enc, m))
    perm = np.random.default_rng(sd).permutation(len(u))
    o = [u[i] for i in perm]
    badc = {D.Xraw[r].tobytes() for r in D.other}
    return [r for r in o if D.Xraw[r].tobytes() not in badc]


def rederive_sdmt(encs, maps):
    lo, out = 1.0, {}
    for enc in encs:
        D = K.Data(enc)
        preps = {p: D.prep(p) for p in ('center', 'whiten')}
        for m in maps:
            if m == 'cr':
                ch = json.load(open(f'{RES}/b66_choice_crgrid_{enc}.json'))['choice']
                key, jd, mm = 'SDMTCR', f'{OUT}/b64cr/jobs', 0
            else:
                ch = json.load(open(f'{RES}/b66_choice_sdmtm{m}_{enc}.json'))['choice']
                key, jd, mm = 'SDMT', f'{OUT}/sdmtm{m}/jobs', m
            locs = {l: locs_m(D, l, enc, mm) for l in ('LA', 'L', 'A')}
            for M in M_GRID:
                for V in VN:
                    rep = int(np.random.default_rng(K.seed('B66RD', enc, m, M, V)).integers(0, TEST_REPS[M]))
                    j = json.load(gzip.open(f'{jd}/{enc}_test_M{M}_V{V}_r{rep}.json.gz', 'rt'))
                    for b in (4, 16):
                        cfg = ch[f'{M}|{V}|{b}|ideal|{key}']['cfg']
                        row = [x for x in j['rows'] if x['cfg'] == cfg and x['bpf'] == b and x['acct'] == 'ideal'][0]
                        pred, N, bt = K.rederive_one(D, preps, locs, enc, M, V, rep, tuple(cfg), b, 'ideal')
                        ag = float(np.mean(np.array(pred) == np.array(row['pred'])))
                        lo = min(lo, ag)
                        out[f'{enc}|{m}|{M}|{V}|{b}'] = {'agree': ag, 'N_log': row['N'], 'N_own': N, 'bits_own': bt,
                                                          'budget': b * M}
                        say(f'rederive SDMt {enc:8s} map {m} M={M:5d} V={V:3d} b={b:3d} rep={rep:2d}: agreement {ag:.4f}; '
                            f'N log {row["N"]} own {N}; own bits {bt:.1f} (budget {b * M})')
    say(f'rederive SDMt: min agreement {lo:.4f}', 'PASS' if lo >= 0.99 else 'FAIL')
    return lo, out


def own_quant(S, q, rng):
    if q == 1:
        out = np.sign(S)
        z = out == 0
        if z.any():
            out[z] = rng.choice([-1.0, 1.0], size=int(z.sum()))
        return out
    sd = float(S.std()) or 1.0
    step = {2: 0.9957, 4: 0.3352}[q] * sd
    L = 2 ** q
    return (np.clip(np.floor(S / step), -L // 2, L // 2 - 1) + 0.5) * step


def own_seed(*parts):
    import hashlib
    return int(hashlib.sha256('|'.join(map(str, parts)).encode()).hexdigest()[:15], 16)


def rederive_vsar(encs, maps, b=8):
    """Own float64 store (first design) on one test store per (encoder, M, V, map) at b bits per fact, ideal."""
    CH, VMAX = 8192, 1024
    HQ = {1: 1.0, 2: 1.9037, 4: 3.6024}
    lo, out = 1.0, {}
    for enc in encs:
        D = K.Data(enc)
        preps = {p: D.prep(p) for p in ('center', 'whiten')}
        n = D.X.shape[1]
        for m in maps:
            ch = json.load(open(f'{RES}/b66_choice_vsarm{m}_{enc}.json'))['choice']
            tg = tag(enc, m)
            for M in M_GRID:
                for V in VN:
                    rep = int(np.random.default_rng(own_seed('B66RV', enc, m, M, V)).integers(0, TEST_REPS[M]))
                    rec = [r for r in jl(f'{OUT}/vsarm/test_{enc}_m{m}_M{M}_V{V}.jsonl.gz') if r['rep'] == rep][0]
                    prep, fmap, q, f = ch[f'{M}|{V}|{b}|ideal|VSAR']['cfg']
                    keys, vals, Q, qg, order = K.store(D, 'test', M, V, rep)
                    assert qg == rec['qgold']
                    Bud = b * M
                    per = HQ[q]
                    d = max(0, int((Bud - 32) // per))
                    while d > 0 and d * per + 32 * math.ceil(d / CH) > Bud:
                        d -= 1
                    ns = max(1, int(math.floor(f * M)))
                    st_seed = own_seed('vsa', tg, 'test', M, V, rep)
                    P = preps[prep]
                    Kp, Qp = P(keys), P(Q)
                    Kf = 1 if fmap in ('lin', 'sgn') else int(fmap[3:])
                    sc = np.zeros((len(qg), V))
                    for j0 in range(0, d, CH):
                        w = min(CH, d - j0)
                        FK = FQ = None
                        for k in range(Kf):
                            G = np.random.default_rng(own_seed('G', tg, prep, fmap, k, j0, CH)).standard_normal(
                                (n, CH), dtype=np.float32).astype(np.float64)[:, :w]
                            pk, pq_ = Kp @ G, Qp @ G
                            if fmap != 'lin':
                                pk, pq_ = np.where(pk >= 0, 1.0, -1.0), np.where(pq_ >= 0, 1.0, -1.0)
                            FK = pk if FK is None else FK * pk
                            FQ = pq_ if FQ is None else FQ * pq_
                        r_ = np.random.default_rng(own_seed('C', tg, prep, fmap, j0, CH))
                        Cv = np.where(r_.random((VMAX, CH), dtype=np.float32) < 0.5, -1.0, 1.0)[:V, :w]
                        S = np.zeros(w)
                        for i in order[:ns]:
                            S += FK[i] * Cv[vals[i]]
                        Sq = own_quant(S, q, np.random.default_rng(own_seed('Q', st_seed, d, q, ns, j0)))
                        sc += (FQ * Sq) @ Cv.T
                    pred = sc.argmax(1) if np.abs(sc).sum() > 0 else np.zeros(len(qg), int)
                    ag = float(np.mean(pred == np.array(rec['pred'][f'VSARm{m}|{b}|ideal'])))
                    lo = min(lo, ag)
                    out[f'{enc}|{m}|{M}|{V}'] = ag
                    say(f'rederive VSAR {enc:8s} map {m} M={M:5d} V={V:3d} b={b} rep={rep:2d} {prep}/{fmap}/q{q}/f{f}: '
                        f'agreement {ag:.4f} (d {d})')
    say(f'rederive VSARm: min agreement {lo:.4f}', 'PASS' if lo >= 0.99 else 'FAIL')
    return lo, out


def main():
    a = sys.argv[1:]
    if a[0] in ('scorer', 'all'):
        scorer()
    if a[0] in ('bits', 'all'):
        bits()
    if a[0] in ('rederive', 'all'):
        rederive_sdmt(PANEL, ['cr'])
        rederive_sdmt(['qwen8', 'llama', 'e5'], [1, 2, 3])
        rederive_vsar(['qwen8', 'llama', 'e5'], [1, 2, 3])
    if a[0] == 'all':
        open(f'{RES}/b66_A_check.txt', 'w').write('\n'.join(LINES) + '\n')


if __name__ == '__main__':
    main()


# ----------------------------------------------------------------------------------------------- (C) plain loops

def cscore():
    """(C): every LAMC / SDMT draw's R1 test accuracy recomputed with plain loops from the raw job files (full-grid
    choices' files b60_out / b64_out, draw-chosen ones b66_out/grids), against results/b66_C_verdict.json."""
    J = json.load(open(f'{RES}/b66_C_choices.json'))
    Vd = json.load(open(f'{RES}/b66_C_verdict.json'))
    worst, n = 0.0, 0
    for enc in ('qwen8', 'b46', 'llama', 'mistral'):
        rows = {}
        for F, dirs in (('LAMC', [f'{ROOT}/b60_out/jobs', f'{OUT}/grids/lamc/jobs']),
                        ('SDMT', [f'{ROOT}/b64_out/jobs', f'{OUT}/grids/sdmt/jobs'])):
            for d in dirs:
                for fn in os.listdir(d):
                    if not fn.startswith(enc + '_') or '_test_' not in fn or (F == 'LAMC' and '_m0_' not in fn):
                        continue
                    r = json.load(gzip.open(os.path.join(d, fn), 'rt'))
                    s = r['spec']
                    for x in r['rows']:
                        if x['acct'] == 'ideal' and 'pred' in x:
                            rows[F, s['M'], s['V'], s['rep'], x['bpf'], json.dumps(x['cfg'])] = (x['pred'], r['qgold'])
        for F in ('LAMC', 'SDMT'):
            for d in ['full'] + list(range(20)):
                tot = 0.0
                for M in M_GRID:
                    for V in VN:
                        for b in BPF:
                            ent = J['enc'][enc][f'{M}|{V}|{b}']
                            cfg = ent['full'][F] if d == 'full' else ent['draws'][d][F]['cfg']
                            ok = cnt = 0
                            for rep in range(TEST_REPS[M]):
                                pred, gold = rows[F, M, V, rep, b, json.dumps(cfg)]
                                for p, g in zip(pred, gold):
                                    ok += int(p == g)
                                    cnt += 1
                            tot += ok / cnt
                reg = tot / 12
                ref = Vd['families'][f'{enc}|{F}']['full' if d == 'full' else 'draws']
                ref = ref if d == 'full' else ref[d]
                worst, n = max(worst, abs(reg - ref)), n + 1
    say(f'(C) plain-loop scorer: {n} family region values (LAMC, SDMT; full grid + 20 draws; 4 language models) from the raw '
        f'job files; max |difference| {worst:.2e}', 'PASS' if worst < 1e-9 else 'FAIL')
    return worst


if __name__ == '__main__' and sys.argv[1:] == ['cscore']:
    cscore()
