"""B74 verdict (prereg/B74_factorial_labels_matched_prereg.md): (A) the 2 x 3 factorial write rule x features at the
store's fixed format, (B) label against alias queries, (C) NLMSc re-chosen from 72-configuration sub-grids.

Arms: existing test predictions as b72_verdict.load reads them (STORE = HR, OPTR = LR, LAMC, NLMSC, VSAS, VSAPC,
VSAPCX), B74's FX test jobs at the committed choices (results/b74_choice_{enc}.json), the label jobs (b74_out/lab*),
the NL72 jobs (b74_out/nl72) at results/b74_nl72_choices.json. Ideal accounting, R1.
Statistics: region statistic = mean over cells of the cell mean of the per-query value; one-level = b36_verdict.
boot_region's scheme (entity clusters resampled within each cell), here computed for all combinations of a cell from
the same resampled clusters (boot_cells; equal to boot_region for one combination, selftest); two-level =
b66_storeboot.cell_draws.

python3 b74_verdict.py selftest
python3 b74_verdict.py part ENC [A,B,C]   # -> results/b74_part_{enc}.json
python3 b74_verdict.py panel              # -> results/b74_verdict.{json,txt}
"""
import gzip
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import b36_verdict as V36  # noqa: E402
import b66_storeboot as SB  # noqa: E402

RES = os.environ.get('B74_RES', f'{ROOT}/results')
OUT = os.environ.get('B74_OUT', f'{ROOT}/b74_out')
PANEL = SB.PANEL
LLM, SENT = PANEL[:4], PANEL[4:]
NAMES = SB.NAMES
M_GRID, VN, BPF = [256, 1024], [2, 16], [4, 8, 16]
REG = SB.REG
RN = ('R1-V2', 'R1-V16', 'R1')
A = 'ideal'
KINDS = ['P', 'T']
NDRAW = 20


def comb(**kw):
    return kw


# (A) combinations of the six arms (HR = STORE, LR = OPTR); keys = arm names in the records
COMBOS_A1 = {
    'W_R': comb(LR=1, HR=-1), 'W_P': comb(LP=1, HP=-1), 'W_T': comb(LT=1, HT=-1),
    'F_P|H': comb(HP=1, HR=-1), 'F_P|L': comb(LP=1, LR=-1), 'F_T|H': comb(HT=1, HR=-1), 'F_T|L': comb(LT=1, LR=-1),
    'I_P': comb(LP=1, HP=-1, LR=-1, HR=1), 'I_T': comb(LT=1, HT=-1, LR=-1, HR=1),
    'G_P': comb(LP=1, HR=-1), 'G_T': comb(LT=1, HR=-1)}
COMBOS_A2 = {
    'Wmain_P': comb(LR=.5, HR=-.5, LP=.5, HP=-.5), 'Fmain_P': comb(HP=.5, HR=-.5, LP=.5, LR=-.5),
    'Wmain_T': comb(LR=.5, HR=-.5, LT=.5, HT=-.5), 'Fmain_T': comb(HT=.5, HR=-.5, LT=.5, LR=-.5),
    'Wmain3': comb(LR=1 / 3, HR=-1 / 3, LP=1 / 3, HP=-1 / 3, LT=1 / 3, HT=-1 / 3),
    'LP-LAMC': comb(LP=1, LAMC=-1), 'LT-LAMC': comb(LT=1, LAMC=-1), 'LR-LAMC': comb(LR=1, LAMC=-1),
    'HR-LAMC': comb(HR=1, LAMC=-1), 'LT-NLMSC': comb(LT=1, NLMSC=-1), 'HT-VSAPCX': comb(HT=1, VSAPCX=-1)}
PAIRS_B = [('STORE', 'LAMC'), ('VSAPCX', 'LAMC'), ('STORE', 'NLMSC'), ('VSAPCX', 'NLMSC')]
ARMS_B = ['STORE', 'VSAS', 'VSAPC', 'VSAPCX', 'LAMC', 'NLMSC']


def lab(lo, hi):
    return '+' if lo > 0 else ('-' if hi < 0 else 'level')


def boot_cells(diffs, cl, B, rng, chunk=1000):
    """boot_region's scheme for one cell and several per-query arrays at once: clusters drawn with replacement
    (k of k), cell draw = sum of the drawn clusters' sums / their query count. -> {name: (B,) draws}."""
    k, inv = V36.clusters(cl)
    n = np.bincount(inv, minlength=k).astype(np.float64)
    sums = {nm: np.bincount(inv, weights=d, minlength=k) for nm, d in diffs.items()}
    out = {nm: np.empty(B) for nm in diffs}
    for b0 in range(0, B, chunk):
        b1 = min(B, b0 + chunk)
        idx = rng.integers(0, k, size=(b1 - b0, k))
        nn = n[idx].sum(1)
        for nm in diffs:
            out[nm][b0:b1] = sums[nm][idx].sum(1) / nn
    return out


def cell_values(recs, b, combos):
    """{name: per-query value} for the linear combinations (keys: arm names)."""
    arms = sorted({a for c in combos.values() for a in c})
    cor = {}
    for a in arms:
        c = SB.correct(recs, f'{a}|{b}|{A}')
        if c is None:
            raise KeyError(f'{a}|{b}')
        cor[a] = c
    return {nm: sum(coef * cor[a] for a, coef in c.items()) for nm, c in combos.items()}


def lin_analyse(data, combos, B, seed0, two=True, regions=REG):
    """data: {(M, V): records}; combos: {name: {arm: coef}} -> {'rn|name': {...}}, {'M|V|b|name': cell mean}."""
    cells = sorted({c for R in regions.values() for c in R})
    one_d, two_d, pts = {}, {}, {}
    for (M, V, b) in cells:
        recs = data[M, V]
        lay, cl = SB.cell_layout(recs)
        vals = cell_values(recs, b, combos)
        rng = np.random.default_rng(seed0 * 1000 + M + 3 * V + 7 * b)
        one_d[M, V, b] = boot_cells(vals, cl, B, rng)
        if two:
            two_d[M, V, b], _ = SB.cell_draws(lay, vals, B, rng)
        for nm, v in vals.items():
            pts[M, V, b, nm] = float(v.mean())
    out, cellout = {}, {}
    for (M, V, b, nm), v in pts.items():
        cellout[f'{M}|{V}|{b}|{nm}'] = v
    for rn, R in regions.items():
        for nm in combos:
            s = float(np.mean([pts[c + (nm,)] for c in R]))
            d1 = np.mean([one_d[c][nm] for c in R], 0)
            e = {'diff': s, 'one': SB.ci(d1), 'n_cells': len(R)}
            e['one_label'] = lab(*e['one'])
            if two:
                e['two'] = SB.ci(np.mean([two_d[c][nm] for c in R], 0))
                e['two_label'] = lab(*e['two'])
            out[f'{rn}|{nm}'] = e
    return out, cellout


# ----------------------------------------------------------------------------------------------- loading

def choice(enc):
    return json.load(open(f'{RES}/b74_choice_{enc}.json'))['choice']


def fx_index(enc):
    """{(w + kind, M, V, rep, bpf, cfg tuple): pred}, {('_q', kind, M, V, rep): qgold} from the FX test jobs."""
    out = {}
    jd = os.path.join(OUT, 'jobs')
    for fn in sorted(os.listdir(jd)):
        if not fn.startswith(f'{enc}_FX') or '_test_' not in fn:
            continue
        r = json.load(gzip.open(os.path.join(jd, fn), 'rt'))
        s = r['spec']
        if s['enc'] != enc:
            continue
        for rep, rr in r['reps'].items():
            out['_q', s['kind'], s['M'], s['V'], int(rep), s['prep'], s['fmap']] = rr['qgold']
            for x in rr['rows']:
                out[x['w'] + s['kind'], s['M'], s['V'], int(rep), x['bpf'], tuple(x['cfg'])] = x['pred']
    return out


def load_A(enc):
    import b72_verdict as V72
    data = V72.load(enc, with_b=False)
    ch = choice(enc)
    ti = fx_index(enc)
    for (M, V), recs in data.items():
        for r in recs:
            rep = r['rep']
            for b in BPF:
                cell = f'{M}|{V}|{b}'
                r['pred'][f'HR|{b}|{A}'] = r['pred'][f'STORE|{b}|{A}']
                r['pred'][f'LR|{b}|{A}'] = r['pred'][f'OPTR|{b}|{A}']
                for kind in KINDS:
                    c = ch[f'{cell}|H{kind}']
                    qk = ('_q', kind, M, V, rep, c['feat'][0], c['feat'][1])
                    assert ti[qk] == r['qgold'], ('FX qgold', enc, qk)
                    for w in 'HL':
                        c = ch[f'{cell}|{w}{kind}']
                        r['pred'][f'{w}{kind}|{b}|{A}'] = ti[w + kind, M, V, rep, b, tuple(c['cfg'])]
    return data


def lab_preds(enc, data):
    """Label-query records {(M, V): [rec]} (qgold = values, qent = entity index) for ARMS_B, and the alias
    reproduction counts {arm: [n_same, n_total, [differing (M, V, b, rep, i)]]}."""
    import b60_verdict as V60
    ch53 = json.load(open(f'{ROOT}/results/b53_choice_{enc}.json'))['choice']
    ch56 = json.load(open(f'{ROOT}/results/b56_choice_{enc}.json'))['choice']
    chpc = json.load(open(f'{ROOT}/results/b60_choicepc_{enc}.json'))['choice']
    ch60 = V60.choice(enc)
    ch62 = json.load(open(f'{ROOT}/results/b62_choice_{enc}.json'))['choice']
    repro = {a: [0, 0, []] for a in ARMS_B if a != 'VSAPCX'}
    out = {}
    jd = os.path.join(OUT, 'jobs')
    for (M, V), recs in data.items():
        dv = [json.loads(l) for l in gzip.open(f'{OUT}/lab56/{enc}/dec_test_{enc}_M{M}_V{V}.jsonl.gz', 'rt')]
        dp = [json.loads(l) for l in gzip.open(f'{OUT}/lab57/{enc}/decp_test_{enc}_M{M}_V{V}.jsonl.gz', 'rt')]
        L = []
        for i, r in enumerate(recs):
            rep = r['rep']
            nq = len(r['qgold'])
            js = json.load(gzip.open(f'{jd}/{enc}_LABS_M{M}_V{V}_r{rep}.json.gz', 'rt'))
            assert js['nq_alias'] == nq and js['qgold'][:nq] == r['qgold']
            vals = js['qgold'][nq:]
            lr = {'rep': rep, 'qgold': vals, 'qent': list(range(len(vals))), 'pred': {}}
            assert dv[i]['rep'] == rep and dp[i]['rep'] == rep
            assert dv[i]['qgold'] == r['qgold'] + vals and dp[i]['qgold'] == r['qgold'] + vals
            lj = {}
            for prep in ('center', 'whiten'):
                for arm, sub in (('LAMC', 'lab60'), ('NLMSC', 'lab62')):
                    nm = (f'{enc}_m0_test_M{M}_V{V}_r{rep}_{prep}' if arm == 'LAMC'
                          else f'{enc}_test_M{M}_V{V}_r{rep}_{prep}')
                    fn = f'{OUT}/{sub}/jobs/{nm}.json.gz'
                    if os.path.exists(fn):
                        lj[arm, prep] = json.load(gzip.open(fn, 'rt'))
            for b in BPF:
                cell = f'{M}|{V}|{b}'
                full = {}
                full['STORE'] = js['pred'][f'STORE|{b}']
                sf = ch56[f'{cell}|ideal|STAR']['fam']
                c = ch56[f'{cell}|ideal|{sf}']['cfg']
                full['VSAS'] = dv[i]['pred']['|'.join(map(str, (sf, b, 'ideal', c[0], c[1])))]
                pc = chpc[f'{cell}|ideal|PSTAR']
                c = chpc[f'{cell}|ideal|{pc["fam"]}']['cfg']
                full['VSAPC'] = dp[i]['pred']['|'.join(map(str, (pc['fam'][1:], b, 'ideal', c[0], c[1])))]
                c = ch60[f'{cell}|ideal|LAMC']['cfg']
                rows = {(tuple(x['cfg']), x['bpf'], x['acct']): x['pred'] for x in lj['LAMC', c[2]]['rows']}
                full['LAMC'] = rows[tuple(c), b, 'ideal']
                c = ch62[f'{cell}|NLMSC']['cfg']
                rows = {(x['arm'], tuple(x['cfg']), x['bpf']): x['pred'] for x in lj['NLMSC', c[2]]['rows']}
                full['NLMSC'] = rows['NLMSC', tuple(c), b]
                for a, p in full.items():
                    old = r['pred'][f'{a}|{b}|{A}']
                    same = [int(x == y) for x, y in zip(p[:nq], old)]
                    repro[a][0] += sum(same)
                    repro[a][1] += nq
                    repro[a][2] += [[M, V, b, rep, j] for j, s_ in enumerate(same) if not s_][:20]
                    lr['pred'][f'{a}|{b}|{A}'] = p[nq:]
                s = ch56[f'{cell}|ideal|STAR']
                src = 'VSAPC' if pc['dev'][pc['fam']] > s['dev'][s['fam']] + 0.001 else 'VSAS'
                lr['pred'][f'VSAPCX|{b}|{A}'] = lr['pred'][f'{src}|{b}|{A}']
                assert r['pred'][f'VSAPCX|{b}|{A}'] == r['pred'][f'{src}|{b}|{A}']
            L.append(lr)
        out[M, V] = L
    return out, repro


def paired(data, ldata, specs, B, seed0, regions=REG):
    """Paired entity bootstrap (one-level): per cell, entity clusters 'rep:e' drawn with replacement; each entity
    carries its label value and its alias values. specs: {name: {arm: coef}} -> for each name the alias region value,
    the label value and Delta = alias - label with 95% intervals."""
    cells = sorted({c for R in regions.values() for c in R})
    dr, pts = {}, {}
    for (M, V, b) in cells:
        recs, lrecs = data[M, V], ldata[M, V]
        va = cell_values(recs, b, specs)
        vl = cell_values(lrecs, b, specs)
        ida, idl = [], []
        for r in recs:
            ida += [f'{r["rep"]}:{e}' for e in r['qent']]
        for r in lrecs:
            idl += [f'{r["rep"]}:{e}' for e in r['qent']]
        u = sorted(set(idl) | set(ida))
        pos = {x: i for i, x in enumerate(u)}
        ia = np.array([pos[x] for x in ida])
        il = np.array([pos[x] for x in idl])
        k = len(u)
        na = np.bincount(ia, minlength=k).astype(np.float64)
        nl = np.bincount(il, minlength=k).astype(np.float64)
        sa = {nm: np.bincount(ia, weights=v, minlength=k) for nm, v in va.items()}
        sl = {nm: np.bincount(il, weights=v, minlength=k) for nm, v in vl.items()}
        rng = np.random.default_rng(seed0 * 1000 + M + 3 * V + 7 * b)
        dd = {nm: (np.empty(B), np.empty(B)) for nm in specs}
        for b0 in range(0, B, 500):
            b1 = min(B, b0 + 500)
            idx = rng.integers(0, k, size=(b1 - b0, k))
            ta, tl = na[idx].sum(1), nl[idx].sum(1)
            for nm in specs:
                dd[nm][0][b0:b1] = sa[nm][idx].sum(1) / ta
                dd[nm][1][b0:b1] = sl[nm][idx].sum(1) / tl
        dr[M, V, b] = dd
        for nm in specs:
            pts[M, V, b, nm] = (float(va[nm].mean()), float(vl[nm].mean()))
    out = {}
    for rn, R in regions.items():
        for nm in specs:
            a = float(np.mean([pts[c + (nm,)][0] for c in R]))
            l_ = float(np.mean([pts[c + (nm,)][1] for c in R]))
            da = np.mean([dr[c][nm][0] for c in R], 0)
            dl = np.mean([dr[c][nm][1] for c in R], 0)
            out[f'{rn}|{nm}'] = {'alias': a, 'label': l_, 'delta': a - l_, 'alias_ci': SB.ci(da), 'label_ci': SB.ci(dl),
                                 'delta_ci': SB.ci(da - dl)}
            out[f'{rn}|{nm}']['delta_label'] = lab(*out[f'{rn}|{nm}']['delta_ci'])
    return out


def nl72_add(enc, data):
    """'N72_{d}|b|ideal' per draw: the draw-chosen NLMSc configuration's test predictions."""
    J = json.load(open(f'{RES}/b74_nl72_choices.json'))['enc'][enc]
    rows = {}
    for (M, V), recs in data.items():
        for r in recs:
            rep = r['rep']
            for b in BPF:
                e = J[f'{M}|{V}|{b}']
                for d, (c, _) in enumerate(e['draws']):
                    if c == e['full'][0]:
                        p = r['pred'][f'NLMSC|{b}|{A}']
                    else:
                        key = (M, V, rep, c[2])
                        if key not in rows:
                            fn = f'{OUT}/nl72/jobs/{enc}_test_M{M}_V{V}_r{rep}_{c[2]}.json.gz'
                            j = json.load(gzip.open(fn, 'rt'))
                            assert j['qgold'] == r['qgold']
                            rows[key] = {(x['arm'], tuple(x['cfg']), x['bpf']): x['pred'] for x in j['rows']}
                        p = rows[key]['NLMSC', tuple(c), b]
                    r['pred'][f'N72_{d}|{b}|{A}'] = p
    return J


# ----------------------------------------------------------------------------------------------- part / panel

def attribution(pt, X):
    """Registered rule over R1 point values pt[name]."""
    wr, wx, fl, fh = pt['W_R'], pt[f'W_{X}'], pt[f'F_{X}|L'], pt[f'F_{X}|H']
    if abs(fl) > abs(wr) and abs(fh) > abs(wx):
        return 'features larger (identified)'
    if abs(wr) > abs(fl) and abs(wx) > abs(fh):
        return 'write larger (identified)'
    return 'path-dependent'


def part(enc, which='A,B,C'):
    ei = PANEL.index(enc)
    which = which.split(',')
    fn = f'{RES}/b74_part_{enc}.json'
    J = json.load(open(fn)) if os.path.exists(fn) else {'enc': enc}
    data = load_A(enc)
    if 'A' in which:
        J1, c1 = lin_analyse(data, COMBOS_A1, 10000, 7400 + ei)
        J2, c2 = lin_analyse(data, COMBOS_A2, 2000, 7450 + ei)
        J['A'] = {**J1, **J2}
        J['A_cells'] = {**c1, **c2}
        J['A_attr'] = {}
        for rn in RN:
            pt = {nm: J['A'][f'{rn}|{nm}']['diff'] for nm in COMBOS_A1}
            for X in KINDS:
                J['A_attr'][f'{rn}|{X}'] = {
                    'rule': attribution(pt, X), 'path1': {'write': pt['W_R'], 'features': pt[f'F_{X}|L']},
                    'path2': {'features': pt[f'F_{X}|H'], 'write': pt[f'W_{X}']},
                    'interaction': pt[f'I_{X}'], 'gap': pt[f'G_{X}'],
                    'material': (J['A'][f'{rn}|I_{X}']['one_label'] != 'level') and abs(pt[f'I_{X}']) >= 0.01,
                    'shapley': {'write': J['A'][f'{rn}|Wmain_{X}']['diff'], 'features': J['A'][f'{rn}|Fmain_{X}']['diff']}}
        acc = {}
        for (M, V), recs in data.items():
            for b in BPF:
                for a in ('HR', 'LR', 'HP', 'LP', 'HT', 'LT', 'LAMC'):
                    acc[f'{M}|{V}|{b}|{a}'] = float(SB.correct(recs, f'{a}|{b}|{A}').mean())
        J['A_acc'] = acc
    if 'B' in which:
        ldata, repro = lab_preds(enc, data)
        J['B_repro'] = repro
        for M, V in data:
            for r in data[M, V]:
                pass
        J['B_label'], _ = lin_analyse(ldata, {f'{x}-{y}': {x: 1, y: -1} for x, y in PAIRS_B}, 10000, 7500 + ei)
        J['B_alias'], _ = lin_analyse(data, {f'{x}-{y}': {x: 1, y: -1} for x, y in PAIRS_B}, 2000, 7550 + ei)
        specs = {f'{x}-{y}': {x: 1, y: -1} for x, y in PAIRS_B}
        specs.update({a: {a: 1} for a in ARMS_B})
        J['B_paired'] = paired(data, ldata, specs, 2000, 7600 + ei)
        rec = {}
        for (M, V) in data:
            for b in BPF:
                for a in ARMS_B:
                    rec[f'{M}|{V}|{b}|{a}'] = [float(SB.correct(ldata[M, V], f'{a}|{b}|{A}').mean()),
                                               float(SB.correct(data[M, V], f'{a}|{b}|{A}').mean())]
        J['B_cells'] = rec
    if 'C' in which and os.path.exists(f'{RES}/b74_nl72_choices.json'):
        Jc = nl72_add(enc, data)
        combos = {}
        for d in range(NDRAW):
            combos[f'STORE-N72_{d}'] = {'STORE': 1, f'N72_{d}': -1}
            combos[f'VSAPCX-N72_{d}'] = {'VSAPCX': 1, f'N72_{d}': -1}
            combos[f'N72_{d}-NLMSC'] = {f'N72_{d}': 1, 'NLMSC': -1}
        combos['STORE-NLMSC'] = {'STORE': 1, 'NLMSC': -1}
        combos['VSAPCX-NLMSC'] = {'VSAPCX': 1, 'NLMSC': -1}
        J['C'], _ = lin_analyse(data, combos, 2000, 7700 + ei, two=False)
        J['C_sel'] = {k: [v['full'][0]] + [c for c, _ in v['draws']] for k, v in Jc.items()}
    json.dump(J, open(fn, 'w'), indent=1)
    print('->', fn)


def fmt(e, two=True):
    s = f'{100 * e["diff"]:+.2f} [{100 * e["one"][0]:+.2f}, {100 * e["one"][1]:+.2f}] {e["one_label"]}'
    if two and 'two_label' in e and e['two_label'] != e['one_label']:
        s += f' (two-level {e["two_label"]})'
    return s


def panel():
    P = {e: json.load(open(f'{RES}/b74_part_{e}.json')) for e in PANEL if os.path.exists(f'{RES}/b74_part_{e}.json')}
    L = ['B74 verdict (points = accuracy points; one-level 95% interval and label; two-level label shown if different)', '']
    if all('A' in J for J in P.values()):
        L.append('(A) 2 x 3 factorial at the store fixed format: HR = STORE, LR = OPTR; P = principal, T = rotated principal')
        for rn in RN:
            L.append(f'-- {rn}')
            for nm in list(COMBOS_A1) + list(COMBOS_A2):
                L.append(f'  {nm}')
                for e, J in P.items():
                    L.append(f'    {NAMES[e]:16s} {fmt(J["A"][f"{rn}|{nm}"])}')
        L.append('')
        L.append('(A) attribution (registered rule, R1; per corner): path 1 write / features; path 2 features / write; I; Shapley')
        for X in KINDS:
            cnt = {}
            for rn in RN:
                for e, J in P.items():
                    a = J['A_attr'][f'{rn}|{X}']
                    if rn == 'R1':
                        cnt[a['rule']] = cnt.get(a['rule'], 0) + 1
                    L.append(f'  {X} {rn:7s} {NAMES[e]:16s} {a["rule"]:30s} p1 {100 * a["path1"]["write"]:+.2f} / '
                             f'{100 * a["path1"]["features"]:+.2f}  p2 {100 * a["path2"]["features"]:+.2f} / '
                             f'{100 * a["path2"]["write"]:+.2f}  I {100 * a["interaction"]:+.2f} '
                             f'{"material" if a["material"] else ""}  Shapley w {100 * a["shapley"]["write"]:+.2f} '
                             f'f {100 * a["shapley"]["features"]:+.2f}')
            L.append(f'  corner {X}, R1 counts: {cnt}')
        L.append('')
    if all('B_label' in J for J in P.values()):
        L.append('(B) label vs alias queries (R1 regions); recall label / alias per arm; contrasts on labels and aliases; Delta = alias - label')
        for rn in RN:
            L.append(f'-- {rn}')
            for e, J in P.items():
                pr = J['B_paired']
                L.append(f'  {NAMES[e]}: ' + '  '.join(f'{a} {100 * pr[f"{rn}|{a}"]["label"]:.1f}/{100 * pr[f"{rn}|{a}"]["alias"]:.1f}'
                                                       for a in ARMS_B))
                for x, y in PAIRS_B:
                    nm = f'{x}-{y}'
                    d = pr[f'{rn}|{nm}']
                    L.append(f'    {nm:14s} labels {fmt(J["B_label"][f"{rn}|{nm}"])} | aliases {fmt(J["B_alias"][f"{rn}|{nm}"])}'
                             f' | Delta {100 * d["delta"]:+.2f} [{100 * d["delta_ci"][0]:+.2f}, {100 * d["delta_ci"][1]:+.2f}]')
            L.append('')
        L.append('  alias reproduction (same / total alias predictions): ' + '; '.join(
            f'{NAMES[e]} ' + ', '.join(f'{a} {v[0]}/{v[1]}' for a, v in J['B_repro'].items()) for e, J in P.items()))
        L.append('')
    if all('C' in J for J in P.values()):
        L.append('(C) NLMSc re-chosen from 20 sub-grids of 72 (of 480): per-draw R1 values, median [min, max], count < 0, count NLMSc72 leads (one-level)')
        for rn in RN:
            L.append(f'-- {rn}')
            for e, J in P.items():
                for base in ('STORE', 'VSAPCX'):
                    v = [J['C'][f'{rn}|{base}-N72_{d}']['diff'] for d in range(NDRAW)]
                    labs = [J['C'][f'{rn}|{base}-N72_{d}']['one_label'] for d in range(NDRAW)]
                    full = J['C'][f'{rn}|{base}-NLMSC']['diff']
                    q = np.quantile(v, [0, 0.25, 0.5, 0.75, 1])
                    L.append(f'  {NAMES[e]:16s} {base}-NLMSC72 median {100 * q[2]:+.2f} [{100 * q[0]:+.2f}, {100 * q[4]:+.2f}]'
                             f' quartiles {100 * q[1]:+.2f} / {100 * q[3]:+.2f}; < 0 in {sum(x < 0 for x in v)}/20; '
                             f'same sign as full grid in {sum(np.sign(x) == np.sign(full) for x in v)}/20; NLMSc72 leads in '
                             f'{sum(l_ == "-" for l_ in labs)}/20; full grid {100 * full:+.2f}')
                w = [J['C'][f'{rn}|N72_{d}-NLMSC']['diff'] for d in range(NDRAW)]
                L.append(f'  {NAMES[e]:16s} NLMSC72 - NLMSC(full) median {100 * np.median(w):+.2f} [{100 * min(w):+.2f}, {100 * max(w):+.2f}]')
    txt = '\n'.join(L)
    open(f'{RES}/b74_verdict.txt', 'w').write(txt + '\n')
    json.dump(P, open(f'{RES}/b74_verdict.json', 'w'))
    print(txt)


def selftest():
    rng = np.random.default_rng(0)
    # boot_cells with one array = boot_region's draws (same generator state)
    d = rng.integers(-1, 2, 500).astype(np.float64)
    cl = np.array([f'0:{i // 3}' for i in range(500)])
    s, lo, hi, dr = V36.boot_region([(d, cl)], 3000, 11)
    dr2 = boot_cells({'x': d}, cl, 3000, np.random.default_rng(11), chunk=3000)['x']
    assert np.allclose(dr, dr2), 'boot_cells differs from boot_region'
    print('boot_cells = boot_region for one array (same draws): ok')
    # attribution rule
    pt = {'W_R': .015, 'W_T': .04, 'F_T|L': .03, 'F_T|H': .005, 'W_P': 0, 'F_P|L': 0, 'F_P|H': 0}
    assert attribution(pt, 'T') == 'path-dependent'
    pt.update({'W_T': .01, 'F_T|H': .035})
    assert attribution(pt, 'T') == 'features larger (identified)'
    print('attribution rule: ok')


if __name__ == '__main__':
    a = sys.argv[1:]
    if a[0] == 'selftest':
        selftest()
    elif a[0] == 'part':
        part(a[1], a[2] if len(a) > 2 else 'A,B,C')
    elif a[0] == 'panel':
        panel()
    else:
        raise SystemExit(__doc__)
