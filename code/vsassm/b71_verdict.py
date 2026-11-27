"""B71 verdict (prereg/B71_hebbian_remedies_prereg.md): parts (A), (B), (C) and the dev audit; (D) is b71_side.py.

Arms: existing test predictions as B60 / B62 / B66's verdicts read them (merged records b59_out/fam/{enc}/all59_*,
b60_verdict.add_lamc / add_vsapx, b62_verdict.add_b62, b66_lsfc.add_lsf) + B71's test jobs (b71_out/jobs) at the
committed choices (results/b71_choice_{enc}.json). Ideal accounting, R1 budgets.
Statistics: b66_storeboot.analyse (one-level = b36_verdict.boot_region; two-level = B66's store-level scheme).

python3 b71_verdict.py part ENC          # -> results/b71_part_{enc}.json
python3 b71_verdict.py audit ENC         # dev audit -> results/b71_audit_{enc}.json
python3 b71_verdict.py panel             # -> results/b71_verdict.{json,txt}
"""
import gzip
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import b66_storeboot as SB  # noqa: E402

RES = os.environ.get('B71_RES', f'{ROOT}/results')
JOBS = os.path.join(os.environ.get('B71_OUT', f'{ROOT}/b71_out'), 'jobs')
PANEL = SB.PANEL
LLM, SENT = PANEL[:4], PANEL[4:]
NAMES = SB.NAMES
M_GRID, VN, BPF = [256, 1024], [2, 16], [4, 8, 16]
TEST_REPS = {256: 20, 1024: 5}
DEV_REPS = {256: 6, 1024: 3}
REG = SB.REG
RN = ('R1-V2', 'R1-V16', 'R1')
A = 'ideal'
PRIMARY = [('VSAPCX', 'LAMC', 'A'), ('VSAPCX', 'NLMSC', 'A'), ('VSAS', 'LAMC', 'A'), ('VSAS', 'NLMSC', 'A'),
           ('VSAPCX', 'LSFM', 'A-dec'), ('LSFM', 'LAMC', 'A-dec'), ('VSAS', 'LSFR', 'A-dec'), ('LSFR', 'LAMC', 'A-dec'),
           ('VSAPCX', 'NLMSFM', 'A-dec'), ('NLMSFM', 'NLMSC', 'A-dec'), ('VSAS', 'NLMSFR', 'A-dec'),
           ('NLMSFR', 'NLMSC', 'A-dec'),
           ('VSAPR', 'LAMC', 'B'), ('VSAPR', 'NLMSC', 'B'),
           ('VSADR1', 'LAMC', 'C'), ('VSADR1', 'NLMSC', 'C'),
           ('NLMSC', 'VSAS', 'C-cause'), ('NLMSC', 'NLMSFR', 'C-cause'), ('NLMSFR', 'VSADR1', 'C-cause'),
           ('VSADR1', 'VSAS', 'C-cause')]
SECONDARY = [('VSAPR', 'VSAPC', 'B2'), ('VSAPR', 'STORE', 'B2'), ('VSAPR', 'VSAPCX', 'B2'), ('VSAPRX', 'LAMC', 'B2'),
             ('VSAPRX', 'NLMSC', 'B2'),
             ('VSADR1', 'STORE', 'C2'), ('VSADRE', 'VSADR1', 'C2'), ('VSADRE', 'LAMC', 'C2'),
             ('VSADRE', 'NLMSC', 'C2'), ('VSADX', 'LAMC', 'C2'), ('VSADX', 'NLMSC', 'C2'), ('VSADP1', 'VSAPC', 'C2'),
             ('NLMSFR', 'LSFR', 'A2'), ('NLMSFP', 'LSFP', 'A2'), ('STORE', 'LAMC', 'ref'), ('STORE', 'NLMSC', 'ref')]
DECOMP = {'VSAPCX-LAMC': ('VSAPCX-LSFM', 'LSFM-LAMC'), 'VSAS-LAMC': ('VSAS-LSFR', 'LSFR-LAMC'),
          'VSAPCX-NLMSC': ('VSAPCX-NLMSFM', 'NLMSFM-NLMSC'), 'VSAS-NLMSC': ('VSAS-NLMSFR', 'NLMSFR-NLMSC')}
CAUSE = ('NLMSC-NLMSFR', 'NLMSFR-VSADR1', 'VSADR1-VSAS')


def choice(enc):
    return json.load(open(f'{RES}/b71_choice_{enc}.json'))['choice']


def test_index(enc):
    """{(armkind, M, V, rep, bpf, cfg tuple, prep, fmap[, f]): pred} from B71's test jobs."""
    out = {}
    for fn in sorted(os.listdir(JOBS)):
        if not fn.startswith(f'{enc}_') or '_test_' not in fn:
            continue
        r = json.load(gzip.open(os.path.join(JOBS, fn), 'rt'))
        s = r['spec']
        if s['enc'] != enc:
            continue
        if s['arm'] == 'VSAPR':
            for rep, rr in r['reps'].items():
                out['_q', 'VSAPR', s['M'], s['V'], int(rep)] = rr['qgold']
                for x in rr['rows']:
                    out['VSAPR', s['M'], s['V'], int(rep), x['bpf'], tuple(x['cfg'])] = x['pred']
        else:
            ak = s['arm'] + s['kind']
            out['_q', ak, s['M'], s['V'], s['rep'], s['prep'], s['fmap']] = r['qgold']
            for x in r['rows']:
                out[ak, s['M'], s['V'], s['rep'], x['bpf'], tuple(x['cfg']), s['prep'], s['fmap']] = x['pred']
    return out


def load(enc):
    import b60_verdict as V60
    import b62_verdict as V62
    import b66_lsfc as LS
    ch60, ch62, ch66, ch = V60.choice(enc), V62.choice(enc), LS.choice(enc), choice(enc)
    ch56 = json.load(open(f'{ROOT}/results/b56_choice_{enc}.json'))['choice']
    chpc = json.load(open(f'{ROOT}/results/b60_choicepc_{enc}.json'))['choice']
    ti = test_index(enc)
    data = {}
    for M in M_GRID:
        for V in VN:
            recs = [json.loads(l) for l in gzip.open(f'{ROOT}/b59_out/fam/{enc}/all59_{enc}_M{M}_V{V}.jsonl.gz', 'rt')]
            recs.sort(key=lambda r: r['rep'])
            assert [r['rep'] for r in recs] == list(range(TEST_REPS[M]))
            for r in recs:
                for b in BPF:
                    for a in ('ideal', 'real'):
                        r['pred'][f'STORE|{b}|{a}'] = r['pred'][f'VSA|{b}|{a}']
            assert V60.add_lamc(recs, enc, M, V, ch60, 0, bpfs=tuple(BPF))
            assert V62.add_b62(recs, enc, M, V, ch62)
            LS.add_lsf(recs, enc, M, V, ch66)
            data[M, V] = recs
    assert V60.add_vsapx(data, enc)
    for M in M_GRID:
        for V in VN:
            for r in data[M, V]:
                rep = r['rep']
                for b in BPF:
                    cell = f'{M}|{V}|{b}'
                    s = ch56[f'{cell}|ideal|STAR']
                    pc = chpc[f'{cell}|ideal|PSTAR']
                    src = 'VSAPC' if pc['dev'][pc['fam']] > s['dev'][s['fam']] + 0.001 else 'VSAS'
                    kd = 'P' if src == 'VSAPC' else 'R'
                    r['pred'][f'LSFM|{b}|{A}'] = r['pred'][f'{"LSFP" if kd == "P" else "LSFR"}|{b}|{A}']
                    for k in ('R', 'P'):
                        c = ch[f'{cell}|NLMSF{k}']
                        key = (f'NLMSF{k}', M, V, rep, b, tuple(c['cfg']), c['feat'][0], c['feat'][1])
                        assert ti['_q', f'NLMSF{k}', M, V, rep, c['feat'][0], c['feat'][1]] == r['qgold']
                        r['pred'][f'NLMSF{k}|{b}|{A}'] = ti[key]
                        for nm in (f'VSAD{k}1', f'VSAD{k}E'):
                            c = ch[f'{cell}|{nm}']
                            key = (f'VSAD{k}', M, V, rep, b, tuple(c['cfg']), c['feat'][0], c['feat'][1])
                            assert ti['_q', f'VSAD{k}', M, V, rep, c['feat'][0], c['feat'][1]] == r['qgold']
                            r['pred'][f'{nm}|{b}|{A}'] = ti[key]
                    r['pred'][f'NLMSFM|{b}|{A}'] = r['pred'][f'NLMSF{kd}|{b}|{A}']
                    c = ch[f'{cell}|VSAPR']
                    assert ti['_q', 'VSAPR', M, V, rep] == r['qgold']
                    r['pred'][f'VSAPR|{b}|{A}'] = ti['VSAPR', M, V, rep, b, tuple(c['cfg'])]
                    r['pred'][f'VSAPRX|{b}|{A}'] = r['pred'][f'{ch[cell + "|VSAPRX"]["src"]}|{b}|{A}']
                    r['pred'][f'VSADX|{b}|{A}'] = r['pred'][f'{ch[cell + "|VSADX"]["src"]}|{b}|{A}']
                    r['pred'][f'VSADR1|{b}|{A}'] = r['pred'][f'VSADR1|{b}|{A}']
    return data


def part(enc):
    data = load(enc)
    ei = PANEL.index(enc)
    J1 = SB.analyse(data, ei, PRIMARY, accts=[A], nb={A: 10000}, regions=REG)
    J2 = SB.analyse(data, ei + 100, SECONDARY, accts=[A], nb={A: 2000}, regions=REG)
    # per-cell accuracies of every arm (for the tables and the plain-loop check)
    arms = sorted({x for c in PRIMARY + SECONDARY for x in c[:2]})
    acc = {}
    for (M, V), recs in data.items():
        for b in BPF:
            for arm in arms:
                c = SB.correct(recs, f'{arm}|{b}|{A}')
                acc[f'{M}|{V}|{b}|{arm}'] = float(c.mean())
    J = {'enc': enc, 'primary': J1['regions'], 'secondary': J2['regions'], 'missing': J1['missing'] + J2['missing'],
         'cell_acc': acc}
    json.dump(J, open(f'{RES}/b71_part_{enc}.json', 'w'))
    print('part', enc, 'missing', len(J['missing']))


# ----------------------------------------------------------------------------------------------- dev audit

def audit(enc):
    """Dev-only audit of B56's retraining and the delta rule on B53's configuration (kind R), per R1 cell."""
    import b71_run as E
    ch = choice(enc)
    acc = E.load_devsum(enc) if os.environ.get('B71_DEVSUM') == '1' else E.load_dev(enc)
    b53 = json.load(open(f'{ROOT}/results/b53_choice_{enc}.json'))['choice']
    cfgR = E.b53_cfg(enc)
    out = {}
    for M in M_GRID:
        for V in VN:
            for b in BPF:
                cell = f'{M}|{V}|{b}'
                prep, fmap, q53, f = cfgR[M, V, b]
                au = ch[f'{cell}|AUDIT']

                def mean(k, fld='acc'):
                    return E.dev_mean(acc, k, DEV_REPS[M], fld)

                def upd(k, e):
                    d = acc[k]
                    if 'm' in d:
                        return d['m']['nupd'][e]
                    return float(np.mean([d[i]['nupd'][e] for i in sorted(d)]))
                qr, er, etar = au['rt_b56grid']['cfg']
                krt = ('RT', M, V, b, (qr, er, etar), prep, fmap)
                kheb = ('HEB', M, V, b, (q53,), prep, fmap)
                d1 = {g: mean(('VSADR', M, V, b, g, prep, fmap)) for g in E.D1_GRID}
                best_n = max(v for g, v in d1.items() if g[1] == 'nlms')
                best_e = max(v for g, v in d1.items() if g[1] == 'err')
                dall = {g: mean(('VSADR', M, V, b, g, prep, fmap)) for g in E.D_GRID}
                c1 = tuple(ch[f'{cell}|VSADR1']['cfg'])
                kd1 = ('VSADR', M, V, b, c1, prep, fmap)
                ns = max(1, int(f * M))
                out[cell] = {
                    'b53_dev': b53[f'{cell}|ideal|VSA']['dev_acc'], 'heb_dev_repro': mean(kheb),
                    'heb_float': mean(kheb, 'acc_float'), 'heb_keys': mean(kheb, 'acc_keys'),
                    'heb_keys_float': mean(kheb, 'acc_keys_float'),
                    'b56_rt_cfg': au['b56_rt_cfg'], 'b56_rt_dev': au['b56_rt_dev'],
                    'rt_cfg_repro': au['rt_b56grid']['cfg'], 'rt_dev_repro': au['rt_b56grid']['dev_acc'],
                    'rt_float': mean(krt, 'acc_float'), 'rt_keys': mean(krt, 'acc_keys'),
                    'rt_upd_frac': [upd(krt, e) / ns for e in range(er)],
                    'rt_ext_cfg': au['rt_ext']['cfg'], 'rt_ext_dev': au['rt_ext']['dev_acc'],
                    'vsad1_cfg': list(c1), 'vsad1_dev': ch[f'{cell}|VSADR1']['dev_acc'],
                    'vsad1_float': mean(kd1, 'acc_float'), 'vsad1_keys': mean(kd1, 'acc_keys'),
                    'vsadE_dev': ch[f'{cell}|VSADRE']['dev_acc'], 'vsadE_cfg': ch[f'{cell}|VSADRE']['cfg'],
                    'd1_best_nlms': best_n, 'd1_best_err': best_e,
                    'dall_best_nlms': max(v for g, v in dall.items() if g[1] == 'nlms'),
                    'dall_best_err': max(v for g, v in dall.items() if g[1] == 'err'),
                    'nlmsfr_dev': ch[f'{cell}|NLMSFR']['dev_acc'], 'vsas_dev': None}
    json.dump({'enc': enc, 'cells': out}, open(f'{RES}/b71_audit_{enc}.json', 'w'), indent=1)
    print('audit', enc, len(out))


# ----------------------------------------------------------------------------------------------- panel

def fmt(e, k='one'):
    lo, hi = e[k]['ci95']
    return f'{100 * e["diff"]:+6.2f} [{100 * lo:+6.2f}, {100 * hi:+6.2f}] {e[k]["label"]}'


def short(e, x, y, k='one'):
    lab = e[k]['label']
    return {f'{x} leads': x, f'{y} leads': y}.get(lab, 'level')


def panel():
    P = {enc: json.load(open(f'{RES}/b71_part_{enc}.json')) for enc in PANEL if os.path.exists(f'{RES}/b71_part_{enc}.json')}
    encs = [e for e in PANEL if e in P]
    L = ['# B71 verdict (prereg/B71_hebbian_remedies_prereg.md). Accuracy points over R1 (V 2 / 16 x 4, 8, 16 bits per '
         'fact x M 256 / 1,024), ideal accounting.',
         '# One-level = 95% cluster bootstrap (entity within store; registered label, B60 rule); two-level = B66 store-level '
         'scheme (stores, then entities).', '']
    J = {'encs': encs, 'tables': {}}
    for grp, title in (('A', '(A) strongest Hebbian variants against least squares (batch LAMc, online NLMSc)'),
                       ('A-dec', '(A) decomposition terms: write rule at identical features / features at identical write'),
                       ('B', '(B) rotated principal features (VSAPR)'), ('C', '(C) the store written by a delta rule (VSADR1, one pass)'),
                       ('C-cause', '(C) cause of B56\'s small gain: NLMSC - VSAS = features + layout + write rule')):
        L.append(f'## {title}')
        for (x, y, g) in PRIMARY:
            if g != grp:
                continue
            nm = f'{x}-{y}'
            L += ['', f'### {x} - {y}', '| encoder | R1-V2 | R1-V16 | R1 (one-level) | R1 (two-level) |', '| --- | --- | --- | --- | --- |']
            cnt = {}
            for enc in encs:
                es = {rn: P[enc]['primary'].get(f'{A}|{rn}|{nm}') for rn in RN}
                if any(v is None for v in es.values()):
                    L.append(f'| {NAMES[enc]} | missing | | | |')
                    continue
                L.append(f'| {NAMES[enc]} | {fmt(es["R1-V2"])} | {fmt(es["R1-V16"])} | {fmt(es["R1"])} | '
                         f'{fmt(es["R1"], "two")} |')
                J['tables'][f'{enc}|{nm}'] = es
                for rn in RN:
                    for k in ('one', 'two'):
                        g2 = 'LLM' if enc in LLM else 'sent'
                        cnt.setdefault((rn, k, g2), {}).setdefault(short(es[rn], x, y, k), 0)
                        cnt[rn, k, g2][short(es[rn], x, y, k)] += 1
            for rn in RN:
                L.append(f'  {rn:7s} one-level: LLM {cnt.get((rn, "one", "LLM"), {})}  sentence {cnt.get((rn, "one", "sent"), {})}'
                         f' | two-level: LLM {cnt.get((rn, "two", "LLM"), {})}  sentence {cnt.get((rn, "two", "sent"), {})}')
        L.append('')
    # decomposition readings
    L += ['## (A) readings: write rule vs features over R1 (|write| >= 2 |feature| -> write; |feature| >= 2 |write| -> features; else both)', '',
          '| encoder | ' + ' | '.join(DECOMP) + ' |', '| --- |' + ' --- |' * len(DECOMP)]
    J['readings'] = {}
    for enc in encs:
        row = []
        for tot, (w, f) in DECOMP.items():
            ew, ef, et = (P[enc]['primary'].get(f'{A}|R1|{k}') for k in (w, f, tot))
            if ew is None or ef is None:
                row.append('missing')
                continue
            dw, df = ew['diff'], ef['diff']
            rd = 'write rule' if abs(dw) >= 2 * abs(df) else ('features' if abs(df) >= 2 * abs(dw) else 'both')
            J['readings'][f'{enc}|{tot}'] = {'write': dw, 'feature': df, 'total': et['diff'], 'reading': rd}
            row.append(f'{100 * et["diff"]:+.2f} = write {100 * dw:+.2f} + features {100 * df:+.2f}: {rd}')
        L.append(f'| {NAMES[enc]} | ' + ' | '.join(row) + ' |')
    # gap closing
    L += ['', '## (B) / (C) closing the gap (registered: R1 remedy - Y "level" or "remedy leads", one-level)', '']
    J['gap'] = {}
    for x in ('VSAPR', 'VSADR1'):
        for y in ('LAMC', 'NLMSC'):
            closed = [enc for enc in encs if P[enc]['primary'][f'{A}|R1|{x}-{y}']['one']['label'] != f'{y} leads']
            share = {}
            for enc in encs:
                st = P[enc]['secondary'].get(f'{A}|R1|STORE-{y}')
                xs = P[enc]['secondary'].get(f'{A}|R1|{x}-STORE')
                if x == 'VSAPR':
                    xs = P[enc]['secondary'].get(f'{A}|R1|VSAPR-STORE')
                if st is not None and xs is not None and st['diff'] != 0:
                    share[enc] = xs['diff'] / (-st['diff'])
            J['gap'][f'{x}-{y}'] = {'closed_on': closed, 'share_closed': share}
            L.append(f'- {x} vs {y}: closed on {len(closed)} of {len(encs)} ({", ".join(NAMES[e] for e in closed) or "none"}); '
                     'share of the store\'s gap closed: ' + ', '.join(f'{NAMES[e]} {100 * v:.0f}%' for e, v in share.items()))
    # cause
    L += ['', '## (C) cause terms over R1 (NLMSC - VSAS = features + layout + write rule)', '',
          '| encoder | total | features NLMSC-NLMSFR | layout NLMSFR-VSADR1 | write rule VSADR1-VSAS | largest |', '| --- | --- | --- | --- | --- | --- |']
    J['cause'] = {}
    means = {k: [] for k in CAUSE}
    for enc in encs:
        es = {k: P[enc]['primary'][f'{A}|R1|{k}']['diff'] for k in CAUSE}
        tot = P[enc]['primary'][f'{A}|R1|NLMSC-VSAS']['diff']
        big = max(CAUSE, key=lambda k: es[k])
        for k in CAUSE:
            means[k].append(es[k])
        J['cause'][enc] = {'total': tot, **es, 'largest': big,
                           'contrib': [k for k in CAUSE if tot != 0 and es[k] / tot >= 1 / 3]}
        L.append(f'| {NAMES[enc]} | {100 * tot:+.2f} | ' + ' | '.join(fmt(P[enc]['primary'][f'{A}|R1|{k}']) for k in CAUSE)
                 + f' | {big} |')
    mm = {k: float(np.mean(v)) for k, v in means.items()}
    main = max(CAUSE, key=lambda k: mm[k])
    nctr = {k: sum(1 for enc in encs if k in J['cause'][enc]['contrib']) for k in CAUSE}
    J['cause_panel'] = {'mean': mm, 'main': main, 'contrib_count': nctr}
    L.append(f'Mean over encoders: ' + ', '.join(f'{k} {100 * v:+.2f}' for k, v in mm.items()) + f'; MAIN CAUSE (registered rule): {main}; '
             f'contributing (>= 1/3 of the total) on: ' + ', '.join(f'{k} {v}' for k, v in nctr.items()))
    # secondary
    L += ['', '## Secondary contrasts (one-level, 2,000 draws)', '']
    for (x, y, g) in SECONDARY:
        nm = f'{x}-{y}'
        L += [f'### {x} - {y}', '| encoder | R1-V2 | R1-V16 | R1 |', '| --- | --- | --- | --- |']
        for enc in encs:
            es = {rn: P[enc]['secondary'].get(f'{A}|{rn}|{nm}') for rn in RN}
            if any(v is None for v in es.values()):
                L.append(f'| {NAMES[enc]} | missing | | |')
                continue
            L.append(f'| {NAMES[enc]} | {fmt(es["R1-V2"])} | {fmt(es["R1-V16"])} | {fmt(es["R1"])} |')
            J['tables'][f'{enc}|{nm}'] = es
        L.append('')
    # region means (raw accuracy) of the main arms
    arms = ['STORE', 'VSAS', 'VSAPC', 'VSAPCX', 'VSAPR', 'VSAPRX', 'VSADR1', 'VSADRE', 'VSADX', 'NLMSFR', 'NLMSFP', 'NLMSFM',
            'LSFR', 'LSFP', 'LSFM', 'NLMSC', 'LAMC']
    L += ['## R1 region means (accuracy %, mean over the 12 cells)', '', '| encoder | ' + ' | '.join(arms) + ' |',
          '| --- |' + ' --- |' * len(arms)]
    for enc in encs:
        ca = P[enc]['cell_acc']
        L.append(f'| {NAMES[enc]} | ' + ' | '.join(
            f'{100 * np.mean([ca[f"{M}|{V}|{b}|{a}"] for M in M_GRID for V in VN for b in BPF]):.2f}'
            if all(f'{M}|{V}|{b}|{a}' in ca for M in M_GRID for V in VN for b in BPF) else '-' for a in arms) + ' |')
    # two-level vs one-level label changes
    ch_n, tot_n = 0, 0
    for enc in encs:
        for (x, y, g) in PRIMARY[:4]:
            for rn in RN:
                e = P[enc]['primary'][f'{A}|{rn}|{x}-{y}']
                tot_n += 1
                ch_n += e['one']['label'] != e['two']['label']
    J['two_level_changes'] = [ch_n, tot_n]
    L += ['', f'Two-level label changes on the four (A) contrasts: {ch_n} of {tot_n}.']
    missing = sum(len(P[e]['missing']) for e in encs)
    L.append(f'Missing contrast-cells: {missing}.')
    open(f'{RES}/b71_verdict.txt', 'w').write('\n'.join(L) + '\n')
    json.dump(J, open(f'{RES}/b71_verdict.json', 'w'), indent=1)
    print('\n'.join(L[:80]))


def main():
    a = sys.argv[1:]
    if a[0] == 'part':
        part(a[1])
    elif a[0] == 'audit':
        audit(a[1])
    elif a[0] == 'panel':
        panel()
    elif a[0] == 'audit_panel':
        audit_panel()
    else:
        raise SystemExit(__doc__)



def audit_panel():
    """Dev audit summary per encoder (means over the 12 R1 cells, points) -> results/b71_audit.txt / .json."""
    L = ['# B71 dev audit of B56\'s retraining and the delta-written store (DEV stores, B53\'s configuration of each R1 cell;',
         '# means over the 12 R1 cells, accuracy points). Registered factor rules: quantisation named if its loss >= 0.5 on >= 4',
         '# encoders; grid named if the extended grid adds >= 0.5 on >= 4; misread-only updates named if nlms beats err by >= 0.5',
         '# on >= 4.', '',
         '| encoder | Hebbian (B53) | Hebbian unquantized | stored-key acc. Hebbian | RT (B56 grid) | RT unquantized | RT quant. loss | '
         'RT ext. grid | ext. gain | RT corrected facts / epoch (first epoch, share) | delta 1 pass (VSADR1) | delta unquant. | '
         'delta stored-key acc. | best nlms - best err (1 pass) | NLMSFR |',
         '| --- |' + ' --- |' * 15]
    J = {}
    cnt = {'quant': 0, 'grid': 0, 'err': 0}
    for enc in PANEL:
        A = json.load(open(f'{RES}/b71_audit_{enc}.json'))['cells']
        m = lambda k: 100 * float(np.mean([a[k] for a in A.values()]))  # noqa: E731
        ql = m('rt_float') - m('rt_dev_repro')
        eg = m('rt_ext_dev') - m('rt_dev_repro')
        ne = m('d1_best_nlms') - m('d1_best_err')
        up = float(np.mean([a['rt_upd_frac'][0] for a in A.values()]))
        J[enc] = {'heb': m('b53_dev'), 'heb_float': m('heb_float'), 'heb_keys': m('heb_keys'), 'rt': m('rt_dev_repro'),
                  'rt_float': m('rt_float'), 'quant_loss': ql, 'rt_ext': m('rt_ext_dev'), 'ext_gain': eg, 'rt_upd1': up,
                  'vsad1': m('vsad1_dev'), 'vsad1_float': m('vsad1_float'), 'vsad1_keys': m('vsad1_keys'),
                  'nlms_minus_err': ne, 'nlmsfr': m('nlmsfr_dev'), 'rt_keys': m('rt_keys')}
        cnt['quant'] += ql >= 0.5
        cnt['grid'] += eg >= 0.5
        cnt['err'] += ne >= 0.5
        e = J[enc]
        L.append(f'| {NAMES[enc]} | {e["heb"]:.2f} | {e["heb_float"]:.2f} | {e["heb_keys"]:.2f} | {e["rt"]:.2f} | '
                 f'{e["rt_float"]:.2f} | {ql:+.2f} | {e["rt_ext"]:.2f} | {eg:+.2f} | {100 * up:.1f}% | {e["vsad1"]:.2f} | '
                 f'{e["vsad1_float"]:.2f} | {e["vsad1_keys"]:.2f} | {ne:+.2f} | {e["nlmsfr"]:.2f} |')
    L += ['', f'Encoders meeting the factor rules: quantisation {cnt["quant"]} of 7; extended grid {cnt["grid"]} of 7; '
          f'misread-only updates (nlms - err >= 0.5) {cnt["err"]} of 7.',
          'Named as contributing (registered rule, >= 4 encoders): ' +
          (', '.join(k for k, v in cnt.items() if v >= 4) or 'none')]
    J['_counts'] = cnt
    open(f'{RES}/b71_audit.txt', 'w').write('\n'.join(L) + '\n')
    json.dump(J, open(f'{RES}/b71_audit.json', 'w'), indent=1)
    print('\n'.join(L))


if __name__ == '__main__':
    main()
