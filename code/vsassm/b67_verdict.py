"""B67 verdict (prereg/B67_theory_vs_measurement_prereg.md): agreement statistics for (a) and the ratios for (b).

python3 b67_verdict.py [ENC ...]     # -> results/b67_verdict.{json,txt}
"""
import json
import math
import os
import sys

import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import b67_theory as T  # noqa: E402

RES = T.RES
NB = 10000


def load_a(encs):
    meas = json.load(open(f'{RES}/b66_B_verdict.json'))['cells']
    rows = []
    for e in encs:
        fn = f'{RES}/b67_a_{e}.json'
        if not os.path.exists(fn):
            continue
        th = json.load(open(fn))['cells']
        for (M, V, b) in T.CELLS:
            c = th[f'{M}|{V}|{b}']
            m = meas[f'{e}|ideal|{M}|{V}|{b}']
            rows.append({'enc': e, 'M': M, 'V': V, 'b': b,
                         'pS': 100 * c['STORE'], 'pR': 100 * c['LSFR'], 'pL': 100 * c['LAMC'],
                         'pR2': 100 * c.get('LSFR2', float('nan')), 'pL2': 100 * c.get('LAMC2', float('nan')),
                         'mS': 100 * m['STORE'], 'mR': 100 * m['LSFR'], 'mL': 100 * m['LAMC'],
                         'cfgS': c['STORE_cfg'], 'cfgR': c['LSFR_cfg'], 'cfgL': c['LAMC_cfg']})
    return rows


def spearman_ci(x, y, rng):
    x, y = np.asarray(x), np.asarray(y)
    r = stats.spearmanr(x, y).statistic
    n = len(x)
    bs = []
    for _ in range(NB):
        i = rng.integers(0, n, n)
        if np.ptp(x[i]) == 0 or np.ptp(y[i]) == 0:
            continue
        bs.append(stats.spearmanr(x[i], y[i]).statistic)
    return float(r), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def verdict_a(rows, out, txt):
    rng = np.random.default_rng(67)
    n = len(rows)
    encs = sorted({r['enc'] for r in rows}, key=T.PANEL.index)
    txt.append(f'## (a) {n} cells, encoders: {", ".join(T.NAMES[e] for e in encs)} (ideal accounting, points)')
    res = {}
    pairs = {'STORE-LSFR': ('pS', 'pR', 'mS', 'mR'), 'STORE-LAMC': ('pS', 'pL', 'mS', 'mL')}
    if all(np.isfinite(r['pR2']) for r in rows):
        pairs.update({"EXPLORATORY STORE-LSFR' (L')": ('pS', 'pR2', 'mS', 'mR'),
                      "EXPLORATORY STORE-LAMC' (L')": ('pS', 'pL2', 'mS', 'mL')})
    for pair, (pa, pb, ma, mb) in pairs.items():
        pg = np.array([r[pa] - r[pb] for r in rows])
        mg = np.array([r[ma] - r[mb] for r in rows])
        rho, lo, hi = spearman_ci(pg, mg, rng)
        pear = float(np.corrcoef(pg, mg)[0, 1])
        sign = float(np.mean(np.sign(pg) == np.sign(mg)))
        b3 = float(np.mean(np.abs(mg - pg) <= 3))
        b5 = float(np.mean(np.abs(mg - pg) <= 5))
        neg = int((pg < 0).sum())
        res[pair] = {'spearman': rho, 'ci': [lo, hi], 'pearson': pear, 'sign_agree': sign, 'band3': b3, 'band5': b5,
                     'pred_neg': neg, 'meas_neg': int((mg < 0).sum()), 'mean_pred': float(pg.mean()),
                     'mean_meas': float(mg.mean())}
        txt.append(f'{pair}: Spearman {rho:+.3f} [{lo:+.3f}, {hi:+.3f}], Pearson {pear:+.3f}; sign agreement '
                   f'{100 * sign:.1f}%; |meas - pred| <= 3: {100 * b3:.1f}%, <= 5: {100 * b5:.1f}%; predicted < 0 in '
                   f'{neg} / {n}, measured < 0 in {int((mg < 0).sum())} / {n}; mean predicted {pg.mean():+.2f}, '
                   f'measured {mg.mean():+.2f}')
        for V in (2, 16):
            sel = np.array([r['V'] == V for r in rows])
            if sel.sum() > 3:
                rv = stats.spearmanr(pg[sel], mg[sel]).statistic
                txt.append(f'   V = {V}: Spearman {rv:+.3f}, sign {100 * np.mean(np.sign(pg[sel]) == np.sign(mg[sel])):.1f}%,'
                           f' mean predicted {pg[sel].mean():+.2f}, measured {mg[sel].mean():+.2f}')
                res[pair][f'V{V}'] = {'spearman': float(rv), 'mean_pred': float(pg[sel].mean()),
                                      'mean_meas': float(mg[sel].mean())}
    txt.append('')
    txt.append('Accuracy levels: measured - predicted (bias), mean absolute error, Spearman (all / V=2 / V=16)')
    res['levels'] = {}
    arms = {'STORE': ('pS', 'mS'), 'LSFR': ('pR', 'mR'), 'LAMC': ('pL', 'mL')}
    if all(np.isfinite(r['pR2']) for r in rows):
        arms.update({"LSFR'": ('pR2', 'mR'), "LAMC'": ('pL2', 'mL')})
    for arm, (p, m) in arms.items():
        pv = np.array([r[p] for r in rows])
        mv = np.array([r[m] for r in rows])
        d = {'bias': float((mv - pv).mean()), 'mae': float(np.abs(mv - pv).mean()),
             'spearman': float(stats.spearmanr(pv, mv).statistic)}
        line = f'  {arm:6s} bias {d["bias"]:+.2f} MAE {d["mae"]:.2f} Spearman {d["spearman"]:.3f}'
        for V in (2, 16):
            sel = np.array([r['V'] == V for r in rows])
            d[f'V{V}'] = {'bias': float((mv[sel] - pv[sel]).mean()), 'mae': float(np.abs(mv[sel] - pv[sel]).mean()),
                          'spearman': float(stats.spearmanr(pv[sel], mv[sel]).statistic)}
            line += (f' | V={V}: bias {d[f"V{V}"]["bias"]:+.2f} MAE {d[f"V{V}"]["mae"]:.2f} '
                     f'rho {d[f"V{V}"]["spearman"]:.3f}')
        for grp, ee in (('LLM', T.LLM), ('sent', T.PANEL[4:])):
            sel = np.array([r['enc'] in ee for r in rows])
            if sel.any():
                d[grp] = {'bias': float((mv[sel] - pv[sel]).mean()), 'mae': float(np.abs(mv[sel] - pv[sel]).mean())}
        res['levels'][arm] = d
        txt.append(line)
    txt.append('')
    txt.append('Per encoder, R1 mean (12 cells): predicted / measured gap, ratio measured / predicted')
    txt.append('| encoder | STORE pred / meas | LSFR pred / meas | LAMc pred / meas | STORE-LSFR pred / meas (ratio) |'
               ' STORE-LAMc pred / meas (ratio) |')
    txt.append('| --- | --- | --- | --- | --- | --- |')
    res['per_enc'] = {}
    for e in encs:
        rr = [r for r in rows if r['enc'] == e]
        mv = {k: float(np.mean([r[k] for r in rr])) for k in ('pS', 'pR', 'pL', 'mS', 'mR', 'mL')}
        gR = (mv['pS'] - mv['pR'], mv['mS'] - mv['mR'])
        gL = (mv['pS'] - mv['pL'], mv['mS'] - mv['mL'])
        pR2 = float(np.mean([r['pR2'] for r in rr]))
        pL2 = float(np.mean([r['pL2'] for r in rr]))
        res['per_enc'][e] = {**mv, 'gapR': gR, 'gapL': gL, 'pR2': pR2, 'pL2': pL2,
                             'gapR2': (mv['pS'] - pR2, gR[1]), 'gapL2': (mv['pS'] - pL2, gL[1])}
        rat = lambda g: (g[1] / g[0]) if abs(g[0]) > 1e-9 else float('nan')  # noqa: E731
        txt.append(f'| {T.NAMES[e]} | {mv["pS"]:.1f} / {mv["mS"]:.1f} | {mv["pR"]:.1f} / {mv["mR"]:.1f} | '
                   f'{mv["pL"]:.1f} / {mv["mL"]:.1f} | {gR[0]:+.2f} / {gR[1]:+.2f} ({rat(gR):+.2f}) | '
                   f'{gL[0]:+.2f} / {gL[1]:+.2f} ({rat(gL):+.2f}) |'
                   + (f" EXPLORATORY L': LSFR' {pR2:.1f}, LAMc' {pL2:.1f}, gaps {mv['pS'] - pR2:+.2f} / "
                      f"{mv['pS'] - pL2:+.2f}" if np.isfinite(pR2) else ''))
    txt.append('')
    txt.append('Per cell (points): enc M V bpf | STORE pred meas | LSFR pred meas | LAMc pred meas | gapR pred meas')
    for r in rows:
        txt.append(f'  {r["enc"]:7s} {r["M"]:5d} {r["V"]:3d} {r["b"]:3d} | {r["pS"]:5.1f} {r["mS"]:5.1f} | '
                   f'{r["pR"]:5.1f} {r["mR"]:5.1f} | {r["pL"]:5.1f} {r["mL"]:5.1f} | L\' {r["pR2"]:5.1f} {r["pL2"]:5.1f} | {r["pS"] - r["pR"]:+6.2f} '
                   f'{r["mS"] - r["mR"]:+6.2f}   store {r["cfgS"]} lsfr {r["cfgR"]}')
    txt.append('')
    # departure map
    txt.append('Largest departures (measured - predicted), per arm, top 8:')
    for arm, (p, m) in {'STORE': ('pS', 'mS'), 'LSFR': ('pR', 'mR'), 'LAMC': ('pL', 'mL')}.items():
        dd = sorted(rows, key=lambda r: -abs(r[m] - r[p]))[:8]
        txt.append(f'  {arm}: ' + '; '.join(f'{r["enc"]} M{r["M"]} V{r["V"]} b{r["b"]} {r[m] - r[p]:+.1f}' for r in dd))
    out['a'] = res
    # predictions
    pr = {}
    R = res['STORE-LSFR']
    L = res['STORE-LAMC']
    nall = len(rows)
    pe = res['per_enc']
    pr['PA1'] = (R['pred_neg'] >= 60 * nall / 84 and all(pe[e]['gapR'][0] < 0 for e in encs)
                 and L['pred_neg'] >= 70 * nall / 84,
                 f"STORE-LSFR predicted < 0 in {R['pred_neg']} / {nall}; R1 predicted < 0 on "
                 f"{sum(pe[e]['gapR'][0] < 0 for e in encs)} / {len(encs)}; STORE-LAMc predicted < 0 in "
                 f"{L['pred_neg']} / {nall}")
    pr['PA2'] = (R['spearman'] >= 0.4 and L['spearman'] >= 0.4, f"Spearman {R['spearman']:+.3f} / {L['spearman']:+.3f}")
    lv = res['levels']
    pr['PA3'] = (all(lv[a]['spearman'] >= 0.9 for a in lv) and lv['STORE']['mae'] <= 5 and lv['LSFR']['mae'] <= 7
                 and lv['LAMC']['mae'] <= 7,
                 '; '.join(f"{a} rho {lv[a]['spearman']:.3f} MAE {lv[a]['mae']:.2f}" for a in lv))
    pr['PA4'] = (R['band3'] >= 0.6, f"band3 {100 * R['band3']:.1f}%")
    pr['PA5'] = (lv['STORE']['V16']['mae'] > lv['STORE']['V2']['mae'] and lv['STORE']['V16']['bias'] < 0,
                 f"store MAE V2 {lv['STORE']['V2']['mae']:.2f} V16 {lv['STORE']['V16']['mae']:.2f}; V16 "
                 f"measured - predicted {lv['STORE']['V16']['bias']:+.2f} (over-prediction = negative)")
    k6 = sum(abs(pe[e]['gapR'][0] - pe[e]['gapR'][1]) <= 2 for e in encs)
    pr['PA6'] = (k6 >= 4 * len(encs) / 7, f'R1 STORE-LSFR within 2 points on {k6} / {len(encs)}')
    out['pred_a'] = {k: {'held': bool(v[0]), 'detail': v[1]} for k, v in pr.items()}
    txt.append('')
    txt.append('Predictions (a):')
    for k, v in pr.items():
        txt.append(f'  {"HELD  " if v[0] else "FAILED"} {k}: {v[1]}')


def verdict_b(out, txt):
    fn = f'{RES}/b67_b.json'
    if not os.path.exists(fn):
        txt.append('(b) not run')
        return
    B = json.load(open(fn))
    txt.append('')
    txt.append(f"## (b) cells per stored pair at 99.5% exact recall ({B['stores']} RULER test stores, structure only; "
               f"N_v {B['Nv_min']}-{B['Nv_max']} (median {B['Nv_median']:.0f}), {B['slots_mean']:.2f} slots per probed "
               f"key, per-slot error budget {B['eps_slot']:.2e})")
    txt.append('| condition | e | dense meas | D-a | D-b | meas/D-a | meas/D-b (steps) | SBC meas | S-a | S-b (lam) | '
               'meas/S-a | meas/S-b (steps) | best table | T-a | T-b | table/T-b |')
    txt.append('| --- ' * 16 + '|')
    res = {}
    for i, k in enumerate(T.KS):
        p = B['pred'][k]
        snap = lambda x: min(T.GRIDB, key=lambda g: abs(g - x))  # noqa: E731  (measured N* are grid values, printed rounded)
        md, ms, mt = snap(T.MEAS['dense'][i]), snap(T.MEAS['sbc'][i]), snap(T.MEAS['best_table'][i])
        g = lambda x: (f'{x:.0f}' if x is not None else '>4096')  # noqa: E731
        sd = T.steps(md, p['D-b']) if p['D-b'] else None
        ss = T.steps(ms, p['S-b']) if p['S-b'] else None
        res[k] = {'dense': md, 'D-a': p['D-a'], 'D-b': p['D-b'], 'r_Da': md / p['D-a'],
                  'r_Db': md / p['D-b'] if p['D-b'] else None, 'steps_Db': sd,
                  'sbc': ms, 'S-a': p['S-a'], 'S-b': p['S-b'], 'S-b_lam': p['S-b_lam'], 'r_Sa': ms / p['S-a'],
                  'r_Sb': ms / p['S-b'] if p['S-b'] else None, 'steps_Sb': ss,
                  'table': mt, 'T-a': p['T-a'], 'T-b': p['T-b'], 'r_Tb': mt / p['T-b'],
                  'D-a_grid': T.gridup(p['D-a']), 'S-a_grid': T.gridup(p['S-a']),
                  'S-a_note': 'no-error bound (meaningful at k1 only)'}
        txt.append(f"| {k} | {T.COND[k]:.4f} | {md:.0f} | {p['D-a']:.1f} | {g(p['D-b'])} | {md / p['D-a']:.2f} | "
                   f"{(md / p['D-b']) if p['D-b'] else float('nan'):.2f} ({sd if sd is None else round(sd, 2)}) | {ms:.0f} | "
                   f"{p['S-a']:.1f} | {g(p['S-b'])} ({p['S-b_lam']}) | {ms / p['S-a']:.2f} | "
                   f"{(ms / p['S-b']) if p['S-b'] else float('nan'):.2f} ({ss if ss is None else round(ss, 2)}) | {mt:.0f} | "
                   f"{p['T-a']:.1f} | {p['T-b']:.1f} | {mt / p['T-b']:.2f} |")
    txt.append(f"B61's registered scratch model (for reference): dense {T.B61_MODEL['dense']}, sparse block "
               f"{T.B61_MODEL['sbc']}")
    out['b'] = res
    pr = {}
    pr['PB1'] = (all(res[k]['r_Da'] >= 1.2 for k in T.KS), 'meas / D-a ' + ', '.join(f"{res[k]['r_Da']:.2f}" for k in T.KS))
    lowk = ['k1', 'k2', 'k4', 'e05', 'k8', 'k16']
    pr['PB2'] = (all(res[k]['steps_Db'] is not None and abs(res[k]['steps_Db']) <= 2 + 1e-9 for k in lowk)
                 and res['k32']['steps_Db'] is not None and res['k32']['steps_Db'] >= 1 - 1e-9,
                 'steps meas - D-b ' + ', '.join(f"{res[k]['steps_Db']:+.2f}" if res[k]['steps_Db'] is not None
                                                 else 'n/a' for k in T.KS))
    pr['PB3'] = (abs(T.steps(res['k1']['sbc'], res['k1']['S-a'])) <= 1 + 1e-9
                 and all(res[k]['steps_Sb'] is not None and abs(res[k]['steps_Sb']) <= 2 + 1e-9 for k in lowk)
                 and res['k32']['steps_Sb'] is not None and res['k32']['steps_Sb'] >= 2 - 1e-9,
                 f"k1 meas vs S-a {T.steps(res['k1']['sbc'], res['k1']['S-a']):+.2f} steps; steps meas - S-b " +
                 ', '.join(f"{res[k]['steps_Sb']:+.2f}" if res[k]['steps_Sb'] is not None else 'n/a' for k in T.KS))
    pr['PB4'] = (all(1.0 <= res[k]['r_Tb'] <= 2.5 for k in T.KS) and res['k1']['r_Tb'] <= 1.6,
                 'table / T-b ' + ', '.join(f"{res[k]['r_Tb']:.2f}" for k in T.KS))
    out['pred_b'] = {k: {'held': bool(v[0]), 'detail': v[1]} for k, v in pr.items()}
    txt.append('Predictions (b):')
    for k, v in pr.items():
        txt.append(f'  {"HELD  " if v[0] else "FAILED"} {k}: {v[1]}')


def main():
    encs = sys.argv[1:] or T.PANEL
    out, txt = {}, ['# B67 verdict (prereg/B67_theory_vs_measurement_prereg.md). Theory from dev keys only; measured = '
                    'test accuracies of B66 / B60 (a) and N* of B38 / B59 / B61 / B63 (b).']
    rows = load_a(encs)
    if rows:
        verdict_a(rows, out, txt)
    verdict_b(out, txt)
    json.dump(out, open(f'{RES}/b67_verdict.json', 'w'), indent=1)
    open(f'{RES}/b67_verdict.txt', 'w').write('\n'.join(txt) + '\n')
    print('\n'.join(txt[:60]))


if __name__ == '__main__':
    main()
