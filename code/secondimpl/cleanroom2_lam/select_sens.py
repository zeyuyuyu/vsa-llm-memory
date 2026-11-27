"""Development selection per condition (best dev accuracy; configs within 0.1 pt tie -> first in grid order)."""
import json, sys, itertools
PREPS = ['centred', 'white']
STORE_GRID = [f'{p}|{m}|{k}|{s}' for p, m, k, s in itertools.product(PREPS, ['lin', 'sign', 'prod3'], [1, 2, 4],
                                                                       [1.0, 0.5, 0.25, 0.125])]
ALPHAS = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]


def ls_grid(V):
    return [f'{p}|{f}|{k}|{C}|{s}|{a}' for p, f, k, C, s, a in itertools.product(
        PREPS, ['random', 'principal'], [1, 2, 4], [V - 1, V], ['percol', 'shared'], ALPHAS)]


def choose(res, grid):
    accs = {g: 100.0 * res[g][0] / res[g][1] for g in grid if g in res}
    best = max(accs.values())
    for g in grid:
        if g in accs and accs[g] >= best - 0.1:
            return g, accs[g], 100.0 * res[g][2] / res[g][3], len(accs)


out = {}
for enc in sys.argv[1:]:
    raw = {'store': {}, 'ls': {}}
    for pp in PREPS:
        part = json.load(open(f'raw_{enc}_{pp}.json' if pp == 'centred' else f'raw_{enc}_whiteE01.json'))
        for mem in ['store', 'ls']:
            for cond, d in part[mem].items():
                raw[mem].setdefault(cond, {}).update(d)
    cells = {}
    for cond in raw['store']:
        M, V, b = cond.split('|')
        gs, ds, ts, ns = choose(raw['store'][cond], STORE_GRID)
        gl, dl, tl, nl = choose(raw['ls'][cond], ls_grid(int(V)))
        cells[f'{enc}|ideal|{M}|{V}|{b}'] = dict(STORE=ts, LAMC=tl, store_cfg=gs, lamc_cfg=gl, store_dev=ds,
                                                 lamc_dev=dl, n_store_cfgs=ns, n_lamc_cfgs=nl,
                                                 n_test_queries=raw['store'][cond][gs][3])
    reg = {}
    for name, f in [('V2', lambda c: c.split('|')[3] == '2'), ('V16', lambda c: c.split('|')[3] == '16'),
                    ('R1', lambda c: True)]:
        ks = [c for c in cells if f(c)]
        reg[name] = sum(cells[c]['STORE'] - cells[c]['LAMC'] for c in ks) / len(ks)
    out[enc] = dict(cells=cells, region_store_minus_lamc=reg)
    print(enc, {k: round(v, 2) for k, v in reg.items()})
    for c in sorted(cells, key=lambda c: (int(c.split('|')[3]), int(c.split('|')[2]), int(c.split('|')[4]))):
        x = cells[c]
        print(f"  {c:28s} store {x['STORE']:6.2f} lamc {x['LAMC']:6.2f} diff {x['STORE']-x['LAMC']:+6.2f}  "
              f"[{x['store_cfg']}] [{x['lamc_cfg']}] nq={x['n_test_queries']}")
json.dump(out, open('my_cells_sens_eps01.json', 'w'), indent=1)
print('wrote my_cells.json')
