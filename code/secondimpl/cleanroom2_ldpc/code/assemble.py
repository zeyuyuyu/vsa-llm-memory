import json, glob, os, statistics
O = '../out'
GRID = [8 * 2 ** (j / 4) for j in range(37)]
def g(x): return None if x is None else round(x, 1)
L = json.load(open(f'{O}/ldpc_requirements.json'))
B = json.load(open(f'{O}/bch_k8.json'))
# BCH requirement per variant: seed 0 is the primary realization; also "all 3 seeds pass"
bch = {}
for vn in sorted(set(r['variant'] for r in B['results'])):
    rows = sorted([r for r in B['results'] if r['variant'] == vn and r['feasible']], key=lambda r: (r['B'], r['seed']))
    Bs = sorted(set(r['B'] for r in rows))
    def req(pred):
        for i, b in enumerate(Bs):
            if all(pred([r for r in rows if r['B'] == bb]) for bb in Bs[i:]):
                return b
    bch[vn] = dict(req_seed0=g(req(lambda rr: [r for r in rr if r['seed'] == 0][0]['share'] >= 0.995)),
                   req_all_seeds=g(req(lambda rr: all(r['share'] >= 0.995 for r in rr))),
                   shares={str(g(b)): [round(r['share'], 4) for r in rows if r['B'] == b] for b in Bs})
extra = {}
for fn in ['bch_k8_naive_dim.json', 'bch_k8_directory_pad.json', 'bch_k8_extra_seeds.json']:
    if os.path.exists(f'{O}/{fn}'): extra[fn] = json.load(open(f'{O}/{fn}'))
seeds = {os.path.basename(f): json.load(open(f)) for f in glob.glob(f'{O}/seeds_*.json')}
paper = {'LDPC': {'k4': 27, 'k8': 54, 'k16': 215, 'k32': 724}, 'BCH': {'k8': 108}}
my = {'grid': [round(x, 1) for x in GRID], 'cell_error_rates': {'k4': 0.021, 'k8': 0.123, 'k16': 0.276, 'k32': 0.374},
      'LDPC_cells_per_pair': {c: (g(r['req_dev_rule']) if r else None) for c, r in L.items()},
      'LDPC_cells_per_pair_strict_dev_argmax': {c: (g(r['req_strict_dev']) if r else None) for c, r in L.items()},
      'LDPC_cells_per_pair_best_test_rate': {c: (g(r['req_oracle']) if r else None) for c, r in L.items()},
      'LDPC_budget_window_evaluated': {c: ([g(x['B']) for x in r['rows']] if r else None) for c, r in L.items()},
      'LDPC_rows': {c: (r['rows'] if r else None) for c, r in L.items()},
      'BCH_k8': bch, 'BCH_k8_variants_extra': extra, 'LDPC_repeat_realizations': seeds,
      }
json.dump(my, open('../my_cells.json', 'w'), indent=1)
print(json.dumps({k: my[k] for k in ['LDPC_cells_per_pair', 'LDPC_cells_per_pair_strict_dev_argmax', 'LDPC_cells_per_pair_best_test_rate']}, indent=1))
print({vn: (v['req_seed0'], v['req_all_seeds']) for vn, v in bch.items()})
