import json, sys, glob, os
O = '../out'
RATES = [(1, 4), (1, 3), (2, 5), (1, 2), (3, 5), (2, 3), (3, 4), (4, 5), (5, 6), (8, 9), (9, 10)]
def req(cond):
    fn = f'{O}/ldpc_{cond}.json'
    if not os.path.exists(fn): return None
    d = json.load(open(fn))
    byB = {}
    for r in d['results']:
        byB.setdefault(r['B'], []).append(r)
    rows = []
    for B in sorted(byB):
        cands = [r for r in byB[B] if not r['pruned'] and r.get('dev') and r.get('test')]
        if not cands:
            rows.append(dict(B=B, dev_rate=None, test_share=0.0, oracle=0.0, strict_test=0.0)); continue
        best_dev = max(c['dev']['share'] for c in cands)
        tied = [c for c in cands if c['dev']['share'] >= best_dev - 0.001]
        tied.sort(key=lambda c: RATES.index(tuple(c['rate'])))
        ch = tied[0]
        strict = max(cands, key=lambda c: (c['dev']['share'], -RATES.index(tuple(c['rate']))))
        orc = max(cands, key=lambda c: c['test']['share'])
        rows.append(dict(B=round(B, 2), dev_rate=ch['rate'], rho=ch['rho'], dev_share=ch['dev']['share'], test_share=ch['test']['share'],
                         oracle=orc['test']['share'], oracle_rate=orc['rate'], oracle_rho=orc['rho'],
                         strict_rate=strict['rate'], strict_rho=strict['rho'], strict_test=strict['test']['share']))
    def first(key):
        Bs = [r['B'] for r in rows]
        for i, r in enumerate(rows):
            if all(rr[key] >= 0.995 for rr in rows[i:]):
                return r['B']
        return None
    return dict(cond=cond, p=d['p'], rows=rows, req_dev_rule=first('test_share'), req_oracle=first('oracle'), req_strict_dev=first('strict_test'),
                max_budget_evaluated=rows[-1]['B'] if rows else None)
res = {c: req(c) for c in ['k4', 'k8', 'k16', 'k32']}
for c, r in res.items():
    if r is None: continue
    print(c, 'p', r['p'], 'req (dev-chosen rate)', r['req_dev_rule'], 'req (best test rate)', r['req_oracle'], 'req (strict dev argmax)', r['req_strict_dev'], 'evaluated up to', r['max_budget_evaluated'])
    for row in r['rows']: print('   ', row)
json.dump(res, open(f'{O}/ldpc_requirements.json', 'w'), indent=1)
