"""V5 verdict: GPT's re-implementation (v5_out/a_out, v5_out/b_out) against the pipeline (B60 / B62; B59 / B61 / B63),
with the agreement criteria of prereg/V5_gpt_reimplementation_prereg.md. Writes results/v5_verdict.txt and .json.

python3 v5_verdict.py
"""
import json
import math
import os

OUT_TXT, OUT_JSON = 'results/v5_verdict.txt', 'results/v5_verdict.json'
ENCS = {'qwen8': 'Qwen3-8B', 'llama': 'Llama-3.1-8B'}


def step(b):
    """Grid index of a budget on 2^(i/4); '> 4096' (None) -> one step beyond 4,096."""
    if b is None:
        return 49
    return int(round(4 * math.log2(b)))


# ---------------------------------------------------------------- GPT side

def gpt_a(enc):
    f = f'v5_out/a_out/{enc}_result.json'
    if not os.path.exists(f):
        return None
    d = json.load(open(f))
    out = {}
    for pair in ('STORE - LAMc', 'STORE - NLMSc', 'NLMSc - LAMc'):
        c = d['comparisons'][pair]
        out[pair] = {h: (100 * c[h]['mean_accuracy_difference'], [100 * x for x in c[h]['ci95']])
                     for h in ('region', 'V2', 'V16')}
    out['cells'] = {(c['M'], c['V'], c['b']): {m: 100 * c['memories'][m]['test_accuracy'] for m in
                                                ('STORE', 'LAMc', 'NLMSc')} for c in d['conditions']}
    out['configs'] = {(c['M'], c['V'], c['b']): {m: c['memories'][m]['configuration'] for m in
                                                  ('STORE', 'LAMc', 'NLMSc')} for c in d['conditions']}
    return out


def gpt_b():
    """{memory: {k: requirement or None}} from the per-memory result files; cuckoo = better of O-BCH and O-LDPC."""
    req, extra = {}, {}
    for mem, sub in (('dense', ['dense']), ('sbc', ['sbc']), ('bch', ['bch']), ('cuckoo', ['O-BCH', 'O-LDPC'])):
        f = f'v5_out/b_out/{mem}_result.json'
        if not os.path.exists(f):
            f = f'v5_out/b_out/{mem}_result.partial.json'   # the script's own partial file (finished (memory, k) only)
            if not os.path.exists(f):
                continue
        d = json.load(open(f))
        extra['device'] = d.get('device')
        for s in sub:
            if s not in d['memories']:
                continue
            for k, r in d['memories'][s].items():
                kk = int(float(k))
                if 'test_requirement' not in r:
                    continue                                   # this k's test window did not finish before the cut
                tr = r['test_requirement']
                req.setdefault(s, {})[kk] = None if isinstance(tr, str) else tr
    if 'O-BCH' in req or 'O-LDPC' in req:
        ks = set(req.get('O-BCH', {})) | set(req.get('O-LDPC', {}))
        req['cuckoo'] = {}
        for k in ks:
            # the better of O-BCH and O-LDPC is defined only where both finished
            if all(s in req and k in req[s] for s in ('O-BCH', 'O-LDPC')):
                have = [req[s][k] for s in ('O-BCH', 'O-LDPC') if req[s][k] is not None]
                req['cuckoo'][k] = min(have) if have else None
    return req, extra


# ---------------------------------------------------------------- pipeline side

def _row(path, section, name):
    """The '| name | V2 [..] | label | V16 [..] | label | R1 [..] | label |' row of a section of a verdict file."""
    import re
    lines = open(path).read().split('\n')
    i = next(j for j, l in enumerate(lines) if l.startswith('## ') and section in l)
    for l in lines[i + 1:]:
        if l.startswith('## '):
            break
        if l.startswith('| ' + name + ' |'):
            nums = [float(x) for x in re.findall(r'[-+]?\d+\.\d+', l[len('| ' + name + ' |'):])]
            return {'V2': nums[0], 'V16': nums[3], 'region': nums[6]}
    raise KeyError(name)


def pipe_a(enc):
    name = ENCS[enc]
    return {'STORE - LAMc': _row('results/b60_verdict.txt', 'PRIMARY (ideal): STORE - LAMc', name),
            'STORE - NLMSc': _row('results/b62_verdict.txt', 'STORE - NLMSC (ideal) [PRIMARY]', name),
            'NLMSc - LAMc': _row('results/b62_verdict.txt', 'NLMSC - LAMC (ideal)', name)}


def part_a():
    rows, met_all, have = [], True, []
    for enc in ENCS:
        g = gpt_a(enc)
        if g is None:
            continue
        have.append(enc)
        p = pipe_a(enc)
        for pair in ('STORE - LAMc', 'STORE - NLMSc', 'NLMSc - LAMc'):
            gv, ci = g[pair]['region']
            pv = p[pair]['region']
            crit = pair != 'NLMSc - LAMc'
            same_sign = (gv > 0) == (pv > 0)
            within = abs(gv - pv) <= 1.5
            met = same_sign and within
            if crit:
                met_all &= met
            rows.append(dict(enc=enc, pair=pair, v5=gv, v5_ci=ci, pipe=pv, diff=gv - pv, same_sign=same_sign,
                             within=within, met=met if crit else None,
                             v5_V2=g[pair]['V2'][0], pipe_V2=p[pair]['V2'], v5_V16=g[pair]['V16'][0],
                             pipe_V16=p[pair]['V16']))
    return rows, met_all, have


def _table1(path, arm):
    """Row of 'Table 1' (columns k1 k2 k4 e05 k8 k16 k32) -> {1: N*, 8: N*, 16: N*}; '>4096' -> None."""
    for l in open(path):
        parts = l.split()
        if parts and parts[0] == arm and len(parts) == 8:
            vals = [None if v.startswith('>') else float(v.replace(',', '')) for v in parts[1:]]
            return {1: vals[0], 8: vals[4], 16: vals[5]}
    raise KeyError((path, arm))


def pipe_b():
    p = {'dense': _table1('results/b59_store_verdict.txt', 'U'),
         'bch': _table1('results/b59_store_verdict.txt', 'E-BCH'),
         'sbc': _table1('results/b61_verdict.txt', 'H-SBC'),
         'O-BCH': _table1('results/b63_verdict.txt', 'O-BCH'),
         'O-LDPC': _table1('results/b63_verdict.txt', 'O-LDPC')}
    p['cuckoo'] = {k: min(x for x in (p['O-BCH'][k], p['O-LDPC'][k]) if x is not None) for k in (1, 8, 16)}
    return p


MEMS = [('dense', 'dense bipolar store (last-value read)'), ('sbc', 'sparse-block store (H-SBC)'),
        ('bch', 'BCH static table (E-BCH)'), ('cuckoo', 'cuckoo insertable table (better of O-BCH, O-LDPC)')]


def part_b():
    g, extra = gpt_b()
    p = pipe_b()
    rows, met_all = [], True
    for m, _ in MEMS:
        for k in (1, 8, 16):
            gv = g.get(m, {}).get(k, 'missing')
            pv = p[m][k]
            if gv == 'missing':
                rows.append(dict(mem=m, k=k, v5=None, pipe=pv, steps=None, met=False, note='no final V5 value'))
                met_all = False
                continue
            d = step(gv) - step(pv)
            met = abs(d) <= 1
            met_all &= met
            rows.append(dict(mem=m, k=k, v5=gv, pipe=pv, steps=d, met=met, note=''))
    sec = []
    for k in (1, 8, 16):
        gv = g.get('O-BCH', {}).get(k)
        sec.append(dict(mem='O-BCH', k=k, v5=gv, pipe=p['O-BCH'][k], steps=(step(gv) - step(p['O-BCH'][k])) if gv else None))
    order = []
    for k in (1, 8, 16):
        have = [m for m, _ in MEMS if g.get(m, {}).get(k, 'missing') != 'missing']
        rev, strict_same = [], True
        for i in range(len(have)):
            for j in range(i + 1, len(have)):
                a, b = have[i], have[j]
                sg = step(g[a][k]) - step(g[b][k])
                sp = step(p[a][k]) - step(p[b][k])
                if sg * sp < 0:
                    rev.append((a, b))
                if (sg > 0) - (sg < 0) != (sp > 0) - (sp < 0):
                    strict_same = False
        all_four = len(have) == len(MEMS)
        order.append(dict(k=k, compared=have, reversals=rev, strict_same=strict_same, all_four=all_four,
                          met=(not rev) and all_four))
        met_all &= (not rev) and all_four
    return rows, sec, order, met_all, extra


def fmt(x):
    return '> 4096' if x is None else ('%g' % round(x)) if x >= 10 else '%.1f' % x


def main():
    ra, met_a, have_a = part_a()
    rb, sec, order, met_b, extra = part_b()
    L = []
    L.append('# V5 verdict (prereg/V5_gpt_reimplementation_prereg.md): code written by gpt-6.1-sol (OpenAI) via the')
    L.append('# api_zheng gateway from v5_out/spec_packet.md; pipeline numbers from results/b60_verdict.txt, b62_verdict.txt,')
    L.append('# b59_store_verdict.txt, b61_verdict.txt, b63_verdict.txt, read only after the V5 outputs were committed.')
    L.append('')
    L.append('## (a) Equal bits, R1 (12 conditions), ideal accounting; region mean of per-condition test-accuracy differences, points')
    L.append('Criterion: same sign and |V5 - pipeline| <= 1.5 points, for STORE - LAMc and STORE - NLMSc on each encoder.')
    L.append('| encoder | difference | V5 [95% CI] | pipeline | V5 - pipeline | criterion | V5 V=2 / pipe | V5 V=16 / pipe |')
    L.append('| --- | --- | --- | --- | --- | --- | --- | --- |')
    for r in ra:
        crit = 'n/a (secondary)' if r['met'] is None else ('MET' if r['met'] else 'NOT MET')
        L.append('| %s | %s | %+.2f [%+.2f, %+.2f] | %+.2f | %+.2f | %s | %+.2f / %+.2f | %+.2f / %+.2f |' % (
            ENCS[r['enc']], r['pair'], r['v5'], r['v5_ci'][0], r['v5_ci'][1], r['pipe'], r['diff'], crit,
            r['v5_V2'], r['pipe_V2'], r['v5_V16'], r['pipe_V16']))
    L.append('(a) overall: %s (encoders in scope: %s)' % ('MET' if met_a else 'NOT MET', ', '.join(ENCS[e] for e in have_a)))
    L.append('')
    L.append('## (b) Equal cells: cells per stored pair for >= 99.5% of the 4,776 probed keys exact, i.i.d. cell errors at the 30-day rate')
    dv = extra.get('device') or {}
    L.append('V5 device-model cell-error rates (10^7 cells): ' + ', '.join('k=%g: %.6f' % (float(k), v['p']) for k, v in sorted(dv.items(), key=lambda x: float(x[0]))))
    L.append('Pipeline cell-error rates (b34_out/ber.json, 30 days): k=1: 0, k=8: 0.122635, k=16: 0.276006')
    L.append('Criterion: |grid index V5 - pipeline| <= 1 (one 19% step) for every memory and k; no reversed pair in the ordering.')
    L.append('| memory | k | V5 | pipeline | steps | criterion |')
    L.append('| --- | --- | --- | --- | --- | --- |')
    for r in rb:
        L.append('| %s | %d | %s | %s | %s | %s |' % (r['mem'], r['k'], 'no final value' if r['v5'] is None else fmt(r['v5']),
                 fmt(r['pipe']), '' if r['steps'] is None else '%+d' % r['steps'], 'MET' if r['met'] else 'NOT MET'))
    L.append('Secondary (not the criterion memory): the cuckoo table with BCH only (O-BCH)')
    for r in sec:
        L.append('| O-BCH | %d | %s | %s | %s |' % (r['k'], fmt(r['v5']) if r['v5'] else 'n/a', fmt(r['pipe']), '' if r['steps'] is None else '%+d' % r['steps']))
    L.append('Ordering checks (memories with a final V5 value):')
    for o in order:
        L.append('  k=%d: compared %s; reversed pairs %s; strict identity %s; all four memories present %s -> %s' % (
            o['k'], o['compared'], o['reversals'] or 'none', o['strict_same'], o['all_four'], 'MET' if o['met'] else 'NOT MET'))
    L.append('(b) overall: %s' % ('MET' if met_b else 'NOT MET'))
    open(OUT_TXT, 'w').write('\n'.join(L) + '\n')
    json.dump({'a': ra, 'a_met': met_a, 'b': rb, 'b_secondary': sec, 'b_order': order, 'b_met': met_b,
               'device': dv}, open(OUT_JSON, 'w'), indent=1, default=str)
    print('\n'.join(L))


if __name__ == '__main__':
    main()
