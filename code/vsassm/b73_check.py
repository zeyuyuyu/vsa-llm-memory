"""B73 (equal bits) checks (prereg/B73_confirmatory_prereg.md 'Checks'), a second angle on the verdict.

python3 b73_check.py score OUTDIR [ENC ...]      # plain-loop scorer from the units' RAW files vs results/b73_verdict.json
python3 b73_check.py bits OUTDIR [ENC ...]       # bits audit (own size rules) + frozen-configuration audit
python3 b73_check.py same OUT_A OUT_B            # determinism: every unit file present in OUT_B equal to OUT_A's
Writes results/b73_check_{score,bits,same}.json and prints a summary.
"""
import gzip
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
PANEL = ['qwen8', 'b46', 'llama', 'mistral', 'e5', 'bge', 'minilm']
DIM = {'qwen8': 4096, 'b46': 4096, 'llama': 4096, 'mistral': 4096, 'e5': 768, 'bge': 384, 'minilm': 384}
CELLS = [(M, V, b) for M in (256, 1024) for V in (2, 16) for b in (4, 8, 16)]
REPS = {256: 20, 1024: 5}
HQ = {1: 1.0, 2: 1.9037, 3: 2.7606, 4: 3.6024}  # entropy of Max's quantizer output (as b32_common.H_Q)
CH = 8192
FZ = None


def frozen():
    global FZ
    if FZ is None:
        FZ = json.load(open(os.path.join(ROOT, 'results', 'b73_frozen.json')))['cells']
    return FZ


def raw_preds(out, enc, M, V):
    """{rep: {'qgold': [...], arm|b: [...]}} read from the raw unit files (no b73_run code)."""
    fz = frozen()
    res = {}
    for arm in ('STORE', 'VSAR'):
        fn = os.path.join(out, arm.lower(), enc, f'{arm.lower()}_{enc}_M{M}_V{V}.jsonl.gz')
        if os.path.exists(fn):
            for line in gzip.open(fn, 'rt'):
                r = json.loads(line)
                d = res.setdefault(r['rep'], {'qgold': r['qgold']})
                for b in (4, 8, 16):
                    d[f'{arm}|{b}'] = r['pred'][f'{arm}|{b}|ideal']
    for arm in ('VSAS', 'VSAPC'):
        fn = os.path.join(out, arm.lower(), enc, f'{arm.lower()}_{enc}_M{M}_V{V}.jsonl.gz')
        if os.path.exists(fn):
            for line in gzip.open(fn, 'rt'):
                r = json.loads(line)
                d = res.setdefault(r['rep'], {'qgold': r['qgold']})
                for b in (4, 8, 16):
                    z = fz[f'{enc}|{M}|{V}|{b}|ideal'][arm]
                    fam = z['fam'] if arm == 'VSAS' else z['fam'][1:]
                    d[f'{arm}|{b}'] = r['pred'][f'{fam}|{b}|ideal|{z["cfg"][0]}|{z["cfg"][1]}']
    for arm in ('LAMC', 'NLMSC', 'SDMT'):
        jd = os.path.join(out, arm.lower(), 'jobs')
        if not os.path.isdir(jd):
            continue
        for rep in range(REPS[M]):
            for fn in sorted(os.listdir(jd)):
                ok = (fn.startswith(f'{enc}_m0_test_M{M}_V{V}_r{rep}_') if arm == 'LAMC' else
                      fn.startswith(f'{enc}_test_M{M}_V{V}_r{rep}_') if arm == 'NLMSC' else
                      fn == f'{enc}_test_M{M}_V{V}_r{rep}.json.gz')
                if not ok:
                    continue
                r = json.load(gzip.open(os.path.join(jd, fn), 'rt'))
                d = res.setdefault(rep, {'qgold': r['qgold']})
                for x in r['rows']:
                    d[f'{arm}|{x["bpf"]}'] = x['pred']
    for rep, d in res.items():
        for b in (4, 8, 16):
            src = fz[f'{enc}|{M}|{V}|{b}|ideal']['VSAPCX']
            if f'{src}|{b}' in d:
                d[f'VSAPCX|{b}'] = d[f'{src}|{b}']
    return res


def score(out, encs):
    """Against the per-encoder part files (results/b73_part_{enc}.json, the verdict's input), else the verdict JSON."""
    V = {'regions': {}}
    for enc in encs:
        fn = os.path.join(ROOT, 'results', f'b73_part_{enc}.json')
        if os.path.exists(fn):
            V['regions'][enc] = json.load(open(fn))['regions']
    if not V['regions']:
        V = json.load(open(os.path.join(ROOT, 'results', 'b73_verdict.json')))
    worst, n = 0.0, 0
    rep_ = {}
    for enc in encs:
        if enc not in V['regions']:
            continue
        cell = {}
        for (M, Vv, b) in CELLS:
            rp = raw_preds(out, enc, M, Vv)
            cell[M, Vv, b] = rp
        for key, e in V['regions'][enc].items():
            acct, rn, nm = key.split('|')
            x, y = nm.split('-')
            cs = [c for c in CELLS if rn == 'R1' or c[1] == int(rn[4:])]
            vals = []
            for (M, Vv, b) in cs:
                s, k = 0.0, 0
                for rep in sorted(cell[M, Vv, b]):
                    d = cell[M, Vv, b][rep]
                    g = d['qgold']
                    px, py = d[f'{x}|{b}'], d[f'{y}|{b}']
                    for i in range(len(g)):
                        s += (1.0 if px[i] == g[i] else 0.0) - (1.0 if py[i] == g[i] else 0.0)
                        k += 1
                vals.append(s / k)
            pt = sum(vals) / len(vals)
            worst = max(worst, abs(pt - e['diff']))
            n += 1
        rep_[enc] = len(V['regions'][enc])
    res = {'n_region_statistics': n, 'max_abs_difference': worst, 'per_enc': rep_}
    json.dump(res, open(os.path.join(ROOT, 'results', 'b73_check_score.json'), 'w'), indent=1)
    print('plain-loop scorer:', res)


def lam_bits(p, c, q, scale):
    if p <= 0:
        return 0.0
    ng = c * math.ceil(p / CH) if scale == 'col' else math.ceil(c * p / CH)
    return c * p * HQ[q] + 32 * ng


def vsa_bits(d, q):
    return d * HQ[q] + 32 * math.ceil(d / CH)


def bits(out, encs):
    fz = frozen()
    over = notmax = cfgbad = rows = 0
    notes = []
    for enc in encs:
        n = DIM[enc]
        for (M, V, b) in CELLS:
            z = fz[f'{enc}|{M}|{V}|{b}|ideal']
            B = b * M
            for arm in ('STORE', 'VSAR'):
                fn = os.path.join(out, arm.lower(), enc, f'{arm.lower()}_{enc}_M{M}_V{V}.jsonl.gz')
                if not os.path.exists(fn):
                    continue
                for line in gzip.open(fn, 'rt'):
                    r = json.loads(line)
                    d, q, ns = r['d'][f'{arm}|{b}|ideal']
                    rows += 1
                    if r['cfg'][f'{arm}|{b}|ideal'] != z[arm]:
                        cfgbad += 1
                    if vsa_bits(d, q) > B + 1e-9:
                        over += 1
                    if vsa_bits(d + 1, q) <= B:
                        notmax += 1
                        notes.append(f'{arm} {enc} {M} {V} {b} d={d}')
            for arm in ('VSAS', 'VSAPC'):
                fn = os.path.join(out, arm.lower(), enc, f'{arm.lower()}_{enc}_M{M}_V{V}.jsonl.gz')
                if not os.path.exists(fn):
                    continue
                fam = z[arm]['fam'] if arm == 'VSAS' else z[arm]['fam'][1:]
                k = f'{fam}|{b}|ideal|{z[arm]["cfg"][0]}|{z[arm]["cfg"][1]}'
                for line in gzip.open(fn, 'rt'):
                    r = json.loads(line)
                    rows += 1
                    if k not in r['pred']:
                        cfgbad += 1
                        continue
                    info = r['info'][k]
                    if info[-2:] != z[arm]['bundle'][:2]:
                        cfgbad += 1
                    d, q = info[0], info[1]
                    if vsa_bits(d, q) > B + 1e-9:
                        over += 1
                    if (arm == 'VSAS' or d < n) and vsa_bits(d + 1, q) <= B:
                        notmax += 1
                        notes.append(f'{arm} {enc} {M} {V} {b} d={d}')
            for arm in ('LAMC', 'NLMSC', 'SDMT'):
                jd = os.path.join(out, arm.lower(), 'jobs')
                if not os.path.isdir(jd):
                    continue
                for rep in range(REPS[M]):
                    for fn in sorted(os.listdir(jd)):
                        ok = (fn.startswith(f'{enc}_m0_test_M{M}_V{V}_r{rep}_') if arm == 'LAMC' else
                              fn.startswith(f'{enc}_test_M{M}_V{V}_r{rep}_') if arm == 'NLMSC' else
                              fn == f'{enc}_test_M{M}_V{V}_r{rep}.json.gz')
                        if not ok:
                            continue
                        r = json.load(gzip.open(os.path.join(jd, fn), 'rt'))
                        for x in r['rows']:
                            if x['bpf'] != b:
                                continue
                            rows += 1
                            cfg = x['cfg']
                            want = z[arm]
                            if arm == 'NLMSC':
                                if list(cfg) != list(want):
                                    cfgbad += 1
                            elif list(cfg) != list(want):
                                cfgbad += 1
                            if x['bits'] > B + 1e-9:
                                over += 1
                            if arm in ('LAMC', 'NLMSC'):
                                scale, t, prep, feat, q = cfg[0], cfg[1], cfg[2], cfg[3], cfg[-1]
                                c = V - 1 if t == 'V-1' else V
                                p = x['p']
                                pmax = 65536 if feat == 'rp' else n
                                if abs(lam_bits(p, c, q, scale) - x['bits']) > 1e-6:
                                    notes.append(f'{arm} {enc} {M} {V} {b} bits formula {lam_bits(p, c, q, scale)} vs {x["bits"]}')
                                    over += 1
                                if p < pmax and lam_bits(p + 1, c, q, scale) <= B:
                                    notmax += 1
                                    notes.append(f'{arm} {enc} {M} {V} {b} p={p}')
    res = {'rows': rows, 'over_budget': over, 'not_maximal': notmax, 'config_mismatch': cfgbad, 'notes': notes[:20]}
    json.dump(res, open(os.path.join(ROOT, 'results', 'b73_check_bits.json'), 'w'), indent=1)
    print('bits / frozen-configuration audit:', {k: v for k, v in res.items() if k != 'notes'}, notes[:5])


def same(a, b):
    """Every unit file under OUT_B (relative path) compared with OUT_A's: predictions identical."""
    n = diff = 0
    det = []
    for dp, _, fs in os.walk(b):
        for f in fs:
            if not f.endswith('.json.gz') and not f.endswith('.jsonl.gz'):
                continue
            rel = os.path.relpath(os.path.join(dp, f), b)
            fa = os.path.join(a, rel)
            if not os.path.exists(fa):
                det.append(f'missing in A: {rel}')
                continue

            def preds(fn):
                out = {}
                if fn.endswith('.jsonl.gz'):
                    for line in gzip.open(fn, 'rt'):
                        r = json.loads(line)
                        for k, v in r['pred'].items():
                            out[r['rep'], k] = v
                else:
                    r = json.load(gzip.open(fn, 'rt'))
                    for x in r['rows']:
                        out[str(x['cfg']), x['bpf']] = x['pred']
                return out
            pa, pb = preds(fa), preds(os.path.join(dp, f))
            n += 1
            if pa != pb:
                diff += 1
                det.append(f'differs: {rel}')
    res = {'files_compared': n, 'files_differing': diff, 'details': det[:20]}
    json.dump(res, open(os.path.join(ROOT, 'results', 'b73_check_same.json'), 'w'), indent=1)
    print('determinism:', res)


if __name__ == '__main__':
    a = sys.argv[1:]
    if a[0] == 'score':
        score(a[1], a[2:] or PANEL)
    elif a[0] == 'bits':
        bits(a[1], a[2:] or PANEL)
    elif a[0] == 'same':
        same(a[1], a[2])
    else:
        raise SystemExit(a[0])
