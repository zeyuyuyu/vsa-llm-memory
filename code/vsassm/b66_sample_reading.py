"""B66 sample reading by the AI-assisted process (prereg 'Checks'): (A1) the Qwen3-8B cell where the two codes' SDMt on
B64's grid differ most; (B) Qwen3-8B and e5, the R1 cell with the lowest STORE - LSFR. Writes
results/b66_sample_reading_raw.txt (the notes are written by hand into results/b66_sample_reading.txt).
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
OUT = os.environ.get('B66_OUT', f'{ROOT}/b66_out')
RES = os.environ.get('B66_RES', f'{ROOT}/results')
L = []


def say(s=''):
    print(s)
    L.append(s)


def a1():
    A1 = json.load(open(f'{RES}/b66_A1.json'))['qwen8']
    P, Cc = A1['pipeline'], A1['clean']
    cells = []
    for c, v in Cc['cells'].items():
        M, V, b = map(int, c.split('|'))
        cells.append((abs(v['sdmT64'] - 100 * P['cells'][f'ideal|{M}|{V}|{b}']['SDMT']), M, V, b))
    cells.sort(reverse=True)
    _, M, V, b = cells[0]
    ch64 = json.load(open(f'{RES}/b64_choice_qwen8.json'))['choice'][f'{M}|{V}|{b}|ideal|SDMT']
    Dd = json.load(open(f'{OUT}/cr/out_qwen8/dev_M{M}_V{V}_b{b}.json'))
    say(f'(A1) Qwen3-8B, cell M={M} V={V} b={b}: SDMt on B64\'s grid, clean-room code {Cc["cells"][f"{M}|{V}|{b}"]["sdmT64"]:.2f} '
        f'vs pipeline code {100 * P["cells"][f"ideal|{M}|{V}|{b}"]["SDMT"]:.2f}')
    say(f'  pipeline dev choice {ch64["cfg"]} (dev {100 * ch64["dev_acc"]:.2f}); clean-room dev choice '
        f'{Dd["choice"]["sdmT64"]} (dev {Dd["choice_dev"]["sdmT64"]:.2f}); clean-room first-design dev {Dd["store_dev"]["first"]:.2f}')
    # the clean room's dev accuracy of the pipeline's chosen configuration, and vice versa
    pc = ch64['cfg']
    key = '|'.join(map(str, ['sdmt', 'centre' if pc[0] == 'center' else pc[0]] + pc[1:]))
    say(f'  clean-room dev accuracy of the pipeline\'s chosen configuration: {Dd["devacc_all"].get(key)}')
    say('  (the two codes use different stores of the same test entities; queries are paired by entity and alias below)')
    # pair queries by (entity label, alias) across codes
    import b32_common as C
    ents = json.load(open(f'{ROOT}/b32_data/entities.json'))['entities']
    test = [e for e in ents if e['split'] == 'test']
    z = np.load(f'{OUT}/cr/out_qwen8/corr_M{M}_V{V}_b{b}.npz')
    import b66_cr as CR
    CR.G['split'] = None
    # rebuild the clean room's test stores' query strings (same seeds as b66_cr.cond_setup)
    strings = json.load(open(f'{ROOT}/b32_data/emb_strings.json'))
    idx = {s: i for i, s in enumerate(strings)}
    split = {'train': [], 'dev': [], 'test': []}
    for e in ents:
        split[e['split']].append((idx[e['name']], [idx[a] for a in e['aliases']]))
    CR.G['split'] = split
    cs = M * 1000 + V * 10 + b
    tst = CR.make_stores('test', M, V, 20 if M == 256 else 5, 2000 + cs)
    crq = []
    for s in tst:
        for a, e in zip(s['q'], s['qe']):
            crq.append((strings[split['test'][e][0]], strings[a]))
    crok = dict()
    for (lab_, al), ok in zip(crq, z['sdmT64']):
        crok.setdefault((lab_, al), []).append(bool(ok))
    import b66_verdict as V66
    recs = V66.merged('qwen8', M, V)
    import b64_verdict as V64
    jr = V64.job_rows('qwen8', M, V)
    uri2name = {e['uri'] if 'uri' in e else e.get('id'): e['name'] for e in ents}
    pok = {}
    for r in recs:
        rows, _ = jr[r['rep']], None
        pred = jr[r['rep']][(tuple(ch64['cfg']), b, 'ideal')][0]
        for i, (qe, al, g) in enumerate(zip(r['qent'], r['qalias'], r['qgold'])):
            nm = uri2name.get(r['uris'][qe], r['uris'][qe])
            pok.setdefault((nm, al), []).append(pred[i] == g)
    common = sorted(set(crok) & set(pok))
    only_cr = [k for k in common if all(crok[k]) and not any(pok[k])]
    only_p = [k for k in common if all(pok[k]) and not any(crok[k])]
    say(f'  queries (label, alias) in both codes\' test stores: {len(common)}; right only in the clean-room code: '
        f'{len(only_cr)}; right only in the pipeline code: {len(only_p)}')
    rng = np.random.default_rng(66)
    for nm, lst in (('clean-room only', only_cr), ('pipeline only', only_p)):
        say(f'  {nm} (10 sampled):')
        for i in rng.permutation(len(lst))[:10]:
            say(f'    label "{lst[i][0]}"  alias "{lst[i][1]}"')
    _ = C


def b():
    import b66_lsfc as LS
    for enc in ('qwen8', 'e5'):
        data, ch = LS.load(enc)
        best = None
        for M in (256, 1024):
            for V in (2, 16):
                for bb in (4, 8, 16):
                    s = np.mean(np.concatenate([np.array(r['pred'][f'STORE|{bb}|ideal']) == np.array(r['qgold']) for r in data[M, V]])) - \
                        np.mean(np.concatenate([np.array(r['pred'][f'LSFR|{bb}|ideal']) == np.array(r['qgold']) for r in data[M, V]]))
                    if best is None or s < best[0]:
                        best = (s, M, V, bb)
        s, M, V, bb = best
        say(f'(B) {enc}: lowest STORE - LSFR cell M={M} V={V} b={bb}: {100 * s:+.2f} points; LSFR cfg '
            f'{ch[f"{M}|{V}|{bb}|ideal|LSFR"]["cfg"]}')
        rows = []
        for r in data[M, V]:
            for i, (al, g, qe) in enumerate(zip(r['qalias'], r['qgold'], r['qent'])):
                ps, pl = r['pred'][f'STORE|{bb}|ideal'][i], r['pred'][f'LSFR|{bb}|ideal'][i]
                rows.append((ps == g, pl == g, al, r['uris'][qe], g, ps, pl, r['rep']))
        rng = np.random.default_rng(67)
        for nm, sel in (('only LSFR right', [x for x in rows if x[1] and not x[0]]),
                        ('only the store right', [x for x in rows if x[0] and not x[1]])):
            say(f'  {nm}: {len(sel)} queries; 10 sampled (alias | entity | gold | store | LSFR | store id):')
            for i in rng.permutation(len(sel))[:10]:
                x = sel[i]
                say(f'    "{x[2]}" | {x[3].rsplit("/", 1)[-1]} | {x[4]} | {x[5]} | {x[6]} | {x[7]}')


if __name__ == '__main__':
    what = sys.argv[1:] or ['a1', 'b']
    if 'a1' in what:
        a1()
    if 'b' in what:
        b()
    open(f'{RES}/b66_sample_reading_raw.txt', 'w').write('\n'.join(L) + '\n')
