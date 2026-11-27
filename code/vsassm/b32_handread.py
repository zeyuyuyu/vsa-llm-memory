"""B32 hand reading: queries on which the two memories disagree, with the alias, the stored label and the label that
the exact-cosine index finds nearest (prereg/B32_vector_form_equal_bits_prereg.md, 'Checks').

python3 b32_handread.py OUTDIR ENC M V BPF ACCT [N]
"""
import gzip
import json
import random
import sys


def main():
    outdir, enc, M, V, b, acct = sys.argv[1:7]
    N = int(sys.argv[7]) if len(sys.argv) > 7 else 10
    E = {e['uri']: e for e in json.load(open('b32_data/entities.json'))['entities']}
    rows = {'VSA only': [], 'IDX only': []}
    for l in gzip.open(f'{outdir}/preds_{enc}_M{M}_V{V}.jsonl.gz', 'rt'):
        r = json.loads(l)
        pv, px = r['pred'][f'VSA|{b}|{acct}'], r['pred'][f'IDX|{b}|{acct}']
        for i, g in enumerate(r['qgold']):
            if (pv[i] == g) == (px[i] == g):
                continue
            nn = r['pred']['EXACT_ENT'][i]
            row = (r['rep'], r['qalias'][i], E[r['uris'][r['qent'][i]]]['name'], E[r['uris'][nn]]['name'],
                   nn == r['qent'][i])
            rows['VSA only' if pv[i] == g else 'IDX only'].append(row)
    rnd = random.Random(7)
    for k, lst in rows.items():
        print(f'## right only for {k.split()[0]}: {len(lst)} queries; {min(N, len(lst))} drawn at random')
        print(f'   share whose entity the exact-cosine index finds: {sum(x[4] for x in lst) / max(1, len(lst)):.3f}')
        for rep, a, name, nn, ok in rnd.sample(lst, min(N, len(lst))):
            print(f'   rep {rep}: alias {a!r} -> label {name!r}; exact nearest {nn!r} ({"right" if ok else "wrong"})')


if __name__ == '__main__':
    main()
