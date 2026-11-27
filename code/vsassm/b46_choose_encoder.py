"""B46 encoder choice (prereg/B46_second_llm_family_prereg.md, 'Encoder'): reads the criterion of every candidate
variant (b32_data/lm_select_{tag}.json, written by B32's b32_lm_select.py, unchanged), applies the registered rule
(highest dev entity retrieval; ties within 0.001 to the earlier model, then mean before eol, then the shallower layer),
stores the chosen codes as float16 (b32_data/emb_b46_{tag}_{variant}_f16.npy, as B32's Qwen3-8B codes) and writes
b46_encoder_choice.json. Run in the B46 box directory after b32_lm_select.py.

python b46_choose_encoder.py TAG1,TAG2,...        # tags in the registered order
"""
import hashlib
import json
import sys

import numpy as np


def main():
    tags = sys.argv[1].split(',')
    rows = []
    for ti, t in enumerate(tags):
        try:
            d = json.load(open(f'b32_data/lm_select_{t}.json'))
        except FileNotFoundError:
            print(f'{t}: no criterion file (candidate dropped)')
            continue
        for v, s in d.items():
            pool, layer = v.split('_L')
            rows.append({'tag': t, 'variant': v, 'score': s, 'rank': (ti, ['mean', 'eol'].index(pool), int(layer))})
    top = max(r['score'] for r in rows)
    tied = sorted([r for r in rows if r['score'] >= top - 0.001], key=lambda r: r['rank'])
    ch = tied[0]
    X = np.load(f'b32_data/emb_{ch["tag"]}_{ch["variant"]}.npy')
    fn = f'emb_b46_{ch["tag"]}_{ch["variant"]}_f16'
    np.save(f'b32_data/{fn}.npy', X.astype(np.float16))
    md5 = hashlib.md5(open(f'b32_data/{fn}.npy', 'rb').read()).hexdigest()
    out = {'chosen': {'tag': ch['tag'], 'variant': ch['variant'], 'score_float32': ch['score'], 'file_tag': fn[4:],
                      'md5_f16': md5, 'n_tied_within_0.001': len(tied)},
           'all': sorted([{k: r[k] for k in ('tag', 'variant', 'score')} for r in rows], key=lambda r: -r['score'])}
    json.dump(out, open('b46_encoder_choice.json', 'w'), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
