"""B32: choose the LM reader's pooling and layer by the registered encoder criterion (dev split only):
exact-cosine entity retrieval over the whole dev pool (every dev label stored, every dev alias a query), keys and
queries centred on the training mean. No value, no memory arm, no test string is scored.

python3 b32_lm_select.py TAG [variants...]     # prints one line per variant, writes b32_data/lm_select_TAG.json
"""
import json
import sys

import numpy as np


def score(X, ents, idx):
    tr = [e for e in ents if e['split'] == 'train']
    dv = [e for e in ents if e['split'] == 'dev']
    rows = [idx[e['name']] for e in tr] + [idx[a] for e in tr for a in e['aliases']]
    mu = X[rows].mean(0)
    Y = X - mu
    Y /= np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
    K = Y[[idx[e['name']] for e in dv]]
    qs = [(idx[a], j) for j, e in enumerate(dv) for a in e['aliases']]
    Q = Y[[q for q, _ in qs]]
    g = np.array([j for _, j in qs])
    return float(((Q @ K.T).argmax(1) == g).mean()), len(dv), len(g)


def main():
    tag, variants = sys.argv[1], sys.argv[2:]
    ents = json.load(open('b32_data/entities.json'))['entities']
    S = json.load(open('b32_data/emb_strings.json'))
    idx = {s: i for i, s in enumerate(S)}
    out = {}
    for v in variants:
        X = np.load(f'b32_data/emb_{tag}_{v}.npy' if v != '-' else f'b32_data/emb_{tag}.npy').astype(np.float32)
        acc, nk, nq = score(X, ents, idx)
        out[v] = acc
        print(f'{tag} {v}: dev entity retrieval {acc:.4f} (stored {nk}, queries {nq})', flush=True)
    json.dump(out, open(f'b32_data/lm_select_{tag}.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
