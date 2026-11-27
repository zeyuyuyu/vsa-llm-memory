"""B46 diagnostics of the codes (prereg/B46, 'Encoder' diagnostic; dev split and training split only; no memory):
- largest |value| per dimension (raw, centred, whitened): outlier / massive-activation dimensions (Sun et al. 2024);
- B36's pre-design measurement, on the dev split: cosine between an alias's code and its label's code (centred,
  whitened), between random pairs of labels, and the chance that all k SimHash bits of an alias equal its label's.
Runs on any encoder's codes for a side-by-side with Qwen3-8B and e5.

python3 b46_diag.py TAG [TAG ...]      # TAG = file tag under b32_data/ (emb_TAG.npy); prints and appends to OUT
"""
import json
import sys

import numpy as np

sys.path.insert(0, '/path/to/vsassm')
import b32_common as C  # noqa: E402


def main():
    out = []
    for tag in sys.argv[1:]:
        pool = C.Pool(tag)
        Tr = pool.train_matrix()
        dv = pool.by_split['dev']
        lab = np.array([pool.idx[e['name']] for e in dv])
        pairs = [(pool.idx[a], j) for j, e in enumerate(dv) for a in e['aliases']]
        qa = np.array([p for p, _ in pairs])
        ql = lab[[j for _, j in pairs]]
        X = pool.X
        out.append(f'## {tag}: {X.shape}; dev labels {len(lab)}, dev aliases {len(qa)}')
        for kind in ('none', 'center', 'whiten'):
            P = C.Prep(kind, Tr)
            Y = P(X) if kind != 'none' else X / np.linalg.norm(X, axis=1, keepdims=True)
            mx = np.abs(Y).max(0)
            top = np.argsort(-mx)[:3]
            share = (np.abs(Y).max(1) > 0.5).mean()
            cs = (Y[qa] * Y[ql]).sum(1)
            rng = np.random.default_rng(0)
            i, j = rng.integers(0, len(lab), 20000), rng.integers(0, len(lab), 20000)
            keep = i != j
            cr = (Y[lab[i[keep]]] * Y[lab[j[keep]]]).sum(1)
            out.append(f'  {kind:6s}: max|value| per dim: median {np.median(mx):.3f}, top-3 dims {list(top)} '
                       f'({", ".join(f"{mx[t]:.3f}" for t in top)}); rows with a coordinate > 0.5: {share:.3f}; '
                       f'alias-label cosine median {np.median(cs):.3f} (10th pct {np.quantile(cs, 0.1):.3f}); '
                       f'random label pairs median {np.median(cr):.3f} (99th pct {np.quantile(cr, 0.99):.3f})')
            if kind != 'none':
                G = np.random.default_rng(46).standard_normal((Y.shape[1], 16)).astype(np.float32)
                sa, sl = (Y[qa] @ G) >= 0, (Y[ql] @ G) >= 0
                eq = sa == sl
                pk = {k: float(eq[:, :k].all(1).mean()) for k in (2, 4, 6, 8, 10, 16)}
                out.append(f'          P(all k SimHash bits equal), k = 2/4/6/8/10/16: ' +
                           ' / '.join(f'{v:.2f}' for v in pk.values()))
    txt = '\n'.join(out)
    print(txt)
    return txt


if __name__ == '__main__':
    main()
