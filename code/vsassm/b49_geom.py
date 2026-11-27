"""B49 geometry of the codes (prereg/B49_panel_mechanism_competitors_prereg.md, 'Mechanism'): descriptive measures of
each encoder's codes on the training split (spectrum) and the dev split (alias-label pairs); no memory, no value, no
test string.

python3 b49_geom.py OUT.json TAG[=FILETAG] ...     # FILETAG under b32_data/ (emb_FILETAG.npy); TAG = name in the output

Per encoder and preprocessing (B32's Prep: 'center', 'whiten' = centre + ZCA, eps 0.1):
  raw_mean_cos     mean cosine between random pairs of training codes before centring (common-direction anisotropy)
  pr, pr_frac      participation ratio (sum l)^2 / sum l^2 of the covariance of the preprocessed (unit-norm) training
                   codes, and pr / n (1 = isotropic); top1, top10: share of variance in the first 1 / 10 components
  mu_pos           mean cosine between a dev alias and its own label (preprocessed)
  sd_neg           standard deviation of the cosine between a dev alias and a random other dev label
  dprime           mu_pos / sd_neg (alias-label signal over the spread of unrelated similarities)
  n_eff_pair       1 / mean squared cosine between random pairs of dev labels
  xtalk_R1         mu_pos / (sd_neg * sqrt(2 M / V)) averaged over R1's (M, V): the signal-to-crosstalk ratio of a summed
                   similarity read in which M / V unrelated keys share each value (the superposed store's structural
                   cross-talk at infinite width)
  nn_dprime        mu_pos / sd_neg * 1 / sqrt(2 ln M) averaged over M in {256, 1024}: the same for a nearest-neighbour
                   read (the alias must beat the largest of M - 1 unrelated similarities)
  sig_top_k        share of the mean alias-label inner product carried by the first k principal components (k = 8, 32,
                   128): how concentrated the signal is in few directions
  simhash_k        P(all k SimHash bits of an alias equal its label's), k = 4, 8, 16 (B36's pre-design measurement)
"""
import json
import math
import sys

import numpy as np

sys.path.insert(0, '/path/to/vsassm')
import b32_common as C  # noqa: E402

R1_MV = [(256, 2), (256, 16), (1024, 2), (1024, 16)]


def measure(filetag):
    pool = C.Pool(filetag)
    Tr = pool.train_matrix()
    dv = pool.by_split['dev']
    lab = np.array([pool.idx[e['name']] for e in dv])
    pairs = [(pool.idx[a], j) for j, e in enumerate(dv) for a in e['aliases']]
    qa = np.array([p for p, _ in pairs])
    qj = np.array([j for _, j in pairs])
    X = pool.X
    rng = np.random.default_rng(49)
    i, j = rng.integers(0, len(Tr), 20000), rng.integers(0, len(Tr), 20000)
    Xn = Tr / np.linalg.norm(Tr, axis=1, keepdims=True)
    out = {'n': int(X.shape[1]), 'raw_mean_cos': float((Xn[i] * Xn[j]).sum(1)[i != j].mean())}
    for kind in ('center', 'whiten'):
        P = C.Prep(kind, Tr)
        T = P(Tr)
        mu = T.mean(0)
        Cv = np.cov((T - mu).T)
        w, U = np.linalg.eigh(Cv)
        o = np.argsort(-w)
        w, U = np.maximum(w[o], 0), U[:, o]
        pr = float(w.sum() ** 2 / (w ** 2).sum())
        Y = P(X)
        A, Lb = Y[qa], Y[lab]
        cpos = (A * Lb[qj]).sum(1)
        r = rng.integers(0, len(lab), len(qa))
        keep = r != qj
        cneg = (A[keep] * Lb[r[keep]]).sum(1)
        a2, b2 = rng.integers(0, len(lab), 20000), rng.integers(0, len(lab), 20000)
        k2 = a2 != b2
        cll = (Lb[a2[k2]] * Lb[b2[k2]]).sum(1)
        mp, sn = float(cpos.mean()), float(cneg.std())
        # signal concentration: alias-label inner product in the PCA basis of the preprocessed training codes
        PA, PL = (A - mu) @ U, (Lb[qj] - mu) @ U
        contrib = (PA * PL).mean(0)
        tot = float(contrib.sum())
        G = np.random.default_rng(46).standard_normal((Y.shape[1], 16)).astype(np.float32)
        eq = ((A @ G) >= 0) == ((Lb[qj] @ G) >= 0)
        out[kind] = {
            'pr': pr, 'pr_frac': pr / X.shape[1], 'top1': float(w[0] / w.sum()), 'top10': float(w[:10].sum() / w.sum()),
            'mu_pos': mp, 'sd_neg': sn, 'mean_neg': float(cneg.mean()), 'dprime': mp / sn,
            'n_eff_pair': float(1 / (cll ** 2).mean()),
            'xtalk_R1': float(np.mean([mp / (sn * math.sqrt(2 * M / V)) for M, V in R1_MV])),
            'nn_dprime': float(np.mean([mp / sn / math.sqrt(2 * math.log(M)) for M in (256, 1024)])),
            'sig_top_k': {k: float(contrib[:k].sum() / tot) for k in (8, 32, 128)},
            'simhash_k': {k: float(eq[:, :k].all(1).mean()) for k in (4, 8, 16)}}
    return out


def main():
    out = {}
    for a in sys.argv[2:]:
        tag, ft = a.split('=') if '=' in a else (a, a)
        out[tag] = measure(ft)
        g = out[tag]
        print(f'{tag:10s} n {g["n"]:5d} raw_cos {g["raw_mean_cos"]:.3f} | ' + ' | '.join(
            f'{k}: PR {g[k]["pr"]:7.1f} ({g[k]["pr_frac"]:.3f}) top10 {g[k]["top10"]:.2f} mu+ {g[k]["mu_pos"]:.3f} '
            f'sd- {g[k]["sd_neg"]:.4f} d\' {g[k]["dprime"]:5.1f} xt {g[k]["xtalk_R1"]:.3f} sigtop8/32/128 '
            f'{g[k]["sig_top_k"][8]:.2f}/{g[k]["sig_top_k"][32]:.2f}/{g[k]["sig_top_k"][128]:.2f} sh8 '
            f'{g[k]["simhash_k"][8]:.2f}' for k in ('center', 'whiten')), flush=True)
    json.dump(out, open(sys.argv[1], 'w'), indent=1)


if __name__ == '__main__':
    main()
