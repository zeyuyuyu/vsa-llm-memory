"""Recompute the forward-pass differences from the archived per-item test predictions.

Task N (arithmetic): trained types only (2,000 items; addition and integer division are held out); task R (chains):
all 2,000 items. Three training seeds per arm. vsa = the vector module, published read; vsaargmax = the same trained
weights read by best match; disc = the discrete module.

Two intervals per difference:
  items      per item, correctness is averaged over seeds; paired bootstrap over items (20,000 resamples, seed 0),
             as in the manuscript's first interval;
  with seeds two-level bootstrap (20,000 resamples, seed 1): the three seeds of each arm are resampled with
             replacement, independently per arm, then the items (paired across arms), so that seed variance is
             included; a Welch t interval over the three seed accuracies of each arm is printed as a check.
Usage: python3 forward_pass.py
"""
import json, math, os
import numpy as np
H = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'inputs', 'forward_pass')


def load(prefix, name, keep):
    M = []
    for s in range(3):
        L = [json.loads(l) for l in open(os.path.join(H, '%s%s_s%d.jsonl' % (prefix, name, s)))]
        M.append([x['correct'] for x in L if keep(x)])
    return np.array(M, float)


trained = lambda x: x['op'] not in ('addition', 'division')
V, A, Dm = load('', 'vsa', trained), load('', 'vsaargmax', trained), load('', 'disc', trained)
RV, RD = load('R_', 'vsa', lambda x: True), load('R_', 'disc', lambda x: True)
print('accuracy: published %.3f  best match %.3f  discrete %.3f  (items %d)' % (V.mean(), A.mean(), Dm.mean(), V.shape[1]))
rng = np.random.default_rng(0)
for label, X in (('published read - discrete', V), ('best match - discrete', A)):
    d = (X - Dm).mean(0)
    bs = d[rng.integers(0, len(d), (20000, len(d)))].mean(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    print('%-28s %+.2f points (95%% %.2f to %.2f)' % (label, 100 * d.mean(), 100 * lo, 100 * hi))

# Student t quantile at 0.975 for a (non-integer) Welch df, with numpy only: the density integrated numerically, then
# bisection on the CDF (agrees with scipy.stats.t.ppf to about 1e-6).
def t975(df):
    c = math.gamma((df + 1) / 2) / (math.sqrt(df * math.pi) * math.gamma(df / 2))
    def cdf(x):
        g = np.linspace(0.0, x, 200001)
        f = c * (1 + g * g / df) ** (-(df + 1) / 2)
        return 0.5 + float(np.sum((f[1:] + f[:-1]) * np.diff(g)) / 2)
    lo, hi = 0.0, 50.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if cdf(mid) < 0.975 else (lo, mid)
    return (lo + hi) / 2

print('\nwith seed variance (task N: 2,000 trained items; task R: 2,000 chain items; three seeds per arm)')
rng2 = np.random.default_rng(1)
for label, X, D in (('N published read - discrete', V, Dm), ('N best match - discrete', A, Dm), ('R chains, vector - discrete', RV, RD)):
    n, B = X.shape[1], 20000
    res = np.empty(B)
    for b in range(B):
        sx, sd, it = rng2.integers(0, 3, 3), rng2.integers(0, 3, 3), rng2.integers(0, n, n)
        res[b] = X[sx][:, it].mean() - D[sd][:, it].mean()
    lo, hi = np.percentile(res, [2.5, 97.5])
    ax, ad = X.mean(1), D.mean(1)
    vx, vd = ax.var(ddof=1) / 3, ad.var(ddof=1) / 3
    se, df = math.sqrt(vx + vd), (vx + vd) ** 2 / (vx ** 2 / 2 + vd ** 2 / 2)
    pt = X.mean() - D.mean()
    print('%-30s %+.2f points; two-level 95%% %.2f to %.2f; Welch t (df %.1f) %.2f to %.2f' % (
        label, 100 * pt, 100 * lo, 100 * hi, df, 100 * (pt - t975(df) * se), 100 * (pt + t975(df) * se)))
