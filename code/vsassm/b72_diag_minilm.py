# B72 check diagnostic (note after the test): the MiniLM OPTP re-derivation stores below 0.99. Run with OMP_NUM_THREADS=1 (as the jobs) and without (as the check).
import sys, math, json
sys.path.insert(0, '/path/to/vsassm')
import numpy as np
import b32_common as C, b32_run as R, b56_run as E56, b66_lsfc as LS, b71_run as E71, b72_run as E72, b72_check as K72, b71_check as K71, b72_verdict as V72
enc = 'minilm'; M, V, rep, b, q = 1024, 16, 0, 4, 4
S = LS.setup(enc); n = S['pool'].X.shape[1]
st = C.make_store(S['pool'], 'test', M, V, rep)
prep, fmap = 'center', 'psgn'; f, alpha = 0.5, 1.0
P = S['preps'][prep]
Fk, Fq, Cv, _ = E71.store_feats_d(enc, 'P', prep, fmap, V, P(st['keys']), P(st['queries']), n, n)
order = R.store_order(st); vals = np.asarray(st['vals']); ns = 512; d = n
X = Fk[order].astype(np.float64) * Cv[0].astype(np.float64)[None, :]
Sp = E72.opt_solve(X, vals[order], V, d, {ns: {alpha}})[ns, alpha]
Xs = X[:ns]; yv = vals[order[:ns]]
lam = alpha * float((Xs ** 2).sum()) / ns / 16
So = np.zeros(d); Hm = K71.sylv16()
for cl in range(1, 16):
    cols = np.arange(cl, d, 16)
    So[cols] = K72.own_ridge_class(Xs[:, cols], Hm[yv, cl], lam)
print('rel diff S', float(np.abs(Sp - So).max() / np.abs(So).max()))
sseed = R.store_seed(enc, st)
Q1 = E56.quant_np(Sp, q, ('B72Q', sseed, 'P', d, q, ns, alpha)); Q2 = K71.own_q_store(So, q, ('B72Q', sseed, 'P', d, q, ns, alpha))
print('quantized differ frac', float((np.abs(Q1 - Q2) > 1e-6 * np.abs(Q2).max()).mean()))
p1 = E56.mf_read(Fq, Cv, Q1); g = st['qgold']
ti = V72.test_index(enc); lg = np.asarray(ti[('OPTP', M, V, rep, prep, fmap, b, q, (f, alpha))])
print('recomputed pipeline vs logged agree', float((p1 == lg).mean()), 'acc', float((p1 == g).mean()), float((lg == g).mean()))
print('sing values of class 1 F:', np.linalg.svd(Xs[:, 1::16], compute_uv=False)[-4:])
