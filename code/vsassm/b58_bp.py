"""B58: sum-product (flooding) LDPC decoder, a numba port of b38_check._sp_decode (B38's second LDPC decoder).

Same algorithm and the same numerical guards as b38_check._sp_decode:
  - channel LLR of bit = 0 in, hard decision tot < 0 -> 1;
  - variable-to-check messages clipped to +-40 before tanh(x / 2); |tanh| floored at 1e-15 (sign kept, +1 for 0);
  - check-to-variable message = 2 atanh(prod), |prod| capped at 1 - 1e-12 (so |message| <= 28.3);
  - stop at the first iteration whose hard decision satisfies every check; at most `iters` iterations.
The extrinsic product over a check is computed, as in _sp_decode, as exp(sum log|t| - log|t_e|) with the parity of
the other edges' signs. decode_many() runs one code over many words (one per row) in a single compiled loop.
Agreement with _sp_decode is checked in b58_check.py (bp_port) on synthetic and re-simulated words.
"""
import math

import numpy as np
from numba import njit


def csr_edges(H):
    """Edge lists of a scipy sparse parity-check matrix: per edge its row (check) and column (bit), edges sorted by
    row (COO order of a CSR matrix), plus per-column edge lists."""
    Hc = H.tocoo()
    order = np.lexsort((Hc.col, Hc.row))
    r = Hc.row[order].astype(np.int64)
    c = Hc.col[order].astype(np.int64)
    return r, c


@njit(cache=True)
def _decode(r, c, M, N, llr, iters, out):
    E = r.shape[0]
    v2c = np.empty(E)
    c2v = np.empty(E)
    lt = np.empty(E)
    sg = np.empty(E, np.int64)
    sum_lt = np.empty(M)
    sum_sg = np.empty(M, np.int64)
    tot = np.empty(N)
    syn = np.empty(M, np.int64)
    for e in range(E):
        v2c[e] = llr[c[e]]
    for i in range(N):
        out[i] = 1 if llr[i] < 0 else 0
    for it in range(iters):
        for m in range(M):
            sum_lt[m] = 0.0
            sum_sg[m] = 0
        for e in range(E):
            x = v2c[e]
            if x > 40.0:
                x = 40.0
            elif x < -40.0:
                x = -40.0
            t = math.tanh(x / 2.0)
            if abs(t) < 1e-15:
                t = 1e-15
            lt[e] = math.log(abs(t))
            sg[e] = 1 if t < 0 else 0
            sum_lt[r[e]] += lt[e]
            sum_sg[r[e]] += sg[e]
        for i in range(N):
            tot[i] = llr[i]
        for e in range(E):
            pa = math.exp(sum_lt[r[e]] - lt[e])
            if pa > 1.0 - 1e-12:
                pa = 1.0 - 1e-12
            if pa < 0.0:
                pa = 0.0
            ps = 1 - 2 * ((sum_sg[r[e]] - sg[e]) % 2)
            c2v[e] = 2.0 * math.atanh(pa * ps)
            tot[c[e]] += c2v[e]
        for i in range(N):
            out[i] = 1 if tot[i] < 0 else 0
        for m in range(M):
            syn[m] = 0
        for e in range(E):
            syn[r[e]] += out[c[e]]
        ok = True
        for m in range(M):
            if syn[m] % 2:
                ok = False
                break
        if ok:
            return True, it + 1
        for e in range(E):
            v2c[e] = tot[c[e]] - c2v[e]
    return False, iters


@njit(cache=True)
def _decode_many(r, c, M, N, LLR, iters, OUT, CONV, NIT):
    for w in range(LLR.shape[0]):
        ok, nit = _decode(r, c, M, N, LLR[w], iters, OUT[w])
        CONV[w] = ok
        NIT[w] = nit


class NbDecoder:
    def __init__(self, H, iters):
        self.r, self.c = csr_edges(H)
        self.M, self.N = H.shape
        self.iters = iters

    def decode_many(self, LLR):
        """LLR: (words, N) channel LLRs of bit = 0. Returns (hard bits uint8 (words, N), converged bool, iterations)."""
        LLR = np.ascontiguousarray(LLR, np.float64)
        W = LLR.shape[0]
        out = np.zeros((W, self.N), np.uint8)
        conv = np.zeros(W, np.bool_)
        nit = np.zeros(W, np.int64)
        _decode_many(self.r, self.c, self.M, self.N, LLR, self.iters, out, conv, nit)
        return out, conv, nit


def sp_decode_batch(H, LLR, iters):
    """b38_check._sp_decode applied to every row of LLR (words x N), vectorized over words with the same numpy
    elementwise functions and the same per-row / per-column summation order, so each word's result is bit-identical
    to _sp_decode on that word (checked in b58_dec.py port). Converged words are frozen and dropped from the batch.
    Returns (hard bits uint8 (words, N), converged bool (words,), iterations (words,))."""
    Hc = H.tocoo()
    r0, c0 = Hc.row.astype(np.int64), Hc.col.astype(np.int64)
    M, N = H.shape
    W = LLR.shape[0]
    out = (LLR < 0).astype(np.uint8)
    conv = np.zeros(W, bool)
    nit = np.full(W, iters, np.int64)
    act = np.arange(W)
    E_ = len(r0)

    def edges(a):
        k = len(a)
        off = np.arange(k, dtype=np.int64)
        return (r0[None, :] + M * off[:, None]).ravel(), (c0[None, :] + N * off[:, None]).ravel()
    r, c = edges(act)
    llr = LLR[act].ravel()
    v2c = llr[c].copy()
    for it in range(iters):
        k = len(act)
        t = np.tanh(np.clip(v2c, -40, 40) / 2)
        t = np.where(np.abs(t) < 1e-15, 1e-15, t)
        lt = np.log(np.abs(t))
        sg = (t < 0).astype(np.int64)
        sum_lt = np.bincount(r, weights=lt, minlength=M * k)
        sum_sg = np.bincount(r, weights=sg, minlength=M * k)
        prod_abs = np.exp(sum_lt[r] - lt)
        prod_sg = 1 - 2 * ((sum_sg[r] - sg) % 2)
        x = np.clip(prod_abs, 0, 1 - 1e-12) * prod_sg
        c2v = 2 * np.arctanh(x)
        tot = llr + np.bincount(c, weights=c2v, minlength=N * k)
        hard = (tot < 0).astype(np.uint8)
        par = np.bincount(r, weights=hard[c], minlength=M * k).reshape(k, M) % 2
        done = ~par.any(1)
        hk = hard.reshape(k, N)
        out[act] = hk
        if done.any():
            conv[act[done]] = True
            nit[act[done]] = it + 1
        if done.all():
            break
        v2c = tot[c] - c2v
        if done.any():
            keep = ~done
            act = act[keep]
            ek = np.repeat(keep, E_)
            v2c = v2c[ek]
            llr = llr.reshape(k, N)[keep].ravel()
            r, c = edges(act)
    return out, conv, nit
