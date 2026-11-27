#!/usr/bin/env python3
# Part B equal-cell simulation (Python 3.10).
#
# Choices where the specification is open:
# * Stable BLAKE2b hashes, with separate namespaces, seed all digital codebooks,
#   layouts and random draws. Codebook randomness is not device storage.
# * Global drift compensation rescales the differential read by
#   25 / E[G_d(SET,t)]. The expectation is estimated in a separate, fixed-seed
#   Monte Carlo calibration, before estimating the sign-error probability.
# * Converter rounding is symmetric nearest-level rounding, with saturation;
#   a zero output receives an independent fair random sign.
# * A shard's unused sparse-block tail is charged, but need not be simulated:
#   its readout can never affect a query. Dense sum ties are independent coins.
# * Sparse codebooks use independent uniformly distributed indices in each
#   block; their seeds are stable across budgets, with geometry-dependent draws.
# * Smoke uses the two smallest development stores, the two smallest test
#   stores, budgets 512 and 4096, and k=8. Its output is explicitly diagnostic,
#   not an estimate of the full-study requirement. Device estimation still
#   uses at least 10^7 cells.
# * No test content is used to select a design. All cell uniforms are seeded
#   without k, so channel draws are shared across noise conditions.
# * The driver and table codecs follow in pieces 2 and 3.

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"

import argparse
import gzip
import hashlib
import json
import math
import time
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy import special
from scipy.stats import binom
from scipy.optimize import minimize_scalar
from numba import njit
from joblib import Parallel, delayed

TEST_PATH = "/path/to/vsassm/b34_out/test_stores.json.gz"
DEV_PATH = "/path/to/vsassm/b34_out/dev_stores.json.gz"
GRID = np.exp2(np.arange(12, 49, dtype=np.float64) / 4)
NOISE_LEVELS = (1.0, 8.0, 16.0)
LAMBDAS = (0.35, 0.5, 0.7, 1.0)
TARGET_SHARE = 0.995
MASTER_SEED = 20250317


def seed_of(*parts):
    payload = json.dumps([MASTER_SEED, *parts], ensure_ascii=False,
                         separators=(",", ":")).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(),
                          "little")


def rng_of(*parts):
    return np.random.default_rng(seed_of(*parts))


def parse_args():
    parser = argparse.ArgumentParser(description="Part B equal-cell simulation")
    parser.add_argument("--memory", choices=("dense", "sbc", "bch", "cuckoo", "all"),
                        default="all")
    parser.add_argument("--k", default="1,8,16")
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=14)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    try:
        args.ks = sorted(set(float(x) for x in args.k.split(",")))
    except ValueError:
        parser.error("--k must be a comma-separated list of numbers")
    if not args.ks or any(not math.isfinite(k) or k < 0 for k in args.ks):
        parser.error("noise multipliers must be finite and nonnegative")
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.smoke:
        args.ks = [8.0]
    return args


@dataclass
class Store:
    sid: str
    keys: list
    values: list
    pair_key: np.ndarray
    pair_value: np.ndarray
    pair_j: np.ndarray
    last: np.ndarray
    truth: list
    probes: np.ndarray
    maxj: int

    @property
    def n(self):
        return len(self.pair_j)

    @property
    def N(self):
        return len(self.values)


def make_store(sid, pairs):
    if not pairs:
        raise ValueError(f"{sid}: empty store")
    keys, values, km, vm = [], [], {}, {}
    pk, pv, pj = [], [], []
    per_key = defaultdict(dict)
    for row in pairs:
        if len(row) != 3:
            raise ValueError(f"{sid}: malformed pair")
        key, value, j = row
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError(f"{sid}: keys and values must be strings")
        if not isinstance(j, int) or j < 0:
            raise ValueError(f"{sid}: invalid position")
        if key not in km:
            km[key] = len(keys)
            keys.append(key)
        if value not in vm:
            vm[value] = len(values)
            values.append(value)
        ki, vi = km[key], vm[value]
        if j in per_key[ki]:
            raise ValueError(f"{sid}: duplicate (key,j)")
        per_key[ki][j] = vi
        pk.append(ki)
        pv.append(vi)
        pj.append(j)
    truth = []
    for ki in range(len(keys)):
        entries = per_key[ki]
        if sorted(entries) != list(range(len(entries))):
            raise ValueError(f"{sid}: noncontiguous key positions")
        truth.append([entries[j] for j in range(len(entries))])
    pk = np.asarray(pk, dtype=np.int32)
    pj = np.asarray(pj, dtype=np.int32)
    last = np.asarray([j == len(truth[k]) - 1 for k, j in zip(pk, pj)],
                      dtype=np.uint8)
    probes = np.arange(0, len(keys), max(1, len(keys) // 150),
                       dtype=np.int32)[:150]
    return Store(str(sid), keys, values, pk, np.asarray(pv, dtype=np.int32),
                 pj, last, truth, probes, int(pj.max()))


def load_stores(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        entries = json.load(handle)
    stores = [make_store(sid, pairs) for sid, pairs in entries]
    if len({s.sid for s in stores}) != len(stores):
        raise ValueError("duplicate store identifiers")
    return stores


def load_data(smoke=False):
    dev, test = load_stores(DEV_PATH), load_stores(TEST_PATH)
    count = sum(len(s.probes) for s in test)
    if len(dev) != 12 or len(test) != 50 or count != 4776:
        raise ValueError(f"dataset mismatch: dev={len(dev)}, test={len(test)}, "
                         f"test probes={count}, expected 12,50,4776")
    if smoke:
        dev = sorted(dev, key=lambda s: (s.n, s.sid))[:2]
        test = sorted(test, key=lambda s: (s.n, s.sid))[:2]
    return dev, test


def geometry(n, budget):
    # Integer budget is authoritative; floor(d) guarantees no overspend.
    cap = math.floor(float(budget) * n)
    shards = max(1, math.ceil(float(budget) * n / 8192))
    d = cap // shards
    if d < 1 or d > 8192:
        raise ValueError("invalid store geometry")
    return shards, d, cap


def shard_ids(store, shards):
    return np.asarray([seed_of("shard", k) % shards for k in store.keys],
                      dtype=np.int32)


def drift_read(target, k, size, rng, read=True):
    t, t0, t1, tref = 30 * 86400.0, 27.36, 55.39, 20.0
    mu = max(-0.0045 * target + 0.08,
             min(0.06, max(-0.00169 * target + 0.0825, 0.055)))
    nu = np.maximum(mu + k / (8 * target + 30.032) *
                    rng.standard_normal(size), 0.0)
    sigma = 1.2 if target else 0.2
    g = np.maximum(target * (t0 / tref) ** (-nu) +
                   k * sigma * rng.standard_normal(size), 0.0)
    g = np.maximum(g * (t1 / t0) ** (-nu) +
                   k * sigma * rng.standard_normal(size), 0.0)
    g *= (t / t1) ** (-nu)
    if read:
        nuq = min(0.081, 0.04 * target + 0.02)
        if target > 13.21:
            nuq -= min(0.035, -0.2048 + 0.0155 * target)
        q = min(0.0388 * target ** (-0.5729), 0.2) if target else 0.2
        q *= (t / 1.089e5) ** (-nuq)
        fmax, fmin, gamma = 1 / (2 * 250e-9), 1 / (t + 250e-9), 1.21
        integ = math.sqrt((fmax ** (1 - gamma) - fmin ** (1 - gamma)) /
                          (1 - gamma))
        g = np.maximum(g + k * q * integ * g * rng.standard_normal(size), 0)
    return g


def estimate_p(k, samples=10_000_000, chunk=250_000):
    if samples < 10_000_000:
        raise ValueError("cell-rate estimation requires at least 10^7 cells")
    calibration = rng_of("device-calibration", k)
    total = 0.0
    for start in range(0, 1_000_000, chunk):
        size = min(chunk, 1_000_000 - start)
        total += float(drift_read(25.0, k, size, calibration, False).sum())
    compensation = 25.0 / max(total / 1_000_000, np.finfo(float).tiny)
    rng = rng_of("device-estimate", k)
    errors = 0
    # Symmetry permits writing +1 throughout; this has precisely the same
    # error distribution as independent equiprobable written signs.
    for start in range(0, samples, chunk):
        size = min(chunk, samples - start)
        diff = compensation * (drift_read(25.0, k, size, rng) -
                               drift_read(0.0, k, size, rng))
        quantized = np.rint(np.clip(diff, -25, 25) * (127 / 25))
        errors += int(np.count_nonzero(quantized < 0))
        zeros = int(np.count_nonzero(quantized == 0))
        errors += int(rng.binomial(zeros, 0.5))
    p = errors / samples
    return {"k": float(k), "p": p, "samples": samples, "errors": errors,
            "standard_error": math.sqrt(p * (1 - p) / samples),
            "global_compensation": compensation}


def channel_rng(store, memory, budget, design):
    return rng_of("channel", store.sid, memory, float(budget), design)


def flip_cells(cells, p, rng):
    result = cells.copy()
    flat = result.reshape(-1)
    for start in range(0, flat.size, 262144):
        view = flat[start:start + 262144]
        mask = rng.random(len(view)) < p
        if result.dtype == np.bool_ or result.dtype == np.uint8:
            view[mask] ^= 1
        else:
            view[mask] *= -1
    return result


def bipolar(namespace, token, d):
    return (2 * rng_of("bipolar", namespace, token).integers(
        0, 2, size=8192, dtype=np.int8)[:d] - 1)


def block_code(namespace, token, blocks, length):
    return rng_of("block", namespace, token).integers(
        0, length, size=blocks, dtype=np.int32)


def dense_run(store, budget, p, design="dense"):
    shards, d, cap = geometry(store.n, budget)
    ids = shard_ids(store, shards)
    kc = np.asarray([bipolar("key", k, d) for k in store.keys])
    vc = np.asarray([bipolar("value", v, d) for v in store.values])
    pc = np.asarray([bipolar("position", j, d) for j in range(store.maxj + 1)])
    last = bipolar("LAST", "", d)
    sums = np.zeros((shards, d), dtype=np.int32)
    for ki, vi, j, flag in zip(store.pair_key, store.pair_value,
                               store.pair_j, store.last):
        code = kc[ki] * pc[j] * vc[vi]
        if flag:
            code = code * last
        sums[ids[ki]] += code
    cells = np.where(sums > 0, 1, -1).astype(np.int8)
    tie_rng = rng_of("sum-ties", store.sid, float(budget))
    for sh in range(shards):
        mask = sums[sh] == 0
        cells[sh, mask] = 2 * tie_rng.integers(
            0, 2, size=int(mask.sum()), dtype=np.int8) - 1
    del sums
    cells = flip_cells(cells, p, channel_rng(store, "dense", budget, design))
    exact = 0
    for ki in store.probes:
        output = []
        for j in range(store.maxj + 1):
            query = (cells[ids[ki]] * kc[ki] * pc[j]).astype(np.float32)
            best, winner, marked = -math.inf, 0, False
            # Plain candidates precede LAST candidates, including across chunks.
            for flag in (False, True):
                q = query * last if flag else query
                for begin in range(0, store.N, 256):
                    scores = vc[begin:begin + 256].astype(np.float32) @ q
                    idx = int(np.argmax(scores))
                    score = float(scores[idx])
                    if score > best:
                        best, winner, marked = score, begin + idx, flag
            output.append(winner)
            if marked:
                break
        exact += output == store.truth[ki]
    return {"exact": int(exact), "probes": len(store.probes),
            "cells": shards * d, "failed_insertions": 0}


def sbc_run(store, budget, p, design):
    lam = float(design["lam"] if isinstance(design, dict) else design)
    shards, d, cap = geometry(store.n, budget)
    length = max(2, round((store.n / shards) / lam))
    blocks = d // length
    ids = shard_ids(store, shards)
    kc = np.asarray([block_code("key", k, blocks, length) for k in store.keys])
    vc = np.asarray([block_code("value", v, blocks, length) for v in store.values])
    pc = np.asarray([block_code("position", j, blocks, length)
                     for j in range(store.maxj + 1)])
    last = block_code("LAST", "", blocks, length)
    cells = np.zeros((shards, blocks * length), dtype=np.uint8)
    offsets = np.arange(blocks, dtype=np.int64) * length
    for ki, vi, j, flag in zip(store.pair_key, store.pair_value,
                               store.pair_j, store.last):
        pos = (kc[ki] + pc[j] + vc[vi] + (last if flag else 0)) % length
        cells[ids[ki], offsets + pos] = 1
    cells = flip_cells(cells, p, channel_rng(store, "sbc", budget, {"lam": lam}))
    exact = 0
    for ki in store.probes:
        output = []
        for j in range(store.maxj + 1):
            base = kc[ki] + pc[j]
            best, winner, marked = -1, 0, False
            for flag in (False, True):
                shift = base + (last if flag else 0)
                for begin in range(0, store.N, 256):
                    pos = (vc[begin:begin + 256] + shift) % length
                    scores = cells[ids[ki], offsets + pos].sum(axis=1)
                    idx = int(np.argmax(scores))
                    score = int(scores[idx])
                    if score > best:
                        best, winner, marked = score, begin + idx, flag
            output.append(winner)
            if marked:
                break
        exact += output == store.truth[ki]
    return {"exact": int(exact), "probes": len(store.probes),
            "cells": shards * d, "failed_insertions": 0}
# BCH polynomials and word coordinates are little-endian: coefficient i
# multiplies x^i. Shortening removes highest systematic coordinates.
# Cuckoo bucket counts are powers of two: XOR partial-key hashing requires
# closure under XOR. The smallest admissible power of two is used.
# BCH design objective uses the union probability of failed codewords touched
# by each probed key; it conservatively treats a failed word as corrupting
# every field touched. Directory words are included for static-table queries.


@lru_cache(None)
def gf_tables(m):
    primitive = {6: 0x43, 7: 0x83, 8: 0x11D, 9: 0x211, 10: 0x409}[m]
    n = (1 << m) - 1
    exp = np.zeros(2 * n, dtype=np.int32)
    log = np.full(n + 1, -1, dtype=np.int32)
    x = 1
    for i in range(n):
        exp[i], log[x] = x, i
        x <<= 1
        if x & (1 << m):
            x ^= primitive
    if x != 1 or len(set(exp[:n])) != n:
        raise RuntimeError("invalid primitive polynomial")
    exp[n:] = exp[:n]
    return exp, log


def gf_mul(a, b, exp, log):
    return int(exp[int(log[a]) + int(log[b])]) if a and b else 0


def gf_div(a, b, exp, log):
    if not b:
        raise ZeroDivisionError("GF division")
    return int(exp[(int(log[a]) - int(log[b])) % (len(exp) // 2)]) if a else 0


def gf_poly_mul(a, b, exp, log):
    c = [0] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            c[i + j] ^= gf_mul(x, y, exp, log)
    return c


@lru_cache(None)
def bch_specs(n):
    m = (n + 1).bit_length() - 1
    exp, log = gf_tables(m)
    roots, generator, specs = set(), [1], []
    for t in range(1, n // 2 + 1):
        for r in (2 * t - 1, 2 * t):
            if r % n in roots:
                continue
            coset, x = [], r % n
            while x not in coset:
                coset.append(x)
                x = 2 * x % n
            roots.update(coset)
            minimal = [1]
            for x in coset:
                minimal = gf_poly_mul(minimal, [int(exp[x]), 1], exp, log)
            if any(v not in (0, 1) for v in minimal):
                raise RuntimeError("nonbinary BCH generator")
            generator = gf_poly_mul(generator, minimal, exp, log)
        k = n - len(generator) + 1
        if k < 2:
            break
        specs.append((n, t, k, tuple(generator)))
    return tuple(specs)


def bch_encode(info, spec):
    n, t, k, generator = spec
    info = np.asarray(info, dtype=np.uint8)
    if not 0 < len(info) <= k:
        raise ValueError("invalid shortened BCH information length")
    r = n - k
    word = np.concatenate((np.zeros(r, dtype=np.uint8), info))
    work = word.copy()
    g = np.asarray(generator, dtype=np.uint8)
    for i in range(len(work) - 1, r - 1, -1):
        if work[i]:
            work[i - r:i + 1] ^= g
    word[:r] = work[:r]
    return word


@njit(cache=True)
def bch_syndromes(word, exp, log, t):
    n = len(exp) // 2
    syn = np.zeros(2 * t, dtype=np.int32)
    for i in range(len(word)):
        if word[i]:
            for j in range(2 * t):
                syn[j] ^= exp[((j + 1) * i) % n]
    return syn


def berlekamp_massey(syn, exp, log):
    c, b, degree, gap, scale = [1], [1], 0, 1, 1
    for step in range(len(syn)):
        discrepancy = int(syn[step])
        for i in range(1, degree + 1):
            if i < len(c):
                discrepancy ^= gf_mul(c[i], int(syn[step - i]), exp, log)
        if not discrepancy:
            gap += 1
            continue
        old = c[:]
        factor = gf_div(discrepancy, scale, exp, log)
        c.extend([0] * max(0, len(b) + gap - len(c)))
        for i, v in enumerate(b):
            c[i + gap] ^= gf_mul(factor, v, exp, log)
        if 2 * degree <= step:
            degree, b, scale, gap = step + 1 - degree, old, discrepancy, 1
        else:
            gap += 1
    return c[:degree + 1]


def bch_decode(received, erased, spec):
    n, t, k, _ = spec
    r = n - k
    raw = np.asarray(received, dtype=np.uint8).copy()
    fallback = raw[r:].copy()
    erasures = np.flatnonzero(erased)
    if len(erasures) > 2 * t:
        return fallback
    exp, log = gf_tables((n + 1).bit_length() - 1)
    syn = bch_syndromes(raw, exp, log, t)
    if not syn.any() and not len(erasures):
        return fallback
    # Forney syndromes eliminate known erasure locations before BM.
    fs = list(map(int, syn))
    eloc = [1]
    for pos in erasures:
        x = int(exp[int(pos)])
        eloc = gf_poly_mul(eloc, [1, x], exp, log)
        fs = [gf_mul(fs[j], x, exp, log) ^ fs[j + 1]
              for j in range(len(fs) - 1)]
    error_loc = berlekamp_massey(fs, exp, log)
    e = len(error_loc) - 1
    if 2 * e + len(erasures) > 2 * t:
        return fallback
    locator = gf_poly_mul(eloc, error_loc, exp, log)
    positions = []
    for pos in range(len(raw)):
        x, val = int(exp[(-pos) % n]), 0
        for coef in reversed(locator):
            val = gf_mul(val, x, exp, log) ^ coef
        if val == 0:
            positions.append(pos)
    v = len(positions)
    if v != len(locator) - 1 or not set(erasures).issubset(positions):
        return fallback
    # Solve error magnitudes, including unknown erased bits, over GF(2^m).
    matrix = [[int(exp[((row + 1) * pos) % n]) for pos in positions] +
              [int(syn[row])] for row in range(v)]
    for col in range(v):
        pivot = next((i for i in range(col, v) if matrix[i][col]), None)
        if pivot is None:
            return fallback
        matrix[col], matrix[pivot] = matrix[pivot], matrix[col]
        divisor = matrix[col][col]
        matrix[col] = [gf_div(x, divisor, exp, log) for x in matrix[col]]
        for i in range(v):
            if i != col and matrix[i][col]:
                factor = matrix[i][col]
                matrix[i] = [x ^ gf_mul(factor, y, exp, log)
                             for x, y in zip(matrix[i], matrix[col])]
    for pos, row in zip(positions, matrix):
        if row[-1] not in (0, 1):
            return fallback
        raw[pos] ^= row[-1]
    if bch_syndromes(raw, exp, log, t).any():
        return fallback
    return raw[r:].copy()


def repetition_counts(length, budget):
    if length > budget:
        return None
    counts = np.full(length, budget // length, dtype=np.int32)
    extra = budget % length
    if extra:
        counts[(np.arange(extra, dtype=np.int64) * length // extra)] += 1
    return counts


def repetition_read(bits, counts, p, rng, soft=False):
    # Binomial inverse CDF uses one fixed uniform per physical cell.
    votes = np.zeros(len(bits), dtype=np.int32)
    for copy in range(int(counts.max())):
        indices = np.flatnonzero(counts > copy)
        flips = rng.random(len(indices)) < p
        votes[indices] += bits[indices] ^ flips.astype(np.uint8)
    if soft:
        return (counts - 2 * votes) * math.log(
            max(1 - p, 1e-15) / max(p, 1e-15))
    return (2 * votes > counts).astype(np.uint8), 2 * votes == counts


def majority_rates(rho, p):
    erasure = float(binom.pmf(rho // 2, rho, p)) if rho % 2 == 0 else 0.0
    error = float(binom.sf(rho // 2, rho, p))
    return error, erasure


def word_failure(length, t, error, erasure):
    if error + erasure == 0:
        return 0.0
    s = np.arange(min(2 * t, length) + 1)
    conditional = error / max(1 - erasure, 1e-300)
    success = np.sum(binom.pmf(s, length, erasure) *
                     binom.cdf((2 * t - s) // 2, length - s, conditional))
    return float(np.clip(1 - success, 0, 1))


def choose_bch(logical_length, cap, p, harder_p=None, touches=None):
    best, answer = None, None
    for n in (63, 127, 255, 511, 1023):
        for spec in bch_specs(n):
            _, t, k, _ = spec
            sizes = np.minimum(k, logical_length - np.arange(
                0, logical_length, k))
            lengths = sizes + n - k
            total = int(lengths.sum())
            if total > cap:
                continue
            rho, fraction = divmod(cap, total)
            weight = fraction / total
            def objective(rate):
                e0, s0 = majority_rates(rho, rate)
                e1, s1 = majority_rates(rho + 1, rate)
                e, s = (1 - weight) * e0 + weight * e1, (
                    1 - weight) * s0 + weight * s1
                failures = np.array([word_failure(int(z), t, e, s)
                                     for z in np.unique(lengths)])
                mapping = dict(zip(np.unique(lengths), failures))
                wf = np.array([mapping[z] for z in lengths])
                if touches is None:
                    return float(1 - np.prod(1 - wf))
                return sum(1 - np.prod(1 - wf[np.unique(
                    np.asarray(indices, dtype=np.int64) // k)])
                           for indices in touches)
            score = (objective(p), objective(harder_p if harder_p is not None
                                            else p), n, t)
            if best is None or score < best:
                best, answer = score, spec
    return answer


def bch_transport(logical, cap, p, rng, spec):
    if spec is None:
        return None
    k = spec[2]
    words = [bch_encode(logical[i:i + k], spec)
             for i in range(0, len(logical), k)]
    encoded = np.concatenate(words)
    counts = repetition_counts(len(encoded), cap)
    if counts is None:
        return None
    received, erased = repetition_read(encoded, counts, p, rng)
    decoded, offset = [], 0
    for word in words:
        end = offset + len(word)
        decoded.append(bch_decode(received[offset:end], erased[offset:end], spec))
        offset = end
    return np.concatenate(decoded)


def uint_bits(values, width):
    a = np.asarray(values, dtype=np.uint64).reshape(-1)
    return ((a[:, None] >> np.arange(width, dtype=np.uint64)) & 1
            ).astype(np.uint8).reshape(-1)


def bits_uint(bits, width):
    if width == 0:
        return np.zeros(0, dtype=np.uint64)
    return (np.asarray(bits).reshape(-1, width).astype(np.uint64) *
            (np.uint64(1) << np.arange(width, dtype=np.uint64))).sum(axis=1)


def pair_token(store, i):
    return (store.keys[int(store.pair_key[i])], int(store.pair_j[i]))


def mph_build(store):
    buckets = max(1, math.ceil(store.n / 5))
    groups = [[] for _ in range(buckets)]
    for i in range(store.n):
        groups[seed_of("mph-bucket", pair_token(store, i)) % buckets].append(i)
    occupied = np.zeros(store.n, dtype=np.bool_)
    displacements = np.zeros(buckets, dtype=np.uint64)
    rows = np.empty(store.n, dtype=np.int32)
    for bucket in sorted(range(buckets), key=lambda x: -len(groups[x])):
        group = groups[bucket]
        if not group:
            continue
        displacement = 0
        while True:
            loc = [seed_of("mph-row", pair_token(store, i), displacement) %
                   store.n for i in group]
            if len(set(loc)) == len(loc) and not occupied[loc].any():
                break
            displacement += 1
            if displacement > 10_000_000:
                raise RuntimeError("MPH displacement search exhausted")
        displacements[bucket] = displacement
        occupied[loc] = True
        rows[loc] = group
    return displacements, rows


def table_fields(store):
    return max(1, (store.N - 1).bit_length()), int(store.maxj > 0)


def static_layout(store):
    disp, order = mph_build(store)
    dw = max(1, int(disp.max()).bit_length())
    vw, flag = table_fields(store)
    rowbits = np.concatenate((uint_bits(store.pair_value[order], vw).reshape(
        store.n, vw), store.last[order, None] if flag else
        np.empty((store.n, 0), dtype=np.uint8)), axis=1).reshape(-1)
    logical = np.concatenate((uint_bits(disp, dw), rowbits))
    touches = []
    base, rw = len(disp) * dw, vw + flag
    inverse = np.empty(store.n, dtype=np.int32)
    inverse[order] = np.arange(store.n)
    for ki in store.probes:
        fields = []
        for i in np.flatnonzero(store.pair_key == ki):
            b = seed_of("mph-bucket", pair_token(store, i)) % len(disp)
            fields.extend(range(b * dw, (b + 1) * dw))
            fields.extend(range(base + inverse[i] * rw,
                                base + (inverse[i] + 1) * rw))
        touches.append(fields)
    return logical, (len(disp), dw, vw, flag), touches


def static_read(store, decoded, meta):
    buckets, dw, vw, flag = meta
    disp = bits_uint(decoded[:buckets * dw], dw)
    rows = decoded[buckets * dw:].reshape(store.n, vw + flag)
    values = bits_uint(rows[:, :vw].reshape(-1), vw)
    exact = 0
    for ki in store.probes:
        output = []
        for j in range(store.maxj + 1):
            token = (store.keys[ki], j)
            bucket = seed_of("mph-bucket", token) % buckets
            row = seed_of("mph-row", token, int(disp[bucket])) % store.n
            output.append(int(values[row]))
            if flag and rows[row, vw]:
                break
        exact += output == store.truth[ki]
    return int(exact)


def bch_run(store, budget, p, design=None, harder_p=None):
    logical, meta, touches = static_layout(store)
    cap = math.floor(budget * store.n)
    spec = choose_bch(len(logical), cap, p, harder_p, touches)
    decoded = bch_transport(logical, cap, p,
                            channel_rng(store, "bch", budget, "closed-form"), spec)
    return {"exact": 0 if decoded is None else static_read(store, decoded, meta),
            "probes": len(store.probes), "cells": cap if decoded is not None else 0,
            "fits": decoded is not None, "failed_insertions": 0}


def cuckoo_address(token, f, buckets):
    fingerprint = 1 + seed_of("fingerprint", token) % ((1 << f) - 1)
    first = seed_of("cuckoo-first", token) % buckets
    delta = seed_of("cuckoo-delta", fingerprint) % buckets
    if buckets > 1 and delta == 0:
        delta = 1
    return fingerprint, first, first ^ delta


def cuckoo_build(store, f):
    needed = max(1, math.ceil(store.n / (0.95 * 4)))
    buckets = 1 << (needed - 1).bit_length()
    slots = np.full((buckets, 4), -1, dtype=np.int32)
    fingerprints = np.zeros((buckets, 4), dtype=np.uint32)
    rng = rng_of("cuckoo-insert", store.sid, f)
    failed = 0
    for i in range(store.n):
        fp, b1, b2 = cuckoo_address(pair_token(store, i), f, buckets)
        empty = [(b, int(s)) for b in sorted(set((b1, b2)))
                 for s in np.flatnonzero(slots[b] < 0)]
        if empty:
            b, s = empty[0]
            slots[b, s], fingerprints[b, s] = i, fp
            continue
        # Roll back failed walks: previously inserted rows are never lost.
        changes, item, fingerprint = [], i, fp
        b = b1 if rng.integers(2) == 0 else b2
        success = False
        for move in range(500):
            s = int(rng.integers(4))
            old, oldfp = int(slots[b, s]), int(fingerprints[b, s])
            changes.append((b, s, old, oldfp))
            slots[b, s], fingerprints[b, s] = item, fingerprint
            item, fingerprint = old, oldfp
            _, a, c = cuckoo_address(pair_token(store, item), f, buckets)
            b = c if b == a else a
            vacancies = np.flatnonzero(slots[b] < 0)
            if len(vacancies):
                s = int(vacancies[0])
                slots[b, s], fingerprints[b, s] = item, fingerprint
                success = True
                break
        if not success:
            failed += 1
            for b, s, old, oldfp in reversed(changes):
                slots[b, s], fingerprints[b, s] = old, oldfp
    vw, flag = table_fields(store)
    flat = slots.reshape(-1)
    vals = np.zeros(len(flat), dtype=np.uint32)
    last = np.zeros(len(flat), dtype=np.uint8)
    used = flat >= 0
    vals[used], last[used] = store.pair_value[flat[used]], store.last[flat[used]]
    rows = np.concatenate((uint_bits(fingerprints, f).reshape(-1, f),
                           uint_bits(vals, vw).reshape(-1, vw),
                           last[:, None] if flag else
                           np.empty((len(flat), 0), dtype=np.uint8)), axis=1)
    return rows.reshape(-1), (buckets, f, vw, flag), failed


def cuckoo_read(store, decoded, meta):
    buckets, f, vw, flag = meta
    rows = decoded.reshape(buckets * 4, f + vw + flag)
    fingerprints = bits_uint(rows[:, :f].reshape(-1), f)
    values = bits_uint(rows[:, f:f + vw].reshape(-1), vw)
    exact = 0
    for ki in store.probes:
        output, valid = [], True
        for j in range(store.maxj + 1):
            fp, a, b = cuckoo_address((store.keys[ki], j), f, buckets)
            locations = [4 * bucket + s for bucket in sorted(set((a, b)))
                         for s in range(4)]
            matches = [loc for loc in locations if fingerprints[loc] == fp]
            if len(matches) != 1:
                valid = False
                break
            loc = matches[0]
            output.append(int(values[loc]))
            if flag and rows[loc, f + vw]:
                break
        exact += valid and output == store.truth[ki]
    return int(exact)
# IRA information-column neighbours are sampled without replacement, with
# stable geometry seeds. Check nodes are not degree-regularised. Short final
# words are constructed at their own information length and requested rate.
# BP uses clipped LLRs for numerical stability; a failed decoder returns its
# current systematic hard decisions. No extra CRC or failure flag is stored.
# Cuckoo designs must fit even when an insertion fails; failed insertions
# contribute lookup failures, rather than removing the store from evaluation.

IRA_RATES = ((1, 4, 6), (1, 3, 6), (1, 2, 5), (2, 3, 6),
             (3, 4, 5), (5, 6, 4), (9, 10, 5))


@lru_cache(maxsize=256)
def ira_graph(k, numerator, denominator, dv):
    checks = math.ceil(k * (denominator - numerator) / numerator)
    # A minimum of dv checks makes distinct information neighbours possible.
    checks = max(checks, dv)
    rng = rng_of("IRA-graph", k, numerator, denominator, dv)
    variables, check_ids = [], []
    for v in range(k):
        neighbours = rng.choice(checks, size=dv, replace=False)
        variables.extend([v] * dv)
        check_ids.extend(map(int, neighbours))
    for c in range(checks):
        variables.append(k + c)
        check_ids.append(c)
        if c:
            variables.append(k + c - 1)
            check_ids.append(c)
    variables = np.asarray(variables, dtype=np.int32)
    check_ids = np.asarray(check_ids, dtype=np.int32)
    order = np.argsort(check_ids, kind="stable").astype(np.int32)
    pointers = np.concatenate(([0], np.cumsum(
        np.bincount(check_ids, minlength=checks)))).astype(np.int32)
    return variables, check_ids, order, pointers, checks


def ira_encode(info, rate):
    k = len(info)
    variables, check_ids, order, pointers, checks = ira_graph(k, *rate)
    rhs = np.zeros(checks, dtype=np.uint8)
    edges = variables < k
    np.bitwise_xor.at(rhs, check_ids[edges], info[variables[edges]])
    return np.concatenate((info, np.bitwise_xor.accumulate(rhs)))


@njit(cache=True)
def ira_bp(channel, variables, order, pointers, iterations=60):
    edges = len(variables)
    outgoing = np.empty(edges, dtype=np.float64)
    incoming = np.zeros(edges, dtype=np.float64)
    posterior = channel.copy()
    hard = np.zeros(len(channel), dtype=np.uint8)
    for e in range(edges):
        outgoing[e] = max(-40.0, min(40.0, channel[variables[e]]))
    maxdegree = 0
    for c in range(len(pointers) - 1):
        maxdegree = max(maxdegree, pointers[c + 1] - pointers[c])
    prefix = np.empty(maxdegree + 1, dtype=np.float64)
    suffix = np.empty(maxdegree + 1, dtype=np.float64)
    for iteration in range(iterations):
        for c in range(len(pointers) - 1):
            begin, end = pointers[c], pointers[c + 1]
            degree = end - begin
            prefix[0] = 1.0
            for j in range(degree):
                e = order[begin + j]
                prefix[j + 1] = prefix[j] * math.tanh(outgoing[e] * 0.5)
            suffix[degree] = 1.0
            for j in range(degree - 1, -1, -1):
                e = order[begin + j]
                suffix[j] = suffix[j + 1] * math.tanh(outgoing[e] * 0.5)
            for j in range(degree):
                product = max(-1 + 1e-12, min(1 - 1e-12,
                                             prefix[j] * suffix[j + 1]))
                incoming[order[begin + j]] = 2 * math.atanh(product)
        posterior[:] = channel
        for e in range(edges):
            posterior[variables[e]] += incoming[e]
        for v in range(len(channel)):
            hard[v] = posterior[v] < 0
        valid = True
        for c in range(len(pointers) - 1):
            parity = 0
            for j in range(pointers[c], pointers[c + 1]):
                parity ^= hard[variables[order[j]]]
            if parity:
                valid = False
                break
        if valid:
            return hard
        for e in range(edges):
            outgoing[e] = max(-40.0, min(40.0,
                posterior[variables[e]] - incoming[e]))
    return hard


def ira_transport(logical, cap, p, rng, rate):
    words = [ira_encode(logical[i:i + 1024], rate)
             for i in range(0, len(logical), 1024)]
    encoded = np.concatenate(words)
    counts = repetition_counts(len(encoded), cap)
    if counts is None:
        return None
    llr = repetition_read(encoded, counts, p, rng, soft=True).astype(np.float64)
    output, offset = [], 0
    for i, word in enumerate(words):
        k = min(1024, len(logical) - 1024 * i)
        variables, _, order, pointers, _ = ira_graph(k, *rate)
        decoded = ira_bp(llr[offset:offset + len(word)],
                         variables, order, pointers)
        output.append(decoded[:k])
        offset += len(word)
    return np.concatenate(output)


def cuckoo_run(store, budget, p, design, harder_p=None):
    f = int(design["f"])
    logical, meta, failures = cuckoo_build(store, f)
    cap = math.floor(budget * store.n)
    codec = design["codec"]
    rng = channel_rng(store, "O-" + codec.upper(), budget, design)
    if codec == "bch":
        spec = choose_bch(len(logical), cap, p, harder_p)
        decoded = bch_transport(logical, cap, p, rng, spec)
    else:
        decoded = ira_transport(logical, cap, p, rng, tuple(design["rate"]))
    return {"exact": 0 if decoded is None else cuckoo_read(store, decoded, meta),
            "probes": len(store.probes), "fits": decoded is not None,
            "cells": cap if decoded is not None else 0,
            "failed_insertions": failures}


def evaluate_store(store, memory, budget, p, design, harder_p):
    if memory == "dense":
        return dense_run(store, budget, p, design)
    if memory == "sbc":
        return sbc_run(store, budget, p, design)
    if memory == "bch":
        return bch_run(store, budget, p, design, harder_p)
    return cuckoo_run(store, budget, p, design, harder_p)


def designs_for(memory):
    if memory == "dense":
        return ["dense"]
    if memory == "sbc":
        return [{"lam": lam} for lam in LAMBDAS]
    if memory == "bch":
        return ["closed-form"]
    codec = "bch" if memory == "O-BCH" else "ldpc"
    if codec == "bch":
        return [{"codec": codec, "f": f} for f in (12, 14, 16, 18)]
    return [{"codec": codec, "f": f, "rate": list(rate)}
            for f in (12, 14, 16, 18) for rate in IRA_RATES]


def pooled_evaluate(pool, stores, memory, budget, p, design, harder_p):
    # Send only a small, self-contained worker function and plain data.
    # Cached functions defined in __main__ otherwise get pickled by reference
    # and cannot be resolved by loky's worker bootstrap module.
    script_path = str(Path(__file__).resolve())
    rows = pool(delayed(isolated_store_worker)(
        script_path, dict(s.__dict__), memory, float(budget), p, design,
        harder_p) for s in stores)
    exact = sum(row["exact"] for row in rows)
    probes = sum(row["probes"] for row in rows)
    return {
        "exact": exact,
        "probes": probes,
        "share": exact / probes,
        "fits": all(row.get("fits", True) for row in rows),
        "failed_insertions": sum(row["failed_insertions"] for row in rows),
        "cells": [row["cells"] for row in rows],
    }


def requirement_index(records):
    # Require a passing suffix of ALL larger evaluated budgets.
    indices = sorted(records)
    candidate = None
    for i in reversed(indices):
        row = records[i]
        if not row.get("fits", True) or row["share"] < TARGET_SHARE:
            break
        candidate = i
    return candidate


def atomic_json(path, data):
    temporary = str(path) + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def record_budget(index, budgets, row, design):
    return {"grid_index": int(index), "budget": float(budgets[index]),
            "design": design, **row}


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    dev, test = load_data(args.smoke)
    budgets = np.asarray([512.0, 4096.0]) if args.smoke else GRID
    memories = (["dense", "sbc", "bch", "O-BCH", "O-LDPC"]
                if args.memory == "all" else
                ["O-BCH", "O-LDPC"] if args.memory == "cuckoo"
                else [args.memory])
    result = {"smoke": args.smoke, "seed": MASTER_SEED,
              "dev_stores": len(dev), "test_stores": len(test),
              "test_probe_count": sum(len(s.probes) for s in test),
              "budget_grid": budgets.tolist(), "device": {}, "memories": {}}
    # Include next harder preregistered condition for design tie-breaking,
    # even when the user requests only a subset of noise conditions.
    calibration_ks = sorted(set(args.ks) | {
        min((q for q in NOISE_LEVELS if q > k), default=k) for k in args.ks})
    for k in calibration_ks:
        print(f"Estimating device p({k:g}), 10^7 cells", flush=True)
        result["device"][str(k)] = estimate_p(k)
        atomic_json(out / "p_device.json", result["device"])
    rates = {k: result["device"][str(k)]["p"] for k in calibration_ks}
    # Process-based parallelism; BLAS and numba remain single-threaded.
    with Parallel(n_jobs=min(args.workers, 14), backend="loky",
                  max_nbytes="2M", pre_dispatch="n_jobs") as pool:
        for memory in memories:
            result["memories"][memory] = {}
            designs = designs_for(memory)
            dev_records = {k: {} for k in args.ks}
            for index, budget in enumerate(budgets):
                evaluations = {}
                needed = sorted(set(args.ks) | {
                    min((q for q in NOISE_LEVELS if q > k), default=k)
                    for k in args.ks})
                for k in needed:
                    evaluations[k] = []
                    harder = min((q for q in NOISE_LEVELS if q > k), default=k)
                    hp = rates.get(harder, rates[k])
                    for di, design in enumerate(designs):
                        row = pooled_evaluate(pool, dev, memory, budget,
                                              rates[k], design, hp)
                        evaluations[k].append(row)
                    print(f"Dev {memory} k={k:g} B={budget:.3f}", flush=True)
                for k in args.ks:
                    harder = min((q for q in NOISE_LEVELS if q > k), default=k)
                    feasible = [i for i, row in enumerate(evaluations[k])
                                if row["fits"] and evaluations[harder][i]["fits"]]
                    if feasible:
                        winner = max(feasible, key=lambda i: (
                            evaluations[k][i]["exact"],
                            evaluations[harder][i]["exact"], -i))
                        row, design = evaluations[k][winner], designs[winner]
                    else:
                        row = {"exact": 0, "probes": sum(len(s.probes) for s in dev),
                               "share": 0.0, "fits": False,
                               "failed_insertions": 0}
                        design = None
                    dev_records[k][index] = record_budget(
                        index, budgets, row, design)
                atomic_json(out / f"dev_{memory}.json",
                            {str(k): list(v.values()) for k, v in dev_records.items()})
            for k in args.ks:
                dr = dev_records[k]
                dev_index = requirement_index(dr)
                entry = {"dev": list(dr.values()),
                         "dev_requirement": "> 4096" if dev_index is None else
                         float(budgets[dev_index]), "test": []}
                result["memories"][memory][str(k)] = entry
                centre = len(budgets) - 1 if dev_index is None else dev_index
                pending = set(range(max(0, centre - 4),
                                    min(len(budgets), centre + 5)))
                tested = {}
                harder = min((q for q in NOISE_LEVELS if q > k), default=k)
                while True:
                    for index in sorted(pending - tested.keys()):
                        design = dr[index]["design"]
                        if design is None:
                            row = {"exact": 0, "probes": result["test_probe_count"],
                                   "share": 0.0, "fits": False,
                                   "failed_insertions": 0}
                        else:
                            row = pooled_evaluate(pool, test, memory, budgets[index],
                                                  rates[k], design,
                                                  rates.get(harder, rates[k]))
                        tested[index] = record_budget(index, budgets, row, design)
                        print(f"Test {memory} k={k:g} B={budgets[index]:.3f}: "
                              f"{row['exact']}/{row['probes']}", flush=True)
                        entry["test"] = [tested[i] for i in sorted(tested)]
                        atomic_json(out / "b_result.partial.json", result)
                    candidate = requirement_index(tested)
                    if candidate is not None:
                        if candidate == 0 or (candidate - 1 in tested and
                                             tested[candidate - 1]["share"] <
                                             TARGET_SHARE):
                            break
                        lo = min(tested)
                        pending.update(range(max(0, lo - 4), lo))
                    else:
                        hi = max(tested)
                        if hi == len(budgets) - 1:
                            break
                        pending.update(range(hi + 1,
                                             min(len(budgets), hi + 5)))
                entry["test_requirement"] = ("> 4096" if candidate is None
                                             else float(budgets[candidate]))
                entry["failed_insertions_by_test_budget"] = {
                    str(row["budget"]): row["failed_insertions"]
                    for row in entry["test"]}
                atomic_json(out / "b_result.partial.json", result)
    if "O-BCH" in result["memories"] and "O-LDPC" in result["memories"]:
        result["cuckoo_better"] = {}
        for k in args.ks:
            choices = [(m, result["memories"][m][str(k)]["test_requirement"])
                       for m in ("O-BCH", "O-LDPC")]
            winner = min(choices, key=lambda x: math.inf if isinstance(x[1], str)
                         else x[1])
            result["cuckoo_better"][str(k)] = {
                "memory": winner[0], "test_requirement": winner[1]}
    result["elapsed_seconds"] = time.time() - started
    atomic_json(out / "b_result.json", result)
    print(f"Finished: {out / 'b_result.json'}", flush=True)


def isolated_store_worker(script_path, store_data, memory, budget, p, design,
                          harder_p):
    """Import the standalone script as a real module inside each worker."""
    import importlib.util
    import sys

    module_name = "_part_b_worker_implementation"
    module = sys.modules.get(module_name)
    if module is None:
        spec = importlib.util.spec_from_file_location(module_name, script_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load simulation script: {script_path}")
        module = importlib.util.module_from_spec(spec)
        # Registration must precede execution for dataclasses and pickle.
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(module_name, None)
            raise

    store = module.Store(**store_data)
    return module.evaluate_store(
        store, memory, budget, p, design, harder_p)


if __name__ == "__main__":
    main()
