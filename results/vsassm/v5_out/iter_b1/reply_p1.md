```python
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
```