```python
@njit(cache=True)
def dense_exact_reads(cells, ids, kc, vc, pc, last, probes, truth_flat,
                      truth_offsets, maxj):
    """Score exhaustively, but stop a query once exact recall is impossible."""
    d = cells.shape[1]
    exact = 0
    query = np.empty(d, dtype=np.int8)
    for ki in probes:
        start, end = truth_offsets[ki], truth_offsets[ki + 1]
        valid = True
        stopped = False
        for j in range(maxj + 1):
            if j >= end - start:
                valid = False
                break
            for component in range(d):
                query[component] = (
                    cells[ids[ki], component] * kc[ki, component] *
                    pc[j, component])
            best = -2147483647
            winner = 0
            marked = False
            # Plain candidates precede marked candidates; strict updates
            # preserve the original first-maximum tie rule.
            for flag in range(2):
                for vi in range(vc.shape[0]):
                    score = 0
                    for component in range(d):
                        term = query[component] * vc[vi, component]
                        if flag:
                            term *= last[component]
                        score += term
                    if score > best:
                        best = score
                        winner = vi
                        marked = flag == 1
            if winner != truth_flat[start + j]:
                valid = False
                break
            if marked:
                stopped = True
                if j + 1 != end - start:
                    valid = False
                break
            if j + 1 == end - start and j < maxj:
                # A longer output cannot become equal to the truth.
                valid = False
                break
        # At maxj, the specified read returns its output even without LAST.
        if valid:
            exact += 1
    return exact


@njit(cache=True)
def sparse_exact_reads(cells, ids, kc, vc, pc, last, probes, truth_flat,
                       truth_offsets, maxj, length):
    blocks = kc.shape[1]
    exact = 0
    for ki in probes:
        start, end = truth_offsets[ki], truth_offsets[ki + 1]
        valid = True
        for j in range(maxj + 1):
            if j >= end - start:
                valid = False
                break
            best, winner, marked = -1, 0, False
            for flag in range(2):
                for vi in range(vc.shape[0]):
                    score = 0
                    for b in range(blocks):
                        pos = kc[ki, b] + pc[j, b] + vc[vi, b]
                        if flag:
                            pos += last[b]
                        score += cells[ids[ki], b * length + pos % length]
                    if score > best:
                        best, winner, marked = score, vi, flag == 1
            if winner != truth_flat[start + j]:
                valid = False
                break
            if marked:
                if j + 1 != end - start:
                    valid = False
                break
            if j + 1 == end - start and j < maxj:
                valid = False
                break
        if valid:
            exact += 1
    return exact


def packed_truth(store):
    offsets = np.concatenate((
        np.array([0], dtype=np.int64),
        np.cumsum([len(v) for v in store.truth], dtype=np.int64)))
    flat = np.asarray([v for row in store.truth for v in row], dtype=np.int32)
    return flat, offsets


def dense_run(store, budget, p, design="dense"):
    shards, d, cap = geometry(store.n, budget)
    ids = shard_ids(store, shards)
    kc = np.asarray([bipolar("key", k, d) for k in store.keys])
    vc = np.asarray([bipolar("value", v, d) for v in store.values])
    pc = np.asarray([bipolar("position", j, d)
                     for j in range(store.maxj + 1)])
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
    flat, offsets = packed_truth(store)
    exact = dense_exact_reads(cells, ids, kc, vc, pc, last, store.probes,
                              flat, offsets, store.maxj)
    return {"exact": int(exact), "probes": len(store.probes),
            "cells": shards * d, "failed_insertions": 0}


def sbc_run(store, budget, p, design):
    lam = float(design["lam"] if isinstance(design, dict) else design)
    shards, d, cap = geometry(store.n, budget)
    length = max(2, round((store.n / shards) / lam))
    blocks = d // length
    ids = shard_ids(store, shards)
    kc = np.asarray([block_code("key", k, blocks, length) for k in store.keys])
    vc = np.asarray([block_code("value", v, blocks, length)
                     for v in store.values])
    pc = np.asarray([block_code("position", j, blocks, length)
                     for j in range(store.maxj + 1)])
    last = block_code("LAST", "", blocks, length)
    cells = np.zeros((shards, blocks * length), dtype=np.uint8)
    offsets = np.arange(blocks, dtype=np.int64) * length
    for ki, vi, j, flag in zip(store.pair_key, store.pair_value,
                               store.pair_j, store.last):
        pos = (kc[ki] + pc[j] + vc[vi] + (last if flag else 0)) % length
        cells[ids[ki], offsets + pos] = 1
    cells = flip_cells(cells, p,
                       channel_rng(store, "sbc", budget, {"lam": lam}))
    flat, truth_offsets = packed_truth(store)
    exact = sparse_exact_reads(cells, ids, kc, vc, pc, last, store.probes,
                               flat, truth_offsets, store.maxj, length)
    return {"exact": int(exact), "probes": len(store.probes),
            "cells": shards * d, "failed_insertions": 0}
```

Compiles dense and sparse scoring with Numba and stops queries once exact recall is impossible, preserving exact-key counts and channel draws.
This addresses the observed dense bottleneck, but **does not establish a 45-minute full-run bound**.
The BCH design search and Python BCH decoder remain substantial bottlenecks; claiming this patch meets the requested total runtime would be unsupported.