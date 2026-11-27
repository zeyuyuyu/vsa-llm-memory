```python
def cuckoo_bch_transport(logical, cap, p, rng, spec):
    """Decode only words whose received information could differ.

    An unchanged received codeword returns the original information under the
    specified decoder, including when there are erasures: either bounded-distance
    decoding returns that codeword or fallback returns the same received bits.
    """
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
    output = np.empty(len(logical), dtype=np.uint8)
    offset = 0
    information_offset = 0
    for word in words:
        end = offset + len(word)
        information_length = len(word) - (spec[0] - spec[2])
        if np.array_equal(received[offset:end], word):
            output[information_offset:
                   information_offset + information_length] = logical[
                       information_offset:
                       information_offset + information_length]
        else:
            output[information_offset:
                   information_offset + information_length] = bch_decode(
                       received[offset:end], erased[offset:end], spec)
        offset = end
        information_offset += information_length
    return output


def cuckoo_ira_transport(logical, cap, p, rng, rate):
    """Use an exact BP fast path when channel hard decisions form a codeword."""
    words = [ira_encode(logical[i:i + 1024], rate)
             for i in range(0, len(logical), 1024)]
    encoded = np.concatenate(words)
    counts = repetition_counts(len(encoded), cap)
    if counts is None:
        return None
    llr = repetition_read(encoded, counts, p, rng, soft=True).astype(np.float64)
    output = np.empty(len(logical), dtype=np.uint8)
    offset = 0
    for i, word in enumerate(words):
        k = min(1024, len(logical) - 1024 * i)
        channel = llr[offset:offset + len(word)]
        hard = (channel < 0).astype(np.uint8)
        variables, _, order, pointers, _ = ira_graph(k, *rate)
        # Only strictly nonzero LLRs qualify: otherwise first-iteration
        # check messages can resolve zero ties differently.
        if (np.all(channel != 0) and
                ira_valid_word(hard, variables, order, pointers)):
            # Every check-to-variable message reinforces the channel sign
            # when the initial hard word satisfies all checks. The original
            # flooding decoder therefore returns this same word at iteration 1.
            output[i * 1024:i * 1024 + k] = hard[:k]
        else:
            output[i * 1024:i * 1024 + k] = ira_bp(
                channel, variables, order, pointers)[:k]
        offset += len(word)
    return output


@njit(cache=True)
def ira_valid_word(hard, variables, order, pointers):
    for c in range(len(pointers) - 1):
        parity = 0
        for j in range(pointers[c], pointers[c + 1]):
            parity ^= hard[variables[order[j]]]
        if parity:
            return False
    return True


@njit(cache=True)
def cuckoo_exact_kernel(rows, fingerprints, values, addresses, offsets,
                        truth_flat, truth_offsets, probes, maxj, flag_column):
    exact = 0
    for probe_number in range(len(probes)):
        ki = probes[probe_number]
        start, end = truth_offsets[ki], truth_offsets[ki + 1]
        valid = True
        for j in range(maxj + 1):
            if j >= end - start:
                valid = False
                break
            fp = addresses[offsets[probe_number] + j, 0]
            a = addresses[offsets[probe_number] + j, 1]
            b = addresses[offsets[probe_number] + j, 2]
            matched = -1
            count = 0
            for s in range(4):
                loc = 4 * a + s
                if fingerprints[loc] == fp:
                    matched = loc
                    count += 1
            if b != a:
                for s in range(4):
                    loc = 4 * b + s
                    if fingerprints[loc] == fp:
                        matched = loc
                        count += 1
            if count != 1 or values[matched] != truth_flat[start + j]:
                valid = False
                break
            if flag_column >= 0 and rows[matched, flag_column]:
                if j + 1 != end - start:
                    valid = False
                break
            if j + 1 == end - start and j < maxj:
                valid = False
                break
        if valid:
            exact += 1
    return exact


def cuckoo_query_plan(store, f, buckets):
    # Keep a bounded worker-local cache; never cache noisy read outcomes.
    cache = getattr(cuckoo_query_plan, "_cache", None)
    if cache is None:
        from collections import OrderedDict
        cache = OrderedDict()
        cuckoo_query_plan._cache = cache
    key = (store.sid, f, buckets)
    if key in cache:
        cache.move_to_end(key)
        return cache[key]
    addresses = []
    offsets = [0]
    for ki in store.probes:
        # Positions beyond the true output length cannot restore exact recall;
        # no address needs to be generated for them.
        for j in range(len(store.truth[ki])):
            addresses.append(cuckoo_address((store.keys[ki], j), f, buckets))
        offsets.append(len(addresses))
    truth_offsets = np.concatenate((
        np.array([0], dtype=np.int64),
        np.cumsum([len(v) for v in store.truth], dtype=np.int64)))
    truth_flat = np.asarray(
        [v for row in store.truth for v in row], dtype=np.int32)
    plan = (np.asarray(addresses, dtype=np.int64),
            np.asarray(offsets, dtype=np.int64), truth_flat, truth_offsets)
    cache[key] = plan
    if len(cache) > 128:
        cache.popitem(last=False)
    return plan


def cuckoo_read(store, decoded, meta):
    buckets, f, vw, flag = meta
    rows = decoded.reshape(buckets * 4, f + vw + flag)
    fingerprints = bits_uint(rows[:, :f].reshape(-1), f)
    values = bits_uint(rows[:, f:f + vw].reshape(-1), vw)
    addresses, offsets, truth_flat, truth_offsets = cuckoo_query_plan(
        store, f, buckets)
    return int(cuckoo_exact_kernel(
        rows, fingerprints, values, addresses, offsets,
        truth_flat, truth_offsets, store.probes, store.maxj,
        f + vw if flag else -1))


def cuckoo_run(store, budget, p, design, harder_p=None):
    f = int(design["f"])
    logical, meta, failures = cuckoo_build(store, f)
    cap = math.floor(budget * store.n)
    codec = design["codec"]
    rng = channel_rng(store, "O-" + codec.upper(), budget, design)
    if codec == "bch":
        spec = choose_bch(len(logical), cap, p, harder_p)
        decoded = cuckoo_bch_transport(logical, cap, p, rng, spec)
    else:
        decoded = cuckoo_ira_transport(
            logical, cap, p, rng, tuple(design["rate"]))
    return {
        "exact": 0 if decoded is None else cuckoo_read(store, decoded, meta),
        "probes": len(store.probes),
        "fits": decoded is not None,
        "cells": cap if decoded is not None else 0,
        "failed_insertions": failures,
    }
```

Adds exact fast paths for unchanged BCH words and already-valid LDPC channel decisions.
Compiles cuckoo lookups and caches their hash addresses, stopping only when exact recall is impossible.
Dense, sparse, static BCH, design selection, and channel draws are unchanged.
These optimizations do **not** establish the requested 35-minute bound; noisy-word decoding remains a bottleneck.