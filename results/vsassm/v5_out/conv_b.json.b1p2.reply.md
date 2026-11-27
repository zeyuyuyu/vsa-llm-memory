```python
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
```