```python
@lru_cache(maxsize=32768)
def majority_rates(rho, p):
    erasure = float(binom.pmf(rho // 2, rho, p)) if rho % 2 == 0 else 0.0
    error = float(binom.sf(rho // 2, rho, p))
    return error, erasure


@lru_cache(maxsize=131072)
def word_failure(length, t, error, erasure):
    if error + erasure == 0:
        return 0.0
    s = np.arange(min(2 * t, length) + 1)
    conditional = error / max(1 - erasure, 1e-300)
    success = np.sum(
        binom.pmf(s, length, erasure) *
        binom.cdf((2 * t - s) // 2, length - s, conditional))
    return float(np.clip(1 - success, 0, 1))


def choose_bch(logical_length, cap, p, harder_p=None, touches=None):
    """Evaluate every admissible strength, reusing equal full-word lengths."""
    if logical_length < 1:
        raise ValueError("BCH logical length must be positive")
    hp = p if harder_p is None else harder_p
    best, answer = None, None
    # All but the final shortened word have the same length and failure rate.
    # Collapse touched-word lists into (full-word count, touches-final-word).
    for n in (63, 127, 255, 511, 1023):
        for spec in bch_specs(n):
            _, t, k, _ = spec
            word_count = (logical_length + k - 1) // k
            final_info = logical_length - (word_count - 1) * k
            final_length = final_info + n - k
            total = (word_count - 1) * n + final_length
            if total > cap:
                continue
            rho, extra = divmod(cap, total)
            weight = extra / total
            if touches is not None:
                full_counts = np.empty(len(touches), dtype=np.int32)
                final_touched = np.empty(len(touches), dtype=np.bool_)
                for i, indices in enumerate(touches):
                    words = np.unique(
                        np.asarray(indices, dtype=np.int64) // k)
                    is_final = bool(len(words) and words[-1] == word_count - 1)
                    full_counts[i] = len(words) - int(is_final)
                    final_touched[i] = is_final

            def objective(rate):
                e0, s0 = majority_rates(rho, rate)
                e1, s1 = majority_rates(rho + 1, rate)
                error = (1 - weight) * e0 + weight * e1
                erasure = (1 - weight) * s0 + weight * s1
                full_success = 1 - word_failure(n, t, error, erasure)
                final_success = 1 - word_failure(
                    final_length, t, error, erasure)
                if touches is None:
                    return float(1 - full_success ** (word_count - 1) *
                                 final_success)
                successes = np.power(full_success, full_counts)
                successes[final_touched] *= final_success
                return float(np.sum(1 - successes))

            primary = objective(p)
            # A secondary objective cannot change a strictly worse primary.
            if best is not None and primary > best[0]:
                continue
            secondary = primary if hp == p else objective(hp)
            score = (primary, secondary, n, t)
            if best is None or score < best:
                best, answer = score, spec
    return answer


@njit(cache=True)
def bch_encode_kernel(info, generator, redundancy):
    word = np.zeros(redundancy + len(info), dtype=np.uint8)
    word[redundancy:] = info
    work = word.copy()
    for i in range(len(work) - 1, redundancy - 1, -1):
        if work[i]:
            for j in range(len(generator)):
                work[i - redundancy + j] ^= generator[j]
    word[:redundancy] = work[:redundancy]
    return word


def bch_encode(info, spec):
    n, t, k, generator = spec
    info = np.asarray(info, dtype=np.uint8)
    if not 0 < len(info) <= k:
        raise ValueError("invalid shortened BCH information length")
    return bch_encode_kernel(
        info, np.asarray(generator, dtype=np.uint8), n - k)


@njit(cache=True)
def bch_locator_roots(locator, length, exp, log):
    n = len(exp) // 2
    result = np.empty(length, dtype=np.int32)
    count = 0
    for pos in range(length):
        x = exp[(-pos) % n]
        value = 0
        for j in range(len(locator) - 1, -1, -1):
            if value and x:
                value = exp[log[value] + log[x]]
            else:
                value = 0
            value ^= locator[j]
        if value == 0:
            result[count] = pos
            count += 1
    return result[:count]


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
    positions = bch_locator_roots(
        np.asarray(locator, dtype=np.int32), len(raw), exp, log)
    v = len(positions)
    if v != len(locator) - 1 or not set(erasures).issubset(positions):
        return fallback
    if not len(erasures):
        # Binary error magnitudes are necessarily one: no GF linear solve
        # is needed in the overwhelmingly common no-erasure case.
        raw[positions] ^= 1
    else:
        matrix = [
            [int(exp[((row + 1) * int(pos)) % n]) for pos in positions] +
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


def install_worker_layout_cache(module):
    """Cache deterministic layouts, not channel outcomes, inside each process."""
    from collections import OrderedDict

    def cached_layout(function, maximum):
        cache = OrderedDict()

        def wrapped(store, *args):
            key = (store.sid, *args)
            if key in cache:
                cache.move_to_end(key)
                return cache[key]
            result = function(store, *args)
            cache[key] = result
            if len(cache) > maximum:
                cache.popitem(last=False)
            return result

        return wrapped

    module.static_layout = cached_layout(module.static_layout, 64)
    module.cuckoo_build = cached_layout(module.cuckoo_build, 128)


def isolated_store_worker(script_path, store_data, memory, budget, p, design,
                          harder_p):
    import importlib.util
    import sys

    module_name = "_part_b_worker_implementation"
    module = sys.modules.get(module_name)
    if module is None:
        spec = importlib.util.spec_from_file_location(module_name, script_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load simulation script: {script_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
            module.install_worker_layout_cache(module)
        except BaseException:
            sys.modules.pop(module_name, None)
            raise
    store = module.Store(**store_data)
    return module.evaluate_store(store, memory, budget, p, design, harder_p)
```
Caches deterministic MPH/cuckoo layouts and repeated probability calculations.
Collapses BCH objectives to full-word and final-shortened-word contributions; all strengths remain evaluated.
Compiles BCH encoding and root search; avoids unnecessary GF solves without erasures.
Dense and sparse code is unchanged. A 40-minute bound remains unverified; this patch does not resolve every bottleneck, particularly LDPC decoding.