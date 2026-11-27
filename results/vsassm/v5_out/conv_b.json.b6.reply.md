```python
def cuckoo_repetition_read(bits, counts, p, rng, soft=False):
    """Batch copy-major channel draws without changing their cell assignment."""
    length = len(bits)
    if length == 0 or len(counts) != length or np.any(counts < 1):
        raise ValueError("invalid repetition layout")
    votes = np.zeros(length, dtype=np.int64)
    minimum = int(counts.min())
    maximum = int(counts.max())
    # repetition_counts produces only rho and rho+1. All rho complete
    # passes therefore have identical indices and can be drawn in batches.
    batch_rows = max(1, 1_048_576 // length)
    completed = 0
    while completed < minimum:
        rows = min(batch_rows, minimum - completed)
        uniforms = rng.random((rows, length))
        flips = uniforms < p
        errors = flips.sum(axis=0, dtype=np.int64)
        votes += np.where(bits != 0, rows - errors, errors)
        completed += rows
    # Preserve the original ascending-index ordering for partial passes.
    for copy in range(minimum, maximum):
        indices = np.flatnonzero(counts > copy)
        flips = rng.random(len(indices)) < p
        votes[indices] += bits[indices] ^ flips.astype(np.uint8)
    if soft:
        return (counts - 2 * votes) * math.log(
            max(1 - p, 1e-15) / max(p, 1e-15))
    return ((2 * votes > counts).astype(np.uint8),
            2 * votes == counts)


def cuckoo_bch_transport(logical, cap, p, rng, spec):
    if spec is None:
        return None
    k = spec[2]
    words = [bch_encode(logical[i:i + k], spec)
             for i in range(0, len(logical), k)]
    encoded = np.concatenate(words)
    counts = repetition_counts(len(encoded), cap)
    if counts is None:
        return None
    received, erased = cuckoo_repetition_read(encoded, counts, p, rng)
    output = np.empty(len(logical), dtype=np.uint8)
    offset = 0
    information_offset = 0
    for word in words:
        end = offset + len(word)
        information_length = len(word) - (spec[0] - spec[2])
        destination = slice(information_offset,
                            information_offset + information_length)
        if np.array_equal(received[offset:end], word):
            output[destination] = logical[destination]
        else:
            output[destination] = bch_decode(
                received[offset:end], erased[offset:end], spec)
        offset = end
        information_offset += information_length
    return output


def cuckoo_ira_transport(logical, cap, p, rng, rate):
    words = [ira_encode(logical[i:i + 1024], rate)
             for i in range(0, len(logical), 1024)]
    encoded = np.concatenate(words)
    counts = repetition_counts(len(encoded), cap)
    if counts is None:
        return None
    llr = cuckoo_repetition_read(
        encoded, counts, p, rng, soft=True).astype(np.float64)
    output = np.empty(len(logical), dtype=np.uint8)
    offset = 0
    for i, word in enumerate(words):
        k = min(1024, len(logical) - 1024 * i)
        channel = llr[offset:offset + len(word)]
        hard = (channel < 0).astype(np.uint8)
        variables, _, order, pointers, _ = ira_graph(k, *rate)
        if (np.all(channel != 0) and
                ira_valid_word(hard, variables, order, pointers)):
            output[i * 1024:i * 1024 + k] = hard[:k]
        else:
            output[i * 1024:i * 1024 + k] = ira_bp(
                channel, variables, order, pointers)[:k]
        offset += len(word)
    return output
```

Batches cuckoo repetition draws, preserving their original ordering and votes, with bounded temporary memory.
Other memory paths are unchanged.
**I cannot honestly claim this meets the 30-minute limit.** It leaves BCH design/decoding and LDPC sweep costs unresolved; the earlier patches were insufficient, and this is not a verified final runtime solution.