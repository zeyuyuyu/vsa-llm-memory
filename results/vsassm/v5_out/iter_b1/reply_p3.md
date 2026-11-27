```python
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
    rows = pool(delayed(evaluate_store)(
        s, memory, float(budget), p, design, harder_p) for s in stores)
    exact = sum(row["exact"] for row in rows)
    probes = sum(row["probes"] for row in rows)
    return {"exact": exact, "probes": probes, "share": exact / probes,
            "fits": all(row.get("fits", True) for row in rows),
            "failed_insertions": sum(row["failed_insertions"] for row in rows),
            "cells": [row["cells"] for row in rows]}


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


if __name__ == "__main__":
    main()
```

- XOR cuckoo hashing uses the smallest admissible power-of-two bucket count.
- BCH selection uses a conservative codeword-failure expectation.
- IRA neighbours are independently sampled without replacement.
- BP clips messages and returns current hard decisions after unsuccessful decoding.
- Smoke results are diagnostic, not full-study requirements.