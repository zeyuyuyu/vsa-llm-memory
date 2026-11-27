```python


def make_stores(m, v, split):
    count = (1 if args.smoke else
             ((6 if m == 256 else 3) if split == "dev"
              else (20 if m == 256 else 5)))
    stores = []
    for number in range(count):
        generator = rng("stores", m, v, split, number)
        ids = generator.choice(splits[split], m, replace=False)
        values = generator.integers(v, size=m, dtype=np.int32)
        query_rows, owners = [], []
        for j, entity in enumerate(ids):
            query_rows.extend(aliases[entity].tolist())
            owners.extend([j] * len(aliases[entity]))
        stores.append(dict(
            ids=ids, values=values,
            rows=np.array(query_rows, np.int32),
            owners=np.array(owners, np.int32),
            tag=(m, v, split, number)))
    return stores


def evaluate_store(store, v, budget, selected=None, details=False):
    values = store["values"]
    owners = store["owners"]
    truth = values[owners]
    m = len(values)
    order = rng("online-order", *store["tag"]).permutation(m)
    majority = int(np.argmax(np.bincount(values, minlength=v)))
    answers = {}
    accuracies = {}

    @lru_cache(maxsize=2)
    def get_features(prep, family, p):
        mat = features[(prep, family)]
        f = np.array(
            mat[labels[store["ids"]], :p], dtype=np.float32, order="C")
        query = np.array(
            mat[store["rows"], :p], dtype=np.float32, order="C")
        return f, query

    @lru_cache(maxsize=2)
    def targets(target):
        t = np.eye(v, dtype=np.float32)[values] - np.float32(1 / v)
        return t[:, 1:] if target == "V-1" else t

    @lru_cache(maxsize=2)
    def ridge_base(prep, family, p, target):
        f, _ = get_features(prep, family, p)
        gram = f @ f.T
        return gram, float(np.trace(gram) / m)

    @lru_cache(maxsize=2)
    def weight(prep, family, p, target, method, strength, output):
        f, _ = get_features(prep, family, p)
        t = targets(target)
        if method == "LAMc":
            gram, average_diag = ridge_base(prep, family, p, target)
            a = gram.astype(np.float64)
            a.flat[::m + 1] += strength * average_diag
            try:
                dual = cho_solve(
                    cho_factor(a, check_finite=False),
                    t, check_finite=False)
            except np.linalg.LinAlgError:
                dual = np.linalg.lstsq(a, t, rcond=None)[0]
            return (f.T @ dual).astype(np.float32)
        return online(
            f, np.ascontiguousarray(t), order, strength, output == "avg")

    for memory, full_grid in grids.items():
        grid = full_grid if selected is None else [selected[memory]]
        scores = []
        memory_answers = []
        for config in grid:
            q = config["q"]
            if memory == "STORE":
                d = dimension(budget, 1, q, "flat", 65536)
                f, query = get_features(config["prep"], config["map"], d)
                generator = rng(
                    args.enc, config["prep"], config["map"], "value-code")
                r = generator.choice(
                    np.array([-1, 1], np.float32), size=d)
                if v == 2:
                    codes = np.stack((r, -r))
                else:
                    codes = (
                        hadamard(16).astype(np.float32)[
                            :, np.arange(d) % 16] * r)
                subset = rng("share", *store["tag"]).permutation(m)[
                    :max(1, int(config["share"] * m))]
                s = np.sum(f[subset] * codes[values[subset]], axis=0)
                s = quantize(
                    s, q, "flat",
                    rng(args.enc, "store-quant", *store["tag"],
                        config["prep"], config["map"], q),
                    store=True)
                prediction = np.argmax(query @ (codes * s).T, axis=1)
                used = bits_used(d, 1, q, "flat")
                dim = d
            else:
                c = v - 1 if config["target"] == "V-1" else v
                p = dimension(
                    budget, c, q, config["scale"],
                    n if config["feat"] == "pca" else 65536)
                used = bits_used(p, c, q, config["scale"])
                dim = p
                if p == 0:
                    prediction = np.full(len(truth), majority, np.int32)
                else:
                    strength = (config["alpha"] if memory == "LAMc"
                                else config["mu"])
                    output = config.get("output", "last")
                    w = weight(
                        config["prep"], config["feat"], p,
                        config["target"], memory, strength, output)
                    wq = quantize(
                        w, q, config["scale"],
                        rng(args.enc, memory, "quant", *store["tag"],
                            json.dumps(config, sort_keys=True)))
                    _, query = get_features(
                        config["prep"], config["feat"], p)
                    score = query @ wq
                    if config["target"] == "V-1":
                        score = np.column_stack(
                            (-score.sum(axis=1), score))
                    prediction = np.argmax(score, axis=1)

            correct = (prediction == truth).astype(np.int32)
            scores.append(
                float(correct.mean()) if len(correct) else 0.0)
            if details:
                counts = np.bincount(owners, minlength=m)
                successes = np.bincount(
                    owners, weights=correct, minlength=m)
                memory_answers.append(dict(
                    counts=counts, successes=successes,
                    bits_used=float(used), dimension=dim))
        accuracies[memory] = scores
        if details:
            answers[memory] = memory_answers[0]

    # These nested caches reference each other. Explicitly clear them so
    # completed jobs do not retain large matrices until cyclic GC runs.
    weight.cache_clear()
    ridge_base.cache_clear()
    get_features.cache_clear()
    targets.cache_clear()
    return accuracies, answers


# Compile the online kernel before worker threads start.
online(
    np.ones((1, 1), np.float32),
    np.ones((1, 1), np.float32),
    np.array([0], np.int64), 1., False)

results = []
clusters = {}
parallel = Parallel(
    n_jobs=args.workers, prefer="threads", batch_size=1)

for m, v, b in conditions:
    progress(f"Condition M={m}, V={v}, b={b}: development search")
    dev = make_stores(m, v, "dev")
    # BLAS is already permanently single-threaded. No threadpool context
    # here or elsewhere in evaluation, so thread counts cannot be restored.
    dev_runs = parallel(
        delayed(evaluate_store)(store, v, m * b) for store in dev)

    chosen = {}
    entry = dict(M=m, V=v, b=b, budget_bits=m * b, memories={})
    for memory in grids:
        means = np.mean([run[0][memory] for run in dev_runs], axis=0)
        best = float(means.max())
        index = int(np.flatnonzero(
            means >= best - .001 - 1e-12)[0])
        chosen[memory] = grids[memory][index]
        entry["memories"][memory] = dict(
            configuration=chosen[memory],
            mean_dev_accuracy=float(means[index]))
        progress(
            f"  {memory}: dev={means[index]:.5f}, {chosen[memory]}")

    test = make_stores(m, v, "test")
    progress("  Selected configurations: test")
    test_runs = parallel(
        delayed(evaluate_store)(
            store, v, m * b, chosen, True)
        for store in test)

    condition_clusters = {}
    for memory in grids:
        counts = np.concatenate([
            run[1][memory]["counts"] for run in test_runs])
        success = np.concatenate([
            run[1][memory]["successes"] for run in test_runs])
        total = int(counts.sum())
        info = entry["memories"][memory]
        info.update(
            test_accuracy=(
                float(success.sum() / total) if total else 0.),
            number_test_queries=total,
            bits_used_by_test_store=[
                run[1][memory]["bits_used"] for run in test_runs],
            dimensions_by_test_store=[
                run[1][memory]["dimension"] for run in test_runs])
        if any(
                used > m * b + 1e-6
                for used in info["bits_used_by_test_store"]):
            raise RuntimeError("Budget exceeded")
        condition_clusters[memory] = (counts, success)

    clusters[(m, v, b)] = condition_clusters
    results.append(entry)
    atomic_json(
        OUT / f"{args.enc}_partial.json",
        dict(encoder=args.enc, smoke=args.smoke, conditions=results))

    # Cluster array positions correspond to (store number, entity position).
    # Save IDs as well, making these positions independently auditable.
    np.savez_compressed(
        OUT / f"{args.enc}_clusters_M{m}_V{v}_b{b}.npz",
        entity_ids=np.concatenate([store["ids"] for store in test]),
        store_indices=np.repeat(np.arange(len(test)), m),
        **{
            f"{memory}_{field}": condition_clusters[memory][j]
            for memory in grids
            for j, field in enumerate(("queries", "correct"))
        })
    del dev_runs, test_runs
    gc.collect()

progress("Paired entity-cluster bootstrap (2000 draws)")
pairs = [
    ("STORE", "LAMc"),
    ("STORE", "NLMSc"),
    ("NLMSc", "LAMc"),
]
bootstrap = {pair: {} for pair in pairs}
point = {pair: {} for pair in pairs}

for condition, data in clusters.items():
    counts = data["STORE"][0]
    number = len(counts)
    for memory in grids:
        if not np.array_equal(counts, data[memory][0]):
            raise RuntimeError("Unpaired test query clusters")
    draws = np.empty((2000, len(pairs)), np.float64)
    generator = rng(args.enc, "bootstrap", *condition)

    # Batch draws to avoid large bootstrap index arrays.
    for begin in range(0, 2000, 50):
        end = min(begin + 50, 2000)
        indices = generator.integers(
            number, size=(end - begin, number))
        denominators = counts[indices].sum(axis=1)
        for j, (a, c) in enumerate(pairs):
            difference = data[a][1] - data[c][1]
            numerator = difference[indices].sum(axis=1)
            draws[begin:end, j] = np.divide(
                numerator, denominators,
                out=np.zeros(end - begin),
                where=denominators > 0)

    for j, pair in enumerate(pairs):
        bootstrap[pair][condition] = draws[:, j]
        difference = data[pair[0]][1] - data[pair[1]][1]
        point[pair][condition] = (
            float(difference.sum() / counts.sum())
            if counts.sum() else 0.)

comparisons = {}
for pair in pairs:
    summary = {}
    for name, value in (
            ("region", None), ("V2", 2), ("V16", 16)):
        subset = [
            condition for condition in conditions
            if value is None or condition[1] == value]
        if not subset:
            summary[name] = None
            continue
        draws = np.mean(
            [bootstrap[pair][condition] for condition in subset],
            axis=0)
        summary[name] = dict(
            mean_accuracy_difference=float(np.mean([
                point[pair][condition] for condition in subset])),
            ci95=[
                float(x) for x in np.quantile(draws, [.025, .975])],
            number_conditions=len(subset),
            bootstrap_draws=2000)
    comparisons[f"{pair[0]} - {pair[1]}"] = summary

output = dict(
    encoder=args.enc,
    smoke=args.smoke,
    conditions=results,
    comparisons=comparisons,
    elapsed_seconds=time.time() - START,
    accounting="quantized Gaussian entropy plus 32-bit scale groups",
    accuracy_units="fraction",
    seed_scheme="SHA256 equal-bit-A-v1",
    training_occurrences=len(train_rows),
    embedding_dimension_used=n,
    grids={memory: len(grid) for memory, grid in grids.items()},
    bootstrap_clusters=(
        "paired (store, entity), including zero-query entities"),
    blas_thread_policy=(
        "initialize at workers; reduce once to one after preprocessing; "
        "never raise runtime limits"),
)
atomic_json(OUT / f"{args.enc}_result.json", output)
progress(f"Finished: {OUT / (args.enc + '_result.json')}")
```