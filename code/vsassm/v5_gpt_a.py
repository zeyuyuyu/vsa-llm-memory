#!/usr/bin/env python3
"""
Equal-bit fact memories, CPU-only, Python 3.10.

Choices where the specification is open:
* Seeds are deterministic SHA-256-derived integers; stores are shared across memories.
* Random projections have iid N(0,1/n) entries and dimension-consistent prefixes.
* V-1 targets are columns 1,...,V-1 of centred one-hot; value 0 is scored
  as minus their sum.
* Training statistics count name/alias occurrences, without deduplication.
* Dev accuracy is the unweighted mean of store accuracies; empty-query stores
  contribute zero. Test accuracy pools queries.
* Population covariance and population standard deviations are used.
* One-bit STORE amplitudes are 1; scale charges are still included.
* Zero-variance multi-bit groups reconstruct as zero.
* Bootstrap resamples paired (store, entity) clusters, including entities
  without aliases, within each condition.
* Smoke uses the first 128 embedding coordinates, condition (256,2,4),
  and reduced configuration grids; it is only a diagnostic.
* Disk-backed feature caches are written beneath --out/cache.

BLAS safety fix:
BLAS starts with --workers threads for sequential preprocessing. Thread limits
are subsequently reduced to one, once, before parallel store evaluation.
They are NEVER raised at runtime. In particular, scipy.linalg.eigh is not
enclosed in a threadpool_limits context: raising an initially single-threaded
OpenBLAS pool crashes on the target machine.
"""
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--enc", required=True, choices=["qwen8", "llama"])
parser.add_argument("--out", required=True)
parser.add_argument("--workers", type=int, default=14)
parser.add_argument("--smoke", action="store_true")
args = parser.parse_args()
if args.workers < 1:
    parser.error("--workers must be positive")

# Set the initial pool size BEFORE importing numerical libraries.
# Do not initialize at one and later raise it with threadpoolctl.
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "MKL_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[variable] = str(args.workers)
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import gc
import json
import math
import time
import hashlib
from pathlib import Path
from functools import lru_cache

import numpy as np
from scipy.linalg import eigh, cho_factor, cho_solve, hadamard
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits
from numba import njit

DATA = Path("/path/to/vsassm/b49_out/b32_data")
OUT = Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)
CACHE = OUT / "cache"
CACHE.mkdir(exist_ok=True)
H = {1: 1.0, 2: 1.9037, 4: 3.6024}
STEPS = {2: 0.9957, 4: 0.3352}
BLOCK = 8192
START = time.time()


def progress(message):
    print(f"[{(time.time()-START)/60:.1f} min] {message}", flush=True)


def seed(*parts):
    text = "|".join(map(str, ("equal-bit-A-v1",) + parts))
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little")


def rng(*parts):
    return np.random.default_rng(seed(*parts))


def atomic_json(path, obj):
    tmp = Path(str(path) + ".tmp")
    with tmp.open("w") as handle:
        json.dump(obj, handle, indent=2, allow_nan=False)
    os.replace(tmp, path)


def bits_used(p, c, q, scale):
    groups = (c * math.ceil(p / BLOCK) if scale == "col"
              else math.ceil(p * c / BLOCK))
    return p * c * H[q] + 32 * groups


def dimension(budget, c, q, scale, cap):
    lo, hi = 0, cap
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if bits_used(mid, c, q, scale) <= budget + 1e-9:
            lo = mid
        else:
            hi = mid - 1
    return lo


def normalize(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-20)


def quant_group(x, q, generator, store=False):
    if q == 1:
        signs = np.where(x > 0, 1.0, -1.0).astype(np.float32)
        zero = x == 0
        if np.any(zero):
            signs[zero] = generator.choice(
                np.array([-1, 1], np.float32), size=int(zero.sum()))
        amplitude = (1.0 if store
                     else float(np.mean(np.abs(x), dtype=np.float64)))
        return signs * amplitude
    step = STEPS[q] * float(np.std(x, dtype=np.float64))
    if step <= 0:
        return np.zeros_like(x)
    k = np.clip(np.floor(x / step), -(2 ** q) // 2,
                (2 ** q) // 2 - 1)
    return ((k + 0.5) * step).astype(np.float32)


def quantize(w, q, scale, generator, store=False):
    if store:
        out = np.empty_like(w)
        for i in range(0, len(w), BLOCK):
            out[i:i + BLOCK] = quant_group(
                w[i:i + BLOCK], q, generator, True)
        return out
    if scale == "col":
        out = np.empty_like(w)
        for j in range(w.shape[1]):
            for i in range(0, w.shape[0], BLOCK):
                out[i:i + BLOCK, j] = quant_group(
                    w[i:i + BLOCK, j], q, generator)
        return out
    flat = w.ravel(order="F")
    result = np.empty_like(flat)
    for i in range(0, len(flat), BLOCK):
        result[i:i + BLOCK] = quant_group(
            flat[i:i + BLOCK], q, generator)
    return result.reshape(w.shape, order="F")


@njit(cache=False)
def online(f, target, order, mu, average):
    m, p = f.shape
    c = target.shape[1]
    w = np.zeros((p, c), np.float32)
    acc = np.zeros((p, c), np.float32)
    error = np.empty(c, np.float32)
    for index in order:
        norm = 1e-12
        for j in range(p):
            norm += f[index, j] * f[index, j]
        for k in range(c):
            pred = 0.0
            for j in range(p):
                pred += f[index, j] * w[j, k]
            error[k] = (target[index, k] - pred) * mu / norm
        for j in range(p):
            for k in range(c):
                w[j, k] += f[index, j] * error[k]
                if average:
                    acc[j, k] += w[j, k]
    if average:
        return acc / m
    return w


with (DATA / "entities.json").open() as handle:
    entities = json.load(handle)["entities"]
with (DATA / "emb_strings.json").open() as handle:
    strings = json.load(handle)
lookup = {s: i for i, s in enumerate(strings)}
if len(lookup) != len(strings):
    raise ValueError("Embedding strings are not distinct")

filename = ("emb_qwen8_mean_L9_f16.npy" if args.enc == "qwen8"
            else "emb_llama31i_mean_L24_f16.npy")
raw = np.load(DATA / filename, mmap_mode="r")
if raw.shape != (len(strings), 4096):
    raise ValueError(f"Unexpected embedding shape {raw.shape}")
if raw.dtype != np.float16:
    raise ValueError(f"Unexpected embedding dtype {raw.dtype}")

n = 128 if args.smoke else 4096
labels = np.array([lookup[e["name"]] for e in entities], np.int32)
aliases = [
    np.array([lookup[a] for a in e["aliases"]], np.int32)
    for e in entities
]
splits = {
    split: np.array([
        i for i, entity in enumerate(entities) if entity["split"] == split
    ], dtype=np.int32)
    for split in ("train", "dev", "test")
}
if any(len(splits[split]) < 1024 for split in splits):
    raise ValueError("Insufficient entities in a split")

train_rows = []
for i in splits["train"]:
    train_rows.append(int(labels[i]))
    train_rows.extend(aliases[i].tolist())

progress(f"Training statistics: {len(train_rows)} occurrences, dimension {n}")
# No runtime thread-limit increase, and no threadpool context around eigh.
training = np.asarray(raw[train_rows, :n], dtype=np.float64)
if not np.isfinite(training).all():
    raise ValueError("Nonfinite training embeddings")
mean = training.mean(axis=0)
training -= mean
cov = training.T @ training / len(training)
ev, u = eigh(cov, check_finite=False, overwrite_a=True)
ev = np.maximum(ev, 0)
regularizer = .1 * ev.mean()
if regularizer <= 0:
    raise ValueError("Training covariance is identically zero")
zca = (u * (1 / np.sqrt(ev + regularizer))) @ u.T
principal = u[:, ::-1].astype(np.float32)
del training, cov, u

x = np.asarray(raw[:, :n], dtype=np.float32) - mean.astype(np.float32)
if not np.isfinite(x).all():
    raise ValueError("Nonfinite embeddings")
preprocessed = {"centre": normalize(x)}
preprocessed["whiten"] = normalize(x @ zca.astype(np.float32))
del x, zca, ev
gc.collect()

conditions = ([(256, 2, 4)] if args.smoke else [
    (m, v, b)
    for m in (256, 1024) for v in (2, 16) for b in (4, 8, 16)
])
store_grid = [
    dict(prep=prep, map=family, q=q, share=share)
    for prep in ("centre", "whiten")
    for family in ("lin", "sgn", "sgn3")
    for q in (1, 2, 4)
    for share in (1., .5, .25, .125)
]
lam_grid = [
    dict(scale=scale, target=target, prep=prep, feat=feat,
         alpha=alpha, q=q)
    for scale in ("col", "flat")
    for target in ("V-1", "V")
    for prep, feat in (
        ("centre", "rp"), ("centre", "pca"),
        ("whiten", "rp"), ("whiten", "pca"))
    for alpha in (1e-4, 1e-2, .1, 1., 10., 100.)
    for q in (1, 2, 4)
]
nlms_grid = [
    dict(scale=scale, target=target, prep=prep, feat=feat,
         mu=mu, output=output, q=q)
    for scale in ("col", "flat")
    for target in ("V-1", "V")
    for prep, feat in (
        ("centre", "rp"), ("centre", "pca"),
        ("whiten", "rp"), ("whiten", "pca"))
    for mu in (1., .5, .25, .1, .03)
    for output in ("last", "avg")
    for q in (1, 2, 4)
]
if args.smoke:
    store_grid = [store_grid[i] for i in (0, 4, 12, 36)]
    lam_grid = [lam_grid[i] for i in (0, 2, 18, 54)]
    nlms_grid = [nlms_grid[i] for i in (0, 3, 30, 60)]
grids = {"STORE": store_grid, "LAMc": lam_grid, "NLMSc": nlms_grid}

required = {}
for m, v, b in conditions:
    budget = m * b
    for memory, grid in grids.items():
        for config in grid:
            if memory == "STORE":
                key = (config["prep"], config["map"])
                p = dimension(budget, 1, config["q"], "flat", 65536)
            else:
                key = (config["prep"], config["feat"])
                c = v - 1 if config["target"] == "V-1" else v
                cap = n if config["feat"] == "pca" else 65536
                p = dimension(budget, c, config["q"], config["scale"], cap)
            required[key] = max(required.get(key, 0), p)

features = {}
for (prep, family), p in required.items():
    progress(f"Preparing {prep}/{family}, {p} features")
    mode = "smoke" if args.smoke else "full"
    path = CACHE / f"{args.enc}_{mode}_{prep}_{family}.npy"
    mat = np.lib.format.open_memmap(
        path, mode="w+", dtype=np.float32, shape=(len(strings), p))
    source = preprocessed[prep]
    if family == "pca":
        # Chunk rows to limit temporary allocations.
        for row in range(0, len(strings), 1024):
            mat[row:row + 1024] = (
                source[row:row + 1024] @ principal[:, :p])
    else:
        # Full 256-column random blocks preserve projection prefixes even
        # when the final requested dimension truncates a block.
        for start in range(0, p, 256):
            width = min(256, p - start)
            product = None
            count = 3 if family == "sgn3" else 1
            for component in range(count):
                generator = rng(
                    args.enc, prep, family, "projection", start, component)
                projection = (
                    generator.standard_normal((n, 256), dtype=np.float32)
                    / np.sqrt(n))[:, :width]
                z = source @ projection
                if family in ("sgn", "sgn3"):
                    z = np.where(z >= 0, 1., -1.).astype(np.float32)
                product = z if product is None else product * z
            mat[:, start:start + width] = product
            del projection, z, product
    mat.flush()
    del mat
    features[(prep, family)] = np.load(path, mmap_mode="r")

# Drop all references to resident preprocessing arrays before evaluation.
del source, preprocessed, principal
gc.collect()

# Reduce BLAS pools ONCE and retain this controller for the rest of the run.
# Do not use a context manager: exiting it would restore/raise thread counts.
evaluation_thread_limit = threadpool_limits(limits=1)
progress("Preprocessing complete; BLAS permanently reduced to one thread")


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
