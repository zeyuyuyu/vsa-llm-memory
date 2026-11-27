"""NSR0 (prereg/NSR0_prereg.md): composition table vs VSA rule memory on CLUTRR gold relation chains.

Every arm shares the learning procedure (k = 2 writes, then 3 closure rounds over k = 3 chains) and the inference (CYK
over any bracketing); only the store differs: T (count table), V-dict (R1: table voted, then written into an FHRR
vector), V-sum (pure additive VSA, no side dictionary), V-delta (pure delta-rule VSA).

python3 nsr/nsr0.py tune      # choose tau / alpha on validation (writes nsr/out/nsr0_tuning.json)
python3 nsr/nsr0.py run       # all registered cells on test (writes nsr/out/nsr0_preds.jsonl.gz, nsr0_summary.json)
"""
import collections
import csv
import gzip
import json
import os
import sys

import numpy as np

ROOT, OUT = "r1_data", "nsr/out"
SEEDS = range(10)


def load(split):
    rows = csv.DictReader(open(f"{ROOT}/{split}.csv", encoding="utf-8-sig"))
    return [{"id": r["id"], "rels": r["f_comb"].split("-"), "target": r["target_text"]} for r in rows]


TRAIN, VAL, TEST = load("train"), load("validation"), load("test")
RELS = sorted({x for r in TRAIN + VAL + TEST for x in r["rels"] + [r["target"]]})
CHAIN_RELS = sorted({x for r in TRAIN for x in r["rels"]})


# ----------------------------------------------------------------------------------------------------------- stores
class Table:
    def __init__(self, **_):
        self.votes = collections.defaultdict(collections.Counter)

    def write(self, a, b, v):
        self.votes[(a, b)][v] += 1

    def known(self, a, b):
        return (a, b) in self.votes

    def read(self, a, b):
        c = self.votes[(a, b)]
        top = max(c.values())
        v = min(x for x, n in c.items() if n == top)
        return v, top / sum(c.values())

    def snapshot(self):
        t = Table()
        t.votes = collections.defaultdict(collections.Counter, {k: collections.Counter(c) for k, c in self.votes.items()})
        return t

    def finalize(self):
        pass


class Codes:
    """Three independent FHRR codebooks (first slot, second slot, value) over the fixed relation list."""

    def __init__(self, D, rng):
        self.D = D
        self.first = {r: np.exp(1j * rng.uniform(-np.pi, np.pi, D)) for r in RELS}
        self.second = {r: np.exp(1j * rng.uniform(-np.pi, np.pi, D)) for r in RELS}
        self.value = {r: np.exp(1j * rng.uniform(-np.pi, np.pi, D)) for r in RELS}
        self.V = np.stack([self.value[r] for r in RELS])  # (n_rel, D)

    def key(self, a, b):
        return self.first[a] * self.second[b]

    def cleanup(self, u):
        sims = np.real(np.conj(self.V) @ u) / self.D  # in units of one stored binding
        i = int(sims.argmax())
        pos = sims[sims > 0].sum()
        return RELS[i], float(sims[i]), float(sims[i] / pos) if pos > 0 else 0.0


class VSum:
    def __init__(self, D, rng, tau, **_):
        self.c, self.tau = Codes(D, rng), tau
        self.M = np.zeros(D, complex)
        self._cache = None  # set on frozen snapshots only

    def write(self, a, b, v):
        self.M += self.c.key(a, b) * self.c.value[v]

    def _clean(self, a, b):
        if self._cache is not None:
            if (a, b) not in self._cache:
                self._cache[(a, b)] = self.c.cleanup(np.conj(self.c.key(a, b)) * self.M)
            return self._cache[(a, b)]
        return self.c.cleanup(np.conj(self.c.key(a, b)) * self.M)

    def known(self, a, b):
        return self._clean(a, b)[1] >= self.tau

    def read(self, a, b):
        v, _, conf = self._clean(a, b)
        return v, conf

    def snapshot(self):
        s = object.__new__(type(self))
        s.__dict__.update(self.__dict__)
        s.M = self.M.copy()
        s._cache = {}
        return s

    def finalize(self):
        pass

    def quantize(self):
        """4-bit phase quantisation of the stored vector, rescaled to the original RMS (one extra scalar)."""
        rms = np.sqrt(np.mean(np.abs(self.M) ** 2))
        ph = np.round(np.angle(self.M) / (2 * np.pi / 16)) * (2 * np.pi / 16)
        self.M = rms * np.exp(1j * ph)


class VDelta(VSum):
    def __init__(self, D, rng, tau, alpha, **_):
        super().__init__(D, rng, tau)
        self.alpha = alpha

    def write(self, a, b, v):
        # Amendment 1: projected delta rule. In FHRR key * conj(key) = 1 elementwise, so the registered form
        # alpha * key * (value - conj(key) * M) equals alpha * (key * value - M) and decays the whole memory.
        # Here only the content along the value codes at this key is corrected.
        k = self.c.key(a, b)
        s = np.real(np.conj(self.c.V) @ (np.conj(k) * self.M)) / self.c.D
        self.M += self.alpha * k * (self.c.value[v] - s @ self.c.V)


class VDict(Table):
    """R1's design: learn as a table; at finalize write each voted rule once into an FHRR vector and read through it."""

    def __init__(self, D, rng, **_):
        super().__init__()
        self.c = Codes(D, rng)
        self.M = None

    def snapshot(self):
        t = Table()
        t.votes = collections.defaultdict(collections.Counter, {k: collections.Counter(c) for k, c in self.votes.items()})
        return t

    def finalize(self):
        self.M = np.zeros(self.c.D, complex)
        for (a, b) in self.votes:
            self.M += self.c.key(a, b) * self.c.value[Table.read(self, a, b)[0]]

    def read(self, a, b):
        v, _, conf = self.c.cleanup(np.conj(self.c.key(a, b)) * self.M)
        return v, conf

    def quantize(self):
        VSum.quantize(self)


# --------------------------------------------------------------------------------------------- learning / inference
def learn(store, stories, rng):
    """Shared procedure: stage 1 (k = 2 writes), stage 2 (3 closure rounds over k = 3, known() on a round snapshot)."""
    order = rng.permutation(len(stories))
    stories = [stories[i] for i in order]
    for s in stories:
        if len(s["rels"]) == 2:
            store.write(s["rels"][0], s["rels"][1], s["target"])
    for _ in range(3):
        snap = store.snapshot()
        for s in stories:
            if len(s["rels"]) != 3:
                continue
            a, b, c = s["rels"]
            if snap.known(a, b):
                x = snap.read(a, b)[0]
                if not snap.known(x, c):
                    store.write(x, c, s["target"])
            if snap.known(b, c):
                y = snap.read(b, c)[0]
                if not snap.known(a, y):
                    store.write(a, y, s["target"])
    store.finalize()
    return store


def rule_tables(store):
    R = {}
    for a in RELS:
        for b in RELS:
            if store.known(a, b):
                R[(a, b)] = store.read(a, b)
    return R


def compose(rels, R):
    """CYK over any bracketing; each cell maps value -> best (max over derivations of min rule confidence)."""
    k = len(rels)
    T = {(i, i): {rels[i]: 1.0} for i in range(k)}
    for span in range(2, k + 1):
        for i in range(k - span + 1):
            j = i + span - 1
            out = {}
            for m in range(i, j):
                for x, cx in T[(i, m)].items():
                    for y, cy in T[(m + 1, j)].items():
                        if (x, y) in R:
                            v, c = R[(x, y)]
                            c = min(c, cx, cy)
                            if c > out.get(v, -1.0):
                                out[v] = c
            T[(i, j)] = out
    res = T[(0, k - 1)]
    if len(res) == 1:
        (v, c), = res.items()
        return v, c
    return ("UNKNOWN", 0.0) if not res else ("AMBIG", 0.0)


IDX = {r: i for i, r in enumerate(RELS)}


def compose_fast(rels, VAL_, CONF, KNOWN):
    """Vectorised CYK, same semantics as compose(): cell = confidence per relation (-1 = not derivable)."""
    n, k = len(RELS), len(rels)
    T = {}
    for i, r in enumerate(rels):
        c = np.full(n, -1.0)
        c[IDX[r]] = 1.0
        T[(i, i)] = c
    for span in range(2, k + 1):
        for i in range(k - span + 1):
            j = i + span - 1
            out = np.full(n, -1.0)
            for m in range(i, j):
                cx, cy = T[(i, m)], T[(m + 1, j)]
                ok = KNOWN & (cx[:, None] >= 0) & (cy[None, :] >= 0)
                if ok.any():
                    comb = np.minimum(np.minimum(cx[:, None], cy[None, :]), CONF)[ok]
                    np.maximum.at(out, VAL_[ok], comb)
            T[(i, j)] = out
    res = T[(0, k - 1)]
    hit = np.flatnonzero(res >= 0)
    if len(hit) == 1:
        return RELS[hit[0]], float(res[hit[0]])
    return ("UNKNOWN", 0.0) if len(hit) == 0 else ("AMBIG", 0.0)


def predict(R, test):
    n = len(RELS)
    VAL_, CONF, KNOWN = np.zeros((n, n), int), np.zeros((n, n)), np.zeros((n, n), bool)
    for (a, b), (v, c) in R.items():
        VAL_[IDX[a], IDX[b]], CONF[IDX[a], IDX[b]], KNOWN[IDX[a], IDX[b]] = IDX[v], c, True
    out = [compose_fast(s["rels"], VAL_, CONF, KNOWN) for s in test]
    return [p for p, _ in out], [c for _, c in out]


def predict_ref(R, test):
    out = [compose(s["rels"], R) for s in test]
    return [p for p, _ in out], [c for _, c in out]


# -------------------------------------------------------------------------------------------------------- sampling
def per_key(m, rng):
    g = collections.defaultdict(list)
    for s in TRAIN:
        g["-".join(s["rels"])].append(s)
    out = []
    for key in sorted(g):
        idx = rng.choice(len(g[key]), size=m, replace=False)
        out += [g[key][i] for i in idx]
    return out


def noisy(stories, kind, p, rng):
    out = []
    for s in stories:
        s = dict(s, rels=list(s["rels"]))
        if rng.random() < p:
            if kind == "L":
                s["target"] = rng.choice([r for r in RELS if r != s["target"]])
            else:
                i = int(rng.integers(len(s["rels"])))
                s["rels"][i] = rng.choice([r for r in CHAIN_RELS if r != s["rels"][i]])
        out.append(s)
    return out


def make(arm, D, rng, params):
    return {"T": lambda: Table(), "V-dict": lambda: VDict(D, rng),
            "V-sum": lambda: VSum(D, rng, params["V-sum"]["tau"]),
            "V-delta": lambda: VDelta(D, rng, params["V-delta"]["tau"], params["V-delta"]["alpha"])}[arm]()


def acc(preds, test):
    return sum(p == s["target"] for p, s in zip(preds, test)) / len(test)


# ------------------------------------------------------------------------------------------------------------ tune
def tune():
    grid_tau, grid_alpha = [0.3, 0.4, 0.5, 0.6, 0.7], [0.3, 0.5, 1.0]
    grid_tau_sum = grid_tau + [1, 2, 3, 5]  # amendment 1
    res = {"V-sum": {}, "V-delta": {}}
    for seed in range(3):
        for cell in ("clean", "L0.2"):
            rng = np.random.default_rng(1000 + seed)
            tr = per_key(3, rng)
            if cell == "L0.2":
                tr = noisy(tr, "L", 0.2, rng)
            for tau in grid_tau_sum:
                for arm, alphas in (("V-sum", [None]), ("V-delta", grid_alpha)):
                    if arm == "V-delta" and tau not in grid_tau:
                        continue
                    for alpha in alphas:
                        st = learn(VSum(2048, np.random.default_rng(seed), tau) if arm == "V-sum" else
                                   VDelta(2048, np.random.default_rng(seed), tau, alpha), tr, np.random.default_rng(seed))
                        a = acc(predict(rule_tables(st), VAL)[0], VAL)
                        res[arm].setdefault(f"{tau}|{alpha}", []).append(a)
    choice = {}
    for arm in res:
        best = max(res[arm].items(), key=lambda kv: np.mean(kv[1]))  # first in grid order on ties
        tau, alpha = best[0].split("|")
        choice[arm] = {"tau": float(tau), "alpha": None if alpha == "None" else float(alpha), "val_acc": np.mean(best[1])}
    table = {arm: {k: float(np.mean(v)) for k, v in d.items()} for arm, d in res.items()}
    os.makedirs(OUT, exist_ok=True)
    json.dump({"grid": table, "choice": choice}, open(f"{OUT}/nsr0_tuning.json", "w"), indent=1)
    print(json.dumps({"grid": table, "choice": choice}, indent=1))


# ------------------------------------------------------------------------------------------------------------- run
PARAMS = None
IDS = [s["id"] for s in TEST]
ARMS = ("T", "V-dict", "V-sum", "V-delta")


def rec(cell, arm, seed, preds, confs, meta):
    return {"cell": cell, "arm": arm, "seed": seed, "ids": IDS, "pred": preds, "conf": confs, "meta": meta}


def train_set(cell, rng):
    if cell.startswith("S_"):
        m = cell[2:][1:] if cell != "S_all" else "all"
        return list(TRAIN) if m == "all" else per_key(int(m), rng)
    if cell.startswith("C_N"):
        return [TRAIN[i] for i in rng.choice(len(TRAIN), size=int(cell[3:]), replace=False)]
    kind, p, m = cell.split("_")  # e.g. L_p0.1_m3
    return noisy(per_key(int(m[1:]), rng), kind, float(p[1:]), rng)


def job(spec):
    """One registered cell (all seeds, all its arms). Returns (records, train-set dumps, log line)."""
    kind, cell = spec
    recs, dumps = [], []
    if kind == "main":
        for seed in SEEDS:
            rng = np.random.default_rng([ord(c) for c in cell] + [seed])
            tr = train_set(cell, rng)
            if seed == 0 and cell in ("S_m1", "L_p0.3_m3", "F_p0.3_m3"):  # for the second implementation of T
                dumps.append({"cell": cell, "seed": seed, "stories": tr})
            for arm in ARMS:
                st = learn(make(arm, 2048, np.random.default_rng([seed, 7]), PARAMS), tr, np.random.default_rng([seed, 11]))
                R = rule_tables(st)
                p, c = predict(R, TEST)
                recs.append(rec(cell, arm, seed, p, c, {"n_rules": len(R), "n_relations": len(RELS), "D": 2048}))
    elif kind == "Z":
        D = int(cell[3:])
        for seed in SEEDS:
            for arm in ("V-dict", "V-sum"):
                st = learn(make(arm, D, np.random.default_rng([seed, 7]), PARAMS), list(TRAIN), np.random.default_rng([seed, 11]))
                R = rule_tables(st)
                p, c = predict(R, TEST)
                recs.append(rec(cell, arm, seed, p, c, {"n_rules": len(R), "n_relations": len(RELS), "D": D}))
                st.quantize()
                R = rule_tables(st)
                p, c = predict(R, TEST)
                recs.append(rec(cell, arm + "-q4", seed, p, c, {"n_rules": len(R), "n_relations": len(RELS), "D": D}))
    elif kind == "Q":
        for seed in SEEDS:
            rng = np.random.default_rng([seed, 99])
            noisy_test = [dict(s, rels=[r if rng.random() >= 0.05 else rng.choice([x for x in CHAIN_RELS if x != r])
                                        for r in s["rels"]]) for s in TEST]
            for arm in ARMS:
                st = learn(make(arm, 2048, np.random.default_rng([seed, 7]), PARAMS), list(TRAIN), np.random.default_rng([seed, 11]))
                p, c = predict(rule_tables(st), noisy_test)
                recs.append(rec("Q", arm, seed, p, c, {"n_relations": len(RELS)}))
    elif kind == "NC":
        for seed in SEEDS:
            rng = np.random.default_rng([seed, 123])
            R = rule_tables(learn(Table(), list(TRAIN), np.random.default_rng([seed, 11])))
            keys = sorted(R)
            vals = [R[k] for k in keys]
            perm = rng.permutation(len(vals))
            p, c = predict({k: vals[perm[i]] for i, k in enumerate(keys)}, TEST)
            recs.append(rec("NC1", "T", seed, p, c, {"n_relations": len(RELS)}))
            labels = [s["target"] for s in TRAIN]
            rng.shuffle(labels)
            tr = [dict(s, target=t) for s, t in zip(TRAIN, labels)]
            p, c = predict(rule_tables(learn(Table(), tr, np.random.default_rng([seed, 11]))), TEST)
            recs.append(rec("NC2", "T", seed, p, c, {"n_relations": len(RELS)}))
    by = collections.defaultdict(list)
    for r in recs:
        by[(r["cell"], r["arm"])].append(acc(r["pred"], TEST))
    line = "  ".join(f"{c}/{a}={np.mean(v):.4f}" for (c, a), v in by.items())
    return recs, dumps, line


def init(params):
    global PARAMS
    PARAMS = params


def run():
    import multiprocessing as mp
    params = json.load(open(f"{OUT}/nsr0_tuning.json"))["choice"]
    cells = ["S_m1", "S_m3", "S_m10", "S_all"] + [f"C_N{n}" for n in (50, 100, 200, 400, 800)]
    cells += [f"{k}_p{p}_m{m}" for k in "LF" for p in (0.1, 0.2, 0.3) for m in (1, 3, 10)]
    specs = [("main", c) for c in cells] + [("Z", f"Z_D{D}") for D in (32, 64, 128, 256, 512, 1024, 2048)]
    specs += [("Q", "Q"), ("NC", "NC")]
    summary = {"acc": {}, "params": params}
    with mp.Pool(14, initializer=init, initargs=(params,)) as pool, \
            gzip.open(f"{OUT}/nsr0_preds.jsonl.gz", "wt") as fout, open(f"{OUT}/nsr0_trainsets.jsonl", "w") as fd:
        for recs, dumps, line in pool.imap(job, specs):  # imap keeps the registered order
            for r in recs:
                fout.write(json.dumps(r) + "\n")
                summary["acc"][f"{r['cell']}|{r['arm']}|{r['seed']}"] = acc(r["pred"], TEST)
            for d in dumps:
                fd.write(json.dumps(d) + "\n")
            print(line, flush=True)
    json.dump(summary, open(f"{OUT}/nsr0_summary.json", "w"))


if __name__ == "__main__":
    {"tune": tune, "run": run}[sys.argv[1]]()
