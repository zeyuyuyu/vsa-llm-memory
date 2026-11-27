"""B21 (prereg/B21_ruler_vsa_prereg.md + amendment 1): RULER with a sharded Fourier HRR store in place of B19's table.

python3 ruler_vsa.py run   OUT.jsonl --model <Qwen3-8B> --root RULER_DIR --length L --tasks t1,t2 --n 100
python3 ruler_vsa.py selftest
python3 ruler_vsa.py score OUT_DIR [--b19 B19_RM_DIR]

The reader, prompts, chunking and grounding check are B19's (ruler_rm.py). Both stores answer from the same reader
output in the same run: the table (B19's facts_of / answer_niah / answer_vt, unchanged) and the vector store below.
Each output line keeps both predictions, the grounded pairs and the reader's raw per-chunk output (for CPU replay).
"""
import argparse
import collections
import glob
import hashlib
import json
import math
import os
import re
import sys

import numpy as np

import ruler_rm as R

D, PER_SHARD, THR = 8192, 16, 0.5          # amendment 1 (before any data): d 4096 -> 8192, 32 -> 16 pairs per shard


def _seed(s):
    return int.from_bytes(hashlib.sha256(s.encode("utf-8")).digest()[:8], "little")


def code(s, slot):
    """Unit-modulus complex code of a string in one slot ("key", "value" or "pos"), from a seeded hash. Keys and values
    have separate codebooks: binding commutes, so with one shared codebook a pair (b, c) would read back as c -> b
    (amendment 1; the same reason as CLUTRR's separate codebooks and arena C's alias)."""
    rng = np.random.default_rng(_seed("code:%s:%s" % (slot, s)))
    return np.exp(1j * rng.uniform(-np.pi, np.pi, D)).astype(np.complex64)


def pos(j):
    return code(str(j), "pos")


def grounded_pairs(reader, ctx):
    """B19's facts_of, returning the ordered (key, value, j) triples it would write, j = index of the value for its key,
    plus the reader's raw output per chunk."""
    cs = R.chunks(ctx)
    raw = reader([R.FACT_PROMPT.format(chunk=c) for c in cs], 400)
    seen, pairs = collections.defaultdict(list), []
    for c, txt in zip(cs, raw):
        for k, v in R.PAIR.findall(txt):
            k, v = re.sub(r"^VAR\s+", "", k.strip()), re.sub(r"^VAR\s+", "", v.strip().rstrip("."))
            if k.lower() in c.lower() and v.lower() in c.lower() and k.lower() != v.lower():
                if v not in seen[k.lower()]:
                    seen[k.lower()].append(v)
                    pairs.append((k.lower(), v, len(seen[k.lower()]) - 1))
    return pairs, raw


def table_of(pairs):
    t = collections.defaultdict(list)
    for k, v, _ in pairs:
        t[k].append(v)
    return t


class Store:
    """Sharded Fourier HRR. bind(key, pos_j, value) is added into shard hash(key) mod S; reading unbinds key and pos_j
    from that shard and cleans up against the codes of the strings in the store's own vocabulary."""

    def __init__(self, pairs):
        self.S = max(1, math.ceil(len(pairs) / PER_SHARD))
        self.mem = np.zeros((self.S, D), np.complex64)
        vocab = []
        for k, v, j in pairs:
            self.mem[self.shard(k)] += code(k, "key") * pos(j) * code(v, "value")
            vocab.append(v)
        self.vocab = list(dict.fromkeys(vocab))
        self.V = np.stack([code(s, "value") for s in self.vocab]) if self.vocab else np.zeros((0, D), np.complex64)
        self.loads = np.bincount([self.shard(k) for k, _, _ in pairs], minlength=self.S) if pairs else np.zeros(1)

    def shard(self, k):
        return _seed("shard:" + k) % self.S

    def read(self, k):
        """Values of key k in order: unbind pos_j for j = 0, 1, ... and stop at the first clean-up below THR."""
        if not len(self.vocab):
            return []
        out, j = [], 0
        x = self.mem[self.shard(k)] * np.conj(code(k, "key"))
        while True:
            y = x * np.conj(pos(j))
            sims = (self.V.conj() @ y).real / D
            b = int(np.argmax(sims))
            if sims[b] < THR:
                return out
            out.append(self.vocab[b])
            j += 1


def answer_niah_vsa(store, q):
    m = re.search(r"for (.+?) mentioned", q)
    keys = re.split(r",\s*and\s+|,\s*|\s+and\s+", m.group(1)) if m else []
    return " ".join(v for k in keys for v in store.read(k.strip().lower()))


def answer_vt_vsa(store, names, q):
    """Resolve each variable by repeated reads from the store; names = the variables the reader named (the codebook of
    variable symbols), in the order in which they were first read."""
    m = re.search(r"assigned the value (\S+?)\s+in", q)
    target = m.group(1) if m else ""
    keyset = set(names)

    def resolve(x, seen=()):
        vals = store.read(x.lower())
        if not vals or x in seen:
            return x
        return resolve(vals[-1], seen + (x,))
    return " ".join(k.upper() for k in names if resolve(k) == target)


def run(args):
    reader = R.CachedReader(args.model)
    with open(args.out, "a") as f:
        for task in args.tasks.split(","):
            for s in R.load(args.root, args.length, task, args.n):
                ctx, q = R.split(s)
                pairs, raw = grounded_pairs(reader, ctx)
                table = table_of(pairs)
                assert dict(table) == dict(R.facts_of(reader, ctx))      # the same table as B19 (cached reader: no extra decoding)
                store = Store(pairs)
                if task.startswith("niah"):
                    p_tab, p_vsa = R.answer_niah(table, q), answer_niah_vsa(store, q)
                else:
                    names = list(dict.fromkeys(k for k, _, _ in pairs))  # variable symbols, in order of first reading
                    p_tab, p_vsa = R.answer_vt(table, q), answer_vt_vsa(store, names, q)
                f.write(json.dumps({"task": task, "length": args.length, "index": s["index"], "outputs": s["outputs"],
                                    "pred_table": p_tab, "pred_vsa": p_vsa, "n_pairs": len(pairs), "shards": store.S,
                                    "max_load": int(np.max(store.loads)), "pairs": pairs,
                                    "raw": raw}) + "\n")
                f.flush()
            print(f"{task} {args.length}: calls {reader.calls} unique {reader.unique} gen tokens {reader.tokens}", flush=True)


def selftest():
    """CPU check before any run: on synthetic facts with thousands of keys and several values per key, the store must
    return exactly what the table returns."""
    rng = np.random.default_rng(0)
    for n_keys, multi in ((50, 1), (2000, 1), (3000, 4)):
        pairs = []
        for i in range(n_keys):
            k = "key-%06x" % rng.integers(1 << 24)
            for j in range(int(rng.integers(1, multi + 1))):
                pairs.append((k, "%08d" % rng.integers(10 ** 8), j))
        seen, clean = collections.defaultdict(list), []
        for k, v, _ in pairs:
            if v not in seen[k]:
                seen[k].append(v); clean.append((k, v, len(seen[k]) - 1))
        t, st = table_of(clean), Store(clean)
        probe = [k for i, k in enumerate(t) if i % max(1, len(t) // 150) == 0]
        bad = sum(st.read(k) != t[k] for k in probe)
        print("selftest keys %d pairs %d shards %d max load %d: %d of %d probed keys differ" % (
            len(t), len(clean), st.S, int(st.loads.max()), bad, len(probe)))
        assert bad == 0
    # a variable chain, as in VT
    chain = [("a", "b", 0), ("b", "c", 0), ("c", "12345", 0), ("x", "y", 0), ("y", "999", 0)]
    st, t = Store(chain), table_of(chain)
    q = "Find all variables that are assigned the value 12345 in the text above."
    assert answer_vt_vsa(st, list(t.keys()), q) == R.answer_vt(t, q) == "A B C", (answer_vt_vsa(st, list(t.keys()), q), R.answer_vt(t, q))
    print("selftest VT chain: ok")


def score(args):
    rows = [json.loads(l) for f in sorted(glob.glob(os.path.join(args.dir, "*.jsonl"))) for l in open(f)]
    by = collections.defaultdict(list)
    for r in rows:
        by[(int(r["length"]), r["task"])].append(r)
    out = {}
    for (L, task), rs in sorted(by.items()):
        tab = 100 * np.mean([R.string_match_all(r["pred_table"], r["outputs"]) for r in rs])
        vsa = 100 * np.mean([R.string_match_all(r["pred_vsa"], r["outputs"]) for r in rs])
        agree = np.mean([r["pred_table"] == r["pred_vsa"] for r in rs])
        out.setdefault(L, {})[task] = {"n": len(rs), "table": round(tab, 2), "vsa": round(vsa, 2), "agree": round(float(agree), 4),
                                       "max_load": max(r["max_load"] for r in rs)}
    print(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "selftest", "score"])
    ap.add_argument("out", nargs="?")
    ap.add_argument("--model"); ap.add_argument("--root"); ap.add_argument("--length", type=int)
    ap.add_argument("--tasks"); ap.add_argument("--n", type=int, default=100)
    a = ap.parse_args()
    if a.mode == "run":
        run(a)
    elif a.mode == "selftest":
        selftest()
    else:
        a.dir = a.out
        score(a)
