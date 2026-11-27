"""B26a (prereg/B26_vector_form_prereg.md): does the vector form itself beat an equally exact table when entity names
vary in surface form? RULER NIAH tasks, B21 reader output, seven memories answering from the same pairs.

python3 b26_fuzzy.py selftest
python3 b26_fuzzy.py dev      OUT.json                      # query-side dev grid (seed-7 dev rows + synthetic stress)
python3 b26_fuzzy.py prompts  SPLIT OUT.jsonl               # write side: the varied needle chunks to re-read (no model)
python3 b26_fuzzy.py run      SPLIT SIDE CHOICE.json OUT.jsonl [--reads READS.jsonl] [--workers 4]
                              # SPLIT dev|test, SIDE query|write; CHOICE = the dev-selected settings
                              # (results/b26_dev_choice.json); READS = the reader's outputs on the prompts (write side)

Arms (each returns, for every queried key, the values of at most one stored key; the key vocabulary is the set of
keys the reader wrote for this text, in first-read order; ties go to the earliest key):
  E   exact table: B21's table and look-up (ruler_rm.answer_niah), unchanged.
  Ev  B21's random-code VSA store (ruler_vsa.answer_niah_vsa), unchanged.
  En  exact table after normalization (lower case, every run of non-alphanumeric characters -> one space).
  L   table + nearest key by edit distance (rapidfuzz; variant chosen on dev) over normalized strings.
  W   table + nearest key by rapidfuzz WRatio with its default processor (a standard general-purpose fuzzy matcher).
  N   table + nearest key by the exact cosine of character n-gram counts (the n-gram set chosen for V).
  V   VSA with similarity-preserving codes: each key's code is the phase-projected bundle of the Fourier HRR codes of its
      character n-grams; the queried key's code is cleaned up against the codes of the keys in the store (top-1), and the
      values of that key are read from the sharded store by B21's read (unbind key and position, clean up values,
      accept >= 0.5). The store binds keys either with their n-gram codes or with B21's random codes (chosen on dev).
"""
import argparse
import collections
import glob
import hashlib
import json
import math
import os
import random
import re
import sys
import uuid as uuidlib

import numpy as np
from scipy import sparse
from rapidfuzz import fuzz, process, utils
from rapidfuzz.distance import OSA, Levenshtein

import ruler_rm as R
import ruler_vsa as V

D = V.D
NIAH = R.NIAH

_CODE, _code0 = {}, V.code


def _cached_code(s, slot):
    """ruler_vsa.code with a cache (identical outputs; cleared per sample)."""
    c = _CODE.get((s, slot))
    if c is None:
        c = _CODE[(s, slot)] = _code0(s, slot)
    return c


V.code = _cached_code


def clear_cache():
    """Per-sample: B21 codes and 4-gram codes are dropped; 1- to 3-gram codes are kept (a bounded set)."""
    _CODE.clear()
    for g in [g for g in _GRAM if len(g) > 3]:
        del _GRAM[g]
TEST_DIR, TEST_ROOT = "b21_dl/b21_test", "/path/to/HOME/bench_dl/ruler"
DEV_DIR, DEV_ROOT = "b24_dev/reader", "/path/to/HOME/bench_dl/ruler_dev"
LENGTHS = {"test": [4096, 8192, 16384, 32768, 65536, 131072], "dev": [4096, 16384]}
NPER = {"test": 100, "dev": 20}

QUERY_CONDS = ["C0", "T1", "T2", "T3", "P", "A"]          # C0 = unchanged (control)
WRITE_CONDS = ["C0", "T2", "P", "A"]                      # write side: C0 = the needle chunks re-read unchanged
RHO = {"T1": 0.05, "T2": 0.10, "T3": 0.20}
ARMS = ["E", "Ev", "En", "L", "W", "N", "V"]

# dev grid (prereg): V over n-gram sets x binding code; L over three edit distances
NSETS = {"3": (3,), "23": (2, 3), "123": (1, 2, 3), "234": (2, 3, 4)}
BINDS = ["ng", "rand"]
LDIST = ["lev", "levnorm", "osa"]

# ---------------------------------------------------------------- the RULER word lists (key = adjective-noun)
try:
    import wonderwords
    _NOUN_L = sorted(set(wonderwords.random_word._get_words_from_text_file("nounlist.txt")))
    _ADJ_L = sorted(set(wonderwords.random_word._get_words_from_text_file("adjectivelist.txt")))
    _NOUNS, _ADJS = set(_NOUN_L), sorted(_ADJ_L, key=len, reverse=True)
except Exception:                                          # pragma: no cover (the split falls back to the first hyphen)
    _NOUNS, _ADJS, _NOUN_L, _ADJ_L = set(), [], [], []

UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


def is_uuid(k):
    return bool(UUID_RE.match(k.lower()))


def split_word_key(k):
    """(adjective, noun) of a RULER word key adj-noun (either may contain a hyphen or a space)."""
    for a in _ADJS:
        if k.startswith(a + "-") and k[len(a) + 1:] in _NOUNS:
            return a, k[len(a) + 1:]
    i = k.find("-")
    return (k[:i], k[i + 1:]) if i > 0 else (k, "")


# ---------------------------------------------------------------- surface variants (prereg: conditions)
_ROWS = ["qwertyuiop", "asdfghjkl", "zxcvbnm", "1234567890"]
_NB = collections.defaultdict(set)
for _r in _ROWS:
    for _i, _c in enumerate(_r):
        for _j in (_i - 1, _i + 1):
            if 0 <= _j < len(_r):
                _NB[_c].add(_r[_j])
for _a, _b in zip(_ROWS[:-1], _ROWS[1:]):               # vertical neighbours (same column)
    for _i in range(min(len(_a), len(_b))):
        _NB[_a[_i]].add(_b[_i]); _NB[_b[_i]].add(_a[_i])
_NB = {c: sorted(s) for c, s in _NB.items()}
HEX = "0123456789abcdef"


def _rng(*parts):
    return random.Random(int.from_bytes(hashlib.sha256(":".join(map(str, parts)).encode()).digest()[:8], "little"))


def _neighbour(c, hexonly, rng):
    if hexonly:
        return rng.choice([h for h in HEX if h != c.lower()])
    nb = _NB.get(c.lower())
    return rng.choice(nb) if nb else rng.choice("abcdefghijklmnopqrstuvwxyz")


def typo(key, rho, rng):
    """n = max(1, floor(rho * len + 0.5)) edits, each a substitution (keyboard neighbour; random hex digit in a uuid),
    deletion, insertion (keyboard neighbour of the character) or transposition of two adjacent characters, at a random
    alphanumeric position. Separators are left alone (they are condition P)."""
    hexonly = is_uuid(key)
    n = max(1, math.floor(rho * len(key) + 0.5))
    for _ in range(50):
        s = list(key)
        for _e in range(n):
            pos = [i for i, c in enumerate(s) if c.isalnum()]
            if not pos:
                break
            i = rng.choice(pos)
            op = rng.choice(["sub", "del", "ins", "swap"])
            if op == "sub":
                s[i] = _neighbour(s[i], hexonly, rng)
            elif op == "del" and len(pos) > 2:
                del s[i]
            elif op == "ins":
                s.insert(i + rng.randint(0, 1), _neighbour(s[i], hexonly, rng))
            elif op == "swap" and i + 1 < len(s) and s[i + 1].isalnum() and s[i + 1] != s[i]:
                s[i], s[i + 1] = s[i + 1], s[i]
            else:
                s[i] = _neighbour(s[i], hexonly, rng)
        out = "".join(s)
        if out.lower() != key.lower():
            return out
    raise RuntimeError("typo: no change for %r" % key)


def fmt(key, rng):
    """Format variant: the separator(s) and the case change. Word keys: the adjective-noun hyphen becomes one of
    ' ', '_', '', ' - ', and the case one of lower / Title / UPPER. UUIDs: dashes removed or replaced by spaces, lower or
    upper case."""
    if is_uuid(key):
        s = key.replace("-", rng.choice(["", " "]))
        return s.upper() if rng.random() < 0.5 else s
    a, n = split_word_key(key)
    s = a + rng.choice([" ", "_", "", " - "]) + n
    c = rng.choice(["lower", "title", "upper"])
    return s.title() if c == "title" else (s.upper() if c == "upper" else s)


def alias(key):
    """Short alias: the noun of a word key; the first 8 hex digits of a uuid (a short id, as for commit hashes)."""
    if is_uuid(key):
        return key[:8]
    return split_word_key(key)[1] or key


def variant(key, cond, *seed):
    if cond == "C0":
        return key
    rng = _rng("b26", cond, *seed)
    if cond in RHO:
        return typo(key, RHO[cond], rng)
    if cond == "P":
        return fmt(key, rng)
    if cond == "A":
        return alias(key)
    raise ValueError(cond)


# ---------------------------------------------------------------- question parsing (B21's, verbatim)
def qkeys(q):
    m = re.search(r"for (.+?) mentioned", q)
    return re.split(r",\s*and\s+|,\s*|\s+and\s+", m.group(1)) if m else []


def vary_question(q, cond, seed):
    """Rebuild the question with every queried key replaced by its variant (key i uses seed + (i,))."""
    m = re.search(r"for (.+?) mentioned", q)
    ks = qkeys(q)
    vs = [variant(k, cond, *seed, i) for i, k in enumerate(ks)]
    if len(ks) == 1:
        body = vs[0]
    else:
        body = ", ".join(vs[:-1]) + ", and " + vs[-1]
    q2 = q[:m.start(1)] + body + q[m.end(1):]
    return q2, ks, vs


def norm(s):
    return re.sub(r"[^0-9a-z]+", " ", s.lower()).strip()


# ---------------------------------------------------------------- codes
_GRAM = {}


def gram_code(g):
    """Fourier HRR code of one character n-gram: unit-modulus complex, phases uniform from a seeded hash of the n-gram
    (its own codebook, separate from B21's key, value and position codebooks)."""
    c = _GRAM.get(g)
    if c is None:
        rng = np.random.default_rng(V._seed("code:ngram:" + g))
        ph = (rng.random(D, dtype=np.float32) * np.float32(2 * np.pi) - np.float32(np.pi))
        c = _GRAM[g] = (np.cos(ph) + 1j * np.sin(ph)).astype(np.complex64)
    return c


def grams(s, nset):
    t = " " + norm(s) + " "
    return [t[i:i + n] for n in nset for i in range(len(t) - n + 1)]


def unit(x):
    """Phase projection of a bundle: every component divided by its modulus (a zero component becomes 1)."""
    m = np.abs(x)
    return np.where(m > 0, x / np.where(m > 0, m, 1), 1).astype(np.complex64)


def ng_code(s, nset):
    """The code of a string: the bundle (sum) of the codes of its character n-grams (the normalized string padded with
    one space at each end; every n in nset), projected to unit modulus."""
    acc = np.zeros(D, np.complex64)
    for g in grams(s, nset):
        acc += gram_code(g)
    return unit(acc)


def ng_codes(strings, nset):
    """ng_code for many strings at once (one sparse count matrix times the n-gram codebook) plus the count matrix itself
    (for arm N). Returns (gram index, counts CSR, key codes)."""
    gi, rows, cols, vals = {}, [], [], []
    for r, s in enumerate(strings):
        for g, c in collections.Counter(grams(s, nset)).items():
            rows.append(r); cols.append(gi.setdefault(g, len(gi))); vals.append(c)
    C = sparse.csr_matrix((np.array(vals, np.float32), (rows, cols)), shape=(len(strings), max(1, len(gi))))
    if not strings:
        return gi, C, np.zeros((0, D), np.complex64)
    G = np.stack([gram_code(g) for g in gi])
    K = np.empty((len(strings), D), np.complex64)
    for a in range(0, len(strings), 512):                   # memory bound
        K[a:a + 512] = unit(C[a:a + 512] @ G)
    return gi, C, K


class FastStore(V.Store):
    """B21's store (ruler_vsa.Store, unchanged construction) with the conjugate value codebook computed once instead of
    at every read step: the same arithmetic, the same results."""

    def __init__(self, pairs):
        super().__init__(pairs)
        self.Vc = self.V.conj()

    def read(self, k):
        if not len(self.vocab):
            return []
        out, j = [], 0
        x = self.mem[self.shard(k)] * np.conj(V.code(k, "key"))
        while True:
            y = x * np.conj(V.pos(j))
            sims = (self.Vc @ y).real / D
            b = int(np.argmax(sims))
            if sims[b] < V.THR:
                return out
            out.append(self.vocab[b])
            j += 1


class NgStore(FastStore):
    """B21's sharded store with the keys bound by their n-gram codes (the values, positions, shards and read rule are
    B21's)."""

    def __init__(self, pairs, kcode):
        self.kcode = kcode
        self.S = max(1, math.ceil(len(pairs) / V.PER_SHARD))
        self.mem = np.zeros((self.S, D), np.complex64)
        vocab = []
        for k, v, j in pairs:
            self.mem[self.shard(k)] += kcode[k] * V.pos(j) * V.code(v, "value")
            vocab.append(v)
        self.vocab = list(dict.fromkeys(vocab))
        self.V = np.stack([V.code(s, "value") for s in self.vocab]) if self.vocab else np.zeros((0, D), np.complex64)
        self.Vc = self.V.conj()
        self.loads = np.bincount([self.shard(k) for k, _, _ in pairs], minlength=self.S) if pairs else np.zeros(1)

    def read(self, k):
        if not len(self.vocab):
            return []
        out, j = [], 0
        x = self.mem[self.shard(k)] * np.conj(self.kcode[k])
        while True:
            y = x * np.conj(V.pos(j))
            sims = (self.Vc @ y).real / D
            b = int(np.argmax(sims))
            if sims[b] < V.THR:
                return out
            out.append(self.vocab[b])
            j += 1


class Memories:
    """All arms over one sample's grounded pairs."""

    def __init__(self, pairs, nsets=("23",), binds=("ng",), ldists=("lev",), b21=True):
        self.pairs = pairs
        self.table = V.table_of(pairs)
        self.keys = list(self.table)                       # first-read order (lower case, as B21 stores them)
        self.kidx = {k: i for i, k in enumerate(self.keys)}
        self.nkeys = [norm(k) for k in self.keys]
        self.byn = {}
        for k, nk in zip(self.keys, self.nkeys):
            self.byn.setdefault(nk, k)
        self.b21 = FastStore(pairs) if (pairs and b21) else None
        self.Kc, self.cnt, self.ng = {}, {}, {}
        for ns in nsets:
            gi, C, K = ng_codes(self.keys, NSETS[ns])
            self.Kc[ns] = K.conj()
            Cd = C.astype(np.float64)
            self.cnt[ns] = (gi, Cd, np.sqrt(np.asarray(Cd.multiply(Cd).sum(1)).ravel()) + 1e-12)
            if "ng" in binds and pairs:
                self.ng[ns] = NgStore(pairs, dict(zip(self.keys, K)))
            del K
        self.ldists = ldists

    # --- key matching: returns the index of the chosen stored key, or None
    def match(self, arm, q, ns="23", ld="lev"):
        if not self.keys:
            return None
        if arm == "E":
            return self.kidx.get(q.strip().lower())
        if arm == "En":
            k = self.byn.get(norm(q))
            return self.kidx[k] if k is not None else None
        if arm == "L":
            scorer = {"lev": Levenshtein.distance, "levnorm": Levenshtein.normalized_distance, "osa": OSA.distance}[ld]
            d = process.cdist([norm(q)], self.nkeys, scorer=scorer, workers=1)[0]
            return int(np.argmin(d))
        if arm == "W":
            s = process.cdist([q.lower()], self.keys, scorer=fuzz.WRatio, processor=utils.default_process, workers=1)[0]
            return int(np.argmax(s))
        if arm == "N":
            gi, M, kn = self.cnt[ns]
            qv = np.zeros(M.shape[1])
            for g, c in collections.Counter(grams(q, NSETS[ns])).items():
                if g in gi:
                    qv[gi[g]] += c
            s = (M @ qv) / kn                                 # the query norm is common to all keys
            return int(np.argmax(np.round(s, 12)))
        if arm == "V":
            sims = (self.Kc[ns] @ ng_code(q, NSETS[ns])).real / D
            return int(np.argmax(sims))
        raise ValueError(arm)

    def values(self, arm, i, ns="23", bind="ng"):
        if i is None:
            return []
        k = self.keys[i]
        if arm == "V":
            return self.ng[ns].read(k) if bind == "ng" else self.b21.read(k)
        return list(self.table[k])

    def answer(self, arm, question, ns="23", bind="ng", ld="lev"):
        """(prediction string, chosen stored key per queried key)."""
        ks = qkeys(question)
        if arm == "E":
            return R.answer_niah(self.table, question), [k.strip().lower() if k.strip().lower() in self.table else None
                                                         for k in ks]
        if arm == "Ev":
            if self.b21 is None:
                return "", [None] * len(ks)
            return V.answer_niah_vsa(self.b21, question), [None] * len(ks)
        out, chosen = [], []
        for k in ks:
            i = self.match(arm, k.strip(), ns, ld)
            chosen.append(self.keys[i] if i is not None else None)
            out += self.values(arm, i, ns, bind)
        return " ".join(out), chosen


def score(pred, outputs):
    return R.string_match_all(pred, outputs)


# ---------------------------------------------------------------- data
def load_split(split):
    """[(cell, pos, row, sample)] for the 8 NIAH tasks: B21-format reader rows aligned with RULER inputs (index and gold
    asserted)."""
    d, root = (TEST_DIR, TEST_ROOT) if split == "test" else (DEV_DIR, DEV_ROOT)
    rows = [json.loads(l) for f in sorted(glob.glob(os.path.join(d, "*.jsonl"))) for l in open(f)]
    by = collections.defaultdict(list)
    for r in rows:
        if r["task"] in NIAH:
            by[(int(r["length"]), r["task"])].append(r)
    out = []
    for (L, t), rs in sorted(by.items()):
        data = R.load(root, L, t, NPER[split])
        assert len(rs) == len(data) == NPER[split], (L, t, len(rs), len(data))
        for p, (r, s) in enumerate(zip(rs, data)):
            assert r["index"] == s["index"] and r["outputs"] == s["outputs"], (L, t, p)
            out.append(((L, t), p, r, s))
    return out


def replay(cs, raw):
    """B21's grounding (ruler_vsa.grounded_pairs) over given chunks and reader outputs, without a model."""
    seen, pairs = collections.defaultdict(list), []
    for c, txt in zip(cs, raw):
        for k, v in R.PAIR.findall(txt):
            k, v = re.sub(r"^VAR\s+", "", k.strip()), re.sub(r"^VAR\s+", "", v.strip().rstrip("."))
            if k.lower() in c.lower() and v.lower() in c.lower() and k.lower() != v.lower():
                if v not in seen[k.lower()]:
                    seen[k.lower()].append(v)
                    pairs.append((k.lower(), v, len(seen[k.lower()]) - 1))
    return pairs


def key_pattern(k):
    return re.compile(r"(?<![\w-])" + re.escape(k) + r"(?![\w-])")


def write_variant(sample, raw, cond, seed):
    """The write-side text: every occurrence of each queried key, in the chunks that hold it, is replaced by its variant
    (the same variant at every occurrence). Chunk boundaries are B21's. Returns (chunks, {chunk index: new chunk})."""
    ctx, q = R.split(sample)
    cs = R.chunks(ctx)
    assert len(cs) == len(raw), (len(cs), len(raw))
    new = {}
    for i, k in enumerate(qkeys(q)):
        k = k.strip()
        v = variant(k, cond, *seed, i)
        pat = key_pattern(k)
        hits = [c for c in range(len(cs)) if pat.search(new.get(c, cs[c]))]
        assert hits, ("key not in text", k)
        for c in hits:
            new[c] = pat.sub(lambda _m: v, new.get(c, cs[c]))
    return cs, new


def prompt_id(text):
    return hashlib.sha256(text.encode()).hexdigest()[:20]


# ---------------------------------------------------------------- synthetic stress dev (no reader, no test data)
STRESS = [("word", 6000), ("word", 1500), ("word", 200), ("uuid", 1700), ("uuid", 400)]


def stress_store(kind, n, s, seed=7):
    """One synthetic store from RULER's key generators (niah.py): adjective-noun word keys (adjective and noun drawn
    uniformly from RULER's wonderwords lists) with 7-digit values, or uuid4 keys with uuid values. Sizes follow the test
    set's largest stores (multikey_2 ~6,000 word keys at 128k; multikey_3 ~1,700 uuid keys) and smaller ones; one queried
    key per store. No reader and no test data."""
    rng = random.Random("b26stress:%d:%s:%d:%d" % (seed, kind, n, s))
    keys, seen = [], set()
    while len(keys) < n:
        k = ("%s-%s" % (rng.choice(_ADJ_L), rng.choice(_NOUN_L))) if kind == "word" else \
            str(uuidlib.UUID(int=rng.getrandbits(128), version=4))
        if k not in seen:
            seen.add(k); keys.append(k)
    vals = [str(rng.randint(10 ** 6, 10 ** 7 - 1)) if kind == "word" else
            str(uuidlib.UUID(int=rng.getrandbits(128), version=4)) for _ in keys]
    qi = rng.randrange(n)
    typ = "number" if kind == "word" else "uuid"
    q = "What is the special magic %s for %s mentioned in the provided text?" % (typ, keys[qi])
    return [(k.lower(), v, 0) for k, v in zip(keys, vals)], q, [vals[qi]]


# ---------------------------------------------------------------- dev grid (query side)
def _dev_one(job):
    cell, pos, pairs, q, outputs, split = job
    clear_cache()
    if cell[0] == "stress":
        pairs, q, outputs = stress_store(cell[1], int(cell[2]), pos)
        cell = ("stress", "%s%s" % (cell[1], cell[2]))
    res = {c: {} for c in QUERY_CONDS}
    qs = {c: vary_question(q, c, (split, cell[0], cell[1], pos))[0] for c in QUERY_CONDS}
    base = Memories(pairs, nsets=(), binds=("rand",))
    for cond in QUERY_CONDS:
        for arm in ("E", "Ev", "En", "W"):
            res[cond][arm] = score(base.answer(arm, qs[cond])[0], outputs)
        for ld in LDIST:
            res[cond]["L:" + ld] = score(base.answer("L", qs[cond], ld=ld)[0], outputs)
    for ns in NSETS:
        mem = Memories(pairs, nsets=(ns,), binds=("ng",), b21=False)
        mem.b21 = base.b21
        for cond in QUERY_CONDS:
            res[cond]["N:" + ns] = score(mem.answer("N", qs[cond], ns=ns)[0], outputs)
            for b in BINDS:
                res[cond]["V:%s:%s" % (ns, b)] = score(mem.answer("V", qs[cond], ns=ns, bind=b)[0], outputs)
        del mem
    return {"cell": list(map(str, cell)), "pos": pos, "res": res}


def dev(out, workers):
    jobs = []
    for kind, n in STRESS:                                   # big ones first
        for s in range(40):
            jobs.append((("stress", kind, str(n)), s, None, None, None, "stress"))
    for cell, pos, row, s in load_split("dev"):
        _, q = R.split(s)
        jobs.append((cell, pos, [tuple(p) for p in row["pairs"]], q, s["outputs"], "dev"))
    from multiprocessing import Pool
    with Pool(workers) as pool:
        rows = list(pool.imap(_dev_one, jobs, chunksize=1))
    json.dump(rows, open(out, "w"))
    tab, score_of = summarize_dev(rows)
    ch = choose(score_of)
    json.dump(ch, open(os.path.splitext(out)[0] + "_choice.json", "w"), indent=1)
    print("choice:", {k: v for k, v in ch.items() if k != "dev_scores"})


def summarize_dev(rows):
    parts = {"real": [r for r in rows if r["cell"][0] != "stress"], "stress": [r for r in rows if r["cell"][0] == "stress"]}
    arms = sorted(rows[0]["res"]["C0"])
    tab = {}
    for name, rs in parts.items():
        for cond in QUERY_CONDS:
            for a in arms:
                tab[(name, cond, a)] = 100 * np.mean([r["res"][cond][a] for r in rs])
    print("dev accuracy (points); real = seed-7 dev rows (4k, 16k; %d samples), stress = synthetic (%d stores)"
          % (len(parts["real"]), len(parts["stress"])))
    print("arm".ljust(14) + "".join(("%s/%s" % (p[:4], c)).rjust(10) for p in parts for c in QUERY_CONDS) + "  devscore")
    score_of = {}
    for a in arms:
        ds = np.mean([np.mean([tab[(p, c, a)] for c in QUERY_CONDS if c != "C0"]) for p in parts])
        score_of[a] = ds
        print(a.ljust(14) + "".join("%10.1f" % tab[(p, c, a)] for p in parts for c in QUERY_CONDS) + "%10.2f" % ds)
    return tab, score_of


def choose(score_of):
    """Prereg selection rule: highest dev score; within 0.1 points of the best, the first in the listed order."""
    def pick(cands):
        best = max(score_of[c] for c in cands)
        return next(c for c in cands if score_of[c] >= best - 0.1)
    v = pick(["V:%s:%s" % (ns, b) for ns in NSETS for b in BINDS])
    l = pick(["L:" + ld for ld in LDIST])
    _, ns, b = v.split(":")
    return {"V_nset": ns, "V_bind": b, "L_dist": l.split(":")[1], "dev_scores": score_of}


# ---------------------------------------------------------------- write side: prompts and splicing
def prompts(split, out):
    seen, n = set(), collections.Counter()
    with open(out, "w") as f:
        for cell, pos, row, s in load_split(split):
            for cond in WRITE_CONDS:
                cs, new = write_variant(s, row["raw"], cond, (split, cell[0], cell[1], pos))
                for c, txt in new.items():
                    p = R.FACT_PROMPT.format(chunk=txt)
                    pid = prompt_id(p)
                    n[cond] += 1
                    if pid not in seen:
                        seen.add(pid)
                        f.write(json.dumps({"id": pid, "prompt": p}) + "\n")
    print("chunks to re-read per condition:", dict(n), "unique prompts:", len(seen))


# ---------------------------------------------------------------- the run (query or write side)
def _run_one(job):
    cell, pos, row, s, side, choice, reads, split = job
    clear_cache()
    ns, bind, ld = choice["V_nset"], choice["V_bind"], choice["L_dist"]
    ctx, q = R.split(s)
    outs = []
    conds = QUERY_CONDS if side == "query" else WRITE_CONDS
    base_pairs = [tuple(p) for p in row["pairs"]]
    mem0 = Memories(base_pairs, nsets=(ns,), binds=(bind,)) if side == "query" else None
    for cond in conds:
        seed = (split, cell[0], cell[1], pos)
        extra = {}
        if side == "query":
            mem = mem0
            qa, ks, vs = vary_question(q, cond, seed)
        else:
            cs, new = write_variant(s, row["raw"], cond, seed)
            raw = list(row["raw"])
            same = 0
            for c, txt in new.items():
                got = reads[prompt_id(R.FACT_PROMPT.format(chunk=txt))]
                same += got == raw[c]
                raw[c] = got
            pairs = replay([new.get(i, c) for i, c in enumerate(cs)], raw)
            mem = Memories(pairs, nsets=(ns,), binds=(bind,))
            qa, ks = q, qkeys(q)
            vs = [variant(k.strip(), cond, *seed, i) for i, k in enumerate(ks)]
            extra = {"reread": len(new), "reread_same": same, "n_pairs": len(pairs),
                     "variant_stored": [v.lower() in mem.table for v in vs]}
        gold_keys = [k.strip().lower() for k in ks]
        r = {"cell": [cell[0], cell[1]], "pos": pos, "index": row["index"], "cond": cond, "side": side,
             "question": qa, "keys": gold_keys, "variants": vs, "outputs": s["outputs"], **extra, "arms": {}}
        for arm in ARMS:
            pred, chosen = mem.answer(arm, qa, ns=ns, bind=bind, ld=ld)
            r["arms"][arm] = {"pred": pred, "score": score(pred, s["outputs"]), "chosen": chosen}
        outs.append(r)
    return outs


def run(split, side, choice_path, out, reads_path, workers):
    choice = json.load(open(choice_path))
    reads = {}
    if side == "write":
        for l in open(reads_path):
            x = json.loads(l)
            reads[x["id"]] = x["text"]
    data = load_split(split)
    if side == "query":                                     # pipeline check: the banked pairs replay from raw output
        for cell, pos, row, s in data[:: max(1, len(data) // 200)]:
            ctx, _ = R.split(s)
            assert [tuple(p) for p in replay(R.chunks(ctx), row["raw"])] == [tuple(p) for p in row["pairs"]]
    jobs = [(cell, pos, row, s, side, choice, reads, split) for cell, pos, row, s in data]
    jobs.sort(key=lambda j: -j[2]["n_pairs"])                # big stores first
    from multiprocessing import Pool
    n = 0
    with Pool(workers) as pool, open(out, "w") as f:
        for outs in pool.imap_unordered(_run_one, jobs, chunksize=1):
            for r in outs:
                f.write(json.dumps(r) + "\n")
            n += 1
            if n % 200 == 0:
                print("done %d / %d" % (n, len(jobs)), flush=True)


# ---------------------------------------------------------------- self-test (synthetic only)
def selftest():
    rng = random.Random(0)
    for k in ["jittery-hospital", "brother-in-law-x", "7faeb748-7781-421e-83c5-01cbfef7d9ea", "ad hoc-contact lens"]:
        print(k, "|", [variant(k, c, "selftest", 0, "t", i) for i, c in enumerate(QUERY_CONDS)])
    print("split", split_word_key("old-fashioned-brother-in-law"), split_word_key("ad hoc-contact lens"))
    pairs = [("jittery-hospital", "111", 0), ("jittery-hospitals", "222", 0), ("faint-water", "333", 0),
             ("faint-water", "444", 1), ("7faeb748-7781-421e-83c5-01cbfef7d9ea", "555", 0)]
    mem = Memories(pairs, nsets=tuple(NSETS), binds=("ng", "rand"), ldists=tuple(LDIST))
    for q in ["What is the special magic number for jittery-hospital mentioned in the provided text?",
              "What is the special magic number for Jittery Hospital mentioned in the provided text?",
              "What is the special magic number for jitery-hospital mentioned in the provided text?",
              "What are all the special magic numbers for faint-water mentioned in the provided text?",
              "What is the special magic uuid for 7faeb748 mentioned in the provided text?",
              "What are all the special magic numbers for water, and 7FAEB748778142 mentioned in the provided text?"]:
        print(q.split(" for ")[1].split(" mentioned")[0], "->",
              {a: mem.answer(a, q)[0] for a in ARMS}, mem.answer("V", q, ns="3", bind="rand")[0],
              mem.answer("L", q, ld="osa")[0])
    # exact queries: every arm returns the table's values
    for k in mem.keys:
        q = "What are all the special magic numbers for %s mentioned in the provided text?" % k
        ans = {a: mem.answer(a, q)[0] for a in ARMS}
        assert len(set(ans.values())) == 1, ans
    # question rebuild keeps the parse
    q = "What are all the special magic numbers for mindless-kilometer, freezing-milk, round-webinar, and worried-purse mentioned in the provided text?"
    for c in QUERY_CONDS:
        q2, ks, vs = vary_question(q, c, ("selftest", 1, "x", 0))
        assert [k.strip() for k in qkeys(q2)] == vs, (q2, vs)
    print("selftest ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["selftest", "dev", "prompts", "run"])
    ap.add_argument("args", nargs="*")
    ap.add_argument("--reads")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    if a.mode == "selftest":
        selftest()
    elif a.mode == "dev":
        dev(a.args[0], a.workers)
    elif a.mode == "prompts":
        prompts(a.args[0], a.args[1])
    else:
        run(a.args[0], a.args[1], a.args[2], a.args[3], a.reads, a.workers)
