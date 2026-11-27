"""B28 common code: the binding encodings under test, the loader for the verbatim ngcls modules, structural alias
flags, and interval helpers. Prereg: prereg/B28_noncommutative_binding_prereg.md.

Every encoding keeps the published codebook (same rows, same seed, d = 2048), the published bundling (a sum of
bound pairs), unbinding by circular correlation, and the published clean-up (cosine argmax over the published
candidate set). Only the way the two arguments of a pair are combined changes. In every task the pair is
(VALUE, KEY): the value is the item the query retrieves, the key is what the query unbinds by. A symbol i has a
key code K(i) and a value code V(i):

  comm   K(i) = V(i) = sym[i]                                       published HRR; bind is commutative
  perm   K(i) = sym[i], V(i) = sym[i][P], P a fixed RANDOM permutation of the d coordinates
                                                                    Plate / Kanerva / Gayler: permute one argument
  roles  K(i) = sym[i], V(i) = sym2[i], sym2 an independent random codebook
                                                                    separate role (key) and filler (value) codebooks
  rtag   K(i) = r_k (*) sym[i], V(i) = r_v (*) sym[i]  (normalised), r_k, r_v random role vectors
                                                                    CONTROL: role vectors bound by convolution
  shift  K(i) = sym[i], V(i) = roll(sym[i], 1)                      CONTROL: the familiar cyclic-shift permutation

A stored pair is bind(V(value), KEY) and a query unbinds by KEY and compares with V(candidates). For comm this is
bit for bit the published computation. rtag and shift are controls registered to FAIL: convolution is
commutative, associative and shift-equivariant, so V(x) = c (*) K(x) for a fixed c in both, and unbinding a
pair whose VALUE is the query returns c (*) (that pair's key) = V(that key) -- the alias survives.
"""
import hashlib, importlib.util, math, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NG_DIR = os.path.join(HERE, "b28_ngcls")
NG_ORDER = ["v13", "v5", "v30", "v37b", "v50", "v52", "v58"]
ENCODINGS = ["comm", "perm", "roles", "rtag", "shift"]
PRIMARY, SECONDARY, CONTROLS = "perm", "roles", ["rtag", "shift"]
OFF = {"perm": 7001, "roles": 7002, "rtag_k": 7003, "rtag_v": 7004}
SEEDS = [42, 123, 456]
VSA_DIM = 2048


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def check_md5():
    want = dict(l.split()[::-1] for l in open(os.path.join(NG_DIR, "MD5SUMS")) if l.strip())
    bad = [f for f, h in want.items() if md5(os.path.join(NG_DIR, f)) != h]
    if bad:
        raise SystemExit("b28_ngcls copies changed: %s" % bad)


def load_ngcls():
    """Import the verbatim copies by path and register them in sys.modules before any of them runs its own
    `import v13` / `from v5 import ...`, so every import resolves to the copies (the originals insert the server
    repository into sys.path; those insertions are undone afterwards)."""
    check_md5()
    keep = list(sys.path)
    mods = {}
    for n in NG_ORDER:
        if n in sys.modules and getattr(sys.modules[n], "__file__", "").startswith(NG_DIR):
            mods[n] = sys.modules[n]; continue
        spec = importlib.util.spec_from_file_location(n, os.path.join(NG_DIR, n + ".py"))
        m = importlib.util.module_from_spec(spec)
        sys.modules[n] = m
        spec.loader.exec_module(m)
        mods[n] = m
    sys.path[:] = keep
    for n, m in mods.items():
        assert os.path.dirname(os.path.abspath(m.__file__)) == NG_DIR, (n, m.__file__)
    return mods


def tokenizer():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


# ------------------------------------------------------------------ the algebra (torch; same primitives as v13)

def bind(a, b):
    import torch
    return torch.fft.irfft(torch.fft.rfft(a, dim=-1) * torch.fft.rfft(b, dim=-1), n=a.size(-1), dim=-1)


def unbind(ab, a):
    import torch
    return torch.fft.irfft(torch.fft.rfft(ab, dim=-1) * torch.conj(torch.fft.rfft(a, dim=-1)),
                           n=ab.size(-1), dim=-1)


def unit_rows(n, d, seed):
    import torch, torch.nn.functional as F
    g = torch.Generator(); g.manual_seed(seed)
    return F.normalize(torch.randn(n, d, generator=g), dim=1)


class Encoding:
    """Key and value codes of every symbol, from the published codebook `sym` (n x d, unit rows). Draws its own
    random material from private generators, so the global torch RNG (and with it every trained arm) is
    untouched by the choice of encoding."""

    def __init__(self, name, sym, seed):
        import torch, torch.nn.functional as F
        assert name in ENCODINGS, name
        self.name = name
        n, d = sym.shape
        dev = sym.device
        if name == "comm":
            self.Kb, self.Vb = sym, sym
        elif name == "perm":
            P = torch.randperm(d, generator=torch.Generator().manual_seed(seed + OFF["perm"])).to(dev)
            self.P = P
            self.Kb, self.Vb = sym, sym[:, P].contiguous()
        elif name == "roles":
            self.Kb, self.Vb = sym, unit_rows(n, d, seed + OFF["roles"]).to(dev)
        elif name == "rtag":
            rk = unit_rows(1, d, seed + OFF["rtag_k"]).to(dev)
            rv = unit_rows(1, d, seed + OFF["rtag_v"]).to(dev)
            self.Kb = F.normalize(bind(rk.expand(n, d), sym), dim=1)
            self.Vb = F.normalize(bind(rv.expand(n, d), sym), dim=1)
        elif name == "shift":
            self.Kb, self.Vb = sym, torch.roll(sym, 1, dims=1)

    def K(self, i):
        return self.Kb[i]

    def V(self, i):
        return self.Vb[i]


# ------------------------------------------------------------------ structural alias flags (gold symbols only)

INV = {"north": "south", "south": "north", "east": "west", "west": "east"}


def alias_task4(stmts, qdir, qroom, label):
    """Under a commutative binding every stored term {dir, a, b} is unordered. The query (qdir, X) matches every
    stored term with that direction that contains X and returns its other room. The instance is ALIASED iff some
    returned room is not the gold answer (every mentioned room is a clean-up candidate)."""
    terms = []
    for a, d, b in stmts:
        terms.append((d, a, b))            # bind(V(a), bind(d, K(b)))
        terms.append((INV[d], b, a))       # bind(V(b), bind(inv d, K(a)))
    ret = set()
    for d, a, b in terms:
        if d != qdir:
            continue
        if b == qroom:
            ret.add(a)
        if a == qroom:
            ret.add(b)
    return int(any(x != label for x in ret))


def alias_task15(cls, mem, q, label):
    """Hop 1 binds class to entity (disjoint codebook rows: no alias). Hop 2 unbinds the class statements by the
    entity's class c; under commutation a statement (c', p') with p' == c returns c'. ALIASED iff such a c' differs
    from the gold property and is a clean-up candidate (the candidates are the properties of the statements)."""
    cl = [c for e, c in mem if e == q]
    if not cl:
        return 0
    c = cl[0]
    cands = {p for _, p in cls}
    spur = {c2 for c2, p2 in cls if p2 == c and c2 != label and c2 in cands}
    return int(bool(spur))


def alias_arenaC(ctx, q, chain_s1, chain_s2):
    """Hop 1: a level-0 pair (a', b') with b' == q returns a' as a second clean candidate; it can win only if a' is
    one of the level-0 fillers (the clean-up set) and differs from s1. Hop 2 likewise with s1 at level 1."""
    def hop(pairs, key, truth):
        fill = {b for a, b in pairs}
        return any(b == key and a != truth and a in fill for a, b in pairs)
    h1 = hop(ctx[0], q, chain_s1)
    h2 = hop(ctx[1], chain_s1, chain_s2)
    return int(h1 or h2), int(h1), int(h2)


# ------------------------------------------------------------------ intervals

def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def boot_mean(correct, B=10000, seed=0):
    """correct: (seeds, n) 0/1 on the same questions; resample questions, mean over seeds."""
    rng = np.random.default_rng(seed)
    C = np.asarray(correct, float)
    if C.ndim == 1:
        C = C[None]
    n = C.shape[1]
    idx = rng.integers(0, n, size=(B, n))
    m = C[:, idx].mean(axis=2).mean(axis=0)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def boot_diff(a, b, B=10000, seed=1):
    """Paired difference of means (a - b), each (seeds, n) on the same questions; resample questions."""
    rng = np.random.default_rng(seed)
    A = np.asarray(a, float); Bm = np.asarray(b, float)
    if A.ndim == 1:
        A = A[None]
    if Bm.ndim == 1:
        Bm = Bm[None]
    n = A.shape[1]
    idx = rng.integers(0, n, size=(B, n))
    d = A[:, idx].mean(axis=2).mean(axis=0) - Bm[:, idx].mean(axis=2).mean(axis=0)
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
