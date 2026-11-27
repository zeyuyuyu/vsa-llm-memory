"""B44 arena-C data (prereg/B44_data_scaling_heldout_cot_crosstalk_prereg.md).

Arena C exactly as B31 / v13.Arena (two hops, 64 shared symbols, N_CTX pairs per level, the query mid-sequence, roles
written in hexadecimal "[Rxx]" / "[Fxx]", fillers in decimal "{n}"), generated here with no file read (no tokenizer, no
machine-dependent word list): the random stream is Python's `random.Random(seed)`, consumed line by line as in
v13.Arena.__init__ (and b31_struct.build_C). The self-test checks that seeds 42 / 777 / 4242 reproduce B31's arena-C
train / dev / test texts exactly.

Variants
  std  v13's generator, unchanged (new seeds for the B44 training sets).
  ho   held-out pairs: a fixed partition of the 64 x 64 (role, filler) pairs, half per role (random, seed 4401). Training
       instances use only TRAIN-half pairs at both levels (distractors and chain); dev and test instances use only
       TEST-half pairs. A pair is the written (role symbol, filler symbol) pair, regardless of level, so no (a, b) that
       appears anywhere in a test instance appears anywhere in a training instance. Negative candidates are drawn from
       the same half of the bridge symbol as the true answer (so the half membership of the candidate carries no
       label information in either split).

Splits (fixed in the prereg)
  std: train stream seed 44042 (80,000; the 800- and 8,000-instance sets are its first 800 / 8,000 instances: nested),
       dev = B31 dev (seed 777, 500), test = B31 test (seed 4242, 500).
  ho : train stream seed 44142 (80,000; nested 800 / 8,000), dev seed 44777 (500), test seed 44242 (500).
  k-sweep (Fig. 2c): std generator with N_CTX = k pairs per level, train seed 42 (800), dev 777 (500), test 4242 (500);
       at k = 32 these are B31's arena-C splits.
"""
import hashlib, json, os, random, re, sys
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

N_SYM, HOPS = 64, 2
LVL = "RFGH"
PAT = re.compile(r"\[([RFGH])([0-9A-F]{2})\]\{(\d+)\}")            # v13.PAT
QPAT = re.compile(r"Q \[R([0-9A-F]{2})\] \{(\d+)\} \?")             # v13.QPAT
HEX = "0123456789ABCDEF"
SIZES = (800, 8000, 80000)
N_MAX = 80000
SEEDS = {"std": {"train": 44042, "dev": 777, "test": 4242},
         "ho": {"train": 44142, "dev": 44777, "test": 44242}}
KS = (2, 4, 8, 16, 24, 32)
KSEEDS = {"train": (800, 42), "dev": (500, 777), "test": (500, 4242)}
PART_SEED = 4401


def h2(n):
    return HEX[(n >> 4) & 0xF] + HEX[n & 0xF]


def partition(seed=PART_SEED):
    """(train_half, test_half): per role symbol a, two disjoint lists of 32 filler symbols."""
    rng = random.Random(seed)
    test_half, train_half = {}, {}
    for a in range(N_SYM):
        p = list(range(N_SYM)); rng.shuffle(p)
        test_half[a] = sorted(p[:N_SYM // 2])
        train_half[a] = sorted(p[N_SYM // 2:])
    return train_half, test_half


def _ns(n_ctx):
    return SimpleNamespace(HOPS=HOPS, PAT=PAT, LVL=LVL, QPAT=QPAT, N_CTX=n_ctx)


def _struct(text, label, q, cand, n_ctx):
    """b31_struct._c_struct (same mention table: context R/F per pair in text order, then query R and candidate F)."""
    import b31_struct as S
    return S._c_struct(text, label, q, cand, _ns(n_ctx))


def gen(n, seed, variant="std", split="train", n_ctx=32, struct=True):
    """n instances of arena C. std: line by line v13.Arena.__init__ / b31_struct.build_C. ho: see module doc."""
    rng = random.Random(seed)
    allowed = None
    if variant == "ho":
        tr, te = partition()
        allowed = tr if split == "train" else te
    out = []
    for _ in range(n):
        if allowed is None:
            chain = [rng.randrange(N_SYM) for _ in range(HOPS + 1)]
        else:
            chain = [rng.randrange(N_SYM)]
            for _L in range(HOPS):
                chain.append(rng.choice(allowed[chain[-1]]))
        levels, toks = [], []
        for L in range(HOPS):
            pairs, seen = [(chain[L], chain[L + 1])], {chain[L]}
            while len(pairs) < n_ctx:
                a = rng.randrange(N_SYM)
                if a in seen:
                    continue
                seen.add(a)
                pairs.append((a, rng.randrange(N_SYM) if allowed is None else rng.choice(allowed[a])))
            rng.shuffle(pairs); levels.append(pairs)
        for L, pairs in enumerate(levels):
            for a, b in pairs:
                toks.append("[%s%s]{%s}" % (LVL[L], h2(a), b))
        label = 1 if rng.random() < 0.5 else 0
        if label == 1:
            cand = chain[-1]
        elif allowed is None:
            cand = rng.choice([s for s in range(N_SYM) if s != chain[-1]])
        else:
            cand = rng.choice([s for s in allowed[chain[-2]] if s != chain[-1]])
        q = "Q [R%s] {%d} ?" % (h2(chain[0]), cand)
        text = " ".join(toks[:len(toks) // 2]) + " " + q + " " + " ".join(toks[len(toks) // 2:])
        if struct:
            s = _struct(text, label, chain[0], cand, n_ctx)
            s["chain"] = chain
            s["levels"] = levels
            out.append(s)
        else:
            out.append({"text": text, "label": label, "chain": chain, "levels": levels, "cand": cand})
    return out


def split(variant, name, n=None, struct=True):
    """The registered splits. name in train / dev / test; n (train only) in SIZES (nested prefixes)."""
    if name == "train":
        n = n or N_MAX
        assert n in SIZES
        return gen(n, SEEDS[variant]["train"], variant, "train", struct=struct)
    return gen(500, SEEDS[variant][name], variant, name, struct=struct)


def ksplit(k, name, struct=True):
    n, seed = KSEEDS[name]
    return gen(n, seed, "std", name, n_ctx=k, struct=struct)


def pairs_of(inst):
    return {(a, b) for lv in inst["levels"] for a, b in lv}


def md5_texts(st):
    return hashlib.md5(json.dumps([s["text"] for s in st]).encode()).hexdigest()


# ------------------------------------------------------------------ gold answers and an independent parse

def dict_answer(text):
    """Dictionary walk from the raw text (regex parse, independent of the generator's own records)."""
    ctx = [{}, {}]
    for L, a, b in re.findall(r"\[([RF])([0-9A-F]{2})\]\{(\d+)\}", text):
        d = ctx["RF".index(L)]
        assert int(a, 16) not in d
        d[int(a, 16)] = int(b)
    m = re.search(r"Q \[R([0-9A-F]{2})\] \{(\d+)\} \?", text)
    k = int(m.group(1), 16)
    bridge = ctx[0][k]
    ans = ctx[1][bridge]
    return {"q": k, "bridge": bridge, "answer": ans, "cand": int(m.group(2)), "label": int(ans == int(m.group(2)))}


# ------------------------------------------------------------------ self-test

def selftest(out_path=None):
    import b31_struct as S
    rep = {}
    for nm in ("train", "dev", "test"):
        b31 = [s["text"] for s in S.build("C", nm)]
        mine = [s["text"] for s in gen(len(b31), {"train": 42, "dev": 777, "test": 4242}[nm], "std", nm)]
        rep["b31_C_%s_identical" % nm] = sum(x == y for x, y in zip(b31, mine))
        rep["b31_C_%s_n" % nm] = len(b31)
    for k in KS:
        for nm in ("train", "dev", "test"):
            st = ksplit(k, nm)
            rep["k%d_%s_dict_equals_label" % (k, nm)] = sum(dict_answer(s["text"])["label"] == s["label"] for s in st)
    tr_half, te_half = partition()
    rep["partition_disjoint"] = all(not set(tr_half[a]) & set(te_half[a]) and len(tr_half[a]) == 32 for a in range(64))
    for v in ("std", "ho"):
        tr = split(v, "train", N_MAX)
        dv, te = split(v, "dev"), split(v, "test")
        rep["%s_train_md5" % v] = md5_texts(tr)
        for nm, st in (("train", tr), ("dev", dv), ("test", te)):
            rep["%s_%s_dict_equals_label" % (v, nm)] = sum(dict_answer(s["text"])["label"] == s["label"] for s in st)
            rep["%s_%s_n" % (v, nm)] = len(st)
            rep["%s_%s_pos_rate" % (v, nm)] = round(sum(s["label"] for s in st) / len(st), 4)
        ttexts = set(s["text"] for s in tr)
        rep["%s_train_texts_in_dev_or_test" % v] = sum(s["text"] in ttexts for s in dv + te)
        test_pairs = set().union(*[pairs_of(s) for s in te])
        for n in SIZES:
            trp = set().union(*[pairs_of(s) for s in tr[:n]])
            rep["%s_n%d_test_pairs_seen_in_train" % (v, n)] = round(len(test_pairs & trp) / len(test_pairs), 4)
            # per test instance: share of its 64 pairs that occur in the training set; and the two chain pairs
            ch = [((s["chain"][0], s["chain"][1]) in trp) and ((s["chain"][1], s["chain"][2]) in trp) for s in te]
            rep["%s_n%d_test_chain_pairs_both_seen" % (v, n)] = round(sum(ch) / len(ch), 4)
        if v == "ho":
            dev_pairs = set().union(*[pairs_of(s) for s in dv])
            rep["ho_dev_pairs_seen_in_train"] = len(dev_pairs & set().union(*[pairs_of(s) for s in tr]))
            # negatives: candidate symbol in the same half of the bridge symbol as the answer, in both splits
            negs = [x for x in tr[:2000] if x["label"] == 0]
            rep["ho_train_neg_cand_in_train_half_of_bridge"] = "%d/%d" % (
                sum(x["mentions"][x["cand"]][2] in tr_half[x["chain"][1]] for x in negs), len(negs))
            negs = [x for x in te if x["label"] == 0]
            rep["ho_test_neg_cand_in_test_half_of_bridge"] = "%d/%d" % (
                sum(x["mentions"][x["cand"]][2] in te_half[x["chain"][1]] for x in negs), len(negs))
    print(json.dumps(rep, indent=1))
    if out_path:
        json.dump(rep, open(out_path, "w"), indent=1)
    return rep


if __name__ == "__main__":
    selftest(sys.argv[1] if len(sys.argv) > 1 else None)
