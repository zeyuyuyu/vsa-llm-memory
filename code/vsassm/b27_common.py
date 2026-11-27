"""B27 common code: loads the ORIGINAL arena modules (verbatim copies in b27_arena/, md5 in b27_arena/MD5SUMS),
builds the three arenas with the published train/test splits plus a new dev split, caches the frozen Phi-2 states
with the original cache functions, and renders the second "view" of each training instance used by the
label-free contrastive arm (same bindings, pair order reshuffled).

Nothing here changes how an original arena instance is generated: train and test rows come from the original
classes with the original seeds; the dev split reuses the original generator code with a new random offset.
"""
import hashlib, importlib.util, os, random, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ARENA_DIR = os.path.join(HERE, "b27_arena")
_ORDER = ["vsa", "systematic_chain_dataset", "systematic_role_binding_dataset", "explicit_chain_models",
          "v13", "v11", "v8", "v9"]
DEV_OFFSET = 777            # dev split of arenas A and B: rng = Random(1000003 * i + 777)
DEV_SEED_C = 777            # dev split of arena C: v13.Arena(tok, n, 777)   (train 42, test 4242)
VIEW2_OFFSET = 990001       # rng of the reshuffled second view: Random(VIEW2_OFFSET + i)
N_SPLIT = {"A": (2000, 500, 1000), "B": (1000, 500, 500), "C": (800, 500, 500)}   # train, dev, test


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def check_md5():
    want = dict(l.split()[::-1] for l in open(os.path.join(ARENA_DIR, "MD5SUMS")) if l.strip())
    bad = [f for f, h in want.items() if md5(os.path.join(ARENA_DIR, f)) != h]
    if bad:
        raise SystemExit("b27_arena copies changed: %s" % bad)


def load_arena_modules():
    """Import the verbatim copies by file path and register them in sys.modules BEFORE any of them runs its own
    `from x import y`, so every import resolves to the copies (the originals insert server paths into sys.path;
    those insertions are removed again afterwards)."""
    check_md5()
    keep = list(sys.path)
    mods = {}
    for n in _ORDER:
        if n in sys.modules and getattr(sys.modules[n], "__file__", "").startswith(ARENA_DIR):
            mods[n] = sys.modules[n]; continue
        spec = importlib.util.spec_from_file_location(n, os.path.join(ARENA_DIR, n + ".py"))
        m = importlib.util.module_from_spec(spec)
        sys.modules[n] = m
        spec.loader.exec_module(m)
        mods[n] = m
    sys.path[:] = keep
    for n, m in mods.items():
        assert os.path.dirname(os.path.abspath(m.__file__)) == ARENA_DIR, (n, m.__file__)
    return mods


def tokenizer():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


# ------------------------------------------------------------------ arena A (SystematicChainDataset, V-1/V-8/V-9)

def chain_cfg(M):
    return M["systematic_chain_dataset"].ChainConfig(num_roles=64, vocab_size=256, num_pairs=16, max_length=512,
                                                     p_positive=0.5)


def gen_chain(idx, offset, cfg, shuffle_rng=None):
    """Line-by-line copy of SystematicChainDataset.__getitem__ up to the text, with the rng offset as a parameter
    (42 = train, 123 = eval/test, DEV_OFFSET = dev). With shuffle_rng given, the three sections are reshuffled
    once more after generation (second view; same bindings, same query, same label)."""
    from systematic_chain_dataset import marker, filler
    rng = random.Random(1000003 * idx + offset)
    roles = list(range(cfg.num_roles)); rng.shuffle(roles); roles = roles[: cfg.num_pairs]
    xs = rng.sample(range(cfg.vocab_size), cfg.num_pairs)
    ys = rng.sample(range(cfg.vocab_size), cfg.num_pairs)
    fs = rng.sample(range(cfg.vocab_size), cfg.num_pairs)
    A = list(zip(roles, xs)); B = list(zip(xs, ys)); C = list(zip(ys, fs))
    rng.shuffle(A); rng.shuffle(B); rng.shuffle(C)
    q_role, q_x = rng.choice(A)
    x_to_y = {x: y for x, y in B}; y_to_f = {y: f_ for y, f_ in C}
    true_f = y_to_f[x_to_y[q_x]]
    label = 1 if rng.random() < cfg.p_positive else 0
    if label == 1:
        cand = true_f
    else:
        other_fs = [f_ for f_ in fs if f_ != true_f]
        cand = rng.choice(other_fs) if other_fs else (true_f + 1) % cfg.vocab_size
    if shuffle_rng is not None:
        shuffle_rng.shuffle(A); shuffle_rng.shuffle(B); shuffle_rng.shuffle(C)
    a_txt = " ".join([marker("R", r) + " " + marker("X", x) for r, x in A])
    b_txt = " ".join([marker("X", x) + " " + marker("Y", y) for x, y in B])
    c_txt = " ".join([marker("Y", y) + " " + filler(f_) for y, f_ in C])
    q_txt = marker("R", q_role) + " " + filler(cand)
    return "A " + a_txt + " B " + b_txt + " C " + c_txt + " Q " + q_txt + " ?", label


class ChainSplit:
    """Dataset-like wrapper with the interface of SystematicChainDataset (items with input_ids, attention_mask,
    labels) for any rng offset; offset 42 / 123 reproduce the original train / eval splits (checked in selftest)."""

    def __init__(self, tok, n, offset, cfg, view2=False):
        import torch
        self.items = []
        for i in range(n):
            text, label = gen_chain(i, offset, cfg, random.Random(VIEW2_OFFSET + i) if view2 else None)
            enc = tok(text, truncation=True, max_length=cfg.max_length, padding="max_length", return_tensors="pt")
            self.items.append({"input_ids": enc["input_ids"].squeeze(0),
                               "attention_mask": enc["attention_mask"].squeeze(0),
                               "labels": torch.tensor(label, dtype=torch.long)})

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


# ------------------------------------------------------------------ arena B (v11 / V-18 span fix)

def make_b_arena(M, tok, n, offset, heldout_query, view2=False):
    """Copy of v11.Arena.__init__ with the rng offset as a parameter. offset 42 + heldout_query False reproduces the
    train split, offset 123 + heldout_query True the test split (checked in selftest). The dev split is
    offset DEV_OFFSET with held-out queries, i.e. drawn like the test split. Also records the token span of the
    query's candidate filler (cand_span), which only the label-free arm with projected candidates reads."""
    V = M["v11"]
    SRB = M["systematic_role_binding_dataset"]
    cfg = SRB.SystematicConfig()
    ds = SRB.SystematicRoleBindingDataset(tok, n, "eval" if heldout_query else "train", cfg)

    class _A:
        pass
    ar = _A(); ar.tok, ar.cfg, ar.rows = tok, cfg, []
    for i in range(n):
        rng = random.Random(1000003 * i + offset)
        qr = rng.randrange(cfg.num_roles)
        tf = rng.choice(ds.train_allowed[qr] if not heldout_query else ds.eval_heldout[qr])
        r2f = ds._sample_context(rng, qr, tf)
        r2f[qr] = tf
        label = 1 if rng.random() < cfg.p_positive else 0
        if label == 1:
            cand = tf
        else:
            wr = rng.choice([r for r in range(cfg.num_roles) if r != qr])
            cand = r2f[wr]
            if cand == tf:
                cand = rng.choice([f for f in range(cfg.num_fillers) if f != tf])
        if view2:
            # same bindings and query, binding order reshuffled with an independent rng
            items = list(r2f.items()); random.Random(VIEW2_OFFSET + i).shuffle(items)
            text = " ".join([SRB.make_role_token(r) + SRB.make_filler_token(f) for r, f in items]) + \
                " Q " + SRB.make_role_token(qr) + SRB.make_filler_token(cand) + " ?"
        else:
            text = ds._render_example(rng, r2f, qr, cand)
        enc = tok(text, truncation=True, max_length=cfg.max_length, padding="max_length",
                  return_tensors="pt", return_offsets_mapping=True)
        offs = enc["offset_mapping"].squeeze(0).tolist()
        pairs = V.parse_pairs(text)
        ctx = [(r, f, V.token_span(offs, cs, ce)) for r, f, cs, ce in pairs[:-1]]
        qr_last, cand_last, ccs, cce = pairs[-1]
        assert qr_last == qr and cand_last == cand
        ar.rows.append({"ids": enc["input_ids"].squeeze(0), "am": enc["attention_mask"].squeeze(0),
                        "label": label, "q_role": qr, "cand": cand, "ctx": ctx,
                        "cand_span": V.token_span(offs, ccs, cce)})
    return ar


# ------------------------------------------------------------------ arena C (v13)

def make_c_view2(M, tok, rows, max_len=768):
    """Second view of arena-C rows: the same pairs in each level, reshuffled; query mid-sequence as in v13."""
    V = M["v13"]
    out = []
    for i, r in enumerate(rows):
        rng = random.Random(VIEW2_OFFSET + i)
        toks = []
        for L in range(V.HOPS):
            pairs = [(a, b) for a, b, _, _ in r["ctx"][L]]
            rng.shuffle(pairs)
            toks += ["[%s%s]{%s}" % (V.LVL[L], V.h2(a), b) for a, b in pairs]
        q = "Q [R%s] {%d} ?" % (V.h2(r["q"]), r["cand"])
        text = " ".join(toks[:len(toks) // 2]) + " " + q + " " + " ".join(toks[len(toks) // 2:])
        out.append(_c_row_from_text(V, tok, text, r["label"], r["q"], r["cand"], max_len))
    return out


def _c_row_from_text(V, tok, text, label, q, cand, max_len=768):
    """Same parsing as v13.Arena.__init__ (role and filler sub-spans, query and candidate spans)."""
    enc = tok(text, truncation=True, max_length=max_len, padding="max_length",
              return_tensors="pt", return_offsets_mapping=True)
    offs = enc["offset_mapping"].squeeze(0).tolist()
    ctx = [[] for _ in range(V.HOPS)]
    for m in V.PAT.finditer(text):
        L = V.LVL.index(m.group(1))
        if L < V.HOPS:
            ctx[L].append((int(m.group(2), 16), int(m.group(3)),
                           V.token_span(offs, *m.span(2)), V.token_span(offs, *m.span(3))))
    qm = V.QPAT.search(text)
    return {"ids": enc["input_ids"].squeeze(0), "am": enc["attention_mask"].squeeze(0), "label": label,
            "q": q, "cand": cand, "ctx": ctx,
            "q_span": V.token_span(offs, *qm.span(1)), "cand_span": V.token_span(offs, *qm.span(2))}


# ------------------------------------------------------------------ split builders

def build_rows(M, tok, arena, split, view2=False):
    """Rows of one split. A: dataset-like ChainSplit; B, C: lists of row dicts (original Arena format)."""
    ntr, ndev, nte = N_SPLIT[arena]
    if arena == "A":
        cfg = chain_cfg(M)
        n, off = {"train": (ntr, 42), "dev": (ndev, DEV_OFFSET), "test": (nte, 123)}[split]
        return ChainSplit(tok, n, off, cfg, view2=view2)
    if arena == "B":
        n, off, held = {"train": (ntr, 42, False), "dev": (ndev, DEV_OFFSET, True),
                        "test": (nte, 123, True)}[split]
        return make_b_arena(M, tok, n, off, held, view2=view2).rows
    V = M["v13"]
    n, sd = {"train": (ntr, 42), "dev": (ndev, DEV_SEED_C), "test": (nte, 4242)}[split]
    rows = V.Arena(tok, n, sd).rows
    return make_c_view2(M, tok, rows) if view2 else rows


def feats_path(root, arena, split, view2=False):
    return os.path.join(root, "%s_%s%s.pt" % (arena, split, "_v2" if view2 else ""))


def cache_feats(M, tok, llm, device, arena, split, root, view2=False):
    """Frozen Phi-2 states with the ORIGINAL cache functions and batch sizes (v13.cache bs 4, v11.cache bs 4,
    v8.cache_hidden bs 8), saved in fp16. For A the saved tuple is (ids, am, H, y) exactly as v8.cache_hidden."""
    import torch
    p = feats_path(root, arena, split, view2)
    if os.path.exists(p):
        return p
    rows = build_rows(M, tok, arena, split, view2)
    if arena == "A":
        obj = M["v8"].cache_hidden(rows, llm, device)
    elif arena == "B":
        class _W:
            pass
        w = _W(); w.rows = rows
        obj = M["v11"].cache(w, llm, device)
    else:
        obj = M["v13"].cache(rows, llm, device)
    torch.save(obj, p + ".tmp")
    os.replace(p + ".tmp", p)
    return p
