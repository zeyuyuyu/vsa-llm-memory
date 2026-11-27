"""B31 preparation (prereg/B31_encoder_equalised_baselines_prereg.md).

  selftest  (CPU) the structure layer reproduces the B27 rows under the Phi-2 tokenizer (token ids; B/C spans; A spans
            against the published token scanner, recording which markers the scanner sees); Qwen3-8B tokenization
            lengths and span decoding; dictionary answers equal the labels; dev/train/test overlap. Writes
            out/selftest_<enc>.json and, for Phi-2, out/scanner_visible_A.pt (the markers the published A scanner
            finds, used ONLY by the G0 emulation of the published A arms).
  feats     (GPU) per encoder, arena and split: the span-mean final state of every mention (fp32), the state at the
            last non-pad token and the mean over the input. Phi-2 reads the B27 caches of the frozen fp16 states (made
            by the original cache functions) and pools them on the GPU exactly as the published arms do; Qwen3-8B is
            run here (bf16, AutoModel last_hidden_state = after the final norm).
"""
import argparse, hashlib, json, os, sys, time
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import b27_common as BC                                                     # noqa: E402
import b31_struct as S                                                      # noqa: E402

QWEN8 = os.environ.get("B31_QWEN8", "/path/to/work/hf/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218")
B27_FEATS = "/path/to/work/b27/feats"
SPLITS = ["train", "dev", "test", "train_v2"]


def tokenizer(enc):
    from transformers import AutoTokenizer
    if enc == "phi2":
        return BC.tokenizer()
    tok = AutoTokenizer.from_pretrained(QWEN8)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def structs(arena, split):
    v2 = split.endswith("_v2")
    return S.build(arena, split[:-3] if v2 else split, view2=v2)


def h(x):
    return hashlib.md5(json.dumps(x).encode()).hexdigest()


# ------------------------------------------------------------------ self-test

def selftest(a):
    M = BC.load_arena_modules()
    tok = tokenizer(a.enc)
    rep = {"enc": a.enc, "padding_side": tok.padding_side}
    vis_all = {}
    for ar in S.ARENAS:
        texts = {}
        for sp in SPLITS:
            st = structs(ar, sp)
            r = {"n": len(st)}
            r["dict_equals_label"] = sum(S.dict_answer(ar, s)[0] == s["label"] for s in st)
            r["mentions_per_instance"] = sorted(set(len(s["mentions"]) for s in st))
            texts[sp] = set(s["text"] for s in st)
            if a.enc == "phi2":
                base = sp[:-3] if sp.endswith("_v2") else sp
                rows = BC.build_rows(M, tok, ar, base, view2=sp.endswith("_v2"))
                ok_ids = ok_sp = 0
                if ar == "A":
                    ECM = M["explicit_chain_models"]
                    scan = type("o", (), {"tokenizer": tok})()
                    vis, drop = [], {}
                    for s, it in zip(st, rows.items):
                        ids, am, spans = S.tokenize(tok, s, S.MAXLEN_PHI2[ar], pad=True)
                        ok_ids += int(torch.equal(torch.tensor(ids), it["input_ids"]))
                        ms = ECM.SymbolicVSAChainClassifier._scan_markers(scan, it["input_ids"])
                        by_start = {(k, v, i): j for k, v, i, j in ms}
                        v_ = []
                        for (key, kind, sym, cs, ce), (lo, hi) in zip(s["mentions"], spans):
                            if kind == "F":
                                v_.append(1); continue
                            j = by_start.get((kind, sym, lo))
                            v_.append(int(j is not None))
                            if j is None:
                                drop[kind] = drop.get(kind, 0) + 1
                            elif j != hi:
                                v_[-1] = -1                     # scanner span differs: counted below
                        ok_sp += int(all(x != -1 for x in v_) and sum(x == 1 for x, m in zip(v_, s["mentions"])
                                                                     if m[1] != "F") == len(ms))
                        vis.append(v_)
                    vis_all[sp] = torch.tensor(vis, dtype=torch.int8)
                    r["scanner_dropped_markers_by_kind"] = drop
                    r["instances_with_a_dropped_marker"] = int(sum(int((t[:-2] == 0).any()) for t in vis_all[sp]))
                else:
                    for s, row in zip(st, rows):
                        ids, am, spans = S.tokenize(tok, s, S.MAXLEN_PHI2[ar], pad=True)
                        ok_ids += int(torch.equal(torch.tensor(ids), row["ids"]))
                        if ar == "B":
                            want = [sp_ for _, _, sp_ in row["ctx"]]
                            got = [spans[f] for _, f in s["ctx"]]
                            ok = got == want and (row["label"], row["q_role"], row["cand"]) == \
                                (s["label"], s["mentions"][s["q"]][2], s["mentions"][s["cand"]][2])
                            if "cand_span" in row:
                                ok = ok and spans[s["cand"]] == row["cand_span"]
                        else:
                            want = [(a_, b_, spr, spf) for L in range(2) for a_, b_, spr, spf in row["ctx"][L]]
                            got = [(s["mentions"][r_][2], s["mentions"][f_][2], spans[r_], spans[f_])
                                   for L in range(2) for r_, f_ in s["lv"][L]]
                            ok = got == want and spans[s["q"]] == row["q_span"] and \
                                spans[s["cand"]] == row["cand_span"] and \
                                (row["label"], row["q"], row["cand"]) == (s["label"], s["mentions"][s["q"]][2],
                                                                          s["mentions"][s["cand"]][2])
                        ok_sp += int(ok)
                r["token_ids_equal_b27_rows"] = ok_ids
                r["spans_equal_published"] = ok_sp
            else:
                lens, bad = [], 0
                for s in st:
                    ids, am, spans = S.tokenize(tok, s)
                    lens.append(len(ids))
                    for (key, kind, sym, cs, ce), (lo, hi) in zip(s["mentions"], spans):
                        d = tok.decode(ids[lo:hi])
                        if s["text"][cs:ce] not in d:
                            bad += 1
                r["max_len"], r["mean_len"] = max(lens), round(sum(lens) / len(lens), 1)
                r["mention_span_decode_failures"] = bad
            rep["%s_%s" % (ar, sp)] = r
            print(ar, sp, json.dumps(r), flush=True)
        rep["%s_overlap" % ar] = {"dev_train": len(texts["dev"] & texts["train"]),
                                  "dev_test": len(texts["dev"] & texts["test"]),
                                  "test_train": len(texts["test"] & texts["train"])}
    os.makedirs(a.out, exist_ok=True)
    json.dump(rep, open(os.path.join(a.out, "selftest_%s.json" % a.enc), "w"), indent=1)
    if a.enc == "phi2":
        torch.save(vis_all, os.path.join(a.out, "scanner_visible_A.pt"))


# ------------------------------------------------------------------ features

def pool_instance(Hi, L, spans, dev):
    """Hi: (T, d) states of one instance on `dev` (fp16 for Phi-2 exactly as cached, bf16 for Qwen). Span means are
    computed as the published arms do: H[s:e].float().mean(0)."""
    F_ = torch.stack([(Hi[s:e] if e > s else Hi[s:s + 1]).float().mean(0) for s, e in spans])
    return F_, Hi[L - 1].float(), Hi[:L].float().mean(0)


def feats(a):
    tok = tokenizer(a.enc)
    dev = a.device
    os.makedirs(a.feats, exist_ok=True)
    llm = None
    for item in a.items:
        ar, sp = item.split(":")
        p = os.path.join(a.feats, "%s_%s_%s.pt" % (a.enc, ar, sp))
        if os.path.exists(p):
            print("exists", p, flush=True); continue
        t0 = time.time()
        st = structs(ar, sp)
        toks = [S.tokenize(tok, s, S.MAXLEN_PHI2[ar], pad=(a.enc == "phi2")) for s in st]
        Fs, Ls, Ms = [], [], []
        if a.enc == "phi2":
            v2 = sp.endswith("_v2")
            obj = torch.load(BC.feats_path(B27_FEATS, ar, sp[:-3] if v2 else sp, v2), weights_only=False)
            Hc = obj[2] if ar == "A" else obj
            ids_c = obj[0] if ar == "A" else None
            for i, (ids, am, spans) in enumerate(toks):
                if ids_c is not None:
                    assert torch.equal(torch.tensor(ids), ids_c[i])
                Hi = Hc[i].to(dev)
                f, l, m = pool_instance(Hi, int(sum(am)), spans, dev)
                Fs.append(f.cpu()); Ls.append(l.cpu()); Ms.append(m.cpu())
            del Hc, obj
        else:
            if llm is None:
                from transformers import AutoModel
                llm = AutoModel.from_pretrained(QWEN8, torch_dtype=torch.bfloat16).to(dev).eval()
                for q in llm.parameters():
                    q.requires_grad_(False)
            order = sorted(range(len(toks)), key=lambda i: len(toks[i][0]))
            res = {}
            with torch.no_grad():
                for k in range(0, len(order), a.bs):
                    idx = order[k:k + a.bs]
                    T = max(len(toks[i][0]) for i in idx)
                    ids = torch.full((len(idx), T), tok.pad_token_id, dtype=torch.long)
                    am = torch.zeros((len(idx), T), dtype=torch.long)
                    for b, i in enumerate(idx):
                        n = len(toks[i][0]); ids[b, :n] = torch.tensor(toks[i][0]); am[b, :n] = 1
                    Hb = llm(input_ids=ids.to(dev), attention_mask=am.to(dev)).last_hidden_state
                    for b, i in enumerate(idx):
                        res[i] = [t.cpu() for t in pool_instance(Hb[b], len(toks[i][0]), toks[i][2], dev)]
            for i in range(len(toks)):
                Fs.append(res[i][0]); Ls.append(res[i][1]); Ms.append(res[i][2])
        obj = {"F": torch.stack(Fs), "last": torch.stack(Ls), "mean": torch.stack(Ms),
               "len": torch.tensor([len(t[0]) if a.enc != "phi2" else int(sum(t[1])) for t in toks]),
               "spans": [t[2] for t in toks], "labels": torch.tensor([s["label"] for s in st]),
               "text_md5": h([s["text"] for s in st]), "enc": a.enc, "arena": ar, "split": sp}
        torch.save(obj, p + ".tmp"); os.replace(p + ".tmp", p)
        print("feats", a.enc, item, tuple(obj["F"].shape), "%.0fs" % (time.time() - t0), flush=True)


def feats_layers(a):
    """Exploratory 1 (prereg/B31_exploratory1.md): Qwen3-8B span means from intermediate layers (hidden_states[L], the
    output of block L; 36 blocks), one forward pass per batch for all requested layers. Files qwen8L<L>_<arena>_<split>.pt
    in the same format as the final-layer files."""
    assert a.enc == "qwen8" and a.layers
    tok = tokenizer("qwen8")
    dev = a.device
    from transformers import AutoModel
    llm = AutoModel.from_pretrained(QWEN8, torch_dtype=torch.bfloat16).to(dev).eval()
    for q in llm.parameters():
        q.requires_grad_(False)
    for item in a.items:
        ar, sp = item.split(":")
        paths = {L: os.path.join(a.feats, "qwen8L%d_%s_%s.pt" % (L, ar, sp)) for L in a.layers}
        if all(os.path.exists(p) for p in paths.values()):
            continue
        t0 = time.time()
        st = structs(ar, sp)
        toks = [S.tokenize(tok, s) for s in st]
        order = sorted(range(len(toks)), key=lambda i: len(toks[i][0]))
        res = {L: {} for L in a.layers}
        with torch.no_grad():
            for k in range(0, len(order), a.bs):
                idx = order[k:k + a.bs]
                T = max(len(toks[i][0]) for i in idx)
                ids = torch.full((len(idx), T), tok.pad_token_id, dtype=torch.long)
                am = torch.zeros((len(idx), T), dtype=torch.long)
                for b, i in enumerate(idx):
                    n = len(toks[i][0]); ids[b, :n] = torch.tensor(toks[i][0]); am[b, :n] = 1
                hs = llm(input_ids=ids.to(dev), attention_mask=am.to(dev), output_hidden_states=True).hidden_states
                for L in a.layers:
                    for b, i in enumerate(idx):
                        res[L][i] = [t.cpu() for t in pool_instance(hs[L][b], len(toks[i][0]), toks[i][2], dev)]
        for L in a.layers:
            obj = {"F": torch.stack([res[L][i][0] for i in range(len(toks))]),
                   "last": torch.stack([res[L][i][1] for i in range(len(toks))]),
                   "mean": torch.stack([res[L][i][2] for i in range(len(toks))]),
                   "len": torch.tensor([len(t[0]) for t in toks]), "spans": [t[2] for t in toks],
                   "labels": torch.tensor([s["label"] for s in st]), "text_md5": h([s["text"] for s in st]),
                   "enc": "qwen8L%d" % L, "arena": ar, "split": sp}
            torch.save(obj, paths[L] + ".tmp"); os.replace(paths[L] + ".tmp", paths[L])
        print("feats layers", a.layers, item, "%.0fs" % (time.time() - t0), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["selftest", "feats", "feats_layers"])
    ap.add_argument("--layers", type=int, nargs="*", default=[])
    ap.add_argument("--enc", required=True, choices=["phi2", "qwen8"])
    ap.add_argument("--items", nargs="*", default=[])
    ap.add_argument("--feats", default="/path/to/work/b31/feats")
    ap.add_argument("--out", default="/path/to/work/b31/out")
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    {"selftest": selftest, "feats": feats, "feats_layers": feats_layers}[a.cmd](a)


if __name__ == "__main__":
    main()
