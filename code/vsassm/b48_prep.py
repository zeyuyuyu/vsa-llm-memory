"""B48 preparation (prereg/B48_one_form_and_alias_naming_prereg.md).

  selftest  (CPU) arena C1: chains, labels and pair order equal arena C's instance by instance (only the role digits
            differ); every symbol's mentions have identical token ids in every position (role, filler, query,
            candidate), under the Phi-2 and the Qwen3-8B tokenizer; dictionary answer = label; no truncation at 768
            (Phi-2); the same for arena C (how many symbols are written with different token ids there). Stories W / P:
            spans decode to the mention; lengths. Writes <out>/selftest_<enc>.json.
  featsC1   (GPU) span-mean final-layer states of every C1 mention, the last-token state and the input mean, in B31's
            format (<feats>/<enc>_C_<split>.pt, text md5 of C1). Phi-2: AutoModel fp16, rows padded to 768 as the
            published arena rows (v13.cache: batch 4, last_hidden_state in fp16), span means in fp32 on the GPU as the
            published arms take them. Qwen3-8B: b31_prep.feats unchanged (bf16, final norm), on the C1 structures.
  featsB    (GPU) Qwen3-8B span means of every story mention at hidden_states[L] for L in --layers (36 = the output of
            the final norm = last_hidden_state), stored in fp16: <feats>/qwen8L<L>_<W|P>_<split>.pt.
"""
import argparse, json, os, sys, time
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import b27_common as BC                                                     # noqa: E402
import b31_struct as S31                                                    # noqa: E402
import b31_prep as P31                                                      # noqa: E402
import b48_struct as S48                                                    # noqa: E402

LAYERS = [9, 18, 27, 36]


def selftest(a):
    tok = P31.tokenizer(a.enc)
    rep = {"enc": a.enc}
    for sp in ("train", "dev", "test"):
        c = S31.build("C", sp)
        c1 = S48.build_C1(sp)
        same = sum(x["chain"] == y["chain"] and x["label"] == y["label"] and x["lv"] == y["lv"] and
                   [m[1:3] for m in x["mentions"]] == [m[1:3] for m in y["mentions"]] for x, y in zip(c, c1))
        r = {"n": len(c1), "same_chain_label_structure_as_C": same,
             "dict_equals_label": sum(S31.dict_answer("C", s)[0] == s["label"] for s in c1)}
        for name, st in (("C1", c1), ("C", c)):
            n_sym, n_multi, n_inst_bad, lens, trunc = 0, 0, 0, [], 0
            for s in st:
                if a.enc == "phi2":
                    ids, am, spans = S31.tokenize(tok, s, 768, pad=True)
                    L = int(sum(am)); trunc += int(L >= 768)
                else:
                    ids, am, spans = S31.tokenize(tok, s)
                    L = len(ids)
                lens.append(L)
                by = {}
                for (key, kind, sym, cs, ce), (lo, hi) in zip(s["mentions"], spans):
                    by.setdefault(sym, set()).add(tuple(ids[lo:hi]))
                n_sym += len(by)
                bad = sum(len(v) > 1 for v in by.values())
                n_multi += bad; n_inst_bad += int(bad > 0)
            r[name] = {"symbols": n_sym, "symbols_with_more_than_one_token_form": n_multi,
                       "instances_with_such_a_symbol": n_inst_bad, "max_len": max(lens), "truncated": trunc}
        rep["C_" + sp] = r
        print(sp, json.dumps(r), flush=True)
    for kind in "WP":
        for sp in ("train", "dev", "test"):
            st = S48.check_B(kind, sp)
            lens, bad = [], 0
            for s in st:
                enc = tok(s["text"], return_offsets_mapping=True)
                lens.append(len(enc["input_ids"]))
                for kd, e, cs, ce, mt in s["mentions"]:
                    lo, hi = S31.token_span(enc["offset_mapping"], cs, ce)
                    if s["text"][cs:ce] not in tok.decode(enc["input_ids"][lo:hi]):
                        bad += 1
            rep["%s_%s" % (kind, sp)] = {"n": len(st), "max_len": max(lens), "mean_len": sum(lens) / len(lens),
                                         "span_decode_failures": bad,
                                         "mentions_per_story": sorted(set(len(s["mentions"]) for s in st))}
            print(kind, sp, json.dumps(rep["%s_%s" % (kind, sp)]), flush=True)
    os.makedirs(a.out, exist_ok=True)
    json.dump(rep, open(os.path.join(a.out, "selftest_%s.json" % a.enc), "w"), indent=1)


def feats_c1(a):
    S48.patch()
    if a.enc == "qwen8":
        a.items = ["C:%s" % sp for sp in a.splits]
        return P31.feats(a)
    tok = BC.tokenizer()
    dev = a.device
    from transformers import AutoModel
    llm = AutoModel.from_pretrained("microsoft/phi-2", trust_remote_code=True, torch_dtype=torch.float16).to(dev).eval()
    os.makedirs(a.feats, exist_ok=True)
    for sp in a.splits:
        p = os.path.join(a.feats, "phi2_C_%s.pt" % sp)
        if os.path.exists(p):
            print("exists", p); continue
        t0 = time.time()
        st = S48.build_C1(sp)
        toks = [S31.tokenize(tok, s, 768, pad=True) for s in st]
        Fs, Ls, Ms = [], [], []
        with torch.no_grad():
            for k in range(0, len(toks), 4):                       # v13.cache: batch 4, fp16 states
                b = toks[k:k + 4]
                H = llm(input_ids=torch.tensor([t[0] for t in b]).to(dev),
                        attention_mask=torch.tensor([t[1] for t in b]).to(dev)).last_hidden_state.half()
                for j, (ids, am, spans) in enumerate(b):
                    f, l, m = P31.pool_instance(H[j], int(sum(am)), spans, dev)
                    Fs.append(f.cpu()); Ls.append(l.cpu()); Ms.append(m.cpu())
        obj = {"F": torch.stack(Fs), "last": torch.stack(Ls), "mean": torch.stack(Ms),
               "len": torch.tensor([int(sum(t[1])) for t in toks]), "spans": [t[2] for t in toks],
               "labels": torch.tensor([s["label"] for s in st]), "text_md5": P31.h([s["text"] for s in st]),
               "enc": "phi2", "arena": "C1", "split": sp}
        torch.save(obj, p + ".tmp"); os.replace(p + ".tmp", p)
        print("feats phi2 C1", sp, tuple(obj["F"].shape), "%.0fs" % (time.time() - t0), flush=True)


def feats_b(a):
    tok = P31.tokenizer("qwen8")
    dev = a.device
    from transformers import AutoModel
    llm = AutoModel.from_pretrained(P31.QWEN8, torch_dtype=torch.bfloat16).to(dev).eval()
    for q in llm.parameters():
        q.requires_grad_(False)
    os.makedirs(a.feats, exist_ok=True)
    for item in a.items:
        kind, sp = item.split(":")
        paths = {L: os.path.join(a.feats, "qwen8L%d_%s_%s.pt" % (L, kind, sp)) for L in a.layers}
        if all(os.path.exists(p) for p in paths.values()):
            print("exists", item); continue
        t0 = time.time()
        st = S48.build_B(kind, sp)
        toks = []
        for s in st:
            enc = tok(s["text"], return_offsets_mapping=True)
            spans = [S31.token_span(enc["offset_mapping"], cs, ce) for _, _, cs, ce, _ in s["mentions"]]
            toks.append((enc["input_ids"], spans))
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
                        Hb = hs[L][b]
                        res[L][i] = torch.stack([Hb[s0:e0].float().mean(0) for s0, e0 in toks[i][1]]).half().cpu()
        for L in a.layers:
            obj = {"F": torch.stack([res[L][i] for i in range(len(toks))]), "len": torch.tensor([len(t[0]) for t in toks]),
                   "spans": [t[1] for t in toks], "labels": torch.tensor([s["label"] for s in st]),
                   "text_md5": P31.h([s["text"] for s in st]), "enc": "qwen8L%d" % L, "kind": kind, "split": sp}
            torch.save(obj, paths[L] + ".tmp"); os.replace(paths[L] + ".tmp", paths[L])
        print("featsB", item, tuple(obj["F"].shape), "%.0fs" % (time.time() - t0), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["selftest", "featsC1", "featsB"])
    ap.add_argument("--enc", default="qwen8", choices=["phi2", "qwen8"])
    ap.add_argument("--splits", nargs="*", default=["train", "dev", "test"])
    ap.add_argument("--items", nargs="*", default=[])
    ap.add_argument("--layers", type=int, nargs="*", default=LAYERS)
    ap.add_argument("--feats", default="feats")
    ap.add_argument("--out", default="out")
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    {"selftest": selftest, "featsC1": feats_c1, "featsB": feats_b}[a.cmd](a)


if __name__ == "__main__":
    main()
