"""B68 (prereg/B68_early_layer_labelfree_prereg.md): span-mean features of the INPUT EMBEDDINGS (hidden_states[0]) and
of EARLY layers (hidden_states[2], hidden_states[4]: the outputs of blocks 2 and 4) of Phi-2 and Qwen3-8B, in the format
of b31_prep.feats / feats_layers (files <enc>L<L>_<arena>_<split>.pt: F = span means fp32, last, mean, len, spans,
labels, text_md5).

Same instances, token ids, spans and pooling as B31:
  phi2   the B27 rows (b27_common.build_rows: same ids, attention masks, padding to the arena's max length) and the
         B27 batch sizes (A 8, B / C 4), AutoModel fp16 (trust_remote_code) exactly as the B27 caches were made;
         states .half() then pooled by b31_prep.pool_instance on the GPU, as b31_prep.feats pools the cached states.
  qwen8  b31_prep.feats_layers' batching (length-sorted, bs 8, right padding), bf16, b31_prep.pool_instance.
Checks (training split of every arena, written to <feats>/b68_check_<enc>.json):
  final  the same forward pass's last_hidden_state, pooled the same way, against B31's final-layer feature file
         (max |diff| of F; must be 0 or at fp16/bf16 rounding level)
  L0     qwen8: hidden_states[0] against B52's embedding-lookup files (qwen8L0_{A,C}_train.pt; must be 0);
         phi2: hidden_states[0] against embed_tokens(ids) (must be 0)

  python b68_feats.py --enc qwen8 --layers 0 2 4 --items A:train A:dev A:test ... --feats /data0/.../b68/feats
"""
import argparse, json, os, sys, time
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import b27_common as BC                                                     # noqa: E402
import b31_prep as BP                                                       # noqa: E402
import b31_struct as S                                                      # noqa: E402

B31_FEATS = "/path/to/work/b31/feats"
BS_PHI2 = {"A": 8, "B": 4, "C": 4}


def save(p, Fs, Ls, Ms, lens, toks, st, enc, ar, sp):
    obj = {"F": torch.stack(Fs), "last": torch.stack(Ls), "mean": torch.stack(Ms), "len": torch.tensor(lens),
           "spans": [t[2] for t in toks], "labels": torch.tensor([s["label"] for s in st]),
           "text_md5": BP.h([s["text"] for s in st]), "enc": enc, "arena": ar, "split": sp}
    torch.save(obj, p + ".tmp"); os.replace(p + ".tmp", p)


def maxdiff(F_new, ref_path):
    if not os.path.exists(ref_path):
        return None
    ref = torch.load(ref_path, weights_only=False)["F"]
    assert ref.shape == F_new.shape, (ref.shape, F_new.shape)
    return float((ref - F_new).abs().max())


def run_phi2(a, layers, check):
    from transformers import AutoModel
    M = BC.load_arena_modules()
    tok = BC.tokenizer()
    dev = a.device
    llm = AutoModel.from_pretrained("microsoft/phi-2", trust_remote_code=True, torch_dtype=torch.float16).to(dev).eval()
    for q in llm.parameters():
        q.requires_grad_(False)
    for item in a.items:
        ar, sp = item.split(":")
        paths = {L: os.path.join(a.feats, "phi2L%d_%s_%s.pt" % (L, ar, sp)) for L in layers}
        if all(os.path.exists(p) for p in paths.values()):
            continue
        t0 = time.time()
        st = BP.structs(ar, sp)
        toks = [S.tokenize(tok, s, S.MAXLEN_PHI2[ar], pad=True) for s in st]
        rows = BC.build_rows(M, tok, ar, sp)
        res = {L: ([], [], []) for L in layers}
        fin = ([], [], [])
        l0diff = 0.0
        bs = BS_PHI2[ar]
        with torch.no_grad():
            for k in range(0, len(st), bs):
                idx = list(range(k, min(k + bs, len(st))))
                if ar == "A":
                    items = [rows[j] for j in idx]
                    ids = torch.stack([it["input_ids"] for it in items]); am = torch.stack([it["attention_mask"] for it in items])
                else:
                    ids = torch.stack([rows[j]["ids"] for j in idx]); am = torch.stack([rows[j]["am"] for j in idx])
                for b, j in enumerate(idx):
                    assert torch.equal(torch.tensor(toks[j][0]), ids[b]), (ar, sp, j)
                out = llm(input_ids=ids.to(dev), attention_mask=am.to(dev), output_hidden_states=True)
                hs = out.hidden_states
                if check is not None and sp == "train":
                    emb = llm.embed_tokens(ids.to(dev)) if hasattr(llm, "embed_tokens") else llm.get_input_embeddings()(ids.to(dev))
                    l0diff = max(l0diff, float((hs[0].float() - emb.float()).abs().max()))
                for b, j in enumerate(idx):
                    L_ = int(sum(toks[j][1]))
                    for L in layers:
                        f, l, m = BP.pool_instance(hs[L][b].half(), L_, toks[j][2], dev)
                        res[L][0].append(f.cpu()); res[L][1].append(l.cpu()); res[L][2].append(m.cpu())
                    if check is not None and sp == "train":
                        f, l, m = BP.pool_instance(out.last_hidden_state[b].half(), L_, toks[j][2], dev)
                        fin[0].append(f.cpu())
        lens = [int(sum(t[1])) for t in toks]
        for L in layers:
            save(paths[L], res[L][0], res[L][1], res[L][2], lens, toks, st, "phi2L%d" % L, ar, sp)
        if check is not None and sp == "train":
            check.setdefault(ar, {})["final_vs_b31"] = maxdiff(torch.stack(fin[0]), os.path.join(B31_FEATS, "phi2_%s_train.pt" % ar))
            check[ar]["L0_vs_embed_lookup"] = l0diff
            check[ar]["n_hidden_states"] = len(hs)
            print("check phi2", ar, json.dumps(check[ar]), flush=True)
        print("feats phi2", layers, item, "%.0fs" % (time.time() - t0), flush=True)


def run_qwen8(a, layers, check):
    from transformers import AutoModel
    tok = BP.tokenizer("qwen8")
    dev = a.device
    llm = AutoModel.from_pretrained(BP.QWEN8, torch_dtype=torch.bfloat16).to(dev).eval()
    for q in llm.parameters():
        q.requires_grad_(False)
    for item in a.items:
        ar, sp = item.split(":")
        paths = {L: os.path.join(a.feats, "qwen8L%d_%s_%s.pt" % (L, ar, sp)) for L in layers}
        if all(os.path.exists(p) for p in paths.values()):
            continue
        t0 = time.time()
        st = BP.structs(ar, sp)
        toks = [S.tokenize(tok, s) for s in st]
        order = sorted(range(len(toks)), key=lambda i: len(toks[i][0]))
        res = {L: {} for L in layers}
        fin = {}
        with torch.no_grad():
            for k in range(0, len(order), a.bs):
                idx = order[k:k + a.bs]
                T = max(len(toks[i][0]) for i in idx)
                ids = torch.full((len(idx), T), tok.pad_token_id, dtype=torch.long)
                am = torch.zeros((len(idx), T), dtype=torch.long)
                for b, i in enumerate(idx):
                    n = len(toks[i][0]); ids[b, :n] = torch.tensor(toks[i][0]); am[b, :n] = 1
                out = llm(input_ids=ids.to(dev), attention_mask=am.to(dev), output_hidden_states=True)
                hs = out.hidden_states
                for b, i in enumerate(idx):
                    n = len(toks[i][0])
                    for L in layers:
                        res[L][i] = [t.cpu() for t in BP.pool_instance(hs[L][b], n, toks[i][2], dev)]
                    if check is not None and sp == "train":
                        fin[i] = BP.pool_instance(out.last_hidden_state[b], n, toks[i][2], dev)[0].cpu()
        lens = [len(t[0]) for t in toks]
        N = len(toks)
        for L in layers:
            save(paths[L], [res[L][i][0] for i in range(N)], [res[L][i][1] for i in range(N)],
                 [res[L][i][2] for i in range(N)], lens, toks, st, "qwen8L%d" % L, ar, sp)
        if check is not None and sp == "train":
            check.setdefault(ar, {})["final_vs_b31"] = maxdiff(torch.stack([fin[i] for i in range(N)]),
                                                               os.path.join(B31_FEATS, "qwen8_%s_train.pt" % ar))
            if 0 in layers:
                check[ar]["L0_vs_b52_lookup"] = maxdiff(torch.stack([res[0][i][0] for i in range(N)]),
                                                        os.path.join(B31_FEATS, "qwen8L0_%s_train.pt" % ar))
            check[ar]["n_hidden_states"] = len(hs)
            print("check qwen8", ar, json.dumps(check[ar]), flush=True)
        print("feats qwen8", layers, item, "%.0fs" % (time.time() - t0), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enc", required=True, choices=["phi2", "qwen8"])
    ap.add_argument("--layers", type=int, nargs="+", default=[0, 2, 4])
    ap.add_argument("--items", nargs="+", required=True)
    ap.add_argument("--feats", required=True)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--check", action="store_true", help="training split: final-layer and L0 checks (b68_check_<enc>.json)")
    a = ap.parse_args()
    os.makedirs(a.feats, exist_ok=True)
    check = {} if a.check else None
    {"phi2": run_phi2, "qwen8": run_qwen8}[a.enc](a, a.layers, check)
    if check is not None:
        p = os.path.join(a.feats, "b68_check_%s.json" % a.enc)
        old = json.load(open(p)) if os.path.exists(p) else {}
        old.update(check)
        json.dump(old, open(p, "w"), indent=1)


if __name__ == "__main__":
    main()
