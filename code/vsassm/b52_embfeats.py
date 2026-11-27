"""B52 (prereg/B52_robustness_layers_maps_1day_prereg.md), part (a): Qwen3-8B INPUT-EMBEDDING span means
(hidden_states[0] = embed_tokens(input_ids), before block 1), in the format of b31_prep.feats_layers
(files qwen8L0_<arena>_<split>.pt: F = span means fp32, last, mean, len, spans, labels, text_md5).

hidden_states[0] of Qwen3Model is the embedding lookup itself (no scaling, no position term), so it is computed by a
lookup into the checkpoint's embed_tokens.weight (bf16), on the same token ids and spans as B31 (b31_struct.tokenize),
pooled by b31_prep.pool_instance. --check N compares the lookup with AutoModel(...).hidden_states[0] for the first N
instances of the first item (must be bit-identical).

  python b52_embfeats.py --items C:train C:dev C:test A:train A:dev A:test --feats /path/to/work/b31/feats --check 2
"""
import argparse, json, os, sys, time
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import b31_prep as BP                                                       # noqa: E402
import b31_struct as S                                                      # noqa: E402


def embed_weight():
    from safetensors import safe_open
    idx = json.load(open(os.path.join(BP.QWEN8, "model.safetensors.index.json")))["weight_map"]
    name = [k for k in idx if k.endswith("embed_tokens.weight")]
    assert len(name) == 1, name
    with safe_open(os.path.join(BP.QWEN8, idx[name[0]]), framework="pt") as f:
        W = f.get_tensor(name[0])
    assert W.dtype == torch.bfloat16, W.dtype
    return W


def check(W, tok, item, n):
    from transformers import AutoModel
    ar, sp = item.split(":")
    st = BP.structs(ar, sp)[:n]
    llm = AutoModel.from_pretrained(BP.QWEN8, torch_dtype=torch.bfloat16).eval()
    out = []
    with torch.no_grad():
        for s in st:
            ids = torch.tensor(S.tokenize(tok, s)[0])
            h0 = llm(input_ids=ids[None], output_hidden_states=True).hidden_states[0][0]
            out.append(float((h0.float() - W[ids].float()).abs().max()))
    print("check hidden_states[0] vs lookup, max |diff| per instance:", out, flush=True)
    assert max(out) == 0.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", nargs="+", required=True)
    ap.add_argument("--feats", default="/path/to/work/b31/feats")
    ap.add_argument("--check", type=int, default=0)
    a = ap.parse_args()
    tok = BP.tokenizer("qwen8")
    W = embed_weight()
    if a.check:
        r = check(W, tok, a.items[0], a.check)
        json.dump({"max_abs_diff": r}, open(os.path.join(a.feats, "qwen8L0_check.json"), "w"))
    for item in a.items:
        ar, sp = item.split(":")
        p = os.path.join(a.feats, "qwen8L0_%s_%s.pt" % (ar, sp))
        if os.path.exists(p):
            continue
        t0 = time.time()
        st = BP.structs(ar, sp)
        toks = [S.tokenize(tok, s) for s in st]
        res = [BP.pool_instance(W[torch.tensor(t[0])], len(t[0]), t[2], "cpu") for t in toks]
        obj = {"F": torch.stack([r[0] for r in res]), "last": torch.stack([r[1] for r in res]),
               "mean": torch.stack([r[2] for r in res]), "len": torch.tensor([len(t[0]) for t in toks]),
               "spans": [t[2] for t in toks], "labels": torch.tensor([s["label"] for s in st]),
               "text_md5": BP.h([s["text"] for s in st]), "enc": "qwen8L0", "arena": ar, "split": sp}
        torch.save(obj, p + ".tmp"); os.replace(p + ".tmp", p)
        print("feats qwen8L0", item, tuple(obj["F"].shape), "%.0fs" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
