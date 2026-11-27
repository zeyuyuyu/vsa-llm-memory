"""B27 preparation: generator self-tests, frozen-state cache, and G0 (the ORIGINAL run functions reproduce the
banked per-seed numbers in this environment).

  selftest   new split code reproduces the original train/test generators exactly; the second views hold the same
             bindings, query and label; token lengths; dev/train/test text overlap. No model is scored.
  cache      frozen Phi-2 states (original load: fp16, trust_remote_code) via the original cache functions.
  g0         original v13.run / v11.run / v8.run / v9.run_arm on the published configuration.
"""
import argparse, json, os, re, sys, time
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import b27_common as C                                                  # noqa: E402

M = C.load_arena_modules()

BANKED = {  # per-seed best-epoch test accuracies behind the paper's Fig. 4a (figdata md5-identical to the box)
    ("C", "parser"): {42: 0.932, 123: 0.932, 456: 0.938},
    ("C", "probe_argmax"): {42: 0.932, 123: 0.932, 456: 0.938},
    ("C", "spans"): {42: 0.472, 123: 0.502, 456: 0.476},
    ("B", "parser"): {42: 0.990, 123: 1.000, 456: 0.998},
    ("B", "probe_argmax"): {42: 0.990, 123: 1.000, 456: 0.998},
    ("B", "spans"): {42: 0.502, 123: 0.498, 456: 0.522},
    ("A", "spans"): {42: 0.511, 123: 0.499, 456: 0.493},
    ("A", "extract_then_bind"): {42: 0.915, 123: 0.913, 456: 0.913},
}


def selftest(a):
    tok = C.tokenizer()
    rep = {"padding_side": tok.padding_side}
    # A: new split code with the original offsets == SystematicChainDataset
    SCD = M["systematic_chain_dataset"]
    cfg = C.chain_cfg(M)
    for split, off, n in (("train", 42, 2000), ("eval", 123, 1000)):
        ref = SCD.SystematicChainDataset(tok, num_samples=n, split=split, cfg=cfg)
        new = C.ChainSplit(tok, n, off, cfg)
        ok = all(torch.equal(ref[i]["input_ids"], new[i]["input_ids"]) and int(ref[i]["labels"]) == int(new[i]["labels"])
                 for i in range(n))
        rep["A_%s_reproduced" % split] = ok
    # B: make_b_arena with the original offsets == v11.Arena
    V11 = M["v11"]; SRB = M["systematic_role_binding_dataset"]
    bcfg = SRB.SystematicConfig()
    for split, off, held, n in (("train", 42, False, 1000), ("eval", 123, True, 500)):
        ref = V11.Arena(tok, n, split, bcfg, 0).rows
        new = C.make_b_arena(M, tok, n, off, held).rows
        ok = all(torch.equal(x["ids"], y["ids"]) and x["ctx"] == y["ctx"] and x["label"] == y["label"] and
                 x["q_role"] == y["q_role"] and x["cand"] == y["cand"] for x, y in zip(ref, new))
        rep["B_%s_reproduced" % split] = ok
    # second views: same bindings / query / label, reshuffled order
    rx = re.compile(r"\[([RXY])([0-9A-F]{2})\] (?:\[([XY])([0-9A-F]{2})\]|\{(\d+)\})")
    def a_sig(ids):
        t = tok.decode(ids[ids != tok.pad_token_id] if tok.pad_token_id is not None else ids)
        return sorted(rx.findall(t)), t.split(" Q ")[-1]
    tr = C.ChainSplit(tok, 2000, 42, cfg); tr2 = C.ChainSplit(tok, 2000, 42, cfg, view2=True)
    rep["A_view2_same_bindings"] = all(a_sig(tr[i]["input_ids"]) == a_sig(tr2[i]["input_ids"]) and
                                       int(tr[i]["labels"]) == int(tr2[i]["labels"]) for i in range(2000))
    rep["A_view2_order_changed"] = sum(not torch.equal(tr[i]["input_ids"], tr2[i]["input_ids"]) for i in range(2000))
    b1 = C.make_b_arena(M, tok, 1000, 42, False).rows; b2 = C.make_b_arena(M, tok, 1000, 42, False, view2=True).rows
    rep["B_view2_same_bindings"] = all(sorted((r, f) for r, f, _ in x["ctx"]) == sorted((r, f) for r, f, _ in y["ctx"])
                                       and x["label"] == y["label"] and x["q_role"] == y["q_role"] and
                                       x["cand"] == y["cand"] for x, y in zip(b1, b2))
    rep["B_view2_order_changed"] = sum(not torch.equal(x["ids"], y["ids"]) for x, y in zip(b1, b2))
    V13 = M["v13"]
    c1 = V13.Arena(tok, 800, 42).rows; c2 = C.make_c_view2(M, tok, c1)
    rep["C_view2_same_bindings"] = all(
        all(sorted((p[0], p[1]) for p in x["ctx"][L]) == sorted((p[0], p[1]) for p in y["ctx"][L]) for L in range(2))
        and (x["label"], x["q"], x["cand"]) == (y["label"], y["q"], y["cand"]) for x, y in zip(c1, c2))
    rep["C_view2_order_changed"] = sum(not torch.equal(x["ids"], y["ids"]) for x, y in zip(c1, c2))
    # C: the query/candidate spans recorded by v13.Arena are what the qc=proj arm reads; check they decode right
    rep["C_q_cand_span_decode_ok"] = all(
        tok.decode(r["ids"][r["q_span"][0]:r["q_span"][1]]).strip() == V13.h2(r["q"]) and
        tok.decode(r["ids"][r["cand_span"][0]:r["cand_span"][1]]).strip() == str(r["cand"]) for r in c1[:200])
    rep["B_cand_span_decode_ok"] = all(tok.decode(r["ids"][r["cand_span"][0]:r["cand_span"][1]]).strip() == str(r["cand"])
                                       for r in b1[:200])
    # lengths and overlap (texts compared as token ids)
    for ar in "ABC":
        L = {}
        keys = {}
        for sp in ("train", "dev", "test"):
            rows = C.build_rows(M, tok, ar, sp)
            ids = [it["input_ids"] for it in rows.items] if ar == "A" else [r["ids"] for r in rows]
            ams = [it["attention_mask"] for it in rows.items] if ar == "A" else [r["am"] for r in rows]
            lens = [int(x.sum()) for x in ams]
            L[sp] = {"n": len(ids), "max_len": max(lens), "mean_len": round(sum(lens) / len(lens), 1),
                     "padded_to": int(ids[0].numel()), "truncated": sum(l == ids[0].numel() for l in lens)}
            keys[sp] = set(tuple(x.tolist()) for x in ids)
        L["dev_in_train"] = len(keys["dev"] & keys["train"]); L["dev_in_test"] = len(keys["dev"] & keys["test"])
        L["test_in_train"] = len(keys["test"] & keys["train"])
        rep["lengths_" + ar] = L
    print(json.dumps(rep, indent=1))
    json.dump(rep, open(a.out, "w"), indent=1)


def load_llm(dev):
    from transformers import AutoModel
    llm = AutoModel.from_pretrained("microsoft/phi-2", trust_remote_code=True, torch_dtype=torch.float16).to(dev).eval()
    for p in llm.parameters():
        p.requires_grad_(False)
    return llm


def cache(a):
    tok = C.tokenizer()
    llm = load_llm(a.device)
    os.makedirs(a.feats, exist_ok=True)
    for item in a.items:
        ar, sp = item.split(":")
        v2 = sp.endswith("_v2")
        t0 = time.time()
        p = C.cache_feats(M, tok, llm, a.device, ar, sp[:-3] if v2 else sp, a.feats, view2=v2)
        print("cached", item, p, "%.0fs" % (time.time() - t0), flush=True)


def g0(a):
    tok = C.tokenizer()
    llm = load_llm(a.device)
    rows = []
    for item in a.items:
        ar, arm = item.split(":")
        for s in a.seeds:
            if ar == "C":
                r = M["v13"].run(arm, s, tok, llm, a.device, 800, 500, 2, 4, 3e-4)
            elif ar == "B":
                r = M["v11"].run(arm, s, M["systematic_role_binding_dataset"].SystematicConfig(), tok, llm,
                                 a.device, 1000, 500, 2, 4, 3e-4)
            elif arm == "spans":
                r = M["v8"].run(s, C.chain_cfg(M), tok, llm, a.device, 2000, 1000, 2, 8, 3e-4); r["arm"] = "spans"
            else:
                r = M["v9"].run_arm(arm, s, C.chain_cfg(M), tok, llm, a.device, 2000, 1000, 2, 8, 3e-4)
            r.update(arena=ar, banked=BANKED[(ar, arm)][s])
            print("G0 %s %s seed %d acc %.4f banked %.3f" % (ar, arm, s, r["acc"], r["banked"]), flush=True)
            rows.append(r)
            json.dump(rows, open(a.out, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["selftest", "cache", "g0"])
    ap.add_argument("--items", nargs="*", default=[])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    ap.add_argument("--feats", default="/path/to/work/b27/feats")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    {"selftest": selftest, "cache": cache, "g0": g0}[a.cmd](a)


if __name__ == "__main__":
    main()
