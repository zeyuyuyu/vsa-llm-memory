"""B33 prompting arm on bAbI task 2 (prereg/B33_fresh_task2_prereg.md).

Exactly the published zero-shot cell of V-49 (verbatim copy b33_ngcls/v49.py: its loc_token_ids and score; frozen
Phi-2 in fp16, right padding, batches of 8, max length 1024, the next-token distribution at the last position restricted
to the six location tokens; text = the story and question on one line + " Answer:"), on the B33 evaluation sets, with
per-question predictions and restricted logits saved. One deterministic decoding (no seeds).
"""
import argparse, json, os, sys, time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import b33_common as B                                                # noqa: E402


@torch.no_grad()
def score_logits(model, tok, texts, ids, device, bs=8):
    """v49.score, returning the restricted logits as well (same batching, padding and truncation)."""
    out, lgs = [], []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)
        lg = model(**enc).logits
        last = enc["attention_mask"].sum(-1) - 1
        sel = lg[torch.arange(lg.size(0), device=device), last][:, ids]
        out += sel.argmax(-1).tolist()
        lgs += sel.float().cpu().tolist()
    return out, lgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", default=B.SETS)
    ap.add_argument("--fresh_file", default="b33_data/b33_fresh.jsonl")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="out")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    M = B.load_v49()
    v49, v5 = M["v49"], M["v5"]
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", trust_remote_code=True,
                                                 torch_dtype=torch.float16).to(a.device).eval()
    d, vocab, sets = B.load_sets(v5, a.fresh_file, [] if a.smoke else a.sets)
    if a.smoke:
        sets = {"smoke_train64": d["train"][:64]}
    ids, kept = v49.loc_token_ids(tok, vocab)
    assert len(ids) == len(vocab)
    remap = {g: j for j, g in enumerate(kept)}
    res = {"vocab": vocab, "kept": kept, "sets": {}}
    for name, split in sets.items():
        texts = [t.strip().replace("\n", " ") + " Answer:" for t, _ in split]
        gold = [remap[int(y)] for _, y in split]
        pred = v49.score(model, tok, texts, ids, a.device)               # the verbatim scorer
        pred2, lgs = score_logits(model, tok, texts, ids, a.device)      # same computation, logits kept
        acc = float(np.mean([p == g for p, g in zip(pred, gold)]))
        res["sets"][name] = {"acc": acc, "pred": pred, "labels": gold, "logits": lgs,
                             "pred_matches_logit_pass": float(np.mean([x == y for x, y in zip(pred, pred2)])),
                             "n_over_1024_tokens": sum(int(len(tok(t)["input_ids"]) > 1024) for t in texts)}
        print("  %-8s n=%4d  zero-shot acc %.4f  (logit pass agrees on %.4f)"
              % (name, len(split), acc, res["sets"][name]["pred_matches_logit_pass"]), flush=True)
    os.makedirs(a.out, exist_ok=True)
    fn = os.path.join(a.out, "b33_prompt%s.json" % ("_smoke" if a.smoke else ""))
    res["secs"] = round(time.time() - t0, 1)
    json.dump(res, open(fn + ".tmp", "w")); os.replace(fn + ".tmp", fn)
    print("B33-PROMPT-DONE (%.0fs) -> %s" % (time.time() - t0, fn), flush=True)


if __name__ == "__main__":
    main()
