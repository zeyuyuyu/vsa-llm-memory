"""V-85 arm 0 (prereg 85461e8): Qwen3-8B's frozen prompt baseline on the paper's 2Wiki slice.

Every threshold in V-85 is relative to this number, so it is measured first and on its own, with no training
of any kind. Deterministic decoding, so a single pass is the measurement.
"""
import argparse, json, os, sys, time
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v85_data import load_rows, prompt_for, em  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--budget", type=int, default=1200)
    ap.add_argument("--out", default="v85_baseline.json")
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    cache = os.environ.get("HF_HOME")
    print("===== V-85 arm 0: frozen prompt baseline (prereg 85461e8) =====", flush=True)
    print("  model: %s" % a.model, flush=True)

    tok = AutoTokenizer.from_pretrained(a.model, cache_dir=cache, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    rows = load_rows(a.n, cache_dir=cache)
    print("  instances: %d (compositional, dev)" % len(rows), flush=True)

    llm = AutoModelForCausalLM.from_pretrained(
        a.model, cache_dir=cache, dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True).eval()
    print("  loaded, %.1fB params" % (sum(p.numel() for p in llm.parameters()) / 1e9), flush=True)

    hits, trunc, t0, preds = [], 0, time.time(), []
    for i, r in enumerate(rows):
        p, tr = prompt_for(r, tok, a.budget)
        trunc += int(tr)
        ids = tok(p, return_tensors="pt", truncation=True, max_length=3500).to("cuda")
        with torch.no_grad():
            o = llm.generate(**ids, max_new_tokens=16, do_sample=False,
                             pad_token_id=tok.pad_token_id)
        txt = tok.decode(o[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
        pred = txt.split("\n")[0].strip()
        hits.append(em(pred, r["answer"]))
        preds.append({"q": r["question"], "gold": r["answer"], "pred": pred})
        if (i + 1) % 50 == 0:
            print("    %d/%d  EM so far %.4f  (%.1f min)"
                  % (i + 1, len(rows), sum(hits) / len(hits), (time.time() - t0) / 60), flush=True)

    acc = sum(hits) / len(hits)
    out = {"prereg": "85461e8", "model": a.model, "n": len(rows), "frozen_prompt_em": acc,
           "trunc_rate": trunc / len(rows), "preds": preds[:40]}
    json.dump(out, open(a.out, "w"), indent=1)
    print("", flush=True)
    print("  >>> Qwen3-8B frozen prompt EM = %.4f (truncation %.3f)" % (acc, trunc / len(rows)), flush=True)
    print("  >>> reference points: phi-2 prompt .2533 | Qwen2.5-7B prompt .2267 | "
          "store w/ gold .9744 | self-extraction .0178-.0311", flush=True)
    print("  >>> GATE for lora_only is therefore %.4f (this + .05)" % (acc + 0.05), flush=True)
    print("V85-BASELINE-DONE", flush=True)


if __name__ == "__main__":
    main()
