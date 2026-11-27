"""V-99 (prereg b32d015): abstention-routed hybrid.

One run trains BOTH routes on the same seed and evaluates them on the SAME instances, so the routing is
aligned per instance rather than assembled from separate runs. Reports the two numbers this project has
never computed: the store's coverage, and its precision on the instances it chooses to answer.

  hybrid(x) = store(x)      if the store answers
              lora_only(x)  if the store abstains

This is not the manuscript's retrieve-then-verbalise hybrid, which sends every instance through the model.
"""
import argparse, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v85_data import load_rows, em                     # noqa: E402
from v98_train import (gold_triples, head_entity, ex_prompt, qa_prompt)   # noqa: E402
import v97_store as ST                                 # noqa: E402


def lora_wrap(base, seed):
    from peft import LoraConfig, get_peft_model
    return get_peft_model(base, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                                           task_type="CAUSAL_LM",
                                           target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))


def train_one(llm, tok, rows, mode, a):
    """mode: 'qa' supervises the answer; 'ex' supervises the gold triples."""
    llm.gradient_checkpointing_enable(); llm.enable_input_require_grads()
    tp = [p for p in llm.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(tp, lr=a.lr)
    t0 = time.time(); tot = nb = 0
    opt.zero_grad()
    for k, r in enumerate(rows):
        if mode == "ex":
            gt = gold_triples(r)
            if not gt:
                continue
            prompt = ex_prompt(r, tok, a.budget, False)
            target = "\n".join("%s | %s | %s" % t for t in gt) + tok.eos_token
        else:
            prompt = qa_prompt(r, tok, a.budget)
            target = " " + str(r["answer"]) + tok.eos_token
        pid = tok(prompt, truncation=True, max_length=a.max_len)["input_ids"]
        aid = tok(target, add_special_tokens=False)["input_ids"]
        ids = torch.tensor([pid + aid], device="cuda")
        lab = torch.tensor([[-100] * len(pid) + aid], device="cuda")
        out = llm(input_ids=ids, labels=lab)
        (out.loss / a.accum).backward()
        tot += float(out.loss); nb += 1
        if (k + 1) % a.accum == 0:
            torch.nn.utils.clip_grad_norm_(tp, 1.0); opt.step(); opt.zero_grad()
        if (k + 1) % 500 == 0:
            print("      [%s] step %d loss %.4f (%.1f min)" % (mode, k + 1, tot / max(nb, 1),
                  (time.time() - t0) / 60), flush=True)
    llm.gradient_checkpointing_disable()
    if hasattr(llm, "config"):
        llm.config.use_cache = True
    llm.eval()
    print("    [%s] train loss %.4f over %d" % (mode, tot / max(nb, 1), nb), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--n_train", type=int, default=10000)
    ap.add_argument("--n_eval", type=int, default=300)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--budget", type=int, default=1200)
    ap.add_argument("--max_len", type=int, default=2048)
    ap.add_argument("--max_new", type=int, default=320)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="v99.json")
    a = ap.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cache = os.environ.get("HF_HOME")
    print("===== V-99 abstention-routed hybrid (prereg b32d015) seed %d =====" % a.seed, flush=True)
    print("  banked: lora_only .6367 | extract_ft(10k) .4678 | frozen_extract .0500", flush=True)
    print("  P2: store precision-when-answering >= .75   P3: hybrid > lora_only + .02", flush=True)

    tok = AutoTokenizer.from_pretrained(a.model, cache_dir=cache, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    ev = load_rows(a.n_eval, split="validation", cache_dir=cache)
    tr = load_rows(a.n_train, split="train", cache_dir=cache)
    print("  eval %d | train %d" % (len(ev), len(tr)), flush=True)

    def fresh():
        return AutoModelForCausalLM.from_pretrained(a.model, cache_dir=cache, dtype=torch.bfloat16,
                                                    device_map="cuda", trust_remote_code=True)

    def gen(llm, prompt, max_new):
        ids = tok(prompt, return_tensors="pt", truncation=True, max_length=3500).to("cuda")
        with torch.no_grad():
            o = llm.generate(**ids, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.pad_token_id)
        return tok.decode(o[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)

    # ---- route A: extraction + store ----
    print("  --- training extraction route ---", flush=True)
    llm = lora_wrap(fresh(), a.seed)
    train_one(llm, tok, tr, "ex", a)
    store_pred = []
    for i, r in enumerate(ev):
        txt = gen(llm, ex_prompt(r, tok, a.budget, False), a.max_new)
        trip, _ = ST.parse_triples(txt)
        ans, _, _ = ST.answer_two_hop(trip, head_entity(r), "cuda")
        store_pred.append(ans)
        if (i + 1) % 100 == 0:
            print("      store %d/%d" % (i + 1, len(ev)), flush=True)
    del llm; torch.cuda.empty_cache()

    # ---- route B: answer-supervised LoRA ----
    print("  --- training QA route ---", flush=True)
    llm = lora_wrap(fresh(), a.seed)
    train_one(llm, tok, tr, "qa", a)
    lora_pred = [gen(llm, qa_prompt(r, tok, a.budget), 16).split("\n")[0].strip() for r in ev]
    del llm; torch.cuda.empty_cache()

    gold = [r["answer"] for r in ev]
    s_ok = [em(p, g) for p, g in zip(store_pred, gold)]
    l_ok = [em(p, g) for p, g in zip(lora_pred, gold)]
    answered = [bool(p.strip()) for p in store_pred]
    hyb_pred = [s if ans else l for s, l, ans in zip(store_pred, lora_pred, answered)]
    h_ok = [em(p, g) for p, g in zip(hyb_pred, gold)]

    n_ans = sum(answered)
    coverage = n_ans / len(ev)
    prec = (sum(o for o, ans in zip(s_ok, answered) if ans) / n_ans) if n_ans else float("nan")
    hall_store = sum(1 for o, ans in zip(s_ok, answered) if ans and not o) / len(ev)
    hall_lora = sum(1 for o in l_ok if not o) / len(ev)

    out = {"prereg": "b32d015", "seed": a.seed, "n_eval": len(ev),
           "lora_only_em": float(np.mean(l_ok)), "store_em": float(np.mean(s_ok)),
           "hybrid_em": float(np.mean(h_ok)), "coverage": coverage,
           "precision_when_answering": prec, "hallucination_store": hall_store,
           "hallucination_lora": hall_lora}
    json.dump(out, open(a.out, "w"), indent=1)
    print("", flush=True)
    print("  >>> lora_only EM %.4f | store EM %.4f | HYBRID EM %.4f" %
          (out["lora_only_em"], out["store_em"], out["hybrid_em"]), flush=True)
    print("  >>> store coverage %.4f | PRECISION-WHEN-ANSWERING %.4f  (P2 bar .75)" % (coverage, prec),
          flush=True)
    print("  >>> hallucination rate: store %.4f vs lora_only %.4f" % (hall_store, hall_lora), flush=True)
    print("  >>> P3: hybrid %.4f vs lora_only+.02 = %.4f -> %s"
          % (out["hybrid_em"], out["lora_only_em"] + 0.02,
             out["hybrid_em"] > out["lora_only_em"] + 0.02), flush=True)
    print("V99-DONE", flush=True)


if __name__ == "__main__":
    main()
