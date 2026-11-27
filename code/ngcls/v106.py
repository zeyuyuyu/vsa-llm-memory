"""V-106: can the store beat a LoRA FINE-TUNE on the evidence-scale axis?

Three routes over the SAME pooled gold facts at each evidence scale:
  prompt : frozen Qwen3-8B, all K facts verbalised into the context
  lora   : the same model with a LoRA adapter trained per V-99's configuration, same pooled-fact prompt
  store  : hash identity, exact two-hop, SHARDED at 64 facts per shard (V-102's operating point)

A LoRA fine-tune learns the task and the answer form, not the facts, so at test time its evidence must come
through the context -- and V-100 measured that the context route pays for evidence scale. V-102 measured that a
sharded store's capacity is per shard, hence flat in total evidence. Nobody, here or elsewhere, has run a
fine-tune at more than ONE evidence budget, so the crossover this experiment tests has only ever been produced
by splicing two of our own runs.

Every arm receives GOLD triples. This measures how to USE evidence at scale, not how to obtain it; extraction
(V-97/V-98: recall .6589, store route .4678) is untouched and remains the binding constraint.

Scoring lives in v106_score.py, committed with the prereg before any dump exists. This script only measures.
"""
import argparse, json, os, random, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v85_data import load_rows, em, norm, FEWSHOT      # noqa: E402
from v100 import pooled, gold_triples, head_entity     # noqa: E402
from v102 import sharded_two_hop, codebook             # noqa: E402
from v98_train import qa_prompt                        # noqa: E402
from v99_train import lora_wrap, train_one             # noqa: E402


def facts_prompt(pool, question):
    """V-100's committed `Facts:` body behind v85_data.FEWSHOT.

    The licensed V-100 run's template was never committed (git log -- v100.py has only the first attempt, whose
    bare body scored .7833 at 141.0 tokens against the licensed .9417 at 246.5). FEWSHOT is ~105 tokens and
    141 + 105 ~ 246, so this is the reconstruction the prereg registers as an inference from token arithmetic.
    """
    shots = "\n\n".join("%s\nQuestion: %s\nAnswer: %s" % f for f in FEWSHOT)
    facts = "\n".join("%s | %s | %s" % t for t in pool)
    body = "Facts:\n%s\n\nAnswer with the entity only.\nQuestion: %s\nAnswer:" % (facts, question)
    return shots + "\n\n" + body


def generate(llm, tok, prompt, max_new=16, max_len=30000):
    ids = tok(prompt, return_tensors="pt", truncation=True, max_length=max_len).to("cuda")
    n_tok = int(ids["input_ids"].shape[1])
    t0 = time.time()
    with torch.no_grad():
        o = llm.generate(**ids, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
    lat = time.time() - t0
    txt = tok.decode(o[0][n_tok:], skip_special_tokens=True).split("\n")[0]
    return txt, n_tok, lat


def llm_sweep(llm, tok, rows, scales, n_eval, seed, label):
    """One LLM arm across evidence scales. Accuracy, mean prompt tokens, mean latency."""
    curve = []
    for K in scales:
        rng = random.Random(seed)
        ok, toks, lats = [], [], []
        for i in range(n_eval):
            r = rows[i]
            pool, own = pooled(rows, i, K, rng)
            if not own:
                continue
            pred, n_tok, lat = generate(llm, tok, facts_prompt(pool, r["question"]))
            ok.append(em(pred, r["answer"])); toks.append(n_tok); lats.append(lat)
        rec = {"n_facts": K, "n": len(ok), "acc": float(np.mean(ok)),
               "tokens": float(np.mean(toks)), "latency_s": float(np.mean(lats))}
        curve.append(rec)
        print("  [%s] facts=%-5d acc %.4f (%.0f tok, %.3fs) n=%d"
              % (label, K, rec["acc"], rec["tokens"], rec["latency_s"], rec["n"]), flush=True)
    return curve


def store_sweep(rows, scales, n_eval, dim, n_rows, seeds):
    """The store arm: shards fixed at 64 facts per shard, V-102's code path verbatim.

    Scored two ways: `em` (the metric shared with the LLM arms) and V-102's own norm-equality, because the
    banked .9194 that P1a gates against was produced under the latter.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    curve = []
    for K in scales:
        S = max(1, K // 64)
        per_seed_em, per_seed_norm, lats = [], [], []
        for sd in seeds:
            CB = codebook(n_rows, dim, 1000 + sd, device)
            rng = random.Random(7000 + sd)
            e_ok, n_ok = [], []
            for i in range(n_eval):
                r = rows[i]
                pool, own = pooled(rows, i, K, rng)
                if not own:
                    continue
                t0 = time.time()
                ans = sharded_two_hop(pool, head_entity(r), S, dim, CB, n_rows, norm)
                lats.append(time.time() - t0)
                e_ok.append(em(ans, r["answer"]))
                n_ok.append(int(norm(ans) == norm(r["answer"])))
            per_seed_em.append(float(np.mean(e_ok))); per_seed_norm.append(float(np.mean(n_ok)))
            del CB
            if device == "cuda":
                torch.cuda.empty_cache()
        rec = {"n_facts": K, "shards": S, "acc": float(np.mean(per_seed_em)),
               "acc_seeds": per_seed_em, "acc_norm": float(np.mean(per_seed_norm)),
               "acc_norm_seeds": per_seed_norm, "latency_s": float(np.mean(lats))}
        curve.append(rec)
        print("  [store] facts=%-5d S=%-3d acc(em) %.4f acc(norm) %.4f (%.4fs) seeds %s"
              % (K, S, rec["acc"], rec["acc_norm"], rec["latency_s"],
                 [round(x, 4) for x in per_seed_em]), flush=True)
    return curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--scales", type=int, nargs="+", default=[8, 64, 128, 512, 1024])
    ap.add_argument("--n_eval", type=int, default=120)
    ap.add_argument("--n_pool", type=int, default=400)
    ap.add_argument("--n_train", type=int, default=10000)      # V-99's configuration
    ap.add_argument("--n_gate_eval", type=int, default=300)    # V-99's configuration
    ap.add_argument("--lr", type=float, default=2e-5)          # V-99's configuration
    ap.add_argument("--accum", type=int, default=8)            # V-99's configuration
    ap.add_argument("--budget", type=int, default=1200)        # V-99's configuration
    ap.add_argument("--max_len", type=int, default=2048)       # V-99's configuration
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dim", type=int, default=2048)           # V-102's configuration
    ap.add_argument("--n_rows", type=int, default=200003)      # V-102's configuration
    ap.add_argument("--cb_seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--adapter_out", default="v106_lora_adapter")
    ap.add_argument("--out", default="v106.json")
    a = ap.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    cache = os.environ.get("HF_HOME")
    print("===== V-106: store vs LoRA on the evidence-scale axis =====", flush=True)
    print("  gates: P1a store K=512/S=8/d=2048 ~ V-102 .9194 (norm metric) | "
          "P1b lora at V-99's own budget ~ .7333 (em) | P1c prompt K=8 ~ V-100 .9417 (em)", flush=True)
    print("  blind on P2 (store-lora>=.05 at K=1024), P3 (lora falls >=.15 over scale), P4, P5", flush=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model, cache_dir=cache, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    rows = load_rows(max(a.n_pool, a.n_eval), split="validation", cache_dir=cache)
    rows = [r for r in rows if gold_triples(r) and head_entity(r)]
    print("  %d usable validation instances; %d evaluated per scale" % (len(rows), a.n_eval), flush=True)

    out = {"prereg": "82da296", "config": vars(a), "scales": a.scales}

    # ---- store arm: no language model, cheapest, and it carries the P1a gate ------------------------------
    out["store"] = store_sweep(rows, a.scales, a.n_eval, a.dim, a.n_rows, a.cb_seeds)
    json.dump(out, open(a.out, "w"), indent=1)

    # ---- frozen prompt arm (carries the P1c reconstruction gate) -----------------------------------------
    llm = AutoModelForCausalLM.from_pretrained(a.model, cache_dir=cache, dtype=torch.bfloat16,
                                               device_map="cuda", trust_remote_code=True).eval()
    out["prompt"] = llm_sweep(llm, tok, rows, a.scales, a.n_eval, a.seed, "prompt")
    json.dump(out, open(a.out, "w"), indent=1)

    # ---- LoRA fine-tune at V-99's configuration ----------------------------------------------------------
    # V-99's train_one is a SINGLE pass over the rows -- there is no epoch loop and v99_train.py has no
    # --epochs flag, so the banked .7333 is 10,000 examples x 1 pass. (The V-98 ledger text claims "1 -> 2
    # epochs" for its 10k arm, but --epochs was declared and never referenced in v97/v98/v103_train.py, so
    # those runs were also single-pass. Recorded as a documentation defect, not inherited here.)
    tr = load_rows(a.n_train, split="train", cache_dir=cache)
    print("  training LoRA on %d train instances, ONE pass (r16 a32 lr %g accum %d budget %d max_len %d)"
          % (len(tr), a.lr, a.accum, a.budget, a.max_len), flush=True)
    llm = lora_wrap(llm, a.seed)
    t0 = time.time()
    train_one(llm, tok, tr, "qa", a)
    out["train_minutes"] = (time.time() - t0) / 60.0
    try:
        llm.save_pretrained(a.adapter_out)
        print("  adapter saved to %s (no adapter had ever been saved in this project)"
              % a.adapter_out, flush=True)
    except Exception as e:                                     # saving must not cost the run
        print("  WARNING: adapter save failed: %r" % (e,), flush=True)

    # ---- P1b gate eval: V-99's own protocol, its own evidence budget, its own row set ---------------------
    # Two details that would silently invalidate this gate if inherited from the sweep instead:
    #   (1) V-99 evaluates load_rows(300, "validation") with NO further filtering, whereas the sweep filters on
    #       gold_triples/head_entity (V-100/V-102's pool). Different row sets -> not a reproduction.
    #   (2) V-99 truncates the GENERATION input at 3500, not at --max_len (2048), which is the TRAINING cap.
    ev = load_rows(a.n_gate_eval, split="validation", cache_dir=cache)
    g_ok = []
    for r in ev:
        pred, _, _ = generate(llm, tok, qa_prompt(r, tok, a.budget), max_new=16, max_len=3500)
        g_ok.append(em(pred, r["answer"]))
    out["lora_gate"] = {"n": len(g_ok), "em": float(np.mean(g_ok)), "protocol": "V-99 qa_prompt budget=%d" % a.budget}
    print("  [lora gate] V-99 protocol em %.4f on n=%d (banked .7333 at n_train=10000)"
          % (out["lora_gate"]["em"], len(g_ok)), flush=True)
    json.dump(out, open(a.out, "w"), indent=1)

    # ---- LoRA arm across evidence scale: the measurement that has never been made -------------------------
    out["lora"] = llm_sweep(llm, tok, rows, a.scales, a.n_eval, a.seed, "lora")
    json.dump(out, open(a.out, "w"), indent=1)
    print("V106-MEASURE-DONE", flush=True)


if __name__ == "__main__":
    main()
