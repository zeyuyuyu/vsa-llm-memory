"""V-98 (prereg 8cf641c), adapted from v97_train.py. Formerly V-97: train EXTRACTION, hash identity, keep the algebra exact.

Three arms, none given gold evidence at test time:
  frozen_extract : the frozen model emits triples, then hash + HRR two-hop. The manuscript's no-gold floor.
  lora_only      : THE GATE. Answer-supervised LoRA, no store. V-86 measured .6267.
  extract_ft     : LoRA supervised on the TRAIN split's gold TRIPLES, then hash + HRR two-hop. THE TEST.

No gradient crosses the discrete interface. Extraction is supervised directly, which is the correction V-96
forced: its discrete path had the right geometry (cross-talk .0004 against a .0989 tolerance) and the model
ignored it anyway, because a straight-through surrogate cannot discover which row is right while the direct
supervision sits unused in the data.

Primary metric is extraction RECALL. P4 checks accuracy against recall squared, since both hops must survive.
"""
import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v85_data import load_rows, build_context, em, norm    # noqa: E402
import v97_store as ST                                     # noqa: E402

EX_INSTR = ("Extract every fact from the context as `subject | relation | object`, one per line. "
            "Use the exact surface forms from the context.")


def gold_triples(row):
    out = []
    for t in row.get("evidences", []):
        if isinstance(t, (list, tuple)) and len(t) >= 3 and all(str(x).strip() for x in t[:3]):
            out.append((str(t[0]).strip(), str(t[1]).strip(), str(t[2]).strip()))
    return out


def chain_triples(row):
    """The gold triples the supporting facts identify -- the only ones a two-hop answer can depend on.
    Overall recall counts every gold triple equally; these are the two that decide the answer."""
    sup = set(zip(row["supporting_facts"]["title"], row["supporting_facts"]["sent_id"]))
    titles = {t for t, _ in sup}
    out = [t for t in gold_triples(row) if norm(t[0]) in {norm(x) for x in titles}]
    return out if out else gold_triples(row)[:2]


def head_entity(row):
    """The chain's starting entity: the subject of the gold triple whose subject appears in the question.
    Used only at EVALUATION as the walk's entry point, and derived from the QUESTION text, not from gold
    objects -- so no answer information enters."""
    q = norm(row["question"])
    cands = [t[0] for t in gold_triples(row)]
    for c in cands:
        if norm(c) and norm(c) in q:
            return c
    return cands[0] if cands else ""


def ex_prompt(row, tok, budget, question_conditioned=False):
    ctx, _ = build_context(row, tok, budget)
    if question_conditioned:
        # PROTOCOL CHANGE, labelled: the store built this way serves one question and cannot be reused,
        # which is the property the blind protocol exists to preserve.
        return ("%s\n\nContext:\n%s\n\nQuestion: %s\n\nFacts needed to answer it:\n"
                % (EX_INSTR, ctx, row["question"]))
    return "%s\n\nContext:\n%s\n\nFacts:\n" % (EX_INSTR, ctx)


def qa_prompt(row, tok, budget):
    from v85_data import prompt_for
    p, _ = prompt_for(row, tok, budget)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--arm", choices=["extract_ft", "extract_ft_10k", "extract_ft_qc"], required=True)
    ap.add_argument("--n_train", type=int, default=3000)
    ap.add_argument("--n_eval", type=int, default=300)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--budget", type=int, default=1200)
    ap.add_argument("--max_len", type=int, default=2048)
    ap.add_argument("--max_new", type=int, default=320)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="v97_arm.json")
    a = ap.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cache = os.environ.get("HF_HOME")
    print("===== V-97 %s (prereg 8cf641c) seed %d =====" % (a.arm, a.seed), flush=True)
    print("  references: frozen prompt .3967 | lora_only .6267 | store w/ gold .9744 | "
          "frozen self-extraction recall .343 acc .018", flush=True)

    tok = AutoTokenizer.from_pretrained(a.model, cache_dir=cache, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    ev = load_rows(a.n_eval, split="validation", cache_dir=cache)
    tr = load_rows(a.n_train, split="train", cache_dir=cache) if a.arm != "frozen_extract" else []
    print("  eval %d (dev, the paper's slice) | train %d" % (len(ev), len(tr)), flush=True)

    llm = AutoModelForCausalLM.from_pretrained(a.model, cache_dir=cache, dtype=torch.bfloat16,
                                               device_map="cuda", trust_remote_code=True)

    qc = (a.arm == "extract_ft_qc")
    if True:
        from peft import LoraConfig, get_peft_model
        llm.gradient_checkpointing_enable(); llm.enable_input_require_grads()
        llm = get_peft_model(llm, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                                             task_type="CAUSAL_LM",
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
        tp = [p for p in llm.parameters() if p.requires_grad]
        print("  trainable %.2fM" % (sum(p.numel() for p in tp) / 1e6), flush=True)
        opt = torch.optim.AdamW(tp, lr=a.lr)

        t0 = time.time(); tot = nb = 0
        opt.zero_grad()
        for k, r in enumerate(tr):
            if True:
                gt = gold_triples(r)
                if not gt:
                    continue
                prompt = ex_prompt(r, tok, a.budget, qc)
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
            if (k + 1) % 300 == 0:
                print("    step %d loss %.4f (%.1f min)" % (k + 1, tot / max(nb, 1),
                      (time.time() - t0) / 60), flush=True)
        print("  train loss %.4f over %d examples" % (tot / max(nb, 1), nb), flush=True)
        llm.gradient_checkpointing_disable()
        if hasattr(llm, "config"):
            llm.config.use_cache = True

    llm.eval()
    hits, recalls, chain_recalls, agrees, bads = [], [], [], [], []
    for i, r in enumerate(ev):
        if False:
            p = qa_prompt(r, tok, a.budget)
            ids = tok(p, return_tensors="pt", truncation=True, max_length=3500).to("cuda")
            with torch.no_grad():
                o = llm.generate(**ids, max_new_tokens=16, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
            pred = tok.decode(o[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).split("\n")[0]
            hits.append(em(pred, r["answer"]))
        else:
            p = ex_prompt(r, tok, a.budget, qc)
            ids = tok(p, return_tensors="pt", truncation=True, max_length=3500).to("cuda")
            with torch.no_grad():
                o = llm.generate(**ids, max_new_tokens=a.max_new, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
            txt = tok.decode(o[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
            trip, bad = ST.parse_triples(txt)
            bads.append(bad)
            recalls.append(ST.triple_recall(trip, gold_triples(r)))
            chain_recalls.append(ST.triple_recall(trip, chain_triples(r)))
            hrr, sym, ok = ST.answer_two_hop(trip, head_entity(r), "cuda")
            if ok is not None:
                agrees.append(ok)          # None = ambiguous walk, excluded by design
            hits.append(em(hrr, r["answer"]))
        if (i + 1) % 50 == 0:
            msg = "    %d/%d  EM %.4f" % (i + 1, len(ev), float(np.mean(hits)))
            if recalls:
                msg += "  recall %.4f" % float(np.nanmean(recalls))
            print(msg, flush=True)

    acc = float(np.mean(hits))
    rec = float(np.nanmean(recalls)) if recalls else float("nan")
    crec = float(np.nanmean(chain_recalls)) if chain_recalls else float("nan")
    out = {"prereg": "8cf641c", "arm": a.arm, "seed": a.seed, "n_eval": len(hits), "em": acc,
           "recall": rec, "chain_recall": crec, "store_agreement": (float(np.mean(agrees)) if agrees else None),
           "malformed_lines_mean": (float(np.mean(bads)) if bads else None),
           "n_train": len(tr), "lr": a.lr}
    json.dump(out, open(a.out, "w"), indent=1)
    print("", flush=True)
    print("  >>> %s seed %d  EM = %.4f  recall = %.4f  CHAIN recall = %.4f"
          % (a.arm, a.seed, acc, rec, crec), flush=True)
    print("  >>> reframing check: |EM - chain_r^2| = %.4f   vs   |EM - r^2| = %.4f"
          % (abs(acc - crec ** 2), abs(acc - rec ** 2)), flush=True)
    if agrees:
        print("  >>> HRR/symbolic agreement %.4f on %d unambiguous instances "
              "(must be ~1.0; lower means the store is broken)"
              % (float(np.mean(agrees)), len(agrees)), flush=True)
        print("  >>> mechanism check: recall^2 = %.4f vs EM %.4f  (P4: within .15)"
              % (rec ** 2, acc), flush=True)
    print("V98-ARM-DONE", flush=True)


if __name__ == "__main__":
    main()
