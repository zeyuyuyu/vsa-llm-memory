"""V-85 training arms (prereg 85461e8): LoRA-only and LoRA+VSA, trained jointly on 2WikiMultihopQA.

Both arms fine-tune Qwen3-8B with LoRA on the 2Wiki TRAIN split and are evaluated on the same 300
compositional DEV instances the paper uses. **No arm is given gold evidence at any point** -- the passages
only. That is the boundary the experiment attacks: the paper's store needs gold triples supplied, and
self-extraction under a frozen model collapses to 0.0178-0.0311.

Fusion, for the lora_vsa arm. The VSA module reads the passage's hidden states, binds a role vector to a
filler vector at every position, superposes them into one memory, unbinds with a question-derived query, and
scores passage positions by how well their filler matches what was retrieved. Those scores are scattered
onto the vocabulary and ADDED to the language model's own logits at every decoding step, with a learned
weight. So the module is an extractive prior that participates in generation rather than a classifier bolted
alongside it, and the gradient from the answer loss flows through the binding algebra into the model's
representations -- exactly the path the frozen protocol forbids.
"""
import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v85_data import load_rows, prompt_for, em          # noqa: E402
from v85_vsa import VSAStore                            # noqa: E402


class Fused(torch.nn.Module):
    """LLM (LoRA) + optional VSA extractive prior fused at the logit level."""

    def __init__(self, llm, use_vsa, d_model, vocab, d_vsa=512):
        super().__init__()
        self.llm = llm
        self.use_vsa = use_vsa
        if use_vsa:
            self.store = VSAStore(d_model, d_vsa)
            # Zero-init: at step 0 the fused model IS the lora_only model, so any use of the VSA prior
            # is learned rather than imposed. Also makes lora_vsa a strict superset of lora_only at init.
            self.alpha = torch.nn.Parameter(torch.tensor(0.0))
        self.vocab = vocab

    def vsa_bias(self, hidden, mask, q_index, input_ids):
        """Scores over passage positions, scattered onto the vocabulary."""
        scores = self.store(hidden, mask, q_index)                      # [B,T]
        bias = hidden.new_full((hidden.shape[0], self.vocab), 0.0)
        bias = bias.scatter_reduce(1, input_ids, scores.to(bias.dtype),
                                   reduce="amax", include_self=True)
        return bias * self.alpha.clamp(-5.0, 5.0)

    def forward(self, input_ids, attn, labels, q_index):
        out = self.llm(input_ids=input_ids, attention_mask=attn, output_hidden_states=self.use_vsa)
        logits = out.logits
        if self.use_vsa:
            h = out.hidden_states[-1]
            # The model's output width is the padded embedding size, which exceeds len(tokenizer) on Qwen.
            # Build the bias at the width the logits actually have.
            self.vocab = logits.shape[-1]
            bias = self.vsa_bias(h, attn.bool(), q_index, input_ids)     # [B,V]
            logits = logits + bias.unsqueeze(1)                          # same prior at every step
        sl = logits[:, :-1].contiguous()
        tl = labels[:, 1:].contiguous()
        return F.cross_entropy(sl.view(-1, sl.size(-1)).float(), tl.view(-1), ignore_index=-100)

    @torch.no_grad()
    def greedy(self, input_ids, attn, q_index, max_new, eos):
        """Greedy decode with the VSA prior fused in at every step."""
        bias = None
        if self.use_vsa:
            o = self.llm(input_ids=input_ids, attention_mask=attn, output_hidden_states=True)
            self.vocab = o.logits.shape[-1]          # padded embedding width, not len(tokenizer)
            bias = self.vsa_bias(o.hidden_states[-1], attn.bool(), q_index, input_ids)
        cur, done = input_ids, torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        outs = []
        past = None
        for _ in range(max_new):
            o = self.llm(input_ids=cur if past is None else cur[:, -1:],
                         attention_mask=attn, past_key_values=past, use_cache=True)
            past = o.past_key_values
            lg = o.logits[:, -1]
            if bias is not None:
                lg = lg + bias
            nxt = lg.argmax(-1)
            nxt = torch.where(done, torch.full_like(nxt, eos), nxt)
            outs.append(nxt)
            done = done | (nxt == eos)
            cur = torch.cat([cur, nxt.unsqueeze(1)], dim=1)
            attn = torch.cat([attn, (~done).long().unsqueeze(1)], dim=1)
            if done.all():
                break
        return torch.stack(outs, dim=1)


def encode(rows, tok, budget, max_len, device, train):
    """Builds prompt(+answer) batches. q_index is the last prompt token, whose state seeds the VSA query."""
    items = []
    for r in rows:
        p, _ = prompt_for(r, tok, budget)
        pid = tok(p, truncation=True, max_length=max_len)["input_ids"]
        if train:
            aid = tok(" " + str(r["answer"]) + tok.eos_token, add_special_tokens=False)["input_ids"]
            ids = pid + aid
            lab = [-100] * len(pid) + aid
        else:
            ids, lab = pid, [-100] * len(pid)
        items.append((ids, lab, len(pid) - 1, r))
    return items


def batches(items, bs, tok, device, train):
    for i in range(0, len(items), bs):
        chunk = items[i:i + bs]
        L = max(len(c[0]) for c in chunk)
        ids = torch.full((len(chunk), L), tok.pad_token_id, dtype=torch.long)
        lab = torch.full((len(chunk), L), -100, dtype=torch.long)
        att = torch.zeros((len(chunk), L), dtype=torch.long)
        qi = torch.zeros(len(chunk), dtype=torch.long)
        for j, (a, b, q, _) in enumerate(chunk):
            ids[j, :len(a)] = torch.tensor(a)
            lab[j, :len(b)] = torch.tensor(b)
            att[j, :len(a)] = 1
            qi[j] = q
        yield (ids.to(device), att.to(device), lab.to(device), qi.to(device),
               [c[3] for c in chunk])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--arm", choices=["lora_only", "lora_vsa"], required=True)
    ap.add_argument("--n_train", type=int, default=4000)
    ap.add_argument("--n_eval", type=int, default=300)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--budget", type=int, default=1200)
    ap.add_argument("--max_len", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="v85_arm.json")
    a = ap.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    cache = os.environ.get("HF_HOME")

    print("===== V-85 %s (prereg 85461e8) seed %d =====" % (a.arm, a.seed), flush=True)
    tok = AutoTokenizer.from_pretrained(a.model, cache_dir=cache, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    tr_rows = load_rows(a.n_train, split="train", cache_dir=cache)
    ev_rows = load_rows(a.n_eval, split="validation", cache_dir=cache)
    print("  train %d (train split) | eval %d (dev split, the paper's slice)"
          % (len(tr_rows), len(ev_rows)), flush=True)

    llm = AutoModelForCausalLM.from_pretrained(a.model, cache_dir=cache, dtype=torch.bfloat16,
                                               device_map="cuda", trust_remote_code=True)
    llm.gradient_checkpointing_enable()
    llm.enable_input_require_grads()
    lcfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    llm = get_peft_model(llm, lcfg)
    d_model = llm.config.hidden_size if hasattr(llm, "config") else llm.base_model.config.hidden_size
    vocab_out = getattr(llm.config, "vocab_size", len(tok))
    print("  vocab: tokenizer %d | model output %d" % (len(tok), vocab_out), flush=True)
    model = Fused(llm, a.arm == "lora_vsa", d_model, vocab_out).to("cuda")
    if a.arm == "lora_vsa":
        model.store.to(torch.bfloat16)

    tp = [p for p in model.parameters() if p.requires_grad]
    print("  trainable params: %.2fM" % (sum(p.numel() for p in tp) / 1e6), flush=True)
    opt = torch.optim.AdamW(tp, lr=a.lr)

    tr_items = encode(tr_rows, tok, a.budget, a.max_len, "cuda", True)
    ev_items = encode(ev_rows, tok, a.budget, a.max_len, "cuda", False)

    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        model.train()
        np.random.shuffle(tr_items)
        tot, nb = 0.0, 0
        opt.zero_grad()
        for k, (ids, att, lab, qi, _) in enumerate(batches(tr_items, a.bs, tok, "cuda", True)):
            loss = model(ids, att, lab, qi) / a.accum
            loss.backward()
            tot += loss.item() * a.accum; nb += 1
            if (k + 1) % a.accum == 0:
                torch.nn.utils.clip_grad_norm_(tp, 1.0)
                opt.step(); opt.zero_grad()
            if (k + 1) % 200 == 0:
                print("    ep%d step %d loss %.4f (%.1f min)" % (ep, k + 1, tot / nb,
                      (time.time() - t0) / 60), flush=True)
        print("  epoch %d mean loss %.4f" % (ep, tot / max(nb, 1)), flush=True)

    model.eval()
    llm.gradient_checkpointing_disable()          # so KV caching works during generation
    if hasattr(llm, "config"):
        llm.config.use_cache = True
    hits = []
    for ids, att, lab, qi, rows in batches(ev_items, 4, tok, "cuda", False):
        gen = model.greedy(ids, att, qi, 16, tok.eos_token_id)
        for j, r in enumerate(rows):
            txt = tok.decode(gen[j], skip_special_tokens=True)
            hits.append(em(txt.split("\n")[0].strip(), r["answer"]))
    acc = float(np.mean(hits))
    out = {"prereg": "85461e8", "arm": a.arm, "seed": a.seed, "n_eval": len(hits), "em": acc,
           "n_train": len(tr_rows), "epochs": a.epochs}
    json.dump(out, open(a.out, "w"), indent=1)
    print("", flush=True)
    print("  >>> %s seed %d EM = %.4f  (frozen baseline .3967, gate .4467)" % (a.arm, a.seed, acc),
          flush=True)
    print("V85-ARM-DONE", flush=True)


if __name__ == "__main__":
    main()
