"""V-15 (prereg e293493): answer "why VSA at all" with a measurement.

LoRA beats every frozen-feature arm on the tasks measured so far, so a referee is entitled to ask what the
symbolic route buys. The candidate property is the one VSA is theoretically for and LoRA has no reason to
have: generalisation to symbol COMBINATIONS never seen in training. Arena B has exactly that split by
construction -- training never contains a held-out (role, filler) pair, evaluation contains only held-out
pairs -- and the symbolic parser arm scores .9960 there without training on pair identity at all.

Bars (prereg e293493): SYMBOLIC ADVANTAGE iff parser - LoRA >= 10pp in 3/3 on at least one arena;
NO ADVANTAGE iff LoRA >= parser - 2pp in 3/3 on both. Banked parser references: B .9960, C .9340.
"""
import argparse, json, os, sys, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
sys.path.insert(0, "/path/to/work/ngcls")
sys.path.insert(0, "/path/to/work/sync_h20/rewardbench_experiment/zeyu/hybrid_neurosymbolic")

PARSER_REF = {"B": 0.9960, "C": 0.9340}


class LoRAClf(nn.Module):
    def __init__(self, rank=16, alpha=32, dropout=0.05):
        super().__init__()
        from transformers import AutoModel
        from peft import LoraConfig, get_peft_model
        base = AutoModel.from_pretrained("microsoft/phi-2", trust_remote_code=True,
                                         torch_dtype=torch.float32)
        cfg = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=dropout, bias="none",
                         target_modules=["q_proj", "k_proj", "v_proj", "dense"])
        self.llm = get_peft_model(base, cfg)
        self.head = nn.Linear(2560, 2)

    def forward(self, ids, am):
        h = self.llm(input_ids=ids, attention_mask=am).last_hidden_state
        idx = am.sum(1).long().clamp(min=1) - 1
        return self.head(h[torch.arange(h.size(0), device=h.device), idx].float())


def get_rows(arena, tok, n_tr, n_te, seed):
    if arena == "B":
        from v11 import Arena as A11
        from systematic_role_binding_dataset import SystematicConfig
        cfg = SystematicConfig()
        return A11(tok, n_tr, "train", cfg, seed).rows, A11(tok, n_te, "eval", cfg, seed).rows
    import v13 as V
    return V.Arena(tok, n_tr, 42).rows, V.Arena(tok, n_te, 4242).rows


def run(arena, seed, tok, device, n_tr, n_te, epochs, bs, lr):
    torch.manual_seed(seed)
    tr, te = get_rows(arena, tok, n_tr, n_te, seed)
    m = LoRAClf().to(device)
    params = [p for p in m.parameters() if p.requires_grad]
    n_tr_p = sum(p.numel() for p in params)
    opt = torch.optim.AdamW(params, lr=lr)

    def ev():
        m.eval(); c = t = 0
        with torch.no_grad():
            for k in range(0, len(te), bs):
                b = te[k:k + bs]
                lg = m(torch.stack([r["ids"] for r in b]).to(device),
                       torch.stack([r["am"] for r in b]).to(device))
                c += (lg.argmax(-1).cpu() == torch.tensor([r["label"] for r in b])).sum().item()
                t += len(b)
        return c / t

    t0, best = time.time(), 0.0
    g = torch.Generator(); g.manual_seed(seed)
    for ep in range(epochs):
        m.train()
        perm = torch.randperm(len(tr), generator=g).tolist()
        for k in range(0, len(perm), bs):
            b = [tr[i] for i in perm[k:k + bs]]
            lg = m(torch.stack([r["ids"] for r in b]).to(device),
                   torch.stack([r["am"] for r in b]).to(device))
            loss = F.cross_entropy(lg, torch.tensor([r["label"] for r in b], device=device))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        a = ev(); best = max(best, a)
        print("      lora/%s ep%d test=%.4f (%.0fs)" % (arena, ep + 1, a, time.time() - t0), flush=True)
    del m; torch.cuda.empty_cache()
    return {"arena": arena, "seed": seed, "acc": best, "trainable": n_tr_p,
            "secs": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arenas", nargs="+", default=["B", "C"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    ap.add_argument("--n_train", type=int, default=600)
    ap.add_argument("--n_test", type=int, default=400)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--device", default="cuda:2")
    ap.add_argument("--out", default="v15_logs/v15.json")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    rows = []
    for arena in a.arenas:
        for s in a.seeds:
            r = run(arena, s, tok, a.device, a.n_train, a.n_test, a.epochs, a.batch_size, a.lr)
            print("  lora arena=%s sd%-4d acc=%.4f trainable=%s %ss"
                  % (arena, s, r["acc"], format(r["trainable"], ","), r["secs"]), flush=True)
            rows.append(r)
            os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
            json.dump(rows, open(a.out, "w"), indent=1)
    print("\n=== V-15 SCORING (prereg e293493) ===", flush=True)
    adv, noadv = False, True
    for arena in a.arenas:
        c = [r["acc"] for r in rows if r["arena"] == arena]
        if len(c) != len(a.seeds):
            continue
        pr = PARSER_REF[arena]
        gaps = [(pr - x) * 100 for x in c]
        print("  arena %s: lora %s mean=%.4f | parser %.4f | parser-lora per seed %s pp"
              % (arena, [round(x, 4) for x in c], float(np.mean(c)), pr,
                 [round(g, 2) for g in gaps]), flush=True)
        if all(g >= 10 for g in gaps):
            adv = True
        if not all(x >= pr - 0.02 for x in c):
            noadv = False
    v = "SYMBOLIC ADVANTAGE ESTABLISHED" if adv else ("NO ADVANTAGE" if noadv else "INTERMEDIATE")
    print("  >>> V-15 VERDICT: %s" % v, flush=True)
    print("V15-DONE", flush=True)


if __name__ == "__main__":
    main()
