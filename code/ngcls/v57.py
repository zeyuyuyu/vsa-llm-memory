"""V-57 (V-56 Part B): does the gain over a supervised fine-tune GROW with hops? (bars at 86d5fec, Part B.)

Arena C is a place where the algebra is exact -- the parser clears its floor by wide margins at every depth we
have run -- so it satisfies the open half of the condition by construction. What is tested here is the measured
half at increasing depth: the model's own single pass should degrade with hops while the handover should not,
until HRR capacity runs out, which V-43 put at four hops rather than three.

Arms at each depth: floor (frozen last-token probe, the paper's convention), parser, probe_argmax, and
lora_direct -- LoRA on the attention projections plus the same last-token two-way head, trained jointly on the
same instances and labels. Synthetic markup, and the manuscript will say so.
"""
import argparse, json, os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
sys.path.insert(0, "/path/to/work/ngcls")
import v13                                                            # noqa: E402


def last_tok(H, r):
    return H[max(int(r["am"].sum()) - 1, 0)].float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depths", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    ap.add_argument("--n_train", type=int, default=800)
    ap.add_argument("--n_test", type=int, default=500)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--probe_epochs", type=int, default=40)
    ap.add_argument("--lora_epochs", type=int, default=2)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="v57_logs/v57.json")
    a = ap.parse_args()

    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print("===== V-57  does the gain over a fine-tune grow with hops? arena C =====", flush=True)
    out = {"depths": a.depths, "cells": {}}

    llm = AutoModel.from_pretrained("microsoft/phi-2", trust_remote_code=True,
                                    torch_dtype=torch.float16).to(a.device).eval()
    for p in llm.parameters():
        p.requires_grad_(False)

    for d in a.depths:
        old = v13.HOPS
        v13.HOPS = d
        cell = {}
        for arm in ("floor", "parser", "probe_argmax"):
            accs = [v13.run(arm, s, tok, llm, a.device, a.n_train, a.n_test,
                            a.epochs, a.bs, a.lr)["acc"] for s in a.seeds]
            cell[arm] = accs
            print("  depth %d %-13s %s mean=%.4f" % (d, arm, [round(x, 4) for x in accs],
                                                     float(np.mean(accs))), flush=True)
        v13.HOPS = old
        out["cells"][str(d)] = cell
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(out, open(a.out, "w"), indent=1)
    del llm
    torch.cuda.empty_cache()

    from peft import LoraConfig, get_peft_model
    for d in a.depths:
        old = v13.HOPS
        v13.HOPS = d
        tr = v13.Arena(tok, a.n_train, 42)
        te = v13.Arena(tok, a.n_test, 4242)
        v13.HOPS = old
        accs = []
        for seed in a.seeds:
            torch.manual_seed(seed)
            m = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", trust_remote_code=True,
                                                     torch_dtype=torch.bfloat16).to(a.device)
            m = get_peft_model(m, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0, bias="none",
                                             target_modules=["q_proj", "k_proj", "v_proj", "dense"],
                                             task_type="CAUSAL_LM"))
            head = nn.Linear(v13.HID, 2).to(a.device).to(torch.bfloat16)
            m.train()
            params = [p for p in m.parameters() if p.requires_grad] + list(head.parameters())
            opt = torch.optim.AdamW(params, lr=1e-4)
            g = torch.Generator().manual_seed(seed)

            def fwd(rows):
                ids = torch.stack([r["ids"] for r in rows]).to(a.device)
                am = torch.stack([r["am"] for r in rows]).to(a.device)
                hs = m(input_ids=ids, attention_mask=am, output_hidden_states=True).hidden_states[-1]
                idx = am.sum(-1) - 1
                return head(hs[torch.arange(hs.size(0), device=a.device), idx]).float()

            for ep in range(a.lora_epochs):
                perm = torch.randperm(len(tr.rows), generator=g).tolist()
                tot, nb = 0.0, 0
                for k in range(0, len(perm), a.bs):
                    rows = [tr.rows[i] for i in perm[k:k + a.bs]]
                    y = torch.tensor([r["label"] for r in rows], device=a.device)
                    loss = F.cross_entropy(fwd(rows), y)
                    opt.zero_grad(); loss.backward()
                    torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
                    tot += float(loss.item()); nb += 1
                print("      depth %d lora_direct seed %d ep%d loss=%.4f"
                      % (d, seed, ep + 1, tot / max(1, nb)), flush=True)
            m.eval()
            ok = []
            with torch.no_grad():
                for k in range(0, len(te.rows), a.bs):
                    rows = te.rows[k:k + a.bs]
                    pred = fwd(rows).argmax(-1).tolist()
                    ok += [int(p == r["label"]) for p, r in zip(pred, rows)]
            accs.append(float(np.mean(ok)))
            del m, head
            torch.cuda.empty_cache()
        out["cells"][str(d)]["lora_direct"] = accs
        print("  depth %d %-13s %s mean=%.4f" % (d, "lora_direct", [round(x, 4) for x in accs],
                                                 float(np.mean(accs))), flush=True)
        json.dump(out, open(a.out, "w"), indent=1)

    print("", flush=True)
    print("===== V-57 SCORING (bars registered at 86d5fec, Part B) =====", flush=True)
    gaps = {}
    for d in a.depths:
        c = out["cells"][str(d)]
        H = float(np.mean(c["parser"])) - float(np.mean(c["floor"]))
        g = [h - l for h, l in zip(c["probe_argmax"], c["lora_direct"])]
        gaps[d] = g
        print("  depth %d  floor %.4f parser %.4f (H %.4f)  handover %.4f  lora_direct %.4f  gap %s"
              % (d, float(np.mean(c["floor"])), float(np.mean(c["parser"])), H,
                 float(np.mean(c["probe_argmax"])), float(np.mean(c["lora_direct"])),
                 [round(x, 4) for x in g]), flush=True)
    out["gaps"] = {str(k): v for k, v in gaps.items()}
    void = [d for d in a.depths
            if float(np.mean(out["cells"][str(d)]["parser"]))
            - float(np.mean(out["cells"][str(d)]["floor"])) < 0.20]
    if void:
        v = "VOID at depths %s (admission parser - floor < 0.20)" % void
    elif 2 in gaps and 3 in gaps:
        at2 = all(x >= 0.10 for x in gaps[2])
        grows = all(x > y for x, y in zip(gaps[3], gaps[2]))
        print("  gap >= .10 at depth 2 in 3/3: %s ; strictly larger at depth 3 in 3/3: %s" % (at2, grows),
              flush=True)
        if at2 and grows:
            v = ("THE GAIN SCALES (the advantage over a supervised fine-tune is at least ten points at two hops "
                 "and strictly larger at three, in three of three seeds)")
        elif not grows:
            v = ("THE GAIN DOES NOT SCALE (it is not larger at three hops than at two; the twenty points stays "
                 "a single-depth result and the manuscript says the gain does not grow)")
        else:
            v = "GRADED (report the table)"
    else:
        v = "INCOMPLETE"
    out["verdict"] = v
    json.dump(out, open(a.out, "w"), indent=1)
    print("  >>> V-57 VERDICT: %s" % v, flush=True)
    print("V57-DONE", flush=True)


if __name__ == "__main__":
    main()
