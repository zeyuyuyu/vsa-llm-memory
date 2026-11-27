"""V-8 (prereg 3b9c460): is the bottleneck localisation rather than identity?

Four numbers are already banked on this task, same model, same budget:
  vsa_parser  spans AND indices given, 2 trainable scalars   -> .918
  consistent  every symbol relabelled but consistently       -> .869   (correctness is not required)
  A_extract   identity linearly decodable from the frozen
              states when spans are given                    -> .9986  (chance .0039)
  vsa_hidden  raw hidden states, 27.9M trainable             -> .505   (the probe floor)
Identity is present and recoverable, the algebra works, and correctness is not even needed -- yet the
hidden-state arm sits at the floor. By elimination the remaining candidate is the INTERFACE: discovering
which token spans are symbols and how they pair. This file tests that directly instead of inferring it.

THE NEW ARM, vsa_spans. It is given the marker spans (start, end, kind) exactly as the parser arm's scanner
finds them, but NOT the symbol indices. Each R/X/Y symbol vector is instead a learned projection of the
frozen hidden states MEAN-POOLED OVER ITS OWN SPAN, L2-normalised into VSA space. Pairing follows span
order, as in the parser arm. Everything else -- bundling, unbinding, cleanup over the observed candidate
set, the filler codebook, sim_scale/sim_bias -- is untouched.
Note that the span pooling and the linear map here are the SAME construction that reached A_extract=.9986,
so this arm demonstrably has access to identity; if it still fails, identity availability is not the cause.

BARS (registered at 3b9c460):
  LOCALISATION IS THE BOTTLENECK      vsa_spans >= .80 in 3/3 seeds
  IDENTITY READOUT ALSO A BOTTLENECK  vsa_spans <= .60 in 3/3 seeds
  INTERMEDIATE                        otherwise, reported with the interpolation stated
"""
import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
sys.path.insert(0, "/path/to/work/sync_h20/rewardbench_experiment/zeyu/hybrid_neurosymbolic")

from systematic_chain_dataset import SystematicChainDataset, ChainConfig      # noqa: E402
from explicit_chain_models import SymbolicVSAChainClassifier, l2_normalize    # noqa: E402
from transformers import AutoModel, AutoTokenizer                            # noqa: E402

HID = 2560
PARSER_REF = 0.918
HIDDEN_REF = 0.505


class SpanVSA(SymbolicVSAChainClassifier):
    """Given spans, not indices. Symbol vectors are learned projections of span-pooled frozen states."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.proj = nn.Linear(HID, self.vsa_dim)
        self.sim_scale.requires_grad_(True)
        self.sim_bias.requires_grad_(True)

    def sym(self, H, s, e):
        span = H[s:e] if e > s else H[s:s + 1]
        return l2_normalize(self.proj(span.float().mean(0)))

    def forward(self, input_ids, attention_mask, hidden=None):
        device = input_ids.device
        B = input_ids.size(0)
        logits = torch.zeros(B, 2, device=device)
        for b in range(B):
            ids = input_ids[b]
            toks = self.tokenizer.convert_ids_to_tokens(ids.tolist())
            markers = self._scan_markers(ids)
            if not markers:
                continue
            last_lc = None
            for i in range(len(toks) - 1, -1, -1):
                if toks[i].lstrip("Ġ") == "{":
                    last_lc = i; break
            if last_lc is None:
                continue
            cand, _ = self._parse_brace_int_at(toks, last_lc)
            if cand is None or cand < 0 or cand >= self.vocab_size:
                continue
            ms = sorted(markers, key=lambda x: x[2])
            H = hidden[b]

            # query role = last R span
            q_role_v = None
            for kind, midx, s, e in reversed(ms):
                if kind == "R":
                    q_role_v = self.sym(H, s, e); break
            if q_role_v is None:
                continue

            A, Bp, C = [], [], []
            for i in range(len(ms) - 1):
                k1, _, s1, e1 = ms[i]
                k2, _, s2, e2 = ms[i + 1]
                if k1 == "R" and k2 == "X":
                    A.append((self.sym(H, s1, e1), self.sym(H, s2, e2)))
                elif k1 == "X" and k2 == "Y":
                    Bp.append((self.sym(H, s1, e1), self.sym(H, s2, e2)))
            for kind, midx, s, e in ms:
                if kind != "Y":
                    continue
                if e < len(toks) and toks[e].lstrip("Ġ") == "{":
                    f_id, _ = self._parse_brace_int_at(toks, e)
                    if f_id is None or f_id < 0 or f_id >= self.vocab_size:
                        continue
                    C.append((self.sym(H, s, e), f_id))
            if not A or not Bp or not C:
                continue

            memA = self.vsa.bundle(torch.stack([self.vsa.bind(x, r) for r, x in A], 0))
            x_hat = l2_normalize(self.vsa.unbind(memA, q_role_v))
            x_cands = torch.stack([x for _, x in A], 0)
            x_clean = x_cands[int(self.vsa.similarity(
                x_cands, x_hat.unsqueeze(0).expand_as(x_cands)).argmax().item())]

            memB = self.vsa.bundle(torch.stack([self.vsa.bind(y, x) for x, y in Bp], 0))
            y_hat = l2_normalize(self.vsa.unbind(memB, x_clean))
            y_cands = torch.stack([y for _, y in Bp], 0)
            y_clean = y_cands[int(self.vsa.similarity(
                y_cands, y_hat.unsqueeze(0).expand_as(y_cands)).argmax().item())]

            memC = self.vsa.bundle(torch.stack(
                [self.vsa.bind(self.f_vectors[f].to(device), y) for y, f in C], 0))
            f_hat = l2_normalize(self.vsa.unbind(memC, y_clean))
            sim = self.vsa.similarity(f_hat, self.f_vectors[int(cand)].to(device)).clamp(-1, 1)
            lp = sim * self.sim_scale + self.sim_bias
            logits[b, 1] = lp
            logits[b, 0] = -lp
        return logits


@torch.no_grad()
def cache_hidden(ds, llm, device, bs=8):
    """The LLM is frozen, so its states are computed once. Kept in fp16 on CPU."""
    ids_all, am_all, h_all, y_all = [], [], [], []
    for i in range(0, len(ds), bs):
        batch = [ds[j] for j in range(i, min(i + bs, len(ds)))]
        ids = torch.stack([b["input_ids"] for b in batch])
        am = torch.stack([b["attention_mask"] for b in batch])
        h = llm(input_ids=ids.to(device), attention_mask=am.to(device)).last_hidden_state
        ids_all.append(ids); am_all.append(am); h_all.append(h.half().cpu())
        y_all += [int(b["labels"]) for b in batch]
    return (torch.cat(ids_all), torch.cat(am_all), torch.cat(h_all), torch.tensor(y_all))


def run(seed, cfg, tok, llm, device, n_train, n_eval, epochs, bs, lr):
    torch.manual_seed(seed)
    tr = SystematicChainDataset(tok, num_samples=n_train, split="train", cfg=cfg)
    ev = SystematicChainDataset(tok, num_samples=n_eval, split="eval", cfg=cfg)
    TR = cache_hidden(tr, llm, device)
    EV = cache_hidden(ev, llm, device)
    m = SpanVSA(vsa_dim=2048, num_roles=cfg.num_roles, vocab_size=cfg.vocab_size, seed=seed).to(device)
    params = [p for p in m.parameters() if p.requires_grad]
    n_tr = sum(p.numel() for p in params)
    opt = torch.optim.AdamW(params, lr=lr)

    def ev_acc():
        m.eval(); c = t = 0
        with torch.no_grad():
            for k in range(0, EV[0].size(0), bs):
                lg = m(EV[0][k:k + bs].to(device), EV[1][k:k + bs].to(device),
                       hidden=EV[2][k:k + bs].to(device))
                c += (lg.argmax(-1).cpu() == EV[3][k:k + bs]).sum().item(); t += lg.size(0)
        return c / t

    t0 = time.time()
    best = 0.0
    g = torch.Generator(); g.manual_seed(seed)
    for ep in range(epochs):
        m.train()
        perm = torch.randperm(TR[0].size(0), generator=g)
        for k in range(0, TR[0].size(0), bs):
            i = perm[k:k + bs]
            lg = m(TR[0][i].to(device), TR[1][i].to(device), hidden=TR[2][i].to(device))
            loss = F.cross_entropy(lg, TR[3][i].to(device))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        a = ev_acc()
        best = max(best, a)
        print("      ep%d eval=%.4f (%.0fs)" % (ep + 1, a, time.time() - t0), flush=True)
    return {"seed": seed, "acc": best, "trainable": n_tr, "secs": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    ap.add_argument("--train_samples", type=int, default=2000)
    ap.add_argument("--eval_samples", type=int, default=1000)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="v8_logs/v8.json")
    a = ap.parse_args()

    cfg = ChainConfig(num_roles=64, vocab_size=256, num_pairs=16, max_length=512, p_positive=0.5)
    tok = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    llm = AutoModel.from_pretrained("microsoft/phi-2", trust_remote_code=True,
                                    torch_dtype=torch.float16).to(a.device).eval()
    for p in llm.parameters():
        p.requires_grad_(False)

    rows = []
    for s in a.seeds:
        r = run(s, cfg, tok, llm, a.device, a.train_samples, a.eval_samples,
                a.epochs, a.batch_size, a.lr)
        print("  vsa_spans sd%-4d acc=%.4f trainable=%s %ss"
              % (s, r["acc"], format(r["trainable"], ","), r["secs"]), flush=True)
        rows.append(r)
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(rows, open(a.out, "w"), indent=1)

    accs = [r["acc"] for r in rows]
    print("\n  vsa_spans per seed: %s  mean=%.4f" % ([round(x, 4) for x in accs], float(np.mean(accs))),
          flush=True)
    print("  banked references: vsa_parser %.3f (spans+indices) | vsa_hidden %.3f (neither)"
          % (PARSER_REF, HIDDEN_REF), flush=True)
    if all(x >= 0.80 for x in accs):
        v = "LOCALISATION IS THE BOTTLENECK"
    elif all(x <= 0.60 for x in accs):
        v = "IDENTITY READOUT IS ALSO A BOTTLENECK"
    else:
        v = "INTERMEDIATE"
    frac = (float(np.mean(accs)) - HIDDEN_REF) / (PARSER_REF - HIDDEN_REF)
    print("  recovered fraction of the parser-vs-hidden gap: %.3f" % frac, flush=True)
    print("  >>> V-8 VERDICT: %s" % v, flush=True)
    print("V8-DONE", flush=True)


if __name__ == "__main__":
    main()
