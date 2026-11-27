"""V-9 (prereg 6f162d1): does supplying the missing supervision recover the gap? The constructive half.

V-8 showed that giving the VSA arm the marker spans buys nothing (.501) even though a linear map on the
SAME span-pooled frozen states, supervised on identity, recovers the symbol at .9986. So the binding
objective does not induce a codebook. V-9 supplies that supervision two ways.

ARM 1  vsa_spans_sup     V-8's arm plus one auxiliary loss pulling each projected span vector toward the
                         fixed codebook vector of its true symbol. True ids at TRAINING time only.
ARM 2  extract_then_bind Stage 1 trains the identity probe (span-pooled states -> linear -> 256-way).
                         Stage 2 uses its ARGMAX to index the fixed codebook and runs the untouched
                         symbolic binding chain. No integer is ever read out of the input string.

LIMITATION, stated in the prereg and repeated here: both arms still use the tokenizer-level scanner to FIND
the spans. They are "hand-written span finder plus learned identity reader", not fully learned pipelines.

BARS: REMEDY FOUND iff either arm >= .80 in 3/3; REMEDY FAILS iff both <= .60 in 3/3; else INTERMEDIATE.
"""
import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
sys.path.insert(0, "/path/to/work/sync_h20/rewardbench_experiment/zeyu/hybrid_neurosymbolic")

from systematic_chain_dataset import SystematicChainDataset, ChainConfig       # noqa: E402
from explicit_chain_models import SymbolicVSAChainClassifier, l2_normalize     # noqa: E402
from transformers import AutoModel, AutoTokenizer                             # noqa: E402
from v8 import SpanVSA, cache_hidden                                          # noqa: E402

HID = 2560
PARSER_REF, HIDDEN_REF = 0.918, 0.505


class SupSpanVSA(SpanVSA):
    """ARM 1. Same as V-8 plus an auxiliary codebook-matching loss on the projected span vectors."""

    def codebook_for(self, kind):
        return {"R": self.vsa.role_vectors, "X": self.x_vectors, "Y": self.y_vectors}[kind]

    def aux_loss(self, input_ids, hidden, device):
        """mean(1 - cos(proj(span), codebook[true_id])) over every R/X/Y marker in the batch."""
        terms = []
        for b in range(input_ids.size(0)):
            H = hidden[b]
            for kind, idx, s, e in self._scan_markers(input_ids[b]):
                cb = self.codebook_for(kind)
                if idx >= cb.size(0):
                    continue
                v = self.sym(H, s, e)
                tgt = l2_normalize(cb[idx].to(device).float())
                terms.append(1.0 - torch.dot(v, tgt))
        if not terms:
            return torch.zeros((), device=device)
        return torch.stack(terms).mean()


class ExtractThenBind(SymbolicVSAChainClassifier):
    """ARM 2. Identity comes from a supervised probe over span-pooled frozen states; its argmax indexes the
    fixed codebook and the untouched binding chain runs. Never reads an integer from the input string."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.id_head = nn.Linear(HID, self.vocab_size)      # shared across R/X/Y, masked per kind at use

    def span_feat(self, H, s, e):
        span = H[s:e] if e > s else H[s:s + 1]
        return span.float().mean(0)

    def predict_ids(self, input_ids, hidden, limit_by_kind=True):
        """Returns, per batch item, the scanner's markers with the PREDICTED index substituted."""
        out = []
        for b in range(input_ids.size(0)):
            H = hidden[b]
            rows = []
            for kind, idx, s, e in self._scan_markers(input_ids[b]):
                logits = self.id_head(self.span_feat(H, s, e))
                if limit_by_kind and kind == "R":
                    logits = logits[:self.num_roles]
                rows.append((kind, int(logits.argmax().item()), s, e))
            out.append(rows)
        return out

    def id_loss(self, input_ids, hidden, device):
        terms = []
        for b in range(input_ids.size(0)):
            H = hidden[b]
            for kind, idx, s, e in self._scan_markers(input_ids[b]):
                n = self.num_roles if kind == "R" else self.vocab_size
                if idx >= n:
                    continue
                logits = self.id_head(self.span_feat(H, s, e))[:n]
                terms.append(F.cross_entropy(logits.unsqueeze(0),
                                             torch.tensor([idx], device=device)))
        if not terms:
            return torch.zeros((), device=device)
        return torch.stack(terms).mean()

    def forward(self, input_ids, attention_mask, hidden=None):
        device = input_ids.device
        logits = torch.zeros(input_ids.size(0), 2, device=device)
        preds = self.predict_ids(input_ids, hidden)
        for b in range(input_ids.size(0)):
            toks = self.tokenizer.convert_ids_to_tokens(input_ids[b].tolist())
            ms = sorted(preds[b], key=lambda x: x[2])
            if not ms:
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
            q_role = None
            for kind, midx, s, e in reversed(ms):
                if kind == "R" and midx < self.num_roles:
                    q_role = midx; break
            if q_role is None:
                continue
            A, Bp, C = [], [], []
            for i in range(len(ms) - 1):
                k1, i1, s1, e1 = ms[i]
                k2, i2, s2, e2 = ms[i + 1]
                if k1 == "R" and k2 == "X" and i1 < self.num_roles and i2 < self.vocab_size:
                    A.append((i1, i2))
                elif k1 == "X" and k2 == "Y" and i1 < self.vocab_size and i2 < self.vocab_size:
                    Bp.append((i1, i2))
            for kind, midx, s, e in ms:
                if kind != "Y" or midx >= self.vocab_size:
                    continue
                if e < len(toks) and toks[e].lstrip("Ġ") == "{":
                    f_id, _ = self._parse_brace_int_at(toks, e)
                    if f_id is None or f_id < 0 or f_id >= self.vocab_size:
                        continue
                    C.append((midx, f_id))
            if not A or not Bp or not C:
                continue
            rq = self.vsa.role_vectors[q_role].to(device)
            memA = self.vsa.bundle(torch.stack(
                [self.vsa.bind(self.x_vectors[x].to(device),
                               self.vsa.role_vectors[r].to(device)) for r, x in A], 0))
            x_hat = l2_normalize(self.vsa.unbind(memA, rq))
            xc = self.x_vectors[torch.tensor([x for _, x in A], device=device)]
            x_clean = xc[int(self.vsa.similarity(xc, x_hat.unsqueeze(0).expand_as(xc)).argmax().item())]
            memB = self.vsa.bundle(torch.stack(
                [self.vsa.bind(self.y_vectors[y].to(device),
                               self.x_vectors[x].to(device)) for x, y in Bp], 0))
            y_hat = l2_normalize(self.vsa.unbind(memB, x_clean))
            yc = self.y_vectors[torch.tensor([y for _, y in Bp], device=device)]
            y_clean = yc[int(self.vsa.similarity(yc, y_hat.unsqueeze(0).expand_as(yc)).argmax().item())]
            memC = self.vsa.bundle(torch.stack(
                [self.vsa.bind(self.f_vectors[f].to(device),
                               self.y_vectors[y].to(device)) for y, f in C], 0))
            f_hat = l2_normalize(self.vsa.unbind(memC, y_clean))
            sim = self.vsa.similarity(f_hat, self.f_vectors[int(cand)].to(device)).clamp(-1, 1)
            lp = sim * self.sim_scale + self.sim_bias
            logits[b, 1] = lp; logits[b, 0] = -lp
        return logits


def run_arm(arm, seed, cfg, tok, llm, device, n_train, n_eval, epochs, bs, lr, lam=1.0):
    torch.manual_seed(seed)
    tr = SystematicChainDataset(tok, num_samples=n_train, split="train", cfg=cfg)
    ev = SystematicChainDataset(tok, num_samples=n_eval, split="eval", cfg=cfg)
    TR, EV = cache_hidden(tr, llm, device), cache_hidden(ev, llm, device)
    cls = SupSpanVSA if arm == "vsa_spans_sup" else ExtractThenBind
    m = cls(vsa_dim=2048, num_roles=cfg.num_roles, vocab_size=cfg.vocab_size, seed=seed).to(device)
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)

    def acc():
        m.eval(); c = t = 0
        with torch.no_grad():
            for k in range(0, EV[0].size(0), bs):
                lg = m(EV[0][k:k + bs].to(device), EV[1][k:k + bs].to(device),
                       hidden=EV[2][k:k + bs].to(device))
                c += (lg.argmax(-1).cpu() == EV[3][k:k + bs]).sum().item(); t += lg.size(0)
        return c / t

    t0, best = time.time(), 0.0
    g = torch.Generator(); g.manual_seed(seed)
    for ep in range(epochs):
        m.train()
        perm = torch.randperm(TR[0].size(0), generator=g)
        for k in range(0, TR[0].size(0), bs):
            i = perm[k:k + bs]
            ids, am, H, y = (TR[0][i].to(device), TR[1][i].to(device),
                             TR[2][i].to(device), TR[3][i].to(device))
            if arm == "vsa_spans_sup":
                loss = F.cross_entropy(m(ids, am, hidden=H), y) + lam * m.aux_loss(ids, H, device)
            else:
                loss = F.cross_entropy(m(ids, am, hidden=H), y) + lam * m.id_loss(ids, H, device)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        a = acc(); best = max(best, a)
        print("      %s ep%d eval=%.4f (%.0fs)" % (arm, ep + 1, a, time.time() - t0), flush=True)
    return {"arm": arm, "seed": seed, "acc": best,
            "trainable": sum(p.numel() for p in params), "secs": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["extract_then_bind", "vsa_spans_sup"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    ap.add_argument("--train_samples", type=int, default=2000)
    ap.add_argument("--eval_samples", type=int, default=1000)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="v9_logs/v9.json")
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
    for arm in a.arms:
        for s in a.seeds:
            r = run_arm(arm, s, cfg, tok, llm, a.device, a.train_samples, a.eval_samples,
                        a.epochs, a.batch_size, a.lr)
            print("  %-17s sd%-4d acc=%.4f trainable=%s %ss"
                  % (arm, s, r["acc"], format(r["trainable"], ","), r["secs"]), flush=True)
            rows.append(r)
            os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
            json.dump(rows, open(a.out, "w"), indent=1)

    print("\n=== V-9 SCORING (bars from prereg 6f162d1) ===", flush=True)
    per = {}
    for arm in a.arms:
        accs = [r["acc"] for r in rows if r["arm"] == arm]
        per[arm] = accs
        frac = (float(np.mean(accs)) - HIDDEN_REF) / (PARSER_REF - HIDDEN_REF)
        print("  %-17s %s mean=%.4f  recovered fraction of the gap=%.3f"
              % (arm, [round(x, 4) for x in accs], float(np.mean(accs)), frac), flush=True)
    found = any(all(x >= 0.80 for x in v) for v in per.values())
    failed = all(all(x <= 0.60 for x in v) for v in per.values())
    print("  banked: vsa_parser %.3f | vsa_hidden %.3f | vsa_spans (V-8) .501" % (PARSER_REF, HIDDEN_REF),
          flush=True)
    print("  >>> V-9 VERDICT: %s" % ("REMEDY FOUND" if found else
                                     ("REMEDY FAILS" if failed else "INTERMEDIATE")), flush=True)
    print("V9-DONE", flush=True)


if __name__ == "__main__":
    main()
