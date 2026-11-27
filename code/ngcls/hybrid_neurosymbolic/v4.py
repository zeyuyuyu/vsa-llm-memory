"""V-4 (prereg 55b38bc, amendment recorded before this file was run): how good must symbol extraction be
before HRR binding pays?

V-1 gave two points: symbol identity read from frozen LLM hidden states through a 27.9M-parameter learned
projection scores 0.505 (the 5.1K probe floor); the same binding algebra with identity handed over by a
hand-written scan of the literal markers, and two trainable scalars, scores 0.918. So the algebra works
and the representation is what fails. V-4 measures HOW MUCH fidelity the algebra needs, and then whether
the LLM's states supply it.

THE CHOKE POINT. Everything downstream of SymbolicVSAChainClassifier._scan_markers consumes its
(kind, idx, start, end) tuples, so overriding that one method is the whole manipulation and nothing else
in the arm changes -- same role vectors, same binding, same two trainable scalars, same data, same budget
as V-1 (2000/1000, 2 epochs, seeds 42/123/456).

TWO CORRUPTION MODES. The prereg specified one; the second is an amendment made before running, because
the contrast between them separates two things the single mode conflates.
  perocc      (registered, PRIMARY) each marker occurrence is independently replaced, with probability
              epsilon, by a uniformly drawn WRONG index of the same kind. A symbol appearing twice can
              receive two different wrong values, so co-reference breaks. This is how a real extractor
              fails: per token, not per symbol.
  consistent  (CONTROL) each distinct symbol is relabelled once and that relabelling is reused for every
              occurrence. Identity is wrong but co-reference is preserved. If accuracy survives here and
              not under perocc, the module needs CONSISTENCY rather than CORRECTNESS -- a sharper claim
              than either mode gives alone.

SCOPE NOTE, stated because it bounds the headline number: the filler ids are parsed from the trailing
{...} by a separate routine, not through _scan_markers, so corruption applies to ROLE and INTERMEDIATE
VARIABLE identity, not to the candidate answer. Corrupting the candidate would inject label noise rather
than representational error, which is not the quantity of interest.
"""
import argparse, json, os, random, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v1_run import build, train_one                                    # noqa: E402
from systematic_chain_dataset import SystematicChainDataset, ChainConfig  # noqa: E402
from explicit_chain_models import SymbolicVSAChainClassifier            # noqa: E402
from transformers import AutoTokenizer                                  # noqa: E402

EPSILONS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
PROBE_FLOOR = 0.509          # V-1's frozen_llm arm, the level epsilon* is defined against


class CorruptedParserVSA(SymbolicVSAChainClassifier):
    def __init__(self, epsilon=0.0, mode="perocc", corrupt_seed=0, **kw):
        super().__init__(**kw)
        self.epsilon, self.mode = epsilon, mode
        self._rng = random.Random(90_000 + corrupt_seed)

    def _wrong(self, kind, idx):
        n = self.num_roles if kind == "R" else self.vocab_size
        new = self._rng.randrange(n - 1)
        return new + 1 if new >= idx else new       # uniform over the wrong values only

    def _scan_markers(self, ids_1d):
        markers = super()._scan_markers(ids_1d)
        if self.epsilon <= 0.0 or not markers:
            return markers
        if self.mode == "consistent":
            remap, out = {}, []
            for kind, idx, s, e in markers:
                key = (kind, idx)
                if key not in remap:
                    remap[key] = self._wrong(kind, idx) if self._rng.random() < self.epsilon else idx
                out.append((kind, remap[key], s, e))
            return out
        return [(k, self._wrong(k, i) if self._rng.random() < self.epsilon else i, s, e)
                for k, i, s, e in markers]


# ------------------------------------------------------------------ P3: can the LLM supply identity?

class MarkerProbe(nn.Module):
    """Linear read-out over the frozen hidden states pooled across each marker's token span, predicting
    that marker's symbol index. A_extract is its top-1 accuracy. This is the most favourable form of the
    question: the probe is TOLD where the markers are, so it only has to recover identity, not location."""
    def __init__(self, hid, n_out):
        super().__init__()
        self.fc = nn.Linear(hid, n_out)

    def forward(self, x):
        return self.fc(x)


@torch.no_grad()
def collect_marker_features(llm, model, loader, device, kinds=("X", "Y"), max_batches=None):
    feats, labels = [], []
    for bi, b in enumerate(loader):
        if max_batches and bi >= max_batches:
            break
        ids = b["input_ids"].to(device)
        am = b["attention_mask"].to(device)
        h = llm(input_ids=ids, attention_mask=am).last_hidden_state.float()
        for j in range(ids.size(0)):
            for kind, idx, s, e in model._scan_markers(ids[j]):
                if kind in kinds and e > s:
                    feats.append(h[j, s:e].mean(0).cpu())
                    labels.append(idx)
    if not feats:
        return None, None
    return torch.stack(feats), torch.tensor(labels)


def run_extract(cfg, tok, device, seed, train_n, eval_n, bs, epochs=40):
    from transformers import AutoModel
    torch.manual_seed(seed)
    llm = AutoModel.from_pretrained("microsoft/phi-2", trust_remote_code=True,
                                    torch_dtype=torch.float16).to(device).eval()
    for p in llm.parameters():
        p.requires_grad_(False)
    ref = CorruptedParserVSA(epsilon=0.0, vsa_dim=2048, num_roles=cfg.num_roles,
                             vocab_size=cfg.vocab_size, seed=seed)
    tr = SystematicChainDataset(tok, num_samples=train_n, split="train", cfg=cfg)
    ev = SystematicChainDataset(tok, num_samples=eval_n, split="eval", cfg=cfg)
    Xtr, ytr = collect_marker_features(llm, ref, DataLoader(tr, batch_size=bs), device)
    Xte, yte = collect_marker_features(llm, ref, DataLoader(ev, batch_size=bs), device)
    del llm; torch.cuda.empty_cache()
    if Xtr is None:
        return {"seed": seed, "A_extract": float("nan"), "n_train": 0}
    probe = MarkerProbe(Xtr.size(1), cfg.vocab_size).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-2)
    Xtr, ytr = Xtr.to(device), ytr.to(device)
    Xte, yte = Xte.to(device), yte.to(device)
    g = torch.Generator(); g.manual_seed(seed)
    for _ in range(epochs):
        probe.train()
        perm = torch.randperm(Xtr.size(0), generator=g).to(device)
        for k in range(0, Xtr.size(0), 256):
            i = perm[k:k + 256]
            loss = F.cross_entropy(probe(Xtr[i]), ytr[i])
            opt.zero_grad(); loss.backward(); opt.step()
    probe.eval()
    with torch.no_grad():
        acc = float((probe(Xte).argmax(-1) == yte).float().mean().item())
    return {"seed": seed, "A_extract": acc, "n_train": int(Xtr.size(0)),
            "n_test": int(Xte.size(0)), "chance": 1.0 / cfg.vocab_size}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="sweep", choices=["sweep", "extract"])
    ap.add_argument("--corrupt", default="perocc", choices=["perocc", "consistent"])
    ap.add_argument("--seeds", default="42,123,456")
    ap.add_argument("--train_samples", type=int, default=2000)
    ap.add_argument("--eval_samples", type=int, default=1000)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    cfg = ChainConfig(num_roles=64, vocab_size=256, num_pairs=16, max_length=512, p_positive=0.5)
    tok = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    seeds = [int(x) for x in a.seeds.split(",")]
    rows = []

    if a.mode == "extract":
        for s in seeds:
            r = run_extract(cfg, tok, a.device, s, a.train_samples, a.eval_samples, a.batch_size)
            print("  A_extract sd%-4d = %.4f  (chance %.4f, n_train=%d n_test=%d)"
                  % (s, r["A_extract"], r.get("chance", 0), r.get("n_train", 0), r.get("n_test", 0)),
                  flush=True)
            rows.append(r)
            if a.out:
                os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
                json.dump(rows, open(a.out, "w"), indent=1)
        accs = [r["A_extract"] for r in rows]
        print("\n  A_extract mean = %.4f over %d seeds" % (float(np.mean(accs)), len(accs)), flush=True)
        print("V4-EXTRACT-DONE", flush=True)
        return

    for eps in EPSILONS:
        for s in seeds:
            torch.manual_seed(s)
            tr = SystematicChainDataset(tok, num_samples=a.train_samples, split="train", cfg=cfg)
            ev = SystematicChainDataset(tok, num_samples=a.eval_samples, split="eval", cfg=cfg)
            m = CorruptedParserVSA(epsilon=eps, mode=a.corrupt, corrupt_seed=s, vsa_dim=2048,
                                   num_roles=cfg.num_roles, vocab_size=cfg.vocab_size, seed=s)
            n_train = sum(p.numel() for p in m.parameters() if p.requires_grad)
            t0 = time.time()
            acc = train_one(m, DataLoader(tr, batch_size=a.batch_size, shuffle=True),
                            DataLoader(ev, batch_size=a.batch_size, shuffle=False),
                            epochs=a.epochs, lr=3e-4, device=a.device)
            r = {"mode": a.corrupt, "epsilon": eps, "seed": s, "acc": float(acc),
                 "trainable": int(n_train), "secs": round(time.time() - t0, 1)}
            print("  %-10s eps=%.2f sd%-4d acc=%.4f trainable=%d %ss"
                  % (a.corrupt, eps, s, r["acc"], n_train, r["secs"]), flush=True)
            rows.append(r)
            del m; torch.cuda.empty_cache()
            if a.out:
                os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
                json.dump(rows, open(a.out, "w"), indent=1)

    # scorer, per the prereg, written before the run
    means = {e: float(np.mean([r["acc"] for r in rows if r["epsilon"] == e])) for e in EPSILONS}
    print("\n  mean accuracy by epsilon: %s"
          % {e: round(v, 4) for e, v in means.items()}, flush=True)
    seq = [means[e] for e in EPSILONS]
    inversions = [(EPSILONS[i], round(seq[i + 1] - seq[i], 4))
                  for i in range(len(seq) - 1) if seq[i + 1] > seq[i] + 0.01]
    print("  P1 monotone (allowing one inversion <= 0.01): %s   inversions=%s"
          % (len(inversions) == 0, inversions), flush=True)
    anchors_ok = seq[0] > 0.85 and seq[-1] < 0.60
    print("  anchors: eps=0 -> %.4f (need >0.85), eps=1 -> %.4f (need <0.60) : %s"
          % (seq[0], seq[-1], anchors_ok), flush=True)
    star = None
    for i in range(len(seq) - 1):
        if seq[i] >= PROBE_FLOOR >= seq[i + 1] and seq[i] > seq[i + 1]:
            f = (seq[i] - PROBE_FLOOR) / (seq[i] - seq[i + 1])
            star = EPSILONS[i] + f * (EPSILONS[i + 1] - EPSILONS[i])
            break
    print("  P2 epsilon* (crossing the %.3f probe floor) = %s ; required symbol fidelity 1-eps* = %s"
          % (PROBE_FLOOR, ("%.4f" % star) if star is not None else "not bracketed",
             ("%.4f" % (1 - star)) if star is not None else "n/a"), flush=True)
    print("V4-SWEEP-DONE", flush=True)


if __name__ == "__main__":
    main()
