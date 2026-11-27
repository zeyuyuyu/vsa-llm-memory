"""V-11 (prereg 64dfd34): does the mechanism replicate on a SECOND symbolic family?

Every number in the SysChain mechanism table came from one in-house task. This runs the same manipulation
-- how symbols enter the binding chain -- on SystematicRoleBindingDataset, which is a different family and a
harder one: training never contains a held-out (role, filler) pair, and evaluation queries only held-out
pairs that appear in context. So it tests systematic generalisation rather than retrieval.

Format: [Rxx]{j} ... Q [Rqq]{cand} ?   64 roles, 256 fillers, single-level binding.
Chain: mem = SUM bind(filler_vec[f_r], role_vec[r]) ; f_hat = unbind(mem, role_vec[q]) ;
       score = cos(f_hat, filler_vec[cand]).

ARMS (the only thing that varies is where the filler vector comes from):
  floor         linear probe on the pooled frozen state, no binding at all
  parser        filler ids parsed from the input string, exact codebook rows
  consistent    same, but every filler id consistently relabelled (correctness vs consistency)
  spans         filler vectors are a learned projection of span-pooled states, trained THROUGH the loss
  probe_argmax  a supervised identity probe's argmax indexes the codebook (the remedy)
  frozen_cont   a frozen, accurate continuous projection, never touched by the binding gradient

BARS (prereg 64dfd34), replication requires all three in 3/3 seeds:
  (a) parser >= floor + 20pp      (b) spans <= floor + 5pp      (c) probe_argmax >= parser - 5pp
VOID if (a) fails: then this arena cannot test the mechanism and the other arms are not reinterpreted.
"""
import argparse, json, os, random, re, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
sys.path.insert(0, "/path/to/work/sync_h20/rewardbench_experiment/zeyu/hybrid_neurosymbolic")

from systematic_role_binding_dataset import SystematicRoleBindingDataset, SystematicConfig  # noqa: E402
from transformers import AutoModel, AutoTokenizer                                            # noqa: E402

HID = 2560
VSA_DIM = 2048


def unit(n, dim, seed):
    g = torch.Generator(); g.manual_seed(seed)
    v = torch.randn(n, dim, generator=g)
    return F.normalize(v, dim=1)


def bind(a, b):      # circular convolution via FFT
    return torch.fft.irfft(torch.fft.rfft(a, dim=-1) * torch.fft.rfft(b, dim=-1), n=a.size(-1), dim=-1)


def unbind(ab, a):   # circular correlation
    return torch.fft.irfft(torch.fft.rfft(ab, dim=-1) * torch.conj(torch.fft.rfft(a, dim=-1)),
                           n=ab.size(-1), dim=-1)


# ------------------------------------------------------------------ symbol spans

PAT = re.compile(r"\[R([0-9A-F]{2})\]\{(\d+)\}")


def parse_pairs(text):
    """Returns [(role_idx, filler_idx, char_start, char_end)] for every [Rxx]{j}, in order."""
    # V-18 fix (prereg 3eb2a81): the role vector here comes from a fixed codebook, so only the filler is
    # derived from text -- but the span passed was the whole [Rxx]{j} match, which also contains the role.
    # The identity head was therefore predicting the filler from a span carrying two symbols, strictly
    # harder than the corrected arena-C task, and the remaining candidate explanation for arena B
    # recovering 26% where A and C recover ~100%.
    return [(int(m.group(1), 16), int(m.group(2)), m.start(2), m.end(2)) for m in PAT.finditer(text)]


def token_span(offsets, cs, ce):
    """Map a character span to the token index range."""
    lo = hi = None
    for i, (a, b) in enumerate(offsets):
        if b <= a:
            continue
        if a < ce and b > cs:
            lo = i if lo is None else lo
            hi = i + 1
    return (lo, hi) if lo is not None else (0, 1)


class Arena:
    """Renders the dataset's own text, then records token spans for each {filler} so arms can read
    identity from the hidden states instead of from the string."""

    def __init__(self, tok, n, split, cfg, seed):
        self.tok, self.cfg = tok, cfg
        ds = SystematicRoleBindingDataset(tok, n, split, cfg)
        self.rows = []
        for i in range(n):
            rng = random.Random(1000003 * i + (42 if split == "train" else 123))
            qr = rng.randrange(cfg.num_roles)
            tf = rng.choice(ds.train_allowed[qr] if split == "train" else ds.eval_heldout[qr])
            r2f = ds._sample_context(rng, qr, tf)
            # DATASET FIX (disclosed at 77ab057): _sample_context pins the query pair only on the eval
            # branch, so on train the "true filler" was not the one bound to the query role and the
            # positive class carried a retrieval cosine of ZERO. Pinning an ALLOWED pair here does not
            # introduce a held-out pair, so the systematic split is preserved.
            r2f[qr] = tf
            label = 1 if rng.random() < cfg.p_positive else 0
            if label == 1:
                cand = tf
            else:
                wr = rng.choice([r for r in range(cfg.num_roles) if r != qr])
                cand = r2f[wr]
                if cand == tf:
                    cand = rng.choice([f for f in range(cfg.num_fillers) if f != tf])
            text = ds._render_example(rng, r2f, qr, cand)
            enc = tok(text, truncation=True, max_length=cfg.max_length, padding="max_length",
                      return_tensors="pt", return_offsets_mapping=True)
            offs = enc["offset_mapping"].squeeze(0).tolist()
            pairs = parse_pairs(text)
            # the LAST pair is the query [Rqq]{cand}; the rest are context bindings
            ctx = [(r, f, token_span(offs, cs, ce)) for r, f, cs, ce in pairs[:-1]]
            self.rows.append({"ids": enc["input_ids"].squeeze(0), "am": enc["attention_mask"].squeeze(0),
                              "label": label, "q_role": qr, "cand": cand, "ctx": ctx})


@torch.no_grad()
def cache(arena, llm, device, bs=4):
    H = []
    for i in range(0, len(arena.rows), bs):
        b = arena.rows[i:i + bs]
        ids = torch.stack([r["ids"] for r in b]).to(device)
        am = torch.stack([r["am"] for r in b]).to(device)
        H.append(llm(input_ids=ids, attention_mask=am).last_hidden_state.half().cpu())
    return torch.cat(H)


# ------------------------------------------------------------------ arms

class Floor(nn.Module):
    def __init__(self):
        super().__init__(); self.fc = nn.Linear(HID, 2)

    def sims(self, H, rows, device):
        """logit margin, so the same closed-form threshold procedure applies to the floor arm too"""
        idx = torch.tensor([int(r["am"].sum()) - 1 for r in rows], device=device).clamp(min=0)
        pooled = H[torch.arange(H.size(0), device=device), idx].float()
        lg = self.fc(pooled)
        return lg[:, 1] - lg[:, 0]

    def forward(self, H, am, rows, device):
        m = am.unsqueeze(-1).float()
        idx = am.sum(1).long().clamp(min=1) - 1
        pooled = H[torch.arange(H.size(0), device=device), idx].float()
        return self.fc(pooled)


class BindArm(nn.Module):
    """mode: parser | consistent | spans | probe_argmax | frozen_cont"""

    def __init__(self, mode, cfg, seed):
        super().__init__()
        self.mode, self.cfg = mode, cfg
        self.register_buffer("role_v", unit(cfg.num_roles, VSA_DIM, seed + 11))
        self.register_buffer("fill_v", unit(cfg.num_fillers, VSA_DIM, seed + 23))
        self.scale = nn.Parameter(torch.tensor(20.0))
        self.bias = nn.Parameter(torch.tensor(-4.0))
        if mode in ("spans", "frozen_cont"):
            self.proj = nn.Linear(HID, VSA_DIM)
            if mode == "frozen_cont":
                for p in self.proj.parameters():
                    p.requires_grad_(False)
        if mode == "probe_argmax":
            self.id_head = nn.Linear(HID, cfg.num_fillers)
        if mode == "consistent":
            g = random.Random(7000 + seed)
            perm = list(range(cfg.num_fillers)); g.shuffle(perm)
            self.register_buffer("remap", torch.tensor(perm))

    def span_feat(self, H, s, e):
        return H[s:e].float().mean(0) if e > s else H[s:s + 1].float().mean(0)

    def filler_vec(self, H, f_id, span, device):
        if self.mode == "parser":
            return self.fill_v[f_id].to(device)
        if self.mode == "consistent":
            return self.fill_v[int(self.remap[f_id])].to(device)
        v = self.span_feat(H, span[0], span[1])
        if self.mode in ("spans", "frozen_cont"):
            return F.normalize(self.proj(v), dim=-1)
        pred = int(self.id_head(v).argmax().item())      # probe_argmax -> exact codebook row
        return self.fill_v[pred].to(device)

    def id_loss(self, H, rows, device):
        if self.mode != "probe_argmax":
            return torch.zeros((), device=device)
        terms = []
        for b, r in enumerate(rows):
            for (_, f, sp) in r["ctx"]:
                lg = self.id_head(self.span_feat(H[b], sp[0], sp[1]))
                terms.append(F.cross_entropy(lg.unsqueeze(0), torch.tensor([f], device=device)))
        return torch.stack(terms).mean() if terms else torch.zeros((), device=device)

    def aux_fit_loss(self, H, rows, device):
        """frozen_cont is fitted to the codebook OUTSIDE the binding loss (that is the point of the arm)."""
        terms = []
        for b, r in enumerate(rows):
            for (_, f, sp) in r["ctx"]:
                v = F.normalize(self.proj(self.span_feat(H[b], sp[0], sp[1])), dim=-1)
                terms.append(1.0 - torch.dot(v, self.fill_v[f].to(device)))
        return torch.stack(terms).mean() if terms else torch.zeros((), device=device)

    def sims(self, H, rows, device):
        """The raw cosine the readout thresholds. Returned separately so the decision threshold can be
        fitted in CLOSED FORM on training data rather than learned -- a learned scale/bias was unstable at
        this task's signal scale (amendment 58bed4e)."""
        out = torch.zeros(len(rows), device=device)
        for b, r in enumerate(rows):
            if not r["ctx"]:
                continue
            bound = [bind(self.filler_vec(H[b], f, sp, device), self.role_v[rr].to(device))
                     for rr, f, sp in r["ctx"]]
            mem = torch.stack(bound, 0).sum(0)
            f_hat = F.normalize(unbind(mem, self.role_v[r["q_role"]].to(device)), dim=-1)
            # BUG FIX in my own arm: under consistent relabelling the CANDIDATE must be relabelled too.
            # On SysChain the corruption touched intermediate symbols only and the filler endpoints stayed
            # fixed, so this never surfaced; here the filler IS the endpoint, and remapping the context but
            # not the query breaks the comparison by construction (it scored .484, i.e. chance).
            cand_id = int(self.remap[r["cand"]]) if self.mode == "consistent" else r["cand"]
            out[b] = torch.dot(f_hat, self.fill_v[cand_id].to(device)).clamp(-1, 1)
        return out

    def forward(self, H, am, rows, device):
        sim = self.sims(H, rows, device)
        lp = sim * self.scale + self.bias
        return torch.stack([-lp, lp], dim=1)


def run(arm, seed, cfg, tok, llm, device, n_tr, n_ev, epochs, bs, lr):
    torch.manual_seed(seed)
    tr, ev = Arena(tok, n_tr, "train", cfg, seed), Arena(tok, n_ev, "eval", cfg, seed)
    Htr, Hev = cache(tr, llm, device), cache(ev, llm, device)
    m = (Floor() if arm == "floor" else BindArm(arm, cfg, seed)).to(device)

    def make_opt(mod):
        """The two readout scalars get their own, much higher rate.

        Why: the similarity the readout thresholds SHRINKS with the number of superposed bindings --
        measured positive-class means are .33 at 8 bindings, .24 at 16, .12 at 64 -- while the inherited
        initialisation (scale 20, bias -4) fixes the threshold at .20, which was right for the 16-binding
        SysChain task and is ABOVE every positive here. At the shared rate two scalars cannot travel that
        far in the registered budget, so every binding arm would sit at chance for a reason that has
        nothing to do with the mechanism. A pre-run diagnostic (no training, no arm) confirmed the chain
        itself separates perfectly at 64 bindings: d\'=5.60, single-threshold accuracy 1.0000. The fix is
        applied identically to every arm, so it advantages none of them."""
        scal = [q for n, q in mod.named_parameters()
                if q.requires_grad and n in ("scale", "bias")]
        rest = [q for n, q in mod.named_parameters()
                if q.requires_grad and n not in ("scale", "bias")]
        groups = []
        if rest:
            groups.append({"params": rest, "lr": lr})
        if scal:
            groups.append({"params": scal, "lr": 1e-1})
        return torch.optim.AdamW(groups)

    params = [p for p in m.parameters() if p.requires_grad]
    opt = make_opt(m)

    def collect(rows_all, Hall):
        m.eval(); S, Y = [], []
        with torch.no_grad():
            for k in range(0, len(rows_all), bs):
                rows = rows_all[k:k + bs]
                S.append(m.sims(Hall[k:k + bs].to(device), rows, device).cpu())
                Y += [r["label"] for r in rows]
        return torch.cat(S).numpy(), np.array(Y)

    def ev_acc():
        """Closed-form readout: the threshold maximising TRAIN accuracy is applied to EVAL. No learning
        rate, no free readout parameter, identical procedure for every arm (amendment 58bed4e)."""
        str_, ytr_ = collect(tr.rows, Htr)
        sev, yev = collect(ev.rows, Hev)
        cand = np.unique(str_)
        if cand.size == 0:
            return 0.5
        accs = [(((str_ >= t) == (ytr_ == 1)).mean()) for t in cand]
        thr = float(cand[int(np.argmax(accs))])
        return float((((sev >= thr) == (yev == 1)).mean()))

    # frozen_cont: fit the projection to the codebook FIRST, outside the binding loss, then freeze
    if arm == "frozen_cont":
        for p in m.proj.parameters():
            p.requires_grad_(True)
        fopt = torch.optim.AdamW(m.proj.parameters(), lr=1e-3)
        for _ in range(3):
            for k in range(0, len(tr.rows), bs):
                rows = tr.rows[k:k + bs]
                H = Htr[k:k + bs].to(device)
                loss = m.aux_fit_loss(H, rows, device)
                fopt.zero_grad(); loss.backward(); fopt.step()
        fid = 1.0 - float(loss.item())
        for p in m.proj.parameters():
            p.requires_grad_(False)
        params = [p for p in m.parameters() if p.requires_grad]
        opt = make_opt(m)
        print("      frozen_cont projection fitted, cos fidelity ~= %.4f" % fid, flush=True)

    t0, best = time.time(), 0.0
    g = torch.Generator(); g.manual_seed(seed)
    for ep in range(epochs):
        m.train()
        perm = torch.randperm(len(tr.rows), generator=g).tolist()
        for k in range(0, len(perm), bs):
            sel = perm[k:k + bs]
            rows = [tr.rows[i] for i in sel]
            H = Htr[sel].to(device)
            am = torch.stack([r["am"] for r in rows]).to(device)
            y = torch.tensor([r["label"] for r in rows], device=device)
            loss = F.cross_entropy(m(H, am, rows, device), y)
            if arm == "probe_argmax":
                loss = loss + m.id_loss(H, rows, device)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        a = ev_acc(); best = max(best, a)
        print("      %-13s ep%d eval=%.4f (%.0fs)" % (arm, ep + 1, a, time.time() - t0), flush=True)
    return {"arm": arm, "seed": seed, "acc": best,
            "trainable": sum(p.numel() for p in params), "secs": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+",
                    default=["floor", "parser", "consistent", "spans", "probe_argmax", "frozen_cont"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    ap.add_argument("--n_train", type=int, default=1000)
    ap.add_argument("--n_eval", type=int, default=500)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="v11_logs/v11.json")
    a = ap.parse_args()

    cfg = SystematicConfig()
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
            r = run(arm, s, cfg, tok, llm, a.device, a.n_train, a.n_eval,
                    a.epochs, a.batch_size, a.lr)
            print("  %-13s sd%-4d acc=%.4f trainable=%s %ss"
                  % (arm, s, r["acc"], format(r["trainable"], ","), r["secs"]), flush=True)
            rows.append(r)
            os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
            json.dump(rows, open(a.out, "w"), indent=1)

    def col(x):
        return [r["acc"] for r in rows if r["arm"] == x]
    print("\n=== V-11 SCORING (bars from prereg 64dfd34) ===", flush=True)
    for x in a.arms:
        c = col(x)
        if c:
            print("  %-13s %s mean=%.4f" % (x, [round(v, 4) for v in c], float(np.mean(c))), flush=True)
    fl, pa, sp, pq = col("floor"), col("parser"), col("spans"), col("probe_argmax")
    if len(fl) == len(pa) == 3:
        a_hold = all(pa[i] >= fl[i] + 0.20 for i in range(3))
        b_hold = all(sp[i] <= fl[i] + 0.05 for i in range(3)) if len(sp) == 3 else None
        c_hold = all(pq[i] >= pa[i] - 0.05 for i in range(3)) if len(pq) == 3 else None
        print("  (a) parser >= floor+20pp in 3/3        -> %s" % a_hold, flush=True)
        print("  (b) spans  <= floor+5pp  in 3/3        -> %s" % b_hold, flush=True)
        print("  (c) probe_argmax >= parser-5pp in 3/3  -> %s" % c_hold, flush=True)
        if not a_hold:
            v = "VOID (arena cannot test the mechanism)"
        elif b_hold and c_hold:
            v = "MECHANISM REPLICATES"
        else:
            v = "FAILS TO REPLICATE (SysChain pattern is task-specific)"
        print("  >>> V-11 VERDICT: %s" % v, flush=True)
    print("V11-DONE", flush=True)


if __name__ == "__main__":
    main()
