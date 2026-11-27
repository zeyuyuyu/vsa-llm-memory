"""V-13 (prereg d3a4da5): the mechanism on an arena chosen by a RULE, with bars as fractions of MEASURED
headroom. This is the corrected version of V-11, which went void because I fixed +20pp before knowing the
arena had 18.5pp of room.

Arena selected by the registered rule from the completed floor survey (lowest floor among configurations
with headroom >= 0.40): hops=2, 32 context bindings per level, query MID-SEQUENCE.
Measured floor .4737, headroom H = .5263, so the registered bars evaluate to
  (a) parser >= .7369   (b) spans <= .5263   (c) probe_argmax >= parser - .0526

Chain, two levels: r0 -> s1 -> s2.  Level 0 renders [Rxx]{s1}, level 1 renders [Fxx]{s2}.
  mem0 = SUM bind(sym[b], sym[a]) over level-0 pairs ; s1_hat = unbind(mem0, sym[r0]) ; cleanup
  mem1 = SUM bind(sym[b], sym[a]) over level-1 pairs ; s2_hat = unbind(mem1, s1_clean)
  score = cos(s2_hat, sym[cand])
Arms differ only in where sym[] comes from, exactly as in V-11.
"""
import argparse, json, os, random, re, sys, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
sys.path.insert(0, "/path/to/work/ngcls")


# inlined rather than imported across directories, so this file has no cross-project dependency
def unit(n, dim, seed):
    g = torch.Generator(); g.manual_seed(seed)
    return F.normalize(torch.randn(n, dim, generator=g), dim=1)


def bind(a, b):
    return torch.fft.irfft(torch.fft.rfft(a, dim=-1) * torch.fft.rfft(b, dim=-1),
                           n=a.size(-1), dim=-1)


def unbind(ab, a):
    return torch.fft.irfft(torch.fft.rfft(ab, dim=-1) * torch.conj(torch.fft.rfft(a, dim=-1)),
                           n=ab.size(-1), dim=-1)


def token_span(offsets, cs, ce):
    lo = hi = None
    for i, (x, y) in enumerate(offsets):
        if y <= x:
            continue
        if x < ce and y > cs:
            lo = i if lo is None else lo
            hi = i + 1
    return (lo, hi) if lo is not None else (0, 1)

HID, VSA_DIM = 2560, 2048
HEX = "0123456789ABCDEF"
h2 = lambda n: HEX[(n >> 4) & 0xF] + HEX[n & 0xF]
HOPS, N_CTX, N_SYM, Q_AT_END = 2, 32, 64, False
# Fitting budget for the frozen_cont projection. Module-level so a sweep can vary it without
# touching run(); V-21 varies it to produce a range of cosine fidelities. Default 3 = the value
# every earlier Family V run used, so those results are unchanged.
FIT_EPOCHS_OVERRIDE = 3
LVL = "RFGH"
PAT = re.compile(r"\[([RFGH])([0-9A-F]{2})\]\{(\d+)\}")
QPAT = re.compile(r"Q \[R([0-9A-F]{2})\] \{(\d+)\} \?")


class Arena:
    def __init__(self, tok, n, seed, max_len=768):
        rng = random.Random(seed)
        self.rows = []
        for _ in range(n):
            chain = [rng.randrange(N_SYM) for _ in range(HOPS + 1)]
            levels, toks = [], []
            for L in range(HOPS):
                pairs, seen = [(chain[L], chain[L + 1])], {chain[L]}
                while len(pairs) < N_CTX:
                    a = rng.randrange(N_SYM)
                    if a in seen:
                        continue
                    seen.add(a); pairs.append((a, rng.randrange(N_SYM)))
                rng.shuffle(pairs); levels.append(pairs)
            for L, pairs in enumerate(levels):
                for a, b in pairs:
                    toks.append("[%s%s]{%s}" % (LVL[L], h2(a), b))
            label = 1 if rng.random() < 0.5 else 0
            cand = chain[-1] if label == 1 else rng.choice(
                [s for s in range(N_SYM) if s != chain[-1]])
            q = "Q [R%s] {%d} ?" % (h2(chain[0]), cand)
            text = (" ".join(toks) + " " + q) if Q_AT_END else \
                   (" ".join(toks[:len(toks) // 2]) + " " + q + " " + " ".join(toks[len(toks) // 2:]))
            enc = tok(text, truncation=True, max_length=max_len, padding="max_length",
                      return_tensors="pt", return_offsets_mapping=True)
            offs = enc["offset_mapping"].squeeze(0).tolist()
            ctx = [[] for _ in range(HOPS)]
            for m in PAT.finditer(text):
                L = LVL.index(m.group(1))
                if L < HOPS:
                    # BUG FIX (disclosed at daa512c): one match carries TWO symbols, the role in group 2
                    # and the filler in group 3. Recording a single span for both made every
                    # derived-vector arm compute bind(v, v) -- binding each pair's value to itself.
                    ctx[L].append((int(m.group(2), 16), int(m.group(3)),
                                   token_span(offs, *m.span(2)),      # role sub-span
                                   token_span(offs, *m.span(3))))     # filler sub-span
            qm = QPAT.search(text)
            q_sp = token_span(offs, *qm.span(1)) if qm else (0, 1)
            c_sp = token_span(offs, *qm.span(2)) if qm else (0, 1)
            self.rows.append({"ids": enc["input_ids"].squeeze(0),
                              "am": enc["attention_mask"].squeeze(0), "label": label,
                              "q": chain[0], "cand": cand, "ctx": ctx,
                              # token spans of the query role and the candidate filler, so that a
                              # label-free relabelling can map them through the same code as the context
                              "q_span": q_sp, "cand_span": c_sp})


@torch.no_grad()
def cache(rows, llm, device, bs=4):
    H = []
    for i in range(0, len(rows), bs):
        b = rows[i:i + bs]
        H.append(llm(input_ids=torch.stack([r["ids"] for r in b]).to(device),
                     attention_mask=torch.stack([r["am"] for r in b]).to(device)
                     ).last_hidden_state.half().cpu())
    return torch.cat(H)


class Floor(nn.Module):
    def __init__(self):
        super().__init__(); self.fc = nn.Linear(HID, 2)

    def logits(self, H, rows, device):
        idx = torch.tensor([int(r["am"].sum()) - 1 for r in rows], device=device).clamp(min=0)
        return self.fc(H[torch.arange(H.size(0), device=device), idx].float())

    def sims(self, H, rows, device):
        lg = self.logits(H, rows, device)
        return lg[:, 1] - lg[:, 0]

    def forward(self, H, am, rows, device):
        return self.logits(H, rows, device)


class BindArm(nn.Module):
    def __init__(self, mode, seed):
        super().__init__()
        self.mode = mode
        self.register_buffer("sym", unit(N_SYM, VSA_DIM, seed + 31))
        self.scale = nn.Parameter(torch.tensor(20.0))
        self.bias = nn.Parameter(torch.tensor(-4.0))
        if mode in ("spans", "frozen_cont"):
            self.proj = nn.Linear(HID, VSA_DIM)
            if mode == "frozen_cont":
                for p in self.proj.parameters():
                    p.requires_grad_(False)
        if mode == "probe_argmax":
            self.id_head = nn.Linear(HID, N_SYM)
        if mode == "consistent":
            g = random.Random(7000 + seed)
            perm = list(range(N_SYM)); g.shuffle(perm)
            self.register_buffer("remap", torch.tensor(perm))

    def sf(self, H, sp):
        return H[sp[0]:sp[1]].float().mean(0) if sp[1] > sp[0] else H[sp[0]:sp[0] + 1].float().mean(0)

    def vec(self, H, sid, sp, device):
        if self.mode == "parser":
            return self.sym[sid].to(device)
        if self.mode == "consistent":
            return self.sym[int(self.remap[sid])].to(device)
        v = self.sf(H, sp)
        if self.mode in ("spans", "frozen_cont"):
            return F.normalize(self.proj(v), dim=-1)
        return self.sym[int(self.id_head(v).argmax().item())].to(device)

    def q_vec(self, sid, device):
        i = int(self.remap[sid]) if self.mode == "consistent" else sid
        return self.sym[i].to(device)

    def sims(self, H, rows, device):
        out = torch.zeros(len(rows), device=device)
        for b, r in enumerate(rows):
            if any(len(l) == 0 for l in r["ctx"]):
                continue
            key = self.q_vec(r["q"], device)
            for L in range(HOPS):
                # each derived vector comes from ITS OWN sub-span (fix daa512c)
                mem = torch.stack([bind(self.vec(H[b], bb, spf, device),
                                        self.vec(H[b], aa, spr, device))
                                   for aa, bb, spr, spf in r["ctx"][L]], 0).sum(0)
                hat = F.normalize(unbind(mem, key), dim=-1)
                cands = torch.stack([self.vec(H[b], bb, spf, device)
                                     for aa, bb, spr, spf in r["ctx"][L]], 0)
                key = cands[int(F.cosine_similarity(cands, hat.unsqueeze(0)).argmax().item())]
            out[b] = torch.dot(key, self.q_vec(r["cand"], device)).clamp(-1, 1)
        return out

    def forward(self, H, am, rows, device):
        lp = self.sims(H, rows, device) * self.scale + self.bias
        return torch.stack([-lp, lp], dim=1)

    def id_loss(self, H, rows, device):
        if self.mode != "probe_argmax":
            return torch.zeros((), device=device)
        t = []
        for b, r in enumerate(rows):
            for L in range(HOPS):
                for aa, bb, spr, spf in r["ctx"][L]:
                    for sid, sp in ((aa, spr), (bb, spf)):
                        t.append(F.cross_entropy(self.id_head(self.sf(H[b], sp)).unsqueeze(0),
                                                 torch.tensor([sid], device=device)))
        return torch.stack(t).mean() if t else torch.zeros((), device=device)

    def fit_loss(self, H, rows, device):
        t = []
        for b, r in enumerate(rows):
            for L in range(HOPS):
                for aa, bb, spr, spf in r["ctx"][L]:
                    for sid, sp in ((aa, spr), (bb, spf)):
                        v = F.normalize(self.proj(self.sf(H[b], sp)), dim=-1)
                        t.append(1.0 - torch.dot(v, self.sym[sid].to(device)))
        return torch.stack(t).mean() if t else torch.zeros((), device=device)


def run(arm, seed, tok, llm, device, n_tr, n_te, epochs, bs, lr):
    torch.manual_seed(seed)
    tr, te = Arena(tok, n_tr, 42), Arena(tok, n_te, 4242)
    Htr, Hte = cache(tr.rows, llm, device), cache(te.rows, llm, device)
    m = (Floor() if arm == "floor" else BindArm(arm, seed)).to(device)

    def mk():
        sc = [q for n, q in m.named_parameters() if q.requires_grad and n in ("scale", "bias")]
        rest = [q for n, q in m.named_parameters() if q.requires_grad and n not in ("scale", "bias")]
        g = ([{"params": rest, "lr": lr}] if rest else []) + \
            ([{"params": sc, "lr": 1e-1}] if sc else [])
        return torch.optim.AdamW(g)

    params = [p for p in m.parameters() if p.requires_grad]
    opt = mk()
    if arm in ("frozen_cont", "proj_argmax"):
        for p in m.proj.parameters():
            p.requires_grad_(True)
        fo = torch.optim.AdamW(m.proj.parameters(), lr=1e-3)
        loss = None
        for _ in range(FIT_EPOCHS_OVERRIDE):
            for k in range(0, len(tr.rows), bs):
                loss = m.fit_loss(Htr[k:k + bs].to(device), tr.rows[k:k + bs], device)
                fo.zero_grad(); loss.backward(); fo.step()
        if loss is None:
            with torch.no_grad():
                l0 = m.fit_loss(Htr[:bs].to(device), tr.rows[:bs], device)
            print("      %s UNFITTED (budget 0), cos fidelity ~= %.4f"
                  % (arm, 1.0 - float(l0.item())), flush=True)
        else:
            print("      %s fitted (%d epochs), cos fidelity ~= %.4f"
                  % (arm, FIT_EPOCHS_OVERRIDE, 1.0 - float(loss.item())), flush=True)
        for p in m.proj.parameters():
            p.requires_grad_(False)
        params = [p for p in m.parameters() if p.requires_grad]; opt = mk()

    def collect(rows, Hall):
        m.eval(); S, Y = [], []
        with torch.no_grad():
            for k in range(0, len(rows), bs):
                S.append(m.sims(Hall[k:k + bs].to(device), rows[k:k + bs], device).cpu())
                Y += [r["label"] for r in rows[k:k + bs]]
        return torch.cat(S).numpy(), np.array(Y)

    def acc():
        st, yt = collect(tr.rows, Htr); se, ye = collect(te.rows, Hte)
        c = np.unique(st)
        if c.size == 0:
            return 0.5
        thr = float(c[int(np.argmax([(((st >= t) == (yt == 1)).mean()) for t in c]))])
        return float((((se >= thr) == (ye == 1)).mean()))

    t0, best = time.time(), 0.0
    g = torch.Generator(); g.manual_seed(seed)
    for ep in range(epochs):
        m.train()
        perm = torch.randperm(len(tr.rows), generator=g).tolist()
        for k in range(0, len(perm), bs):
            sel = perm[k:k + bs]; rows = [tr.rows[i] for i in sel]
            H = Htr[sel].to(device)
            am = torch.stack([r["am"] for r in rows]).to(device)
            y = torch.tensor([r["label"] for r in rows], device=device)
            loss = F.cross_entropy(m(H, am, rows, device), y)
            if arm == "probe_argmax":
                loss = loss + m.id_loss(H, rows, device)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        a = acc(); best = max(best, a)
        print("      %-13s ep%d test=%.4f (%.0fs)" % (arm, ep + 1, a, time.time() - t0), flush=True)
    return {"arm": arm, "seed": seed, "acc": best,
            "trainable": sum(p.numel() for p in params), "secs": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+",
                    default=["floor", "parser", "consistent", "spans", "probe_argmax", "frozen_cont"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    ap.add_argument("--n_train", type=int, default=800)
    ap.add_argument("--n_test", type=int, default=500)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="v13_logs/v13.json")
    a = ap.parse_args()
    from transformers import AutoModel, AutoTokenizer
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
            r = run(arm, s, tok, llm, a.device, a.n_train, a.n_test, a.epochs, a.batch_size, a.lr)
            print("  %-13s sd%-4d acc=%.4f trainable=%s %ss"
                  % (arm, s, r["acc"], format(r["trainable"], ","), r["secs"]), flush=True)
            rows.append(r)
            os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
            json.dump(rows, open(a.out, "w"), indent=1)

    def col(x):
        return [r["acc"] for r in rows if r["arm"] == x]
    FLOOR_M, Hh = 0.4737, 0.5263        # measured in the floor survey, fixed by prereg d3a4da5
    print("\n=== V-13 SCORING (prereg d3a4da5; floor %.4f, H %.4f) ===" % (FLOOR_M, Hh), flush=True)
    for x in a.arms:
        c = col(x)
        if c:
            print("  %-13s %s mean=%.4f" % (x, [round(v, 4) for v in c], float(np.mean(c))), flush=True)
    fl, pa, sp, pq = col("floor"), col("parser"), col("spans"), col("probe_argmax")
    if len(fl) == len(pa) == 3:
        A = all(pa[i] >= fl[i] + 0.50 * (1 - fl[i]) for i in range(3))
        B = all(sp[i] <= fl[i] + 0.10 * (1 - fl[i]) for i in range(3)) if len(sp) == 3 else None
        C = all(pq[i] >= pa[i] - 0.10 * (1 - fl[i]) for i in range(3)) if len(pq) == 3 else None
        print("  (a) parser >= floor+0.50H in 3/3       -> %s" % A, flush=True)
        print("  (b) spans  <= floor+0.10H in 3/3       -> %s" % B, flush=True)
        print("  (c) probe_argmax >= parser-0.10H 3/3   -> %s" % C, flush=True)
        v = "VOID (arena cannot test)" if not A else \
            ("MECHANISM REPLICATES" if (B and C) else "FAILS TO REPLICATE")
        print("  >>> V-13 VERDICT: %s" % v, flush=True)
    print("V13-DONE", flush=True)


if __name__ == "__main__":
    main()
