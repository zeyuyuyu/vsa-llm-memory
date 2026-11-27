"""B31 memory arms on the frozen states (prereg/B31_encoder_equalised_baselines_prereg.md), for either encoder.

Every arm hands one code per symbol mention to the published HRR chain of its arena and differs only in where the codes
come from:
  exact  exact codebook rows of the gold symbols (the hand-written parser; no state is read)
  name   naming head: linear head on the span-mean frozen state, trained with identity labels; its arg-max picks the
         codebook row (snapped)
  soft   naming head read without snapping (the published "frozen continuous projection"): a linear map fitted with
         the same identity labels to the codebook row (1 - cosine), then frozen and read continuously
  bind   linear map trained only through the memory's answer (the binding loss; published "spans" arm)
  nce    the same map plus the label-free InfoNCE term between the code of each mention and the code of the same
         mention in a reshuffled rendering of the instance (B27's best label-free family)

Encodings (B28): comm = published HRR, K(i) = V(i) = codebook row; perm = the value argument permuted by a fixed random
permutation P of the coordinates, V(i) = K(i)[P] (private generator seed + 7001; for learned codes V = code[P]).
Conventions: index = the published one (A: fillers and the candidate are codebook rows by index; B: roles, the query
role and the candidate by index; C: the query and the candidate by index); named = every symbol, including the query
and the candidate, comes from its own span (no index enters the store).

The chains, optimisers, batch sizes, clipping and closed-form threshold are the published ones (A: v8/v9 on
SymbolicVSAChainClassifier with the vsa.py operators; B: v11.BindArm; C: v13.BindArm), vectorised over a batch. Mode
g0 runs the published configuration (comm, index, published identity-label sets, 2 epochs, best test epoch; arena A
with the published token scanner's view of the input, i.e. the markers it finds and its consecutive-marker pairing,
and the published learned scale/bias read-out) and must reproduce the banked per-seed numbers.
"""
import argparse, json, math, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import b31_struct as S                                                      # noqa: E402

D = 2048
NCLS = {"A": 256, "B": 256, "C": 64}
NROLE = 64
KINDS = {"A": "RXYF", "B": "RF", "C": "RF"}
BS = {"A": 8, "B": 4, "C": 4}
SEEDS = [42, 123, 456]
BANKED = {  # published per-seed test accuracies (best test epoch of 2), figdata / b27_prep.BANKED
    ("A", "exact"): [0.918, 0.919, 0.918], ("A", "name"): [0.915, 0.913, 0.913], ("A", "bind"): [0.511, 0.499, 0.493],
    ("B", "exact"): [0.990, 1.000, 0.998], ("B", "name"): [0.990, 1.000, 0.998], ("B", "bind"): [0.502, 0.498, 0.522],
    ("C", "exact"): [0.932, 0.932, 0.938], ("C", "name"): [0.932, 0.932, 0.938], ("C", "bind"): [0.472, 0.502, 0.476]}


# ------------------------------------------------------------------ layout (fixed per arena; see b31_struct)

def layout(arena):
    """Mention kind per position, the hops as (key positions, value positions), query and candidate positions, and
    the published 'index' positions."""
    if arena == "A":
        kinds = ["R", "X"] * 16 + ["X", "Y"] * 16 + ["Y", "F"] * 16 + ["R", "F"]
        hops = [(list(range(0, 32, 2)), list(range(1, 32, 2))), (list(range(32, 64, 2)), list(range(33, 64, 2))),
                (list(range(64, 96, 2)), list(range(65, 96, 2)))]
        q, c = 96, 97
        index = [i for i, k in enumerate(kinds) if k == "F"]
    elif arena == "B":
        kinds = ["R", "F"] * 64 + ["R", "F"]
        hops = [(list(range(0, 128, 2)), list(range(1, 128, 2)))]
        q, c = 128, 129
        index = [i for i, k in enumerate(kinds) if k == "R"] + [c]
    else:
        kinds = ["R", "F"] * 64 + ["R", "F"]
        hops = [(list(range(0, 64, 2)), list(range(1, 64, 2))), (list(range(64, 128, 2)), list(range(65, 128, 2)))]
        q, c = 128, 129
        index = [q, c]
    return kinds, hops, q, c, index


def published_idset(arena, kinds, q, c):
    """Mentions the published naming head is trained on: A all R/X/Y markers incl. the query's R (v9 id_loss over the
    scanner's markers); B the context fillers (v11); C the context roles and fillers (v13)."""
    if arena == "A":
        return [i for i, k in enumerate(kinds) if k != "F"]
    if arena == "B":
        return [i for i, k in enumerate(kinds[:128]) if k == "F"]
    return list(range(128))


# ------------------------------------------------------------------ data

class Split:
    def __init__(self, enc, arena, split, feats_root, struct_cache=None):
        p = os.path.join(feats_root, "%s_%s_%s.pt" % (enc, arena, split))
        obj = torch.load(p, weights_only=False)
        st = struct_cache if struct_cache is not None else \
            S.build(arena, split[:-3] if split.endswith("_v2") else split, view2=split.endswith("_v2"))
        assert obj["text_md5"] == __import__("b31_prep").h([s["text"] for s in st]), "feats/structure mismatch"
        self.st = st
        self.F = obj["F"]                                       # (N, M, d) fp32
        self.y = torch.tensor([s["label"] for s in st])
        self.sym = torch.tensor([[m[2] for m in s["mentions"]] for s in st])
        self.keys = [[m[0] for m in s["mentions"]] for s in st]
        self.n = len(st)


# ------------------------------------------------------------------ operators (published forms)

def bindA(a, b):
    return torch.fft.ifft(torch.fft.fft(a.float(), dim=-1) * torch.fft.fft(b.float(), dim=-1), dim=-1).real


def unbindA(ab, a):
    return torch.fft.ifft(torch.fft.fft(ab.float(), dim=-1) * torch.conj(torch.fft.fft(a.float(), dim=-1)),
                          dim=-1).real


def l2n(x, eps=1e-8):
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def simA(a, b):
    return (l2n(a) * l2n(b)).sum(-1)


def bindC(a, b):
    return torch.fft.irfft(torch.fft.rfft(a, dim=-1) * torch.fft.rfft(b, dim=-1), n=a.size(-1), dim=-1)


def unbindC(ab, a):
    return torch.fft.irfft(torch.fft.rfft(ab, dim=-1) * torch.conj(torch.fft.rfft(a, dim=-1)), n=ab.size(-1), dim=-1)


def unitBC(n, d, seed):                     # v11.unit / v13.unit
    g = torch.Generator(); g.manual_seed(seed)
    return F.normalize(torch.randn(n, d, generator=g), dim=1)


def unitA(n, d, seed):                      # explicit_chain_models._init_random_unit_vectors
    g = torch.Generator(); g.manual_seed(seed)
    v = torch.randn(n, d, generator=g)
    return v / (v.norm(dim=-1, keepdim=True) + 1e-8)


# ------------------------------------------------------------------ the arm

class Arm(nn.Module):
    def __init__(self, arena, arm, seed, hid):
        """Global-RNG consumption in the published order: A draws the role vectors (VSAModule, torch.randn) before any
        layer; then the head / projection (nn.Linear). Codebooks of X, Y, F (A) and of B, C use private generators."""
        super().__init__()
        self.arena, self.arm, self.seed = arena, arm, seed
        self.kinds, self.hops, self.q, self.c, self.index = layout(arena)
        if arena == "A":
            role = torch.randn(NROLE, D)
            role = role / role.norm(dim=-1, keepdim=True)         # VSAModule._init_random_vectors
            books = {"R": role, "X": unitA(256, D, seed + 11), "Y": unitA(256, D, seed + 23),
                     "F": unitA(256, D, seed + 37)}
        elif arena == "B":
            books = {"R": unitBC(64, D, seed + 11), "F": unitBC(256, D, seed + 23)}
        else:
            s_ = unitBC(64, D, seed + 31)
            books = {"R": s_, "F": s_}
        order = list(dict.fromkeys(KINDS[arena]))
        off, rows = {}, []
        for k in order:
            if arena == "C" and k == "F":
                off[k] = off["R"]; continue
            off[k] = sum(r.size(0) for r in rows); rows.append(books[k])
        self.off = off
        self.register_buffer("table", torch.cat(rows))
        P = torch.randperm(D, generator=torch.Generator().manual_seed(seed + 7001))
        self.register_buffer("P", P)
        self.register_buffer("koff", torch.tensor([off[k] for k in self.kinds]))
        self.register_buffer("is_r", torch.tensor([k == "R" for k in self.kinds]))
        if arena == "A":
            self.sim_scale = nn.Parameter(torch.tensor(20.0)); self.sim_bias = nn.Parameter(torch.tensor(-4.0))
        else:
            self.scale = nn.Parameter(torch.tensor(20.0)); self.bias = nn.Parameter(torch.tensor(-4.0))
        if arm == "name":
            self.head = nn.Linear(hid, NCLS[arena])
        elif arm in ("soft", "bind", "nce"):
            self.proj = nn.Linear(hid, D)
            if arm == "soft":
                self.proj.requires_grad_(False)

    # ---- codes
    def logits_id(self, Fb, pos=None):
        """Head logits (B, M, ncls); an R mention is read over the 64 roles only (v9: logits[:num_roles])."""
        lg = self.head(Fb)
        if self.arena in ("A", "B"):
            isr = self.is_r if pos is None else self.is_r[torch.as_tensor(pos, device=lg.device)]
            lg = torch.where(isr[None, :, None] & (torch.arange(lg.size(-1), device=lg.device) >= NROLE),
                             torch.full_like(lg, -float("inf")), lg)
        return lg

    def gold_rows(self, sym):
        return self.koff[None, :] + sym

    def kcodes(self, Fb, sym, conv):
        """K codes (B, M, D) of every mention under convention conv (index | named)."""
        if self.arm == "exact":
            K = self.table[self.gold_rows(sym)]
        elif self.arm == "name":
            pred = self.logits_id(Fb).argmax(-1)
            K = self.table[self.koff[None, :] + pred]
        else:
            z = self.proj(Fb)
            K = l2n(z) if self.arena == "A" else F.normalize(z, dim=-1)
        if conv == "index" and self.arm != "exact":
            K = K.clone()
            idx = torch.tensor(self.index, device=K.device)
            K[:, idx] = self.table[self.gold_rows(sym)[:, idx]]
        return K

    def chain(self, K, enc, struct=None):
        """Published chain of the arena on K codes; V = K (comm) or K[..., P] (perm). Returns (score (B,), winners
        per cleaned hop (B, hops-1 or hops), ok mask (B,)). struct (A, g0 only): per-instance hop pairs + masks."""
        V = K[..., self.P] if enc == "perm" else K
        B = K.size(0)
        ar = torch.arange(B, device=K.device)
        if self.arena == "A":
            if struct is None:
                hops = [(torch.tensor(k, device=K.device).expand(B, -1), torch.tensor(v, device=K.device).expand(B, -1),
                         torch.ones(B, len(k), dtype=torch.bool, device=K.device)) for k, v in self.hops]
                qpos = torch.full((B,), self.q, device=K.device, dtype=torch.long)
            else:
                hops, qpos = struct["hops"], struct["q"]
            ok = torch.ones(B, dtype=torch.bool, device=K.device)
            key = K[ar, qpos]
            wins = []
            for h, (kp, vp, mk) in enumerate(hops):
                ok = ok & mk.any(1)
                Kk = K[ar[:, None], kp]; Vv = V[ar[:, None], vp]
                bound = bindA(Vv, Kk) * mk[..., None]
                s_ = bound.sum(1)
                mem = s_ / (s_.norm(dim=-1, keepdim=True) + 1e-8)                  # vsa.bundle
                hat = l2n(unbindA(mem, key))
                if h < len(hops) - 1:
                    sims = simA(Vv, hat[:, None, :].expand_as(Vv)).masked_fill(~mk, -float("inf"))
                    w = sims.argmax(1)
                    wins.append(vp[ar, w])
                    key = K[ar, vp[ar, w]]
                else:
                    score = simA(hat, V[:, self.c]).clamp(-1, 1)
            score = torch.where(ok, score, torch.zeros_like(score))
            return score, torch.stack(wins, 1), ok
        if self.arena == "B":
            kp, vp = (torch.tensor(x, device=K.device) for x in self.hops[0])
            mem = bindC(V[:, vp], K[:, kp]).sum(1)
            hat = F.normalize(unbindC(mem, K[:, self.q]), dim=-1)
            score = (hat * V[:, self.c]).sum(-1).clamp(-1, 1)
            return score, torch.zeros(B, 0, dtype=torch.long, device=K.device), torch.ones(B, dtype=torch.bool,
                                                                                          device=K.device)
        key = K[:, self.q]
        wins = []
        for kp, vp in self.hops:
            kp = torch.tensor(kp, device=K.device); vp = torch.tensor(vp, device=K.device)
            mem = bindC(V[:, vp], K[:, kp]).sum(1)
            hat = F.normalize(unbindC(mem, key), dim=-1)
            cands = V[:, vp]
            w = F.cosine_similarity(cands, hat[:, None, :], dim=-1).argmax(1)
            wins.append(vp[w])
            key = K[ar, vp[w]]
        score = (key * K[:, self.c]).sum(-1).clamp(-1, 1)
        return score, torch.stack(wins, 1), torch.ones(B, dtype=torch.bool, device=K.device)

    def answer_logits(self, score):
        sc, bi = (self.sim_scale, self.sim_bias) if self.arena == "A" else (self.scale, self.bias)
        lp = score * sc + bi
        return torch.stack([-lp, lp], dim=1)

    def id_loss(self, Fb, sym, pos, vis=None):
        """Mean cross-entropy of the head over the mention positions `pos` (R mentions over the 64 roles)."""
        lg = self.logits_id(Fb[:, pos], pos)
        t = sym[:, pos]
        ce = F.cross_entropy(lg.reshape(-1, lg.size(-1)), t.reshape(-1), reduction="none")
        if vis is not None:
            m = vis[:, pos].reshape(-1).float()
            return (ce * m).sum() / m.sum().clamp(min=1)
        return ce.mean()

    def fit_loss(self, Fb, sym, pos):
        v = F.normalize(self.proj(Fb[:, pos]), dim=-1)
        tgt = self.table[self.gold_rows(sym)[:, pos]]
        return (1.0 - (v * tgt).sum(-1)).mean()


# ------------------------------------------------------------------ A: the published token scanner's view (g0 only)

def scanner_struct(vis_rows, dev):
    """vis_rows: (B, 98) int8, 1 where the published scanner finds the marker (F positions are 1). Rebuilds the
    published pairing: consecutive visible markers (R, X) -> hop 1, (X, Y) -> hop 2 (including pairs across a section
    boundary when a marker in between is missed), visible section-C Y markers -> hop 3 with their filler; query = last
    visible R."""
    kinds, hops, q, c, _ = layout("A")
    B = vis_rows.size(0)
    H = [[[], []] for _ in range(3)]
    qs = []
    for b in range(B):
        v = vis_rows[b].tolist()
        seq = [i for i in range(97) if kinds[i] != "F" and v[i] == 1]
        p1, p2 = [], []
        for i1, i2 in zip(seq[:-1], seq[1:]):
            if kinds[i1] == "R" and kinds[i2] == "X":
                p1.append((i1, i2))
            elif kinds[i1] == "X" and kinds[i2] == "Y":
                p2.append((i1, i2))
        p3 = [(i, i + 1) for i in range(64, 96, 2) if v[i] == 1]
        for h, pl in enumerate((p1, p2, p3)):
            H[h][0].append(pl); H[h][1].append(len(pl))
        qs.append(max(i for i in seq if kinds[i] == "R"))
    out = []
    for h in range(3):
        P_ = max(1, max(H[h][1]))
        kp = torch.zeros(B, P_, dtype=torch.long); vp = torch.zeros(B, P_, dtype=torch.long)
        mk = torch.zeros(B, P_, dtype=torch.bool)
        for b, pl in enumerate(H[h][0]):
            for j, (i1, i2) in enumerate(pl):
                kp[b, j], vp[b, j], mk[b, j] = i1, i2, True
        out.append((kp.to(dev), vp.to(dev), mk.to(dev)))
    return {"hops": out, "q": torch.tensor(qs, device=dev)}


# ------------------------------------------------------------------ scoring helpers

def thr_acc(st, yt, se, ye):
    """Closed-form threshold as v13.run / v11.run: the training-accuracy-maximising candidate, applied to se."""
    c = np.unique(st)
    if c.size == 0:
        return 0.5, None, 0.5
    accs = [(((st >= t) == (yt == 1)).mean()) for t in c]
    i = int(np.argmax(accs))
    thr = float(c[i])
    return float((((se >= thr) == (ye == 1)).mean())), thr, float(accs[i])


GEO_SETS = {"A": "RXY", "B": "F", "C": "RF"}


@torch.no_grad()
def geometry(m, Fs, sym, conv, n_inst=120, per_sym=4, n_sym=40):
    """V-165 protocol as in B27: first 120 instances, up to 4 occurrences per symbol, first 40 symbols; mean cosine
    between different symbols (first occurrence each) and between occurrences of one symbol. Context mentions only;
    labels group the vectors for this measurement only."""
    kinds = m.kinds
    pos = [i for i in range(len(kinds)) if i not in (m.q, m.c) and kinds[i] in GEO_SETS[m.arena]]
    by = {}
    K = m.kcodes(Fs[:n_inst], sym[:n_inst], conv)
    for b in range(K.size(0)):
        for i in pos:
            sid = (kinds[i], int(sym[b, i])) if m.arena == "A" else int(sym[b, i])
            if len(by.setdefault(sid, [])) < per_sym:
                by[sid].append(K[b, i])
    ks = [k for k in by if by[k]][:n_sym]
    first = torch.stack([by[k][0] for k in ks])
    G = first @ first.t()
    diff = G[~torch.eye(len(ks), dtype=torch.bool, device=G.device)].mean().item()
    same = [float(torch.dot(by[k][i], by[k][j])) for k in ks for i in range(len(by[k])) for j in range(i + 1, len(by[k]))]
    return {"different": diff, "same": float(np.mean(same)) if same else None, "symbols": len(ks)}


# ------------------------------------------------------------------ runs

def run(a, seed, D_, dev):
    arena = a.arena
    torch.manual_seed(seed)
    hid = D_["train"].F.size(-1)
    m = Arm(arena, a.arm, seed, hid).to(dev)
    kinds, hops, q, c, index = m.kinds, m.hops, m.q, m.c, m.index
    g0 = a.mode == "g0"
    train_enc, train_conv = a.enc, a.conv
    readings = [(a.enc, a.conv)] + [tuple(r.split("/")) for r in a.readings if tuple(r.split("/")) != (a.enc, a.conv)]
    if a.arm in ("name", "soft"):
        idpos = published_idset(arena, kinds, q, c) if a.idset == "published" else list(range(len(kinds)))
    vis = {}
    if g0 and arena == "A":
        V_ = torch.load(os.path.join(a.out_root, "scanner_visible_A.pt"))
        vis = {k: V_[k] for k in ("train", "test")}
    sc = [p for n, p in m.named_parameters() if p.requires_grad and n in ("scale", "bias")]
    rest = [p for n, p in m.named_parameters() if p.requires_grad and n not in ("scale", "bias")]
    if arena == "A":
        opt = torch.optim.AdamW(rest, lr=a.lr) if rest else None     # v8/v9: one group (sim_scale/bias at lr)
    else:
        opt = torch.optim.AdamW(([{"params": rest, "lr": a.lr}] if rest else []) +
                                ([{"params": sc, "lr": 1e-1}] if sc else []))
    params = [p for p in m.parameters() if p.requires_grad]
    bs = BS[arena]
    tr = D_["train"]

    def scores(sp_name, Dsp, enc, conv):
        m.eval(); out, wins, named, oks, preds = [], [], [], [], []
        with torch.no_grad():
            for k in range(0, Dsp.n, 64):
                Fb = Dsp.F[k:k + 64].to(dev); sy = Dsp.sym[k:k + 64].to(dev)
                st = scanner_struct(vis[sp_name][k:k + 64], dev) if (g0 and arena == "A") else None
                K = m.kcodes(Fb, sy, conv)
                s_, w, ok = m.chain(K, enc, st)
                if g0 and arena == "A":
                    s_ = torch.where(ok, s_, torch.zeros_like(s_))
                out.append(s_.cpu()); wins.append(w.cpu()); oks.append(ok.cpu())
                if a.arm == "name":
                    pred = m.logits_id(Fb).argmax(-1)
                    named.append((pred == sy).cpu()); preds.append(pred.cpu())
        res = {"s": torch.cat(out).numpy(), "w": torch.cat(wins), "ok": torch.cat(oks)}
        if named:
            res["named"] = torch.cat(named); res["pred"] = torch.cat(preds)
        return res

    def evaluate(ep):
        e = {"epoch": ep}
        for enc, conv in readings:
            tag = "%s/%s" % (enc, conv)
            rt = scores("train", tr, enc, conv)
            yt = tr.y.numpy()
            for sp in [s for s in ("dev", "test") if s in D_]:
                Dsp = D_[sp]
                rs = scores(sp, Dsp, enc, conv)
                ye = Dsp.y.numpy()
                if a.readout == "logit":       # v8/v9: argmax of the two learned logits; unparsed rows have logits 0
                    def lpred(rr):
                        with torch.no_grad():
                            lg = m.answer_logits(torch.tensor(rr["s"]).to(dev)).cpu()
                        lg = torch.where(rr["ok"][:, None], lg, torch.zeros_like(lg))
                        return lg.argmax(1).numpy()
                    acc = float((lpred(rs) == ye).mean()); thr = None
                    tracc = float((lpred(rt) == yt).mean())
                else:
                    acc, thr, tracc = thr_acc(rt["s"], yt, rs["s"], ye)
                r = {"acc": acc, "thr": thr, "train_acc": tracc}
                if a.readout == "logit":
                    r["scale_bias"] = [float(p) for p in (m.sim_scale, m.sim_bias)]
                if sp == "test" or a.save_dev_scores:
                    r["scores"] = [round(float(x), 6) for x in rs["s"]]
                    r["wins"] = rs["w"].tolist()
                if "named" in rs:
                    nm = rs["named"]
                    r["naming_acc_mentions"] = float(nm.float().mean())
                    r["naming_all_ok"] = nm.all(1).int().tolist() if sp == "test" else float(nm.all(1).float().mean())
                    if sp == "test":
                        r["misnamed"] = [[(j, int(rs["pred"][b_, j])) for j in torch.nonzero(~nm[b_]).flatten().tolist()]
                                         for b_ in range(nm.size(0))]
                if (sp == "test" or (sp == "dev" and "test" not in D_)) and (enc, conv) == readings[0] \
                        and a.arm != "exact":
                    r["geom"] = geometry(m, Dsp.F[:120].to(dev), Dsp.sym[:120].to(dev), conv)
                e["%s:%s" % (tag, sp)] = r
        return e

    rec = {"seed": seed, "epochs": []}
    t0 = time.time()
    if a.arm == "soft":
        for p in m.proj.parameters():
            p.requires_grad_(True)
        fo = torch.optim.AdamW(m.proj.parameters(), lr=1e-3)
        for ep in range(a.fit_epochs):
            m.train()
            for k in range(0, tr.n, bs):                                   # published: data order, no shuffle
                loss = m.fit_loss(tr.F[k:k + bs].to(dev), tr.sym[k:k + bs].to(dev), idpos)
                fo.zero_grad(); loss.backward(); fo.step()
            e = evaluate(ep + 1); e["fit_loss_last"] = float(loss); e["secs"] = round(time.time() - t0, 1)
            rec["epochs"].append(e); log(arena, a.arm, seed, e, readings[0])
        return rec
    if a.arm == "exact" and a.readout == "thr":
        e = evaluate(0); e["secs"] = round(time.time() - t0, 1)
        rec["epochs"].append(e); log(arena, a.arm, seed, e, readings[0])
        return rec
    g = torch.Generator(); g.manual_seed(seed)
    for ep in range(a.epochs):
        m.train()
        perm = torch.randperm(tr.n, generator=g)
        tl, rl, nb = 0.0, 0.0, 0
        for k in range(0, tr.n, bs):
            i = perm[k:k + bs]
            Fb = tr.F[i].to(dev); sy = tr.sym[i].to(dev); y = tr.y[i].to(dev)
            st = scanner_struct(vis["train"][i], dev) if (g0 and arena == "A") else None
            K = m.kcodes(Fb, sy, train_conv)
            s_, _, ok = m.chain(K, train_enc, st)
            lg = m.answer_logits(s_)
            if g0 and arena == "A":
                lg = torch.where(ok[:, None], lg, torch.zeros_like(lg))     # unparsed rows: zero logits (v8/v9)
            loss = F.cross_entropy(lg, y)
            tl += float(loss); nb += 1
            if a.arm == "name":
                loss = loss + m.id_loss(Fb, sy, idpos, vis["train"][i].to(dev) if (g0 and arena == "A") else None)
            if a.arm == "nce":
                tr2 = D_["train_v2"]
                F2 = tr2.F[i].to(dev)
                K1 = K
                K2 = m.kcodes(F2, tr2.sym[i].to(dev), train_conv)
                learned = [j for j in range(len(kinds)) if train_conv == "named" or j not in index]
                c1, c2 = [], []
                for bb, ii in enumerate(i.tolist()):
                    k2 = {kk: jj for jj, kk in enumerate(tr2.keys[ii])}
                    sel = [j for j in learned if tr.keys[ii][j] in k2]
                    c1.append(K1[bb, sel]); c2.append(K2[bb, [k2[tr.keys[ii][j]] for j in sel]])
                c1, c2 = torch.cat(c1), torch.cat(c2)
                lgn = c1 @ c2.t() / a.tau
                yy = torch.arange(c1.size(0), device=dev)
                rv = 0.5 * (F.cross_entropy(lgn, yy) + F.cross_entropy(lgn.t(), yy))
                rl += float(rv)
                loss = loss + a.lam * rv
            if opt is not None:
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        e = evaluate(ep + 1)
        e["task_loss"], e["reg"], e["secs"] = tl / nb, (rl / nb if a.arm == "nce" else None), round(time.time() - t0, 1)
        rec["epochs"].append(e); log(arena, a.arm, seed, e, readings[0])
    return rec


def log(arena, arm, seed, e, rd):
    tag = "%s/%s" % rd
    parts = []
    for sp in ("dev", "test"):
        k = "%s:%s" % (tag, sp)
        if k in e:
            r = e[k]
            parts.append("%s %.4f (train %.4f%s%s)" % (sp, r["acc"], r["train_acc"],
                                                    ", named %.4f" % r["naming_acc_mentions"]
                                                    if "naming_acc_mentions" in r else "",
                                                    ", geom %.3f/%.3f" % (r["geom"]["different"], r["geom"]["same"])
                                                    if "geom" in r and r["geom"]["same"] is not None else ""))
    print("  %s %s seed %d ep%d [%s] %s (%.0fs)" % (arena, arm, seed, e["epoch"], tag, " | ".join(parts),
                                                    e.get("secs", 0)), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enc_model", required=True, choices=["phi2", "qwen8", "qwen8L9", "qwen8L18", "qwen8L27"],
                    help="frozen encoder of the states (qwen8L<L>: exploratory 1, intermediate layer L)")
    ap.add_argument("--arena", required=True, choices=["A", "B", "C"])
    ap.add_argument("--arm", required=True, choices=["exact", "name", "soft", "bind", "nce"])
    ap.add_argument("--mode", required=True, choices=["g0", "tune", "final"])
    ap.add_argument("--enc", default="perm", choices=["comm", "perm"], help="binding encoding used in training")
    ap.add_argument("--conv", default="named", choices=["index", "named"])
    ap.add_argument("--readings", nargs="*", default=[], help="extra enc/conv readings of the same model")
    ap.add_argument("--idset", default="all", choices=["all", "published"])
    ap.add_argument("--readout", default="thr", choices=["thr", "logit"])
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lam", type=float, default=0.0)
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--fit_epochs", type=int, default=6)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--save_dev_scores", action="store_true")
    ap.add_argument("--feats", default="/path/to/work/b31/feats")
    ap.add_argument("--out_root", default="/path/to/work/b31/out")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.mode == "g0":
        a.enc, a.conv, a.idset, a.epochs, a.lr, a.readings = "comm", "index", "published", 2, 3e-4, []
        a.readout = "logit" if a.arena == "A" else "thr"
        assert a.enc_model == "phi2" and a.arm in ("exact", "name", "bind")
    if os.path.exists(a.out):
        print("exists, skip", a.out); return
    splits = {"g0": ["train", "test"], "tune": ["train", "dev"], "final": ["train", "dev", "test"]}[a.mode]
    if a.arm == "nce":
        splits.append("train_v2")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    D_ = {sp: Split(a.enc_model, a.arena, sp, a.feats) for sp in splits}
    out = {"args": vars(a), "host": os.uname()[1], "gpu": torch.cuda.get_device_name(a.device),
           "start": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "runs": []}
    for s in a.seeds:
        rec = run(a, s, D_, a.device)
        if a.mode == "g0":
            accs = [e["comm/index:test"]["acc"] for e in rec["epochs"]]
            rec["g0_best"], rec["g0_banked"] = max(accs), BANKED[(a.arena, a.arm)][SEEDS.index(s)]
            print("  G0 %s %s seed %d best test %.4f banked %.3f" % (a.arena, a.arm, s, rec["g0_best"],
                                                                      rec["g0_banked"]), flush=True)
        out["runs"].append(rec)
        json.dump(out, open(a.out + ".part", "w"))
    out["end"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out + ".tmp", "w"))
    os.replace(a.out + ".tmp", a.out)
    if os.path.exists(a.out + ".part"):
        os.remove(a.out + ".part")


if __name__ == "__main__":
    main()
