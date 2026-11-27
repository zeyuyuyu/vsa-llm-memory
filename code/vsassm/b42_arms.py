"""B42 label-free discrete interfaces (prereg/B42_label_free_discrete_prereg.md) on B31's frozen states.

Every arm hands one code per symbol mention to the published HRR chain of its arena (b31_arms.Arm.chain; perm
encoding; named convention: every symbol, including the query and the candidate, comes from its own span) and is
trained only through the memory's answer (cross-entropy on the chain's score, exactly the loss of the B31
binding-loss map). No identity label enters any label-free arm. Families:

  vq      linear map -> l2-normalised z; nearest code (cosine) in a learned per-kind codebook updated by EMA (0.99),
          k-means++ initialisation on training mentions, dead-code restart; commitment loss (beta 0.25);
          straight-through gradient (van den Oord et al. 2017; Razavi et al. 2019; Dhariwal et al. 2020; Lancucki et
          al. 2020; Yu et al. 2022)
  dkvb    discrete key-value bottleneck (Trauble et al. 2023): H fixed random projections of the standardised state,
          frozen keys per head from k-means on training mentions (label-free), one learned D-vector per key, averaged
          over heads
  gumbel  linear head (the naming head's architecture) -> Gumbel-softmax over the fixed random codebook rows the
          naming head snaps to, temperature annealed 2 -> 0.1 (Jang et al. 2017; Maddison et al. 2017); optional
          batch-entropy diversity term (Baevski et al. 2020); hard arg-max at evaluation (snapping)
  st      the same head, straight-through arg-max (forward one-hot, backward softmax; Bengio et al. 2013), optional
          diversity term; hard arg-max at evaluation
  cw      the binding-loss map with input ZCA whitening and/or code centring (global training mean, or the mean of
          the instance's own codes) or code ZCA whitening; statistics from all training mentions, refreshed every
          50 steps, no gradient through the statistics
  bind    the B31 binding-loss map unchanged (gate G0), read also with post-hoc centring (pc) and post-hoc ZCA (pz) of
          its codes (statistics of the training mentions at that epoch)
Diagnostics that use labels (never label-free, reported apart): dkvbor (keys = the gold surface form, kind + symbol;
learned values), and --idlam > 0 (the naming head's identity loss added to gumbel / st: positive control G1).

Train-time clean-up (A, C): hard = the published arg-max; st = the same forward pass, with the backward pass also
through softmax(cosine / 0.05) over the candidates, so that the answer loss reaches the codes on every hop. Evaluation
always uses the published hard chain.
"""
import argparse, json, math, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import b31_arms as B                                                       # noqa: E402

D = B.D
TAU_C = 0.05
PRIMARY = "perm/named"
FAMS = ["vq", "dkvb", "gumbel", "st", "cw", "bind", "dkvbor"]
DISCRETE = ("vq", "dkvb", "gumbel", "st", "dkvbor")


def kind_groups(arena, kinds):
    """(name, positions, row offset in the fixed table, rows) per kind; arena C: one group (shared 64 symbols)."""
    if arena == "C":
        return [("RF", list(range(len(kinds))), 0, 64)]
    out, off = [], 0
    for k in dict.fromkeys(B.KINDS[arena]):
        n = 64 if k == "R" else 256
        out.append((k, [i for i, kk in enumerate(kinds) if kk == k], off, n))
        off += n
    return out


def kmeans(X, k, iters, seed, cosine):
    """k-means++ initialisation and Lloyd iterations on X (n, d) (on X's device). cosine: spherical k-means."""
    g = torch.Generator().manual_seed(seed)
    if cosine:
        X = F.normalize(X, dim=-1)
    n = X.size(0)
    first = int(torch.randint(n, (1,), generator=g))
    C = [X[first]]
    d2 = ((X - X[first]) ** 2).sum(-1)
    for _ in range(1, k):
        p = (d2 / d2.sum().clamp(min=1e-12)).double().cpu()
        i = int(torch.multinomial(p, 1, generator=g)) if float(p.sum()) > 0 else int(torch.randint(n, (1,), generator=g))
        C.append(X[i])
        d2 = torch.minimum(d2, ((X - X[i]) ** 2).sum(-1))
    C = torch.stack(C)
    for _ in range(iters):
        if cosine:
            a = (X @ F.normalize(C, dim=-1).t()).argmax(1)
        else:
            a = torch.cdist(X, C).argmin(1)
        S_ = torch.zeros_like(C).index_add_(0, a, X)
        cnt = torch.bincount(a, minlength=k).float()
        nz = cnt > 0
        C[nz] = S_[nz] / cnt[nz, None]
    if cosine:
        C = F.normalize(C, dim=-1)
    return C


def zca(X, dev, rel_eps=1e-3, chunk=32768):
    """Mean and ZCA whitening matrix of the rows of X (n, d; any device), float64 in chunks on dev:
    W = U diag((lam + eps)^-1/2) U^T, eps = rel_eps * mean(lam)."""
    n, d = X.shape
    mu = torch.zeros(d, dtype=torch.float64, device=dev)
    for k in range(0, n, chunk):
        mu += X[k:k + chunk].to(dev).double().sum(0)
    mu /= n
    C = torch.zeros(d, d, dtype=torch.float64, device=dev)
    for k in range(0, n, chunk):
        xc = X[k:k + chunk].to(dev).double() - mu
        C += xc.t() @ xc
    C /= max(1, n - 1)
    lam, U = torch.linalg.eigh(C)
    lam = lam.clamp(min=0)
    eps = rel_eps * float(lam.mean())
    W = (U * (lam + eps).rsqrt()) @ U.t()
    return mu.float(), W.float()


class Arm42(B.Arm):
    def __init__(self, a, seed, hid, train_split, dev):
        base = {"vq": "bind", "cw": "bind", "bind": "bind", "gumbel": "name", "st": "name", "dkvb": "exact",
                "dkvbor": "exact"}[a.fam]
        super().__init__(a.arena, base, seed, hid)    # global RNG: (A: role vectors), then proj / head, as in B31
        self.fam, self.a = a.fam, a
        self.groups = kind_groups(a.arena, self.kinds)
        self.reading = "plain"
        self.to(dev)
        self.dev = dev
        Ftr = train_split.F                                                    # (N, M, hid) on CPU
        if a.fam in ("cw", "bind", "vq"):
            self.inw = bool(a.inw)
            if self.inw:
                mu, W = zca(Ftr.reshape(-1, hid), dev)
                self.register_buffer("in_mu", mu); self.register_buffer("in_W", W)
            self.register_buffer("z_mu", torch.zeros(D, device=dev)); self.register_buffer("z_W", torch.eye(D, device=dev))
        if a.fam == "vq":
            # z is centred by the training-mention mean (refreshed every 50 steps) before l2-normalisation, so that the
            # common direction of the frozen states does not make every code alike; codebook initialised by spherical
            # k-means++ on the centred, l2-normalised z of training mentions under the initial projection
            self.refresh_stats(Ftr, mean_only=True)
            for gi, (gname, pos, off, n) in enumerate(self.groups):
                with torch.no_grad():
                    Z = self._z_all(Ftr, pos, max_n=30000, seed=seed + 501 + gi)
                C = kmeans(Z, a.mult * n, 20, seed + 601 + gi, cosine=True)
                self.register_buffer("cbm%d" % gi, C.clone())
                self.register_buffer("cbn%d" % gi, torch.ones(C.size(0), device=dev))
            self.steps_since = 0
        if a.fam in ("dkvb", "dkvbor"):
            flat = Ftr.reshape(-1, hid)
            sub = flat[torch.randperm(flat.size(0), generator=torch.Generator().manual_seed(seed + 701))[:200000]].to(dev)
            self.register_buffer("f_mu", sub.mean(0)); self.register_buffer("f_sd", sub.std(0).clamp(min=1e-6))
            H = a.heads if a.fam == "dkvb" else 1
            self.H = H
            dh = 256 // H
            g = torch.Generator().manual_seed(seed + 9001)
            self.register_buffer("Rproj", (torch.randn(hid, H * dh, generator=g) / math.sqrt(hid)).to(dev))
            vals, self.key_off = [], []
            tot = 0
            for gi, (gname, pos, off, n) in enumerate(self.groups):
                if a.fam == "dkvb":
                    with torch.no_grad():
                        P_ = self._proj_all(Ftr, pos, max_n=30000, seed=seed + 801 + gi)   # (n, H, dh)
                    for h in range(H):
                        Ck = kmeans(P_[:, h], a.mult * n, 20, seed + 901 + 31 * gi + h, cosine=False)
                        self.register_buffer("key%d_%d" % (gi, h), Ck)
                    ng = a.mult * n
                else:
                    ng = 2 * n if a.arena == "C" else n          # surface forms: C hex roles and decimal fillers apart
                self.key_off.append(tot); tot += ng
            gv = torch.Generator().manual_seed(seed + 9101)
            self.values = nn.Parameter(F.normalize(torch.randn(H, tot, D, generator=gv), dim=-1).to(dev))

    # ---------------------------------------------------------------- helpers
    def _nrm(self, x):
        return B.l2n(x) if self.arena == "A" else F.normalize(x, dim=-1)

    def _fin(self, Fb):
        return (Fb - self.in_mu) @ self.in_W if getattr(self, "inw", False) else Fb

    def _z_all(self, Ftr, pos, max_n, seed):
        X = Ftr[:, pos].reshape(-1, Ftr.size(-1))
        if X.size(0) > max_n:
            X = X[torch.randperm(X.size(0), generator=torch.Generator().manual_seed(seed))[:max_n]]
        return F.normalize(self.proj(self._fin(X.to(self.dev))) - self.z_mu, dim=-1)

    def _proj_all(self, Ftr, pos, max_n, seed):
        X = Ftr[:, pos].reshape(-1, Ftr.size(-1))
        if X.size(0) > max_n:
            X = X[torch.randperm(X.size(0), generator=torch.Generator().manual_seed(seed))[:max_n]]
        X = (X.to(self.dev) - self.f_mu) / self.f_sd
        return (X @ self.Rproj).view(X.size(0), self.H, -1)

    @torch.no_grad()
    def refresh_stats(self, Ftr, bs=256, mean_only=False):
        """cw / bind / vq: mean (and ZCA) of z = proj(input) over ALL training mentions (no gradient)."""
        if mean_only:
            tot, n = torch.zeros(D, dtype=torch.float64, device=self.dev), 0
            for k in range(0, Ftr.size(0), bs):
                z = self.proj(self._fin(Ftr[k:k + bs].to(self.dev))).reshape(-1, D)
                tot += z.double().sum(0); n += z.size(0)
            self.z_mu.copy_((tot / n).float())
            return
        zs = []
        for k in range(0, Ftr.size(0), bs):
            zs.append(self.proj(self._fin(Ftr[k:k + bs].to(self.dev))).reshape(-1, D))
        mu, W = zca(torch.cat(zs), self.dev)
        self.z_mu.copy_(mu); self.z_W.copy_(W)

    # ---------------------------------------------------------------- codes
    def codes(self, Fb, sym, train, tau=1.0):
        """K codes (B, M, D) and aux: {'commit', 'div', 'id', 'assign' (B, M) int64 or None}."""
        aux = {"assign": None}
        fam = self.fam
        if fam in ("cw", "bind"):
            z = self.proj(self._fin(Fb))
            mode = self.a.code if fam == "cw" else "none"
            if not train and fam == "bind":
                mode = {"plain": "none", "pc": "center", "pz": "zca"}[self.reading]
            if mode == "center":
                z = z - self.z_mu
            elif mode == "inst":
                z = z - z.mean(1, keepdim=True)
            elif mode == "zca":
                z = (z - self.z_mu) @ self.z_W
            return self._nrm(z), aux
        if fam == "vq":
            zn = F.normalize(self.proj(self._fin(Fb)) - self.z_mu, dim=-1)
            K = torch.empty_like(zn)
            assign = torch.zeros(zn.shape[:2], dtype=torch.long, device=zn.device)
            commit = 0.0
            base = 0
            for gi, (gname, pos, off, n) in enumerate(self.groups):
                E = F.normalize(getattr(self, "cbm%d" % gi), dim=-1)
                zg = zn[:, pos]
                idx = (zg @ E.t()).argmax(-1)
                e = E[idx]
                assign[:, pos] = idx + base
                base += E.size(0)
                if train:
                    K[:, pos] = zg + (e - zg).detach()
                    commit = commit + ((zg - e.detach()) ** 2).sum(-1).mean() * len(pos)
                    self._ema(gi, zg.detach().reshape(-1, D), idx.reshape(-1))
                else:
                    K[:, pos] = e
            aux["commit"] = commit / zn.size(1) if train else None
            aux["assign"] = assign
            return K, aux
        if fam in ("dkvb", "dkvbor"):
            Bn, M = Fb.shape[:2]
            vals = torch.zeros(Bn, M, D, device=Fb.device)
            assign = torch.zeros(Bn, M, self.H, dtype=torch.long, device=Fb.device)
            if fam == "dkvb":
                Pp = (((Fb - self.f_mu) / self.f_sd) @ self.Rproj).view(Bn, M, self.H, -1)
            for gi, (gname, pos, off, n) in enumerate(self.groups):
                for h in range(self.H):
                    if fam == "dkvb":
                        Kh = getattr(self, "key%d_%d" % (gi, h))
                        x = Pp[:, pos, h].reshape(-1, Pp.size(-1))
                        idx = torch.cdist(x, Kh).argmin(1).view(Bn, len(pos))
                    else:
                        idx = sym[:, pos].clone()
                        if self.arena == "C":
                            isf = torch.tensor([k == "F" for k in self.kinds], device=Fb.device)[pos]
                            idx = idx + n * isf[None, :].long()
                    vals[:, pos] = vals[:, pos] + self.values[h, self.key_off[gi] + idx]
                    assign[:, pos, h] = idx
            aux["assign"] = assign                       # group-local key index per head
            return self._nrm(vals / self.H), aux
        # gumbel / st: the naming head's logits over the fixed table rows of the mention's kind
        lg = self.logits_id(Fb)
        K = torch.empty(Fb.size(0), Fb.size(1), D, device=Fb.device)
        assign = torch.zeros(Fb.shape[:2], dtype=torch.long, device=Fb.device)
        div = 0.0
        for gi, (gname, pos, off, n) in enumerate(self.groups):
            T = self.table[off:off + n]
            lgg = lg[:, pos, :n]
            idx = lgg.argmax(-1)
            assign[:, pos] = idx + off
            if train:
                if fam == "gumbel":
                    u = torch.rand_like(lgg).clamp(1e-10, 1 - 1e-10)
                    y = torch.softmax((lgg - torch.log(-torch.log(u))) / tau, -1)
                else:
                    p = torch.softmax(lgg, -1)
                    y = F.one_hot(idx, n).float() + p - p.detach()
                K[:, pos] = F.normalize(y @ T, dim=-1)
                pb = torch.softmax(lgg, -1).reshape(-1, n).mean(0)
                div = div + (1.0 - (-(pb * torch.log(pb + 1e-12)).sum()) / math.log(n))
            else:
                K[:, pos] = T[idx]
        aux["assign"] = assign
        if train:
            aux["div"] = div / len(self.groups)
            if self.a.idlam > 0:
                aux["id"] = self.id_loss(Fb, sym, list(range(len(self.kinds))))
        return K, aux

    @torch.no_grad()
    def _ema(self, gi, z, idx, gamma=0.99):
        m, nn_ = getattr(self, "cbm%d" % gi), getattr(self, "cbn%d" % gi)
        k = m.size(0)
        cnt = torch.bincount(idx, minlength=k).float()
        s = torch.zeros_like(m).index_add_(0, idx, z)
        nn_.mul_(gamma).add_(cnt, alpha=1 - gamma)
        m.mul_(gamma).add_(s, alpha=1 - gamma)
        self._last_batch = getattr(self, "_last_batch", {}); self._last_batch[gi] = z

    @torch.no_grad()
    def dead_restart(self, gen, thr=0.1):
        """Codes whose EMA usage per batch fell below thr are reset to random encoder outputs of the current batch."""
        n_dead = 0
        for gi in range(len(self.groups)):
            m, nn_ = getattr(self, "cbm%d" % gi), getattr(self, "cbn%d" % gi)
            dead = torch.nonzero(nn_ < thr).flatten()
            z = self._last_batch.get(gi)
            if dead.numel() == 0 or z is None:
                continue
            pick = torch.randint(z.size(0), (dead.numel(),), generator=gen).to(z.device)
            m[dead] = z[pick]; nn_[dead] = 1.0
            n_dead += dead.numel()
        return n_dead

    def kcodes(self, Fb, sym, conv):          # evaluation codes (used by b31_arms.geometry)
        return self.codes(Fb, sym, train=False)[0]

    # ---------------------------------------------------------------- chain with optional straight-through clean-up
    def chain42(self, K, enc, cleanup):
        if cleanup == "hard" or self.arena == "B":
            return self.chain(K, enc)
        V = K[..., self.P] if enc == "perm" else K
        Bn = K.size(0)
        ar = torch.arange(Bn, device=K.device)
        wins = []
        if self.arena == "A":
            key = K[ar, self.q]
            for h, (kp, vp) in enumerate(self.hops):
                kp = torch.tensor(kp, device=K.device); vp = torch.tensor(vp, device=K.device)
                Kk, Vv = K[:, kp], V[:, vp]
                s_ = B.bindA(Vv, Kk).sum(1)
                mem = s_ / (s_.norm(dim=-1, keepdim=True) + 1e-8)
                hat = B.l2n(B.unbindA(mem, key))
                if h < len(self.hops) - 1:
                    sims = B.simA(Vv, hat[:, None, :].expand_as(Vv))
                    w = sims.argmax(1)
                    wins.append(vp[w])
                    soft = (torch.softmax(sims / TAU_C, 1)[..., None] * K[:, vp]).sum(1)
                    key = K[ar, vp[w]] + soft - soft.detach()
                else:
                    score = B.simA(hat, V[:, self.c]).clamp(-1, 1)
            return score, torch.stack(wins, 1), torch.ones(Bn, dtype=torch.bool, device=K.device)
        key = K[:, self.q]
        for kp, vp in self.hops:
            kp = torch.tensor(kp, device=K.device); vp = torch.tensor(vp, device=K.device)
            mem = B.bindC(V[:, vp], K[:, kp]).sum(1)
            hat = F.normalize(B.unbindC(mem, key), dim=-1)
            cs = F.cosine_similarity(V[:, vp], hat[:, None, :], dim=-1)
            w = cs.argmax(1)
            wins.append(vp[w])
            soft = (torch.softmax(cs / TAU_C, 1)[..., None] * K[:, vp]).sum(1)
            key = K[ar, vp[w]] + soft - soft.detach()
        score = (key * K[:, self.c]).sum(-1).clamp(-1, 1)
        return score, torch.stack(wins, 1), torch.ones(Bn, dtype=torch.bool, device=K.device)


# ------------------------------------------------------------------ discrete statistics (labels group, never train)

def discrete_stats(assign, sym, kinds, arena):
    """Mention consistency (mentions on their symbol's modal code), purity (mentions on their code's modal symbol),
    codes used; arena C also the alias rate (symbols whose modal code as a hex role equals that as a decimal filler)."""
    N_, M_ = assign.shape[:2]
    kk = [kinds[j] for j in range(M_)] * N_
    if assign.ndim == 3:                          # dkvb: the H-tuple of keys within the mention's kind group
        A_ = [(k if arena != "C" else "", tuple(r)) for k, r in zip(kk, assign.reshape(N_ * M_, -1).tolist())]
    else:
        A_ = assign.reshape(-1).tolist()
    S_ = sym.reshape(-1).tolist()
    gold = [s if arena == "C" else (k, s) for k, s in zip(kk, S_)]
    from collections import Counter, defaultdict
    by_s, by_c = defaultdict(Counter), defaultdict(Counter)
    for g, c in zip(gold, A_):
        by_s[g][c] += 1; by_c[c][g] += 1
    tot = len(A_)
    out = {"consistency": sum(max(v.values()) for v in by_s.values()) / tot,
           "purity": sum(max(v.values()) for v in by_c.values()) / tot,
           "codes_used": len(by_c), "symbols": len(by_s)}
    if arena == "C":
        surf = defaultdict(Counter)
        for k, s, c in zip(kk, S_, A_):
            surf[(k, s)][c] += 1
        out["surface_consistency"] = sum(max(v.values()) for v in surf.values()) / tot
        both = [s for s in set(S_) if ("R", s) in surf and ("F", s) in surf]
        out["alias_rate"] = float(np.mean([surf[("R", s)].most_common(1)[0][0] == surf[("F", s)].most_common(1)[0][0]
                                           for s in both])) if both else None
    return out


# ------------------------------------------------------------------ run

def readings_of(a):
    return ["plain", "pc", "pz"] if a.fam == "bind" else ["plain"]


def tag_of(rd):
    return PRIMARY if rd == "plain" else "%s+%s" % (PRIMARY, rd)


def run(a, seed, D_, dev):
    torch.manual_seed(seed)
    tr = D_["train"]
    hid = tr.F.size(-1)
    m = Arm42(a, seed, hid, tr, dev)
    arena = a.arena
    sc = [p for n, p in m.named_parameters() if p.requires_grad and n in ("scale", "bias")]
    rest = [p for n, p in m.named_parameters() if p.requires_grad and n not in ("scale", "bias")]
    lr = a.lr
    if arena == "A":
        opt = torch.optim.AdamW(rest, lr=lr)
    else:
        opt = torch.optim.AdamW(([{"params": rest, "lr": lr}] if rest else []) + ([{"params": sc, "lr": 1e-1}] if sc else []))
    params = [p for p in m.parameters() if p.requires_grad]
    bs = B.BS[arena]
    nsteps = a.epochs * math.ceil(tr.n / bs)
    gen = torch.Generator(); gen.manual_seed(seed)
    rgen = torch.Generator(); rgen.manual_seed(seed + 3)

    def tau_at(step):
        f = min(1.0, step / max(1, 0.7 * nsteps))
        return a.tau0 * (a.tau_end / a.tau0) ** f

    def scores(Dsp, want_assign=False):
        m.eval(); out, wins, asg = [], [], []
        with torch.no_grad():
            for k in range(0, Dsp.n, 64):
                Fb = Dsp.F[k:k + 64].to(dev); sy = Dsp.sym[k:k + 64].to(dev)
                K, aux = m.codes(Fb, sy, train=False)
                s_, w, ok = m.chain(K, "perm")
                out.append(s_.cpu()); wins.append(w.cpu())
                if want_assign and aux["assign"] is not None:
                    asg.append(aux["assign"].cpu())
        return torch.cat(out).numpy(), torch.cat(wins), (torch.cat(asg) if asg else None)

    def evaluate(ep):
        e = {"epoch": ep}
        if a.fam == "bind" and any(r != "plain" for r in readings_of(a)):
            m.refresh_stats(tr.F)                         # post-hoc statistics of this epoch's codes
        snap = {}
        for rd in readings_of(a):
            m.reading = rd
            st_, _, _ = scores(tr)
            yt = tr.y.numpy()
            for sp in [s for s in ("dev", "test") if s in D_]:
                Dsp = D_[sp]
                se, wi, asg = scores(Dsp, want_assign=True)
                acc, thr, tracc = B.thr_acc(st_, yt, se, Dsp.y.numpy())
                r = {"acc": acc, "thr": thr, "train_acc": tracc}
                if sp == "test" or a.save_dev_scores:
                    r["scores"] = [round(float(x), 6) for x in se]
                    r["wins"] = wi.tolist()
                if sp == "test" or "test" not in D_:
                    r["geom"] = B.geometry(m, Dsp.F[:120].to(dev), Dsp.sym[:120].to(dev), "named")
                    if asg is not None and rd == "plain":
                        r["discrete"] = discrete_stats(asg.numpy(), Dsp.sym.numpy(), m.kinds, arena)
                if sp == "test" and rd == "plain" and asg is not None:
                    snap["assign_test"] = asg.to(torch.int64)
                e["%s:%s" % (tag_of(rd), sp)] = r
        m.reading = "plain"
        if snap and a.mode == "final":
            if a.fam == "vq":
                snap["codebook"] = [F.normalize(getattr(m, "cbm%d" % gi), dim=-1).cpu().clone() for gi in range(len(m.groups))]
            if a.fam in ("dkvb", "dkvbor"):
                # clone: on a CPU run .cpu() returns the live parameter itself, which later epochs would overwrite
                snap["values"] = m.values.detach().cpu().clone(); snap["key_off"] = list(m.key_off)
            if a.fam in ("gumbel", "st"):
                snap["table"] = m.table.cpu().clone()
        return e, snap

    rec = {"seed": seed, "epochs": []}
    snaps = []
    t0 = time.time()
    step = 0
    for ep in range(a.epochs):
        m.train()
        perm = torch.randperm(tr.n, generator=gen)
        tl, cl, dl, nb, ndead = 0.0, 0.0, 0.0, 0, 0
        for k in range(0, tr.n, bs):
            if a.fam in ("cw",) and a.code in ("center", "zca") and step % 50 == 0:
                m.refresh_stats(tr.F)
            if a.fam == "vq" and step % 50 == 0 and step > 0:
                m.refresh_stats(tr.F, mean_only=True)
            i = perm[k:k + bs]
            Fb = tr.F[i].to(dev); sy = tr.sym[i].to(dev); y = tr.y[i].to(dev)
            K, aux = m.codes(Fb, sy, train=True, tau=tau_at(step))
            s_, _, _ = m.chain42(K, "perm", a.cleanup)
            loss = F.cross_entropy(m.answer_logits(s_), y)
            tl += float(loss); nb += 1
            if a.fam == "vq":
                loss = loss + a.beta * aux["commit"]; cl += float(aux["commit"])
            if a.fam in ("gumbel", "st"):
                if a.div > 0:
                    loss = loss + a.div * aux["div"]
                dl += float(aux["div"])
                if a.idlam > 0:
                    loss = loss + a.idlam * aux["id"]
            if params:
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
            step += 1
            if a.fam == "vq" and step % 50 == 0:
                ndead += m.dead_restart(rgen)
        if a.fam == "cw" and a.code in ("center", "zca"):
            m.refresh_stats(tr.F)
        e, snap = evaluate(ep + 1)
        e.update({"task_loss": tl / nb, "commit": cl / nb if a.fam == "vq" else None,
                  "div": dl / nb if a.fam in ("gumbel", "st") else None, "dead_restarts": ndead,
                  "tau": tau_at(step) if a.fam == "gumbel" else None, "secs": round(time.time() - t0, 1)})
        rec["epochs"].append(e); snaps.append(snap)
        log(arena, a.fam, seed, e)
    # dev-selected epoch of the primary reading: keep its test assignments and codebook for the second scorer
    accs = [ep_["%s:dev" % PRIMARY]["acc"] for ep_ in rec["epochs"]]
    j = max(range(len(accs)), key=lambda t: (accs[t], -t))
    rec["dev_selected_epoch"] = j + 1
    return rec, snaps[j]


def log(arena, fam, seed, e):
    parts = []
    for k in sorted(k for k in e if ":" in k):
        r = e[k]
        g = r.get("geom")
        dsc = r.get("discrete")
        parts.append("%s %.4f (tr %.3f%s%s)" % (k, r["acc"], r["train_acc"],
                                               ", geom %.3f/%.3f" % (g["different"], g["same"]) if g and g["same"] is not None else "",
                                               ", cons %.3f pur %.3f%s" % (dsc["consistency"], dsc["purity"],
                                                                          " alias %.2f" % dsc["alias_rate"] if dsc.get("alias_rate") is not None else "")
                                               if dsc else ""))
    print("  %s %s seed %d ep%d loss %.4f | %s (%.0fs)" % (arena, fam, seed, e["epoch"], e["task_loss"], " | ".join(parts),
                                                       e.get("secs", 0)), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enc_model", required=True, choices=["phi2", "qwen8"])
    ap.add_argument("--arena", required=True, choices=["A", "B", "C"])
    ap.add_argument("--fam", required=True, choices=FAMS)
    ap.add_argument("--mode", required=True, choices=["tune", "final"])
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--cleanup", default="hard", choices=["hard", "st"])
    ap.add_argument("--mult", type=int, default=1, help="vq / dkvb: codes (keys) per symbol the naming head separates")
    ap.add_argument("--beta", type=float, default=0.25)
    ap.add_argument("--heads", type=int, default=1)
    ap.add_argument("--div", type=float, default=0.0)
    ap.add_argument("--tau0", type=float, default=2.0)
    ap.add_argument("--tau_end", type=float, default=0.1)
    ap.add_argument("--inw", type=int, default=0)
    ap.add_argument("--code", default="none", choices=["none", "center", "inst", "zca"])
    ap.add_argument("--idlam", type=float, default=0.0, help="> 0 only for the labelled positive control G1")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--save_dev_scores", action="store_true")
    ap.add_argument("--feats", default="/path/to/work/b31/feats")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.fam in ("gumbel", "st"):
        assert a.idlam == 0 or "g1" in os.path.basename(a.out), "identity loss only in the G1 control"
    if os.path.exists(a.out):
        print("exists, skip", a.out); return
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    splits = {"tune": ["train", "dev"], "final": ["train", "dev", "test"]}[a.mode]
    D_ = {sp: B.Split(a.enc_model, a.arena, sp, a.feats) for sp in splits}
    for sp in D_:                                 # the frozen features live on the GPU (no numeric change)
        D_[sp].F = D_[sp].F.to(a.device)
    out = {"args": vars(a), "host": os.uname()[1],
           "gpu": torch.cuda.get_device_name(a.device) if a.device.startswith("cuda") else "cpu (%d threads)" % torch.get_num_threads(),
           "start": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "runs": []}
    side = {}
    for s in a.seeds:
        rec, snap = run(a, s, D_, a.device)
        out["runs"].append(rec)
        if snap:
            side[s] = snap
        json.dump(out, open(a.out + ".part", "w"))
    out["end"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    if side and a.mode == "final":
        torch.save(side, a.out.replace(".json", ".side.pt"))
    json.dump(out, open(a.out + ".tmp", "w"))
    os.replace(a.out + ".tmp", a.out)
    if os.path.exists(a.out + ".part"):
        os.remove(a.out + ".part")


if __name__ == "__main__":
    main()
