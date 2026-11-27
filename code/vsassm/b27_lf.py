"""B27 label-free arms (prereg/B27_arena_referee_baselines_prereg.md): the projection trained through the binding
loss, exactly as published in each arena (v8.SpanVSA on A, v11.BindArm "spans" on B, v13.BindArm "spans" on C),
plus an anti-collapse term that uses NO identity label:

  orth  mean squared cosine between the codes of every pair of different mentions in an instance
  vic   VICReg-style variance hinge + covariance penalty over the codes of all mentions in the batch
  nce   InfoNCE between the code of each mention and the code of the same mention in a second rendering of the
        same instance with the pair order reshuffled (positives come from the augmentation, not from labels)

Option qc (arenas B and C): "index" = the query role / candidate enter as codebook rows, as in the published arm;
"proj" = they enter through the same projection, read from their own token spans, so no codebook row and no
identity reaches the store at all. In arena A the published arm already projects the query role; the final
filler hop uses codebook rows for every arm (including the naming head), so A has no qc option.

Training follows the original loops line by line (same optimiser groups, batch size, clipping, RNG order, closed-
form threshold on the training instances for B/C, learned scale/bias + argmax for A); the only additions are the
regulariser term, more epochs, a dev split and per-epoch logging. With reg "none" and qc "index" the code path is
the published arm (checked by mode g0 against the banked numbers).

Modes: tune (train + dev only; the test split is never built), final (train + dev + test), g0 (published config,
2 epochs, train + test, reg none).
"""
import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import b27_common as C                                                  # noqa: E402

M = C.load_arena_modules()
V13, V11, V8 = M["v13"], M["v11"], M["v8"]

BANKED_SPANS = {"A": {42: 0.511, 123: 0.499, 456: 0.493},             # v8_logs/v8.json
                "B": {42: 0.502, 123: 0.498, 456: 0.522},             # v11_logs/v18b_spans.json
                "C": {42: 0.472, 123: 0.502, 456: 0.476}}             # v13_logs/v13_fixed.json


# ------------------------------------------------------------------ regularisers (no identity labels anywhere)

def reg_orth(per_instance):
    t = []
    for c in per_instance:
        n = c.size(0)
        if n < 2:
            continue
        G = c @ c.t()
        t.append((G[~torch.eye(n, dtype=torch.bool, device=c.device)] ** 2).mean())
    return torch.stack(t).mean()


def reg_vic(codes, cov_w=0.04):
    """VICReg variance + covariance on the unit codes scaled by sqrt(d), so that isotropic codes have unit
    per-dimension variance (gamma = 1). The variance hinge also forbids a shared mean direction: the total variance
    of unit vectors is 1 - |mean|^2, so per-dimension variance >= 1/d everywhere forces the mean to zero."""
    N, d = codes.shape
    u = codes * (d ** 0.5)
    u = u - u.mean(0, keepdim=True)
    std = torch.sqrt(u.var(0) + 1e-4)
    v = F.relu(1.0 - std).mean()
    cov = (u.t() @ u) / (N - 1)
    off = cov.pow(2).sum() - cov.diagonal().pow(2).sum()
    return v + cov_w * off / d


def reg_nce(c1, c2, tau):
    lg = c1 @ c2.t() / tau
    y = torch.arange(c1.size(0), device=c1.device)
    return 0.5 * (F.cross_entropy(lg, y) + F.cross_entropy(lg.t(), y))


# ------------------------------------------------------------------ arms

class LFArmC(V13.BindArm):
    def __init__(self, seed, qc="index"):
        super().__init__("spans", seed)
        self.qc = qc

    def codes(self, Hb, spans):
        return F.normalize(self.proj(torch.stack([self.sf(Hb, sp) for sp in spans])), dim=-1)

    def sims(self, H, rows, device):
        if self.qc == "index":
            return super().sims(H, rows, device)
        out = torch.zeros(len(rows), device=device)
        for b, r in enumerate(rows):
            if any(len(l) == 0 for l in r["ctx"]):
                continue
            key = self.codes(H[b], [r["q_span"]])[0]
            for L in range(V13.HOPS):
                mem = torch.stack([V13.bind(self.vec(H[b], bb, spf, device), self.vec(H[b], aa, spr, device))
                                   for aa, bb, spr, spf in r["ctx"][L]], 0).sum(0)
                hat = F.normalize(V13.unbind(mem, key), dim=-1)
                cands = torch.stack([self.vec(H[b], bb, spf, device) for aa, bb, spr, spf in r["ctx"][L]], 0)
                key = cands[int(F.cosine_similarity(cands, hat.unsqueeze(0)).argmax().item())]
            out[b] = torch.dot(key, self.codes(H[b], [r["cand_span"]])[0]).clamp(-1, 1)
        return out

    def mention_spans(self, r):
        out = []
        for L in range(V13.HOPS):
            for aa, bb, spr, spf in r["ctx"][L]:
                out += [((L, aa, "r"), spr), ((L, aa, "f"), spf)]
        if self.qc == "proj":
            out += [(("q",), r["q_span"]), (("cand",), r["cand_span"])]
        return out

    def symbol_mentions(self, r):          # for the cross-talk measurement only (labels group, never train)
        return [(sid, sp) for L in range(V13.HOPS) for aa, bb, spr, spf in r["ctx"][L]
                for sid, sp in ((aa, spr), (bb, spf))]


class LFArmB(V11.BindArm):
    def __init__(self, cfg, seed, qc="index"):
        super().__init__("spans", cfg, seed)
        self.qc = qc

    def codes(self, Hb, spans):
        return F.normalize(self.proj(torch.stack([self.span_feat(Hb, s, e) for s, e in spans])), dim=-1)

    def sims(self, H, rows, device):
        if self.qc == "index":
            return super().sims(H, rows, device)
        out = torch.zeros(len(rows), device=device)
        for b, r in enumerate(rows):
            if not r["ctx"]:
                continue
            bound = [V11.bind(self.filler_vec(H[b], f, sp, device), self.role_v[rr].to(device))
                     for rr, f, sp in r["ctx"]]
            mem = torch.stack(bound, 0).sum(0)
            f_hat = F.normalize(V11.unbind(mem, self.role_v[r["q_role"]].to(device)), dim=-1)
            out[b] = torch.dot(f_hat, self.codes(H[b], [r["cand_span"]])[0]).clamp(-1, 1)
        return out

    def mention_spans(self, r):
        out = [((rr,), sp) for rr, f, sp in r["ctx"]]
        if self.qc == "proj":
            out.append((("cand",), r["cand_span"]))
        return out

    def symbol_mentions(self, r):
        return [(f, sp) for rr, f, sp in r["ctx"]]


class LFArmA(V8.SpanVSA):
    """Forward is v8.SpanVSA.forward unchanged; this class only adds the mention list used by the regulariser."""

    def codes(self, Hb, spans):
        feats = torch.stack([(Hb[s:e] if e > s else Hb[s:s + 1]).float().mean(0) for s, e in spans])
        return F.normalize(self.proj(feats), dim=-1)

    def mention_list(self, ids_1d):
        """Every span whose projection enters v8.SpanVSA.forward, keyed by (section, kind, index). The index is
        read from the string only to pair a mention with the same mention in the reshuffled view (bookkeeping of
        the augmentation, equivalent to tracking the shuffle); it never reaches the projection or the loss."""
        toks = self.tokenizer.convert_ids_to_tokens(ids_1d.tolist())
        ms = sorted(self._scan_markers(ids_1d), key=lambda x: x[2])
        out = []
        for kind, idx, s, e in reversed(ms):
            if kind == "R":
                out.append((("Q", "R", idx), (s, e))); break
        for i in range(len(ms) - 1):
            k1, i1, s1, e1 = ms[i]
            k2, i2, s2, e2 = ms[i + 1]
            if k1 == "R" and k2 == "X":
                out += [(("A", "R", i1), (s1, e1)), (("A", "X", i2), (s2, e2))]
            elif k1 == "X" and k2 == "Y":
                out += [(("B", "X", i1), (s1, e1)), (("B", "Y", i2), (s2, e2))]
        for kind, idx, s, e in ms:
            if kind == "Y" and e < len(toks) and toks[e].lstrip("Ġ") == "{":
                out.append((("C", "Y", idx), (s, e)))
        return out


# ------------------------------------------------------------------ regulariser and geometry on a batch

def reg_value(arena, m, kind, tau, H, rows, H2=None, rows2=None):
    per, keys = [], []
    for b in range(H.size(0)):
        ms = m.mention_list(rows[b]) if arena == "A" else m.mention_spans(rows[b])
        keys.append([k for k, _ in ms])
        per.append(m.codes(H[b], [sp for _, sp in ms]))
    if kind == "orth":
        return reg_orth(per)
    if kind == "vic":
        return reg_vic(torch.cat(per))
    c1, c2 = [], []
    for b in range(H.size(0)):
        ms2 = dict(m.mention_list(rows2[b]) if arena == "A" else m.mention_spans(rows2[b]))
        keep = [j for j, k in enumerate(keys[b]) if k in ms2]
        c1.append(per[b][keep])
        c2.append(m.codes(H2[b], [ms2[keys[b][j]] for j in keep]))
    return reg_nce(torch.cat(c1), torch.cat(c2), tau)


@torch.no_grad()
def geometry(arena, m, H, rows, dev, n_inst=120, per_sym=4, n_sym=40):
    """V-165 protocol (crosstalk_audit2): first 120 instances, up to 4 occurrences per symbol, first 40 symbols;
    mean cosine between different symbols (first occurrence each) and between occurrences of the same symbol.
    Symbol labels group the vectors for this measurement only."""
    by = {}
    for b in range(min(n_inst, H.size(0))):
        Hb = H[b].to(dev)
        if arena == "A":
            ml = [((k[1], k[2]), sp) for k, sp in m.mention_list(rows[b]) if k[0] != "Q"]
        else:
            ml = m.symbol_mentions(rows[b])
        need = [(sid, sp) for sid, sp in ml if len(by.get(sid, [])) < per_sym]
        if not need:
            continue
        cs = m.codes(Hb, [sp for _, sp in need])
        for (sid, _), c in zip(need, cs):
            if len(by.setdefault(sid, [])) < per_sym:
                by[sid].append(c)
    ks = [k for k in by if by[k]][:n_sym]
    first = torch.stack([by[k][0] for k in ks])
    G = first @ first.t()
    diff = G[~torch.eye(len(ks), dtype=torch.bool, device=G.device)].mean().item()
    same = [float(torch.dot(by[k][i], by[k][j])) for k in ks for i in range(len(by[k]))
            for j in range(i + 1, len(by[k]))]
    return {"different": diff, "same": float(np.mean(same)) if same else None, "symbols": len(ks)}


# ------------------------------------------------------------------ runs

def thr_acc(st, yt, se, ye):
    """Closed-form threshold exactly as v13.run / v11.run: the training-accuracy-maximising candidate."""
    c = np.unique(st)
    if c.size == 0:
        return 0.5, None, 0.5
    accs = [(((st >= t) == (yt == 1)).mean()) for t in c]
    i = int(np.argmax(accs))
    thr = float(c[i])
    return float((((se >= thr) == (ye == 1)).mean())), thr, float(accs[i])


def run_bc(arena, seed, D, a, dev, mode):
    torch.manual_seed(seed)
    if arena == "C":
        m = LFArmC(seed, a.qc).to(dev)
    else:
        m = LFArmB(M["systematic_role_binding_dataset"].SystematicConfig(), seed, a.qc).to(dev)
    sc = [q for n, q in m.named_parameters() if q.requires_grad and n in ("scale", "bias")]
    rest = [q for n, q in m.named_parameters() if q.requires_grad and n not in ("scale", "bias")]
    opt = torch.optim.AdamW(([{"params": rest, "lr": a.lr}] if rest else []) +
                            ([{"params": sc, "lr": 1e-1}] if sc else []))
    params = [p for p in m.parameters() if p.requires_grad]
    bs = 4
    tr, Htr = D["train"]

    def collect(rows, Hall):
        m.eval(); S, Y = [], []
        with torch.no_grad():
            for k in range(0, len(rows), bs):
                S.append(m.sims(Hall[k:k + bs].to(dev), rows[k:k + bs], dev).cpu())
                Y += [r["label"] for r in rows[k:k + bs]]
        return torch.cat(S).numpy(), np.array(Y)

    rec = {"epochs": []}
    g = torch.Generator(); g.manual_seed(seed)
    t0 = time.time()
    for ep in range(a.epochs):
        m.train()
        perm = torch.randperm(len(tr), generator=g).tolist()
        tl, rl, nb = 0.0, 0.0, 0
        for k in range(0, len(perm), bs):
            sel = perm[k:k + bs]; rows = [tr[i] for i in sel]
            H = Htr[sel].to(dev)
            am = torch.stack([r["am"] for r in rows]).to(dev)
            y = torch.tensor([r["label"] for r in rows], device=dev)
            loss = F.cross_entropy(m(H, am, rows, dev), y)
            tl += float(loss); nb += 1
            if a.reg != "none":
                if a.reg == "nce":
                    tr2, Htr2 = D["train_v2"]
                    rv = reg_value(arena, m, "nce", a.tau, H, rows, Htr2[sel].to(dev), [tr2[i] for i in sel])
                else:
                    rv = reg_value(arena, m, a.reg, a.tau, H, rows)
                rl += float(rv)
                loss = loss + a.lam * rv
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        st, yt = collect(tr, Htr)
        e = {"epoch": ep + 1, "task_loss": tl / nb, "reg": rl / nb if a.reg != "none" else None,
             "secs": round(time.time() - t0, 1)}
        for split in [s for s in ("dev", "test") if s in D]:
            rows_s, H_s = D[split]
            se, ye = collect(rows_s, H_s)
            acc, thr, tracc = thr_acc(st, yt, se, ye)
            e[split] = acc; e["thr"] = thr; e["train_acc"] = tracc
            if mode == "final":
                e[split + "_scores"] = [round(float(x), 6) for x in se]
            gsplit = "test" if mode == "final" else "dev"
            if split == gsplit and (mode == "final" or ep + 1 in (1, a.epochs) or a.geom_every):
                m.eval(); e["geom_" + split] = geometry(arena, m, H_s, rows_s, dev)
        rec["epochs"].append(e)
        print("    %s ep%d task %.4f reg %s train %.4f %s (%.0fs)" % (
            arena, ep + 1, e["task_loss"], "-" if e["reg"] is None else "%.4f" % e["reg"], e["train_acc"],
            " ".join("%s %.4f" % (s, e[s]) for s in ("dev", "test") if s in e), e["secs"]), flush=True)
    return rec


def run_a(seed, D, a, dev, mode):
    torch.manual_seed(seed)
    m = LFArmA(vsa_dim=2048, num_roles=64, vocab_size=256, seed=seed).to(dev)
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr)
    bs = 8
    TR = D["train"]

    def scores(T):
        m.eval(); S = []
        with torch.no_grad():
            for k in range(0, T[0].size(0), bs):
                lg = m(T[0][k:k + bs].to(dev), T[1][k:k + bs].to(dev), hidden=T[2][k:k + bs].to(dev))
                S.append((lg[:, 1] - lg[:, 0]).cpu())
        return torch.cat(S).numpy()

    def acc(T, s):     # v8: argmax of the two logits; ties (both 0, unparsable rows) go to class 0
        return float(((s > 0).astype(int) == T[3].numpy()).mean())

    rec = {"epochs": []}
    g = torch.Generator(); g.manual_seed(seed)
    t0 = time.time()
    for ep in range(a.epochs):
        m.train()
        perm = torch.randperm(TR[0].size(0), generator=g)
        tl, rl, nb = 0.0, 0.0, 0
        for k in range(0, TR[0].size(0), bs):
            i = perm[k:k + bs]
            H = TR[2][i].to(dev)
            lg = m(TR[0][i].to(dev), TR[1][i].to(dev), hidden=H)
            loss = F.cross_entropy(lg, TR[3][i].to(dev))
            tl += float(loss); nb += 1
            if a.reg != "none":
                ids = [TR[0][j] for j in i.tolist()]
                if a.reg == "nce":
                    T2 = D["train_v2"]
                    rv = reg_value("A", m, "nce", a.tau, H, ids, T2[2][i].to(dev), [T2[0][j] for j in i.tolist()])
                else:
                    rv = reg_value("A", m, a.reg, a.tau, H, ids)
                rl += float(rv)
                loss = loss + a.lam * rv
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        e = {"epoch": ep + 1, "task_loss": tl / nb, "reg": rl / nb if a.reg != "none" else None}
        e["train_acc"] = acc(TR, scores(TR))
        for split in [s for s in ("dev", "test") if s in D]:
            T = D[split]
            s = scores(T)
            e[split] = acc(T, s)
            if mode == "final":
                e[split + "_scores"] = [round(float(x), 6) for x in s]
            gsplit = "test" if mode == "final" else "dev"
            if split == gsplit and (mode == "final" or ep + 1 in (1, a.epochs) or a.geom_every):
                m.eval(); e["geom_" + split] = geometry("A", m, T[2], [T[0][j] for j in range(T[0].size(0))], dev)
        e["secs"] = round(time.time() - t0, 1)
        rec["epochs"].append(e)
        print("    A ep%d task %.4f reg %s train %.4f %s (%.0fs)" % (
            ep + 1, e["task_loss"], "-" if e["reg"] is None else "%.4f" % e["reg"], e["train_acc"],
            " ".join("%s %.4f" % (s, e[s]) for s in ("dev", "test") if s in e), e["secs"]), flush=True)
    return rec


def load_feats(root, arena, splits):
    """Rows (for B/C), and cached states. The rows are rebuilt from the generators (cheap, deterministic) and
    checked against the cached token ids, so a stale cache cannot be paired with different rows."""
    from b27_common import build_rows, feats_path
    tok = C.tokenizer()
    D = {}
    for sp in splits:
        v2 = sp.endswith("_v2")
        base = sp[:-3] if v2 else sp
        obj = torch.load(feats_path(root, arena, base, v2))
        rows = build_rows(M, tok, arena, base, v2)
        if arena == "A":
            ids = torch.stack([it["input_ids"] for it in rows.items])
            assert torch.equal(ids, obj[0]), "cache/rows mismatch A %s" % sp
            D[sp] = obj
        else:
            assert len(rows) == obj.size(0)
            assert all(torch.equal(r["ids"], rows[j]["ids"]) for j, r in enumerate(rows))
            D[sp] = (rows, obj)
    return D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arena", required=True, choices=["A", "B", "C"])
    ap.add_argument("--mode", required=True, choices=["tune", "final", "g0"])
    ap.add_argument("--reg", default="none", choices=["none", "orth", "vic", "nce"])
    ap.add_argument("--lam", type=float, default=0.0)
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--qc", default="index", choices=["index", "proj"])
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--feats", default="/path/to/work/b27/feats")
    ap.add_argument("--geom_every", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.arena == "A" and a.qc != "index":
        raise SystemExit("arena A has no qc option")
    if a.mode == "g0":
        a.reg, a.qc, a.lr, a.epochs = "none", "index", 3e-4, 2
    splits = {"tune": ["train", "dev"], "final": ["train", "dev", "test"], "g0": ["train", "test"]}[a.mode]
    if a.reg == "nce":
        splits.append("train_v2")
    if os.path.exists(a.out):
        print("exists, skip", a.out); return
    D = load_feats(a.feats, a.arena, splits)
    out = {"args": vars(a), "host": os.uname()[1], "gpu": torch.cuda.get_device_name(a.device),
           "start": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "runs": []}
    for s in a.seeds:
        t0 = time.time()
        rec = run_a(s, D, a, a.device, a.mode) if a.arena == "A" else run_bc(a.arena, s, D, a, a.device, a.mode)
        rec["seed"], rec["secs"] = s, round(time.time() - t0, 1)
        if a.mode == "g0":
            best = max(e["test"] for e in rec["epochs"])
            rec["g0_best"], rec["g0_banked"] = best, BANKED_SPANS[a.arena][s]
            print("  G0 %s seed %d best test %.4f banked %.3f" % (a.arena, s, best, BANKED_SPANS[a.arena][s]),
                  flush=True)
        out["runs"].append(rec)
    out["end"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out + ".tmp", "w"))
    os.replace(a.out + ".tmp", a.out)


if __name__ == "__main__":
    main()
