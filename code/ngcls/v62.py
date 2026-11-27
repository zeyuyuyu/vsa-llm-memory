"""V-62: FSQ binding head (prereg ba52d77, bars registered before this file).

Finite Scalar Quantization: proj -> 6 dims -> tanh -> round each dim to 7 levels (straight-through). No learnable
codebook, so nothing can collapse and nothing can drift toward the encoder. The quantised point is expanded to
VSA_DIM by a FIXED random matrix and normalised. Declared in the prereg: code similarity inherits grid distance,
so distinct symbols must be placed far apart on the grid by the encoder itself.
"""
import argparse, json, os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
sys.path.insert(0, "/path/to/work/ngcls")
import v13                                                            # noqa: E402

D_FSQ, LEVELS = 6, 7


class FSQHead(nn.Module):
    def __init__(self, seed, inj):
        super().__init__()
        self.proj = nn.Linear(v13.HID, D_FSQ)
        g = torch.Generator().manual_seed(seed + 977)
        self.register_buffer("expand", torch.randn(v13.VSA_DIM, D_FSQ, generator=g) / (D_FSQ ** 0.5))
        self.scale = nn.Parameter(torch.tensor(20.0))
        self.bias = nn.Parameter(torch.tensor(-4.0))
        self.inj = inj
        self.half = (LEVELS - 1) / 2.0

    def quantise(self, h):
        """FSQ: z in (-1,1)^D scaled to the grid, rounded with a straight-through estimator."""
        z = torch.tanh(self.proj(h)) * self.half
        q = z + (z.round() - z).detach()
        code = F.normalize(self.expand @ q, dim=-1)
        return code, tuple(int(t) for t in z.round().tolist()), z

    def instance_sim(self, H, r, device):
        aux = torch.zeros((), device=device)
        cache = {}

        def code(sp):
            key = (sp[0], sp[1])
            if key not in cache:
                cache[key] = self.quantise(self.sf(H, sp))
            return cache[key]

        if any(len(l) == 0 for l in r["ctx"]):
            return None, aux
        qc, _, _ = code(r["q_span"])
        key = qc
        for L in range(v13.HOPS):
            mems, cands, role_zs = [], [], []
            for aa, bb, spr, spf in r["ctx"][L]:
                fc, _, fz = code(spf)
                rc, _, rz = code(spr)
                mems.append(v13.bind(fc, rc)); cands.append(fc); role_zs.append(rz)
            mem = torch.stack(mems, 0).sum(0)
            hat = F.normalize(v13.unbind(mem, key), dim=-1)
            C = torch.stack(cands, 0)
            key = C[int(F.cosine_similarity(C, hat.unsqueeze(0)).argmax().item())]
            if self.inj and len(role_zs) > 1:
                # structure-derived injectivity pressure in GRID space: roles within one level are distinct
                # symbols by construction, so their pre-round coordinates are pushed apart. No labels.
                Z = torch.stack(role_zs, 0)
                d = torch.cdist(Z, Z) / (2 * self.half * (D_FSQ ** 0.5))
                n = Z.size(0)
                off = (1.0 - d)[~torch.eye(n, dtype=torch.bool, device=device)]
                aux = aux + 0.5 * off.clamp(min=0).mean()
        cc, _, _ = code(r["cand_span"])
        return torch.dot(key, cc).clamp(-1, 1), aux

    @staticmethod
    def sf(H, sp):
        return H[sp[0]:sp[1]].float().mean(0) if sp[1] > sp[0] else H[sp[0]].float()


def run(arm, seed, tok, llm, device, n_tr, n_te, epochs, bs, lr):
    torch.manual_seed(seed)
    tr, te = v13.Arena(tok, n_tr, 42), v13.Arena(tok, n_te, 4242)
    Htr, Hte = v13.cache(tr.rows, llm, device), v13.cache(te.rows, llm, device)
    m = FSQHead(seed, inj=(arm == "fsq_inj")).to(device)
    opt = torch.optim.AdamW([
        {"params": list(m.proj.parameters()), "lr": lr},
        {"params": [m.scale, m.bias], "lr": 1e-1}])

    def sims_of(rows, Hall):
        outs, auxs = [], []
        for b, r in enumerate(rows):
            s_, a_ = m.instance_sim(Hall[b].to(device), r, device)
            outs.append(s_ if s_ is not None else torch.zeros((), device=device))
            auxs.append(a_)
        return torch.stack(outs), torch.stack(auxs).mean()

    def acc():
        m.eval()
        with torch.no_grad():
            st, _ = sims_of(tr.rows, Htr)
            se, _ = sims_of(te.rows, Hte)
        st = st.cpu().numpy(); se = se.cpu().numpy()
        yt = np.array([r["label"] for r in tr.rows]); ye = np.array([r["label"] for r in te.rows])
        c = np.unique(st)
        thr = float(c[int(np.argmax([(((st >= t) == (yt == 1)).mean()) for t in c]))]) if c.size else 0.0
        m.train()
        return float((((se >= thr) == (ye == 1)).mean()))

    best = 0.0
    g = torch.Generator(); g.manual_seed(seed)
    for ep in range(epochs):
        m.train()
        perm = torch.randperm(len(tr.rows), generator=g).tolist()
        tot, nb = 0.0, 0
        for k in range(0, len(perm), bs):
            sel = perm[k:k + bs]
            rows = [tr.rows[i] for i in sel]
            s_, aux = sims_of(rows, Htr[sel])
            y = torch.tensor([r["label"] for r in rows], device=device)
            lp = s_ * m.scale + m.bias
            loss = F.cross_entropy(torch.stack([-lp, lp], 1), y) + aux
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad], 1.0)
            opt.step()
            tot += float(loss.item()); nb += 1
        a = acc(); best = max(best, a)
        print("      %-9s seed %d ep%d loss=%.4f test=%.4f" % (arm, seed, ep + 1, tot / max(1, nb), a),
              flush=True)

    # diagnostics: distinct tuples used, within-level role distinctness, same-symbol consistency,
    # and the registered grid-correlation check (pairwise code cosine between DISTINCT symbols)
    m.eval()
    used = set()
    distinct_ok, level_n = 0, 0
    same_pairs, same_hits = 0, 0
    cos_diff = []
    with torch.no_grad():
        for b, r in enumerate(te.rows[:150]):
            H = Hte[b].to(device)
            assign, codevec = {}, {}
            for L in range(v13.HOPS):
                tuples = []
                for aa, bb, spr, spf in r["ctx"][L]:
                    rc, rt, _ = m.quantise(m.sf(H, spr))
                    fc, ft, _ = m.quantise(m.sf(H, spf))
                    used.add(rt); used.add(ft)
                    tuples.append(rt)
                    assign.setdefault(aa, []).append(rt)
                    assign.setdefault(bb, []).append(ft)
                    codevec.setdefault(aa, rc); codevec.setdefault(bb, fc)
                level_n += 1
                distinct_ok += int(len(set(tuples)) == len(tuples))
            for sid, lst in assign.items():
                if len(lst) > 1:
                    same_pairs += 1
                    same_hits += int(len(set(lst)) == 1)
            ks = list(codevec.keys())[:10]
            for i in range(len(ks)):
                for j in range(i + 1, len(ks)):
                    cos_diff.append(float(torch.dot(codevec[ks[i]], codevec[ks[j]])))
    diag = {"tuples_used": len(used), "role_distinct_rate": distinct_ok / max(1, level_n),
            "same_symbol_consistency": same_hits / max(1, same_pairs),
            "distinct_symbol_code_cosine": float(np.mean(cos_diff)) if cos_diff else float("nan")}
    print("      DIAG %-9s tuples=%d role-distinct=%.4f same-symbol=%.4f diff-symbol-cos=%.4f"
          % (arm, diag["tuples_used"], diag["role_distinct_rate"],
             diag["same_symbol_consistency"], diag["distinct_symbol_code_cosine"]), flush=True)
    return best, diag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["floor", "parser", "fsq_spans", "fsq_inj"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    ap.add_argument("--n_train", type=int, default=800)
    ap.add_argument("--n_test", type=int, default=500)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="v62_logs/v62.json")
    a = ap.parse_args()

    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    llm = AutoModel.from_pretrained("microsoft/phi-2", trust_remote_code=True,
                                    torch_dtype=torch.float16).to(a.device).eval()
    for p in llm.parameters():
        p.requires_grad_(False)

    print("===== V-62  FSQ binding head: %d dims x %d levels =====" % (D_FSQ, LEVELS), flush=True)
    out = {"d_fsq": D_FSQ, "levels": LEVELS, "arms": {}, "diag": {}}
    for arm in a.arms:
        if arm in ("floor", "parser"):
            accs = [v13.run(arm, s, tok, llm, a.device, a.n_train, a.n_test, 2, a.batch_size, a.lr)["acc"]
                    for s in a.seeds]
        else:
            accs, dgs = [], []
            for s in a.seeds:
                acc, dg = run(arm, s, tok, llm, a.device, a.n_train, a.n_test, a.epochs, a.batch_size, a.lr)
                accs.append(acc); dgs.append(dg)
            out["diag"][arm] = dgs
        out["arms"][arm] = accs
        print("  %-10s %s mean=%.4f" % (arm, [round(x, 4) for x in accs], float(np.mean(accs))), flush=True)
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(out, open(a.out, "w"), indent=1)

    m = {k: float(np.mean(v)) for k, v in out["arms"].items()}
    fl, pa = m["floor"], m["parser"]
    H = pa - fl
    print("", flush=True)
    print("===== V-62 SCORING (bars registered at ba52d77) =====", flush=True)
    print("  floor=%.4f parser=%.4f H=%.4f  works >= %.4f partial >= %.4f fails <= %.4f"
          % (fl, pa, H, fl + 0.90 * H, fl + 0.50 * H, fl + 0.10 * H), flush=True)
    if H < 0.20:
        v = "VOID (admission)"
    else:
        parts = []
        for arm in ("fsq_spans", "fsq_inj"):
            acc = out["arms"].get(arm)
            if not acc:
                continue
            if all(x >= fl + 0.90 * H for x in acc):
                parts.append("%s: FSQ WORKS (first end-to-end success; enters Results)" % arm)
            elif all(x >= fl + 0.50 * H for x in acc):
                parts.append("%s: PARTIAL" % arm)
            elif all(x <= fl + 0.10 * H for x in acc):
                parts.append("%s: FAILS (collapse and drift excluded by construction, so the blocker is the "
                             "encoder's separation under a task-only gradient)" % arm)
            else:
                parts.append("%s: GRADED" % arm)
        v = " | ".join(parts)
    out["verdict"] = v
    json.dump(out, open(a.out, "w"), indent=1)
    print("  >>> V-62 VERDICT: %s" % v, flush=True)
    print("V62-DONE", flush=True)


if __name__ == "__main__":
    main()
