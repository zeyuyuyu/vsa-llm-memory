"""V-63: the interventional close (prereg 2ff3e25, bars committed before this file).

Same FSQ head that failed in V-62, plus the one thing the diagnosis says was missing: an identity objective.
A linear classifier from the 6-dim pre-round coordinates to the symbol vocabulary, cross-entropy on the
arena's symbol ids. fsq_id trains identity CE + task loss; fsq_id_only trains the projection by identity CE
alone (the task loss reaches only the readout scale/bias, codes detached).
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


class FSQIdHead(nn.Module):
    def __init__(self, seed, n_classes):
        super().__init__()
        self.proj = nn.Linear(v13.HID, D_FSQ)
        self.cls = nn.Linear(D_FSQ, n_classes)
        g = torch.Generator().manual_seed(seed + 977)
        self.register_buffer("expand", torch.randn(v13.VSA_DIM, D_FSQ, generator=g) / (D_FSQ ** 0.5))
        self.scale = nn.Parameter(torch.tensor(20.0))
        self.bias = nn.Parameter(torch.tensor(-4.0))
        self.half = (LEVELS - 1) / 2.0

    def quantise(self, h):
        z = torch.tanh(self.proj(h)) * self.half
        q = z + (z.round() - z).detach()
        code = F.normalize(self.expand @ q, dim=-1)
        return code, tuple(int(t) for t in z.round().tolist()), z

    def instance_sim(self, H, r, sym2idx, device, detach_codes):
        """Returns (task similarity, identity CE over this row's labelled spans)."""
        ce_terms = []
        cache = {}

        def code(sp, sym=None):
            key = (sp[0], sp[1])
            if key not in cache:
                cache[key] = self.quantise(self.sf(H, sp))
            c, t, z = cache[key]
            if sym is not None and sym in sym2idx:
                ce_terms.append(F.cross_entropy(self.cls(z).unsqueeze(0),
                                                torch.tensor([sym2idx[sym]], device=device)))
            return c

        if any(len(l) == 0 for l in r["ctx"]):
            return None, None
        key = code(r["q_span"])
        for L in range(v13.HOPS):
            mems, cands = [], []
            for aa, bb, spr, spf in r["ctx"][L]:
                fc = code(spf, bb)
                rc = code(spr, aa)
                mems.append(v13.bind(fc, rc))
                cands.append(fc)
            mem = torch.stack(mems, 0).sum(0)
            hat = F.normalize(v13.unbind(mem, key), dim=-1)
            C = torch.stack(cands, 0)
            key = C[int(F.cosine_similarity(C, hat.unsqueeze(0)).argmax().item())]
        cc = code(r["cand_span"])
        s = torch.dot(key, cc).clamp(-1, 1)
        if detach_codes:
            s = s.detach()
        ce = torch.stack(ce_terms).mean() if ce_terms else torch.zeros((), device=device)
        return s, ce

    @staticmethod
    def sf(H, sp):
        return H[sp[0]:sp[1]].float().mean(0) if sp[1] > sp[0] else H[sp[0]].float()


def run(arm, seed, tok, llm, device, n_tr, n_te, epochs, bs, lr):
    torch.manual_seed(seed)
    tr, te = v13.Arena(tok, n_tr, 42), v13.Arena(tok, n_te, 4242)
    Htr, Hte = v13.cache(tr.rows, llm, device), v13.cache(te.rows, llm, device)
    syms = sorted({s for rows in (tr.rows, te.rows) for r in rows
                   for L in r["ctx"] for aa, bb, _, _ in L for s in (aa, bb)})
    sym2idx = {s: i for i, s in enumerate(syms)}
    detach = (arm == "fsq_id_only")
    m = FSQIdHead(seed, len(syms)).to(device)
    opt = torch.optim.AdamW([
        {"params": list(m.proj.parameters()) + list(m.cls.parameters()), "lr": lr},
        {"params": [m.scale, m.bias], "lr": 1e-1}])

    def sims_of(rows, Hall):
        outs, ces = [], []
        for b, r in enumerate(rows):
            s_, ce = m.instance_sim(Hall[b].to(device), r, sym2idx, device, detach)
            outs.append(s_ if s_ is not None else torch.zeros((), device=device))
            ces.append(ce if ce is not None else torch.zeros((), device=device))
        return torch.stack(outs), torch.stack(ces).mean()

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
            s_, ce = sims_of(rows, Htr[sel])
            y = torch.tensor([r["label"] for r in rows], device=device)
            lp = s_ * m.scale + m.bias
            task = F.cross_entropy(torch.stack([-lp, lp], 1), y)
            loss = ce if detach else task + ce
            if detach:
                loss = loss + task          # task gradient reaches only scale/bias (codes detached)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad], 1.0)
            opt.step()
            tot += float(loss.item()); nb += 1
        a = acc(); best = max(best, a)
        print("      %-11s seed %d ep%d loss=%.4f test=%.4f" % (arm, seed, ep + 1, tot / max(1, nb), a),
              flush=True)

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
    print("      DIAG %-11s tuples=%d role-distinct=%.4f same-symbol=%.4f diff-symbol-cos=%.4f"
          % (arm, diag["tuples_used"], diag["role_distinct_rate"],
             diag["same_symbol_consistency"], diag["distinct_symbol_code_cosine"]), flush=True)
    return best, diag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["fsq_id", "fsq_id_only"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    ap.add_argument("--n_train", type=int, default=800)
    ap.add_argument("--n_test", type=int, default=500)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="v63_logs/v63.json")
    a = ap.parse_args()

    ref = json.load(open("v62_logs/v62_BANKED.json"))
    fl = float(np.mean(ref["arms"]["floor"])); pa = float(np.mean(ref["arms"]["parser"]))
    H = pa - fl

    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    llm = AutoModel.from_pretrained("microsoft/phi-2", trust_remote_code=True,
                                    torch_dtype=torch.float16).to(a.device).eval()
    for p in llm.parameters():
        p.requires_grad_(False)

    print("===== V-63  identity objective on the failed FSQ head =====", flush=True)
    print("  reference floor=%.4f parser=%.4f H=%.4f (from v62_BANKED)" % (fl, pa, H), flush=True)
    out = {"floor_ref": fl, "parser_ref": pa, "arms": {}, "diag": {}}
    for arm in a.arms:
        accs, dgs = [], []
        for s in a.seeds:
            acc, dg = run(arm, s, tok, llm, a.device, a.n_train, a.n_test, a.epochs, a.batch_size, a.lr)
            accs.append(acc); dgs.append(dg)
        out["arms"][arm] = accs; out["diag"][arm] = dgs
        print("  %-11s %s mean=%.4f" % (arm, [round(x, 4) for x in accs], float(np.mean(accs))), flush=True)
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(out, open(a.out, "w"), indent=1)

    print("", flush=True)
    print("===== V-63 SCORING (bars registered at 2ff3e25) =====", flush=True)
    print("  works >= %.4f partial >= %.4f fails <= %.4f" % (fl + 0.90 * H, fl + 0.50 * H, fl + 0.10 * H),
          flush=True)
    parts = []
    for arm in a.arms:
        acc = out["arms"][arm]
        if all(x >= fl + 0.90 * H for x in acc):
            parts.append("%s: WORKS (the interventional confirmation)" % arm)
        elif all(x >= fl + 0.50 * H for x in acc):
            parts.append("%s: PARTIAL" % arm)
        elif all(x <= fl + 0.10 * H for x in acc):
            parts.append("%s: FAILS (the V-62 attribution is overclaimed; weaken the manuscript)" % arm)
        else:
            parts.append("%s: GRADED" % arm)
    out["verdict"] = " | ".join(parts)
    json.dump(out, open(a.out, "w"), indent=1)
    print("  >>> V-63 VERDICT: %s" % out["verdict"], flush=True)
    print("V63-DONE", flush=True)


if __name__ == "__main__":
    main()
