"""B28 arena C (prereg/B28_noncommutative_binding_prereg.md): exact codebook rows (parser) and the naming head
(probe_argmax) under the five encodings of b28_common, seeds 42/123/456.

The arena module is the verbatim copy b27_arena/v13.py (md5 checked by b27_common). EncArm subclasses
v13.BindArm and overrides only sims(): pairs are stored as bind(V(filler), K(role)), the hop is read by unbinding
the key, cleaned up against V(that level's fillers), and the winner's KEY code is the next key; the final score
is <key, K(candidate)>. For encoding comm this is the published computation step for step. run() is v13.run with
the arena and cache built outside (neither draws from the torch RNG) and with per-epoch, per-instance recording.
Scoring as published: threshold fitted on the training sims, best test epoch (final epoch also recorded).

--split test (registered runs) | dev (smoke: v13.Arena(tok, 500, 777), the B27 dev split; never the test split).
"""
import argparse, json, os, random, sys, time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import b27_common as BC                                               # noqa: E402
import b28_common as C                                                # noqa: E402

M = BC.load_arena_modules()
v13 = M["v13"]


class EncArm(v13.BindArm):
    def __init__(self, mode, seed, enc_name):
        super().__init__(mode, seed)          # same RNG consumption as the published arm
        self.enc_name, self.seed_ = enc_name, seed
        self._enc = None

    def enc(self, device):
        if self._enc is None or self._enc.Kb.device != torch.device(device):
            self._enc = C.Encoding(self.enc_name, self.sym.to(device), self.seed_)
        return self._enc

    def nid(self, H, sid, sp):
        if self.mode == "parser":
            return sid
        return int(self.id_head(self.sf(H, sp)).argmax().item())

    def path(self, H, r, device):
        """(score, retrieved filler id per hop, named-ok) for one row."""
        E = self.enc(device)
        key = E.K(r["q"])
        hops, ok = [], True
        for L in range(v13.HOPS):
            pr = []
            for aa, bb, spr, spf in r["ctx"][L]:
                ri, fi = self.nid(H, aa, spr), self.nid(H, bb, spf)
                ok = ok and ri == aa and fi == bb
                pr.append((ri, fi))
            mem = torch.stack([v13.bind(E.V(fi), E.K(ri)) for ri, fi in pr], 0).sum(0)
            hat = F.normalize(v13.unbind(mem, key), dim=-1)
            cands = torch.stack([E.V(fi) for _, fi in pr], 0)
            w = int(F.cosine_similarity(cands, hat.unsqueeze(0)).argmax().item())
            key = E.K(pr[w][1])
            hops.append(int(r["ctx"][L][w][1]))          # gold id of the winning mention
        return torch.dot(key, E.K(r["cand"])).clamp(-1, 1), hops, ok

    def sims(self, H, rows, device):
        out = torch.zeros(len(rows), device=device)
        for b, r in enumerate(rows):
            if any(len(l) == 0 for l in r["ctx"]):
                continue
            out[b] = self.path(H[b], r, device)[0]
        return out


def chain_of(r):
    s1 = [b for a, b, _, _ in r["ctx"][0] if a == r["q"]][0]
    s2 = [b for a, b, _, _ in r["ctx"][1] if a == s1][0]
    return s1, s2


def run(arm, enc_name, seed, tr_rows, te_rows, Htr, Hte, device, epochs=2, bs=4, lr=3e-4):
    torch.manual_seed(seed)
    m = EncArm(arm, seed, enc_name).to(device)

    def mk():
        sc = [q for n, q in m.named_parameters() if q.requires_grad and n in ("scale", "bias")]
        rest = [q for n, q in m.named_parameters() if q.requires_grad and n not in ("scale", "bias")]
        g = ([{"params": rest, "lr": lr}] if rest else []) + ([{"params": sc, "lr": 1e-1}] if sc else [])
        return torch.optim.AdamW(g)

    params = [p for p in m.parameters() if p.requires_grad]
    opt = mk()

    def collect(rows, Hall):
        m.eval(); S, Y = [], []
        with torch.no_grad():
            for k in range(0, len(rows), bs):
                S.append(m.sims(Hall[k:k + bs].to(device), rows[k:k + bs], device).cpu())
                Y += [r["label"] for r in rows[k:k + bs]]
        return torch.cat(S).numpy(), np.array(Y)

    def acc():
        st, yt = collect(tr_rows, Htr); se, ye = collect(te_rows, Hte)
        c = np.unique(st)
        if c.size == 0:
            return 0.5, None, se, ye
        thr = float(c[int(np.argmax([(((st >= t) == (yt == 1)).mean()) for t in c]))])
        return float((((se >= thr) == (ye == 1)).mean())), thr, se, ye

    t0, best, eps = time.time(), 0.0, []
    g = torch.Generator(); g.manual_seed(seed)
    for ep in range(epochs):
        m.train()
        perm = torch.randperm(len(tr_rows), generator=g).tolist()
        for k in range(0, len(perm), bs):
            sel = perm[k:k + bs]; rows = [tr_rows[i] for i in sel]
            H = Htr[sel].to(device)
            am = torch.stack([r["am"] for r in rows]).to(device)
            y = torch.tensor([r["label"] for r in rows], device=device)
            loss = F.cross_entropy(m(H, am, rows, device), y)
            if arm == "probe_argmax":
                loss = loss + m.id_loss(H, rows, device)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        a_, thr, se, ye = acc(); best = max(best, a_)
        # retrieval path and naming on the scored split (diagnostics; no effect on the score)
        m.eval(); hops, named = [], []
        with torch.no_grad():
            for b, r in enumerate(te_rows):
                _, hp, ok = m.path(Hte[b].to(device), r, device)
                hops.append(hp); named.append(int(ok))
        eps.append({"epoch": ep + 1, "acc": a_, "thr": thr, "sims": [float(x) for x in se],
                    "pred": [int(x >= thr) if thr is not None else 0 for x in se], "hops": hops,
                    "named_ok": named})
        print("      %-12s %-5s seed %3d ep%d acc=%.4f (%.0fs)" % (arm, enc_name, seed, ep + 1, a_,
                                                                 time.time() - t0), flush=True)
    return {"arm": arm, "enc": enc_name, "seed": seed, "acc_best": best, "acc_final": eps[-1]["acc"],
            "epochs": eps, "secs": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["parser", "probe_argmax"])
    ap.add_argument("--encodings", nargs="+", default=C.ENCODINGS)
    ap.add_argument("--seeds", type=int, nargs="+", default=C.SEEDS)
    ap.add_argument("--split", default="test", choices=["test", "dev"])
    ap.add_argument("--feats", default="/path/to/work/b28/feats")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="out")
    ap.add_argument("--cache_only", action="store_true")
    a = ap.parse_args()
    tok = BC.tokenizer()
    tr = v13.Arena(tok, 800, 42).rows
    te = v13.Arena(tok, 500, 4242 if a.split == "test" else 777).rows
    os.makedirs(a.feats, exist_ok=True)

    def cached(name, rows):
        p = os.path.join(a.feats, "C_%s.pt" % name)
        if os.path.exists(p):
            obj = torch.load(p)
            assert obj["n"] == len(rows) and torch.equal(obj["ids"], torch.stack([r["ids"] for r in rows]))
            return obj["H"]
        from transformers import AutoModel
        llm = AutoModel.from_pretrained("microsoft/phi-2", trust_remote_code=True,
                                        torch_dtype=torch.float16).to(a.device).eval()
        for q in llm.parameters():
            q.requires_grad_(False)
        H = v13.cache(rows, llm, a.device)
        del llm; torch.cuda.empty_cache()
        torch.save({"n": len(rows), "ids": torch.stack([r["ids"] for r in rows]), "H": H}, p + ".tmp")
        os.replace(p + ".tmp", p)
        return H

    Htr = cached("train", tr)
    Hte = cached(a.split, te)
    if a.cache_only:
        print("B28-C-CACHED", flush=True)
        return
    gold = []
    for r in te:
        s1, s2 = chain_of(r)
        al, h1, h2 = C.alias_arenaC([[(x, y) for x, y, _, _ in r["ctx"][L]] for L in range(2)], r["q"], s1, s2)
        gold.append({"q": r["q"], "s1": s1, "s2": s2, "cand": r["cand"], "label": r["label"], "alias": al,
                     "alias_h1": h1, "alias_h2": h2,
                     "ctx": [[(x, y) for x, y, _, _ in r["ctx"][L]] for L in range(2)]})
    gold_tr = [{"q": r["q"], "cand": r["cand"], "label": r["label"],
                "ctx": [[(x, y) for x, y, _, _ in r["ctx"][L]] for L in range(2)]} for r in tr]
    os.makedirs(a.out, exist_ok=True)
    for arm in a.arms:
        for enc in a.encodings:
            fn = os.path.join(a.out, "b28_C_%s_%s_%s.json" % (a.split, arm, enc))
            res = {"split": a.split, "arm": arm, "enc": enc, "gold": gold, "gold_train": gold_tr, "runs": []}
            for s in a.seeds:
                r = run(arm, enc, s, tr, te, Htr, Hte, a.device)
                res["runs"].append(r)
                al = np.array([g_["alias"] for g_ in gold])
                hp = r["epochs"][-1]["hops"]
                ret = np.array([int(h[0] == g_["s1"] and h[1] == g_["s2"]) for h, g_ in zip(hp, gold)])
                print("  %-12s %-5s seed %3d best %.4f final %.4f | retrieval alias %d: %.3f, non-alias: %.3f"
                      % (arm, enc, s, r["acc_best"], r["acc_final"], al.sum(), ret[al == 1].mean(),
                         ret[al == 0].mean()), flush=True)
                json.dump(res, open(fn + ".tmp", "w")); os.replace(fn + ".tmp", fn)
    print("B28-C-DONE", flush=True)


if __name__ == "__main__":
    main()
