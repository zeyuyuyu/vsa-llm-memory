"""B47 consistency sweep (prereg/B47_phi2_finetunes_hierboot_consistency_prereg.md): within-symbol inconsistency
across mentions at zero code coherence, in B31's registered arena-C pipeline. No encoder, nothing trained.

Codes. B31's arena-C codebook (64 shared rows, d = 2,048; codebook seeds 42 / 123 / 456: b31_arms.Arm("C", "exact", s)
.table). Coherence c (the published sweep, B44 / V-95): row t -> t' = l2n(sqrt(1 - c) t + sqrt(c) u), u one fixed random
direction (generator seed + 4242). Consistency rho: every MENTION m of symbol s in every instance gets its own code
    k_m = rho t'_s + sqrt(1 - rho^2) n_m,   n_m a unit vector orthogonal to t'_s (Gaussian draw, t'_s projected out),
so cos(k_m, t'_s) = rho exactly and two mentions of one symbol have cosine ~ rho^2; different symbols stay as
orthogonal as the random rows (c = 0). The noise draws n_m are fixed per (codebook seed, split, instance, mention)
(generator seed + 9001 + split offset) and shared by every rho and c level (common random numbers).

Noise structures:
  mention (primary)  independent n_m for every mention (the query and the candidate included)
  form    (secondary) one code per (symbol, written form): the hexadecimal form ("[Rxx]", "[Fxx]", "Q [Rxx]": every key
                      and the query) is t'_s; the decimal form ("{n}": every value and the candidate) is
                      rho t'_s + sqrt(1 - rho^2) n_s with n_s fixed per symbol (generator seed + 9101): consistent within
                      a form, cosine rho across the two forms (the arena-C alias)

Read with B31's arena-C chain (b31_arms.Arm.chain: two hops, clean-up to the instance's own value mentions after each,
the winner's code as the next key, final cosine with the candidate's code), perm (B31 primary) and comm; read-out =
B31's closed-form threshold fitted on the training split (800); accuracy on dev (500) and test (500); AUROC; clean-up
correctness at hop 1, hop 2 and both (gold positions from the generator's chain); geometry of the codes as bound
(B31 protocol on the first 120 instances of the split: mean cosine between different symbols and between mentions of
one symbol).

  python b47_sweep.py --what consistency --out b47_out/sweep_consistency.json      (c = 0; S1 + form, perm + comm)
  python b47_sweep.py --what grid --out b47_out/sweep_grid.json                    (c in 0.1/0.2/0.3 x rho; perm)
  python b47_sweep.py --what coherence --out b47_out/sweep_coherence.json          (rho = 1, B44's 14 c levels: G1b)
"""
import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import b31_arms as B                                                       # noqa: E402
import b44_data as BD                                                      # noqa: E402
import b44_xtalk as X                                                      # noqa: E402

RHOS = [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2]
CS_PUB = X.CS                                    # the 14 published coherence levels (B44)
CS_GRID = [0.1, 0.2, 0.3]
SEEDS = [42, 123, 456]
SPLIT_OFF = {"train": 0, "dev": 1, "test": 2}
SPLITS = {"train": (800, 42), "dev": (500, 777), "test": (500, 4242)}
BS = 50


def load_split(name):
    st = BD.gen(*SPLITS[name], "std", name)
    sym = torch.tensor([[mm[2] for mm in s["mentions"]] for s in st])
    y = np.array([s["label"] for s in st])
    return st, sym, y


def gold_symbols(sym):
    """Gold symbols of the two hops from the generator's mentions: hop 1 = the filler of the R-table key that equals
    the query symbol (the bridge); hop 2 = the filler of the F-table key that equals the bridge (the answer). A hop is
    correct when the clean-up winner's SYMBOL is the gold one (a filler can occur twice in one table)."""
    kinds, hops, q, c, _ = B.layout("C")
    (k1, v1), (k2, v2) = hops
    g1, g2 = [], []
    for b in range(sym.size(0)):
        s = sym[b].tolist()
        j1 = [s[v] for k, v in zip(k1, v1) if s[k] == s[q]]
        j2 = [s[v] for k, v in zip(k2, v2) if s[k] == j1[0]]
        assert len(j1) == 1 and len(j2) == 1
        g1.append(j1[0]); g2.append(j2[0])
    return torch.tensor(g1), torch.tensor(g2)


class Codes:
    """Mention codes for one codebook seed. The raw noise of every split is drawn once (generator seed + 9001 + 1000 x
    split offset; one (130, d) Gaussian block per instance, in instance order) and reused by every level."""
    def __init__(self, m, seed, data, need_noise=True):
        self.m = m
        self.seed = seed
        g = torch.Generator().manual_seed(seed + 4242)
        self.u = F.normalize(torch.randn(B.D, generator=g), dim=-1)
        self.form_noise = torch.randn(64, B.D, generator=torch.Generator().manual_seed(seed + 9101))
        self.is_dec = torch.tensor([k == "F" for k in ["R", "F"] * 65])    # value / candidate positions: decimal form
        self.noise = {}
        if need_noise:
            for sp in ("train", "dev", "test"):
                gen = torch.Generator().manual_seed(seed + 9001 + 1000 * SPLIT_OFF[sp])
                n = data[sp][1].size(0)
                self.noise[sp] = torch.randn(n, 130, B.D, generator=gen)

    def rows(self, c):
        return F.normalize(np.sqrt(max(0.0, 1 - c)) * self.m.table + np.sqrt(max(0.0, c)) * self.u[None], dim=-1)

    @staticmethod
    def mix(t, n, rho):
        """rho t + sqrt(1 - rho^2) n_perp, n_perp = unit(n - (n . t) t); t unit rows (..., d)."""
        if rho >= 1.0:
            return t
        n = n - (n * t).sum(-1, keepdim=True) * t
        n = F.normalize(n, dim=-1)
        return rho * t + np.sqrt(max(0.0, 1 - rho * rho)) * n

    def batches(self, sym, split, c, rho, structure, limit=None):
        T = self.rows(c)
        N = sym.size(0) if limit is None else min(limit, sym.size(0))
        for k in range(0, N, BS):
            sb = sym[k:min(k + BS, N)]
            t = T[sb]
            if structure == "mention":
                K = t if rho >= 1.0 else self.mix(t, self.noise[split][k:k + sb.size(0)], rho)
            else:
                dec = self.mix(T, self.form_noise, rho)[sb]
                K = torch.where(self.is_dec[None, :, None], dec, t)
            yield K, slice(k, k + sb.size(0))


def run_cell(m, codes, data, c, rho, structure, enc):
    out = {}
    for sp in ("train", "dev", "test"):
        st, sym, y = data[sp][:3]
        S_, W_ = [], []
        with torch.no_grad():
            for K, sl in codes.batches(sym, sp, c, rho, structure):
                s_, w, _ = m.chain(K, enc)
                S_.append(s_); W_.append(w)
        out[sp] = {"s": torch.cat(S_).numpy().astype(np.float64), "w": torch.cat(W_)}
    acc, thr, tracc = B.thr_acc(out["train"]["s"], data["train"][2], out["test"]["s"], data["test"][2])
    dacc = float((((out["dev"]["s"] >= thr) == (data["dev"][2] == 1)).mean()))
    res = {"acc": acc, "dev_acc": dacc, "thr": thr, "train_acc": tracc,
           "auroc": X.auroc(out["test"]["s"], data["test"][2])}
    sym = data["test"][1]
    g1, g2 = data["test"][3]
    w = out["test"]["w"]
    ar = torch.arange(sym.size(0))
    h1 = (sym[ar, w[:, 0]] == g1); h2 = (sym[ar, w[:, 1]] == g2)
    res.update(hop1=float(h1.float().mean()), hop2=float(h2.float().mean()), both=float((h1 & h2).float().mean()))
    pred = (out["test"]["s"] >= thr).astype(int); yy = data["test"][2]
    ok = (pred == yy)
    hb = (h1 & h2).numpy()
    res["acc_chain_right"] = float(ok[hb].mean()) if hb.any() else None
    res["acc_chain_wrong"] = float(ok[~hb].mean()) if (~hb).any() else None
    res["test_scores"] = [round(float(x), 6) for x in out["test"]["s"]]
    res["test_both_hops_right"] = hb.astype(int).tolist()
    return res


def geometry(codes, data, c, rho, structure):
    """B31 protocol (b44_xtalk.geometry_codes: context mentions of the first 120 test instances, up to 4 occurrences
    per symbol, first 40 symbols) on the codes as bound (K)."""
    sym = data["test"][1]
    K = torch.cat([K for K, sl in codes.batches(sym, "test", c, rho, structure, limit=120)])
    return X.geometry_codes(codes.m, K, sym[:120])


def cells_of(what):
    if what == "consistency":
        return [(0.0, r, st_, enc) for st_ in ("mention", "form") for r in RHOS for enc in ("perm", "comm")]
    if what == "grid":
        return [(c, r, "mention", "perm") for c in CS_GRID for r in RHOS]
    return [(c, 1.0, "mention", enc) for c in CS_PUB for enc in ("perm", "comm")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", choices=["consistency", "grid", "coherence"])
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--nparts", type=int, default=1)
    ap.add_argument("--merge", nargs="*", default=None, help="merge part files into --out")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    if a.merge:
        parts = [json.load(open(p)) for p in a.merge]
        res = {k: parts[0][k] for k in ("what", "seeds", "rhos")}
        res["parts"] = [{"start": p["start"], "end": p["end"], "host": p.get("host"), "part": p.get("part")} for p in parts]
        order = {tuple(c): i for i, c in enumerate(cells_of(res["what"]))}
        allc = [c for p in parts for c in p["cells"]]
        res["cells"] = sorted(allc, key=lambda r: order[(r["c"], r["rho"], r["structure"], r["enc"])])
        assert len(res["cells"]) == len(order) and len({(r["c"], r["rho"], r["structure"], r["enc"]) for r in allc}) == len(order)
        json.dump(res, open(a.out + ".tmp", "w")); os.replace(a.out + ".tmp", a.out)
        print("merged", len(res["cells"]), "cells"); return
    torch.set_num_threads(a.threads)
    data = {}
    for sp in ("train", "dev", "test"):
        st, sym, y = load_split(sp)
        data[sp] = (st, sym, y, gold_symbols(sym))
    cells = [x for i, x in enumerate(cells_of(a.what)) if i % a.nparts == a.part]
    res = {"what": a.what, "seeds": SEEDS, "rhos": RHOS, "part": [a.part, a.nparts], "host": os.uname()[1],
           "start": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "cells": []}
    rows = {cell: {"c": cell[0], "rho": cell[1], "structure": cell[2], "enc": cell[3], "per_seed": []} for cell in cells}
    t0 = time.time()
    need_noise = any(cell[1] < 1.0 and cell[2] == "mention" for cell in cells)
    for s in SEEDS:
        torch.manual_seed(s)
        m = B.Arm("C", "exact", s, 8)
        codes = Codes(m, s, data, need_noise)
        for cell in cells:
            c, rho, structure, enc = cell
            r = run_cell(m, codes, data, c, rho, structure, enc)
            r["geom"] = geometry(codes, data, c, rho, structure)
            r["seed"] = s
            rows[cell]["per_seed"].append(r)
            print("seed %d c %.2f rho %.2f %-7s %s acc %.3f (dev %.3f) hop1 %.3f hop2 %.3f geom %.3f/%.3f [%.0fs]" % (
                s, c, rho, structure, enc, r["acc"], r["dev_acc"], r["hop1"], r["hop2"], r["geom"]["different"],
                r["geom"]["same"], time.time() - t0), flush=True)
        del codes
    for cell in cells:
        row = rows[cell]
        for k in ("acc", "dev_acc", "auroc", "hop1", "hop2", "both"):
            row[k + "_mean"] = float(np.mean([r[k] for r in row["per_seed"]]))
        row["geom_different_mean"] = float(np.mean([r["geom"]["different"] for r in row["per_seed"]]))
        row["geom_same_mean"] = float(np.mean([r["geom"]["same"] for r in row["per_seed"]]))
        res["cells"].append(row)
    res["end"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(res, open(a.out + ".tmp", "w")); os.replace(a.out + ".tmp", a.out)


if __name__ == "__main__":
    main()
