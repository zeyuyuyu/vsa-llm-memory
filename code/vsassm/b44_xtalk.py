"""B44 cross-talk sweep with centred and whitened codes, and the Fig. 2b data under the B31 protocol
(prereg/B44_data_scaling_heldout_cot_crosstalk_prereg.md).

sweep (no encoder, nothing trained). Arena C, B31's splits (train 800 seed 42 for the threshold, test 500 seed 4242),
B31's codebooks (64 shared rows, d = 2,048, seeds 42 / 123 / 456: b31_arms.unitBC(64, 2048, seed + 31)). Cross-talk is
imposed as in the published sweep (V-95 ConeArm, V-163): row s -> l2n(sqrt(1 - c) s + sqrt(c) u), u one fixed random
direction (generator seed + 4242), at the 14 published levels c. Each code set is then read in three forms before
binding:
  raw       as published
  centred   code - mu, then l2-normalised; mu = mean of the codes of all training-split mentions
  whitened  ZCA: l2n(W (code - mu)), W = S_a^(-1/2), S_a = (1 - a) S + a (tr S / d) I, S the covariance of the
            training-split mention codes, a = 0.1 (primary) and 0.5 (sensitivity)
and chained with B31's arena-C chain (b31_arms.Arm.chain: two hops, clean-up after each) under both encodings (perm =
B31 primary, comm = the published sweep). Read-out: B31's closed-form threshold fitted on the training split; also the
AUROC of the final cosine. Measured cross-talk = mean cosine between the codes of different symbols (all 64 rows; the
V-95 measure) in the form that is bound.

init (Fig. 2b, needs the B31 features): the binding-loss map (perm, named) at initialisation, per encoder and seed (the
same initial parameters as B31's runs: torch.manual_seed(seed) then b31_arms.Arm), geometry by B31's protocol and test
accuracy with the training-fitted threshold, raw and centred (training-mention mean of the codes).

  python b44_xtalk.py sweep --out results/b44_xtalk_sweep.json
  python b44_xtalk.py init --feats /path/to/work/b31/feats --out out/xtalk_init.json
"""
import argparse, json, os, sys
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import b31_arms as B                                                        # noqa: E402
import b44_data as BD                                                       # noqa: E402

CS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
SEEDS = [42, 123, 456]
ALPHAS = [0.1, 0.5]


def auroc(s, y):
    order = np.argsort(s, kind="mergesort"); r = np.empty(len(s)); i = 0; ss = s[order]
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    npos, nneg = int((y == 1).sum()), int((y == 0).sum())
    return float((r[y == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def crosstalk(rows):
    G = F.normalize(rows, dim=-1) @ F.normalize(rows, dim=-1).t()
    return float(G[~torch.eye(G.size(0), dtype=torch.bool)].mean())


def transform(rows, tr_sym, form, alpha=0.1):
    """rows (64, d) code per symbol; statistics over all training-split mentions (each mention's code is its row)."""
    if form == "raw":
        return rows
    X = rows[tr_sym.reshape(-1)].double()
    mu = X.mean(0)
    if form == "centred":
        return F.normalize((rows.double() - mu), dim=-1).float()
    Xc = X - mu
    S = Xc.t() @ Xc / X.size(0)
    d = S.size(0)
    Sa = (1 - alpha) * S + alpha * (torch.trace(S) / d) * torch.eye(d, dtype=S.dtype)
    ev, U = torch.linalg.eigh(Sa)
    W = U @ torch.diag(ev.clamp(min=1e-12).rsqrt()) @ U.t()
    return F.normalize((rows.double() - mu) @ W.t(), dim=-1).float()


def chain_scores(m, table, sym, enc, bs=100):
    out = []
    with torch.no_grad():
        for k in range(0, sym.size(0), bs):
            K = table[sym[k:k + bs]]
            s, _, _ = m.chain(K, enc)
            out.append(s)
    return torch.cat(out).numpy().astype(np.float64)


def sweep(a):
    tr = BD.gen(800, 42, "std", "train"); te = BD.gen(500, 4242, "std", "test")
    trs = torch.tensor([[mm[2] for mm in s["mentions"]] for s in tr]); tes = torch.tensor([[mm[2] for mm in s["mentions"]] for s in te])
    yt = np.array([s["label"] for s in tr]); ye = np.array([s["label"] for s in te])
    forms = [("raw", None), ("centred", None)] + [("whitened", al) for al in ALPHAS]
    res = {"levels": CS, "seeds": SEEDS, "curve": []}
    for c in CS:
        row = {"c": c}
        for form, al in forms:
            fk = form if al is None else "%s_a%g" % (form, al)
            for enc in ("perm", "comm"):
                accs, xts, aucs, thrs = [], [], [], []
                for s in SEEDS:
                    torch.manual_seed(s)
                    m = B.Arm("C", "exact", s, 8)
                    g = torch.Generator().manual_seed(s + 4242)
                    u = F.normalize(torch.randn(B.D, generator=g), dim=-1)
                    rows = F.normalize(np.sqrt(max(0.0, 1 - c)) * m.table + np.sqrt(max(0.0, c)) * u[None], dim=-1)
                    rows = transform(rows, trs, form, al if al is not None else 0.1)
                    st = chain_scores(m, rows, trs, enc); se = chain_scores(m, rows, tes, enc)
                    acc, thr, _ = B.thr_acc(st, yt, se, ye)
                    accs.append(acc); thrs.append(thr); aucs.append(auroc(se, ye)); xts.append(crosstalk(rows))
                row["%s/%s" % (fk, enc)] = {"acc": accs, "acc_mean": float(np.mean(accs)), "auroc": aucs,
                                            "auroc_mean": float(np.mean(aucs)), "crosstalk": xts,
                                            "crosstalk_mean": float(np.mean(xts)), "thr": thrs}
        res["curve"].append(row)
        print("c=%.2f " % c + " | ".join("%s %.3f (xt %.3f)" % (k, v["acc_mean"], v["crosstalk_mean"])
                                         for k, v in row.items() if k != "c" and k.endswith("/perm")), flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


def init(a):
    """Binding-loss map at initialisation (B31 config, perm / named), raw and centred, per encoder and seed."""
    out = {}
    for enc in ("phi2", "qwen8"):
        D_ = {sp: B.Split(enc, "C", sp, a.feats) for sp in ("train", "test")}
        hid = D_["train"].F.size(-1)
        for s in SEEDS:
            torch.manual_seed(s)
            m = B.Arm("C", "bind", s, hid).to(a.device).eval()
            with torch.no_grad():
                def codes(Dsp):
                    return torch.cat([m.kcodes(Dsp.F[k:k + 64].to(a.device), Dsp.sym[k:k + 64].to(a.device), "named")
                                      for k in range(0, Dsp.n, 64)])
                Ktr, Kte = codes(D_["train"]), codes(D_["test"])
                mu = Ktr.reshape(-1, Ktr.size(-1)).mean(0)
                rec = {}
                for form in ("raw", "centred"):
                    f = (lambda K: K) if form == "raw" else (lambda K: F.normalize(K - mu, dim=-1))
                    sc = {}
                    for nm, K in (("train", f(Ktr)), ("test", f(Kte))):
                        sc[nm] = torch.cat([m.chain(K[k:k + 64], "perm")[0] for k in range(0, K.size(0), 64)]).cpu().numpy()
                    acc, thr, tracc = B.thr_acc(sc["train"], D_["train"].y.numpy(), sc["test"], D_["test"].y.numpy())
                    geo = geometry_codes(m, f(Kte[:120]), D_["test"].sym[:120].to(a.device))
                    rec[form] = {"acc": acc, "thr": thr, "train_acc": tracc, "geom": geo}
                out["%s_seed%d" % (enc, s)] = rec
                print(enc, s, json.dumps({k: (round(v["acc"], 4), round(v["geom"]["different"], 4),
                                              round(v["geom"]["same"], 4)) for k, v in rec.items()}), flush=True)
    json.dump(out, open(a.out, "w"), indent=1)


def geometry_codes(m, K, sym, per_sym=4, n_sym=40):
    """b31_arms.geometry on given codes (same protocol: context mentions, up to 4 occurrences, first 40 symbols)."""
    kinds = m.kinds
    pos = [i for i in range(len(kinds)) if i not in (m.q, m.c) and kinds[i] in B.GEO_SETS["C"]]
    by = {}
    for b in range(K.size(0)):
        for i in pos:
            sid = int(sym[b, i])
            if len(by.setdefault(sid, [])) < per_sym:
                by[sid].append(K[b, i])
    ks = [k for k in by if by[k]][:n_sym]
    first = torch.stack([by[k][0] for k in ks])
    G = first @ first.t()
    diff = G[~torch.eye(len(ks), dtype=torch.bool, device=G.device)].mean().item()
    same = [float(torch.dot(by[k][i], by[k][j])) for k in ks for i in range(len(by[k])) for j in range(i + 1, len(by[k]))]
    return {"different": diff, "same": float(np.mean(same)), "symbols": len(ks)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["sweep", "init"])
    ap.add_argument("--feats", default="/path/to/work/b31/feats")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    {"sweep": sweep, "init": init}[a.cmd](a)


if __name__ == "__main__":
    main()
