"""B68 self-test on TRAINING rows only (prereg/B68_early_layer_labelfree_prereg.md). No dev or test row is read.

For every feature set (phi2L0/L2/L4, qwen8L0/L2/L4) and arena (A, B, C), training split:
  forms    written forms (kind, symbol; arena C: a symbol's hex role and decimal filler are two forms), distinct
           span-feature vectors among the mentions, forms whose mentions do not all share one vector (L0 must be 0:
           the input embeddings carry no context), and forms that share a vector with another form (collisions, e.g.
           anagrams under single-digit tokenisation);
  geom     mean cosine of the centred span features between mentions of one form and between different forms of one
           kind (first 120 instances, labels group the vectors for this measurement only);
  hash     the 'feature hash' reader: one fixed random code (seed 42 table) per distinct span-feature vector, the arena's
           published chain (perm, named), the closed-form threshold: the accuracy on training rows of a reader that
           gives identical inputs identical codes and distinct inputs unrelated codes, i.e. the ceiling for any
           per-mention reader of these features without context (exact for L0; a lower bound for L2 / L4 readers,
           whose inputs differ across mentions of a form);
  smoke    each B42 family (vq, dkvb, gumbel, st, cw, bind) and the naming head through the unchanged code path for one
           epoch on the first 64 training instances, with those same training rows standing in for the dev split.

  python b68_selftest.py --feats /data0/path/to/HOME/b68/feats --out out/selftest.json [--smoke]
"""
import argparse, copy, json, os, sys, time
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import b31_arms as B                                                        # noqa: E402
import b31_struct as S                                                      # noqa: E402

ENCS = ["%sL%d" % (e, L) for e in ("qwen8", "phi2") for L in (0, 2, 4)]


def forms_and_hash(enc, ar, feats, dev):
    st = S.build(ar, "train")
    sp = B.Split(enc, ar, "train", feats, struct_cache=st)
    kinds, _, q, c, _ = B.layout(ar)
    Fm = sp.F                                                      # (N, M, d)
    N, M, d = Fm.shape
    keyb = {}
    vid = np.zeros((N, M), dtype=np.int64)
    flat = Fm.reshape(-1, d).numpy()
    for i in range(flat.shape[0]):
        b = hash(flat[i].tobytes())
        vid.flat[i] = keyb.setdefault(b, len(keyb))
    sym = sp.sym.numpy()
    form = [[(kinds[j], int(sym[n, j])) for j in range(M)] for n in range(N)]
    by_form = {}
    for n in range(N):
        for j in range(M):
            by_form.setdefault(form[n][j], set()).add(int(vid[n, j]))
    multi = sum(1 for v in by_form.values() if len(v) > 1)
    owner = {}
    for f_, vs in by_form.items():
        for v in vs:
            owner.setdefault(v, set()).add(f_)
    collide = sorted(set(f_ for v, fs in owner.items() if len(fs) > 1 for f_ in fs))
    out = {"forms": len(by_form), "distinct_vectors": len(keyb), "forms_with_several_vectors": multi,
           "forms_sharing_a_vector": len(collide), "collision_examples": [list(x) for x in collide[:8]]}
    # centred geometry on the first 120 instances
    X = Fm[:120].reshape(-1, d).double()
    X = X - X.mean(0)
    X = F.normalize(X, dim=-1).reshape(min(120, N), M, d)
    fl = [f_ for row in form[:120] for f_ in row]
    Xf = X.reshape(-1, d)
    idx = {}
    for i, f_ in enumerate(fl):
        idx.setdefault(f_, []).append(i)
    same = [float(Xf[v[0]] @ Xf[v[1]]) for v in idx.values() if len(v) > 1][:2000]
    firsts = {}
    for f_, v in idx.items():
        firsts.setdefault(f_[0], []).append(v[0])
    diff = []
    for kd, v in firsts.items():
        v = v[:40]
        G = Xf[v] @ Xf[v].t()
        diff += G[~torch.eye(len(v), dtype=torch.bool)].tolist()
    out["geom_centred_same_form"] = float(np.mean(same)) if same else None
    out["geom_centred_different_forms"] = float(np.mean(diff)) if diff else None
    # feature-hash reader
    m = B.Arm(ar, "exact", 42, d).to(dev)
    g = torch.Generator().manual_seed(4242)
    T = F.normalize(torch.randn(len(keyb), B.D, generator=g), dim=-1)
    scores = []
    with torch.no_grad():
        for k in range(0, N, 64):
            K = T[torch.tensor(vid[k:k + 64])].to(dev)
            if ar == "A":
                K = B.l2n(K)
            s_, _, _ = m.chain(K, "perm")
            scores.append(s_.cpu())
    s_ = torch.cat(scores).numpy()
    y = sp.y.numpy()
    acc, thr, tracc = B.thr_acc(s_, y, s_, y)
    out["hash_reader_train_acc"] = tracc
    return out


def sub(sp, n):
    s2 = copy.copy(sp)
    s2.F, s2.y, s2.sym, s2.st, s2.keys, s2.n = sp.F[:n], sp.y[:n], sp.sym[:n], sp.st[:n], sp.keys[:n], n
    return s2


def smoke(enc, ar, feats, dev):
    import b42_arms as A42
    tr = sub(B.Split(enc, ar, "train", feats), 64)
    tr.F = tr.F.to(dev)
    res = {}
    base = dict(enc_model=enc, arena=ar, mode="tune", lr=3e-4, cleanup="st", mult=1, beta=0.25, heads=8, div=0.1,
                tau0=2.0, tau_end=0.1, inw=0, code="center", idlam=0.0, epochs=1, seeds=[42], save_dev_scores=False,
                feats=feats, device=dev)
    for fam in ["vq", "dkvb", "gumbel", "st", "cw", "bind"]:
        a = SimpleNamespace(**dict(base, fam=fam))
        t0 = time.time()
        rec, snap = A42.run(a, 42, {"train": tr, "dev": tr}, dev)
        e = rec["epochs"][-1]
        res[fam] = {"train_rows_acc": e["perm/named:dev"]["acc"], "secs": round(time.time() - t0, 1)}
    a = SimpleNamespace(enc_model=enc, arena=ar, arm="name", mode="tune", enc="perm", conv="named", readings=[],
                        idset="all", readout="thr", lr=3e-4, lam=0.0, tau=0.1, epochs=1, fit_epochs=1, seeds=[42],
                        save_dev_scores=False, feats=feats, out_root="", device=dev)
    rec = B.run(a, 42, {"train": tr, "dev": tr}, dev)
    res["name"] = {"train_rows_acc": rec["epochs"][-1]["perm/named:dev"]["acc"],
                   "train_rows_naming": rec["epochs"][-1]["perm/named:dev"]["naming_acc_mentions"]}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--encs", nargs="*", default=ENCS)
    ap.add_argument("--arenas", default="ABC")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    rep = json.load(open(a.out)) if os.path.exists(a.out) else {}
    for enc in a.encs:
        for ar in a.arenas:
            k = "%s_%s" % (enc, ar)
            if not os.path.exists(os.path.join(a.feats, "%s_%s_train.pt" % (enc, ar))):
                print("missing", k); continue
            r = rep.get(k, {})
            if "forms" not in r:
                r.update(forms_and_hash(enc, ar, a.feats, a.device))
            if a.smoke and "smoke" not in r:
                r["smoke"] = smoke(enc, ar, a.feats, a.device)
            rep[k] = r
            print(k, json.dumps(r), flush=True)
            json.dump(rep, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
