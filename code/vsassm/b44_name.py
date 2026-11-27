"""B44 memory arms and probe on the frozen states, at 800 / 8,000 / 80,000 training instances and in the Fig. 2c
pairs-per-level sweep (prereg/B44_data_scaling_heldout_cot_crosstalk_prereg.md).

The arms are B31's code, unchanged (b31_arms.Arm and b31_arms.run; b31_probe.train_eval), run on B44's splits:
  name   naming head + VSA, perm encoding, named convention, identity loss on every mention, AdamW 3e-4, batch 4, clip
         1.0, 4 epochs; the test is read at each seed's dev-best epoch (scores of every epoch are saved, as in B31)
  exact  exact codebook rows + VSA (perm; comm is read from the same arm), threshold fitted on the training split
  probe  the equalised probe: B31's grid (feature sets last / eq / eqx, logistic and MLP), selected on dev, retrained
         with seeds 42 / 123 / 456 at the selected step
The only differences from B31 are where the features come from (b44_feats: fp16 storage, read back as fp32) and, in
the Fig. 2c sweep, the number of pairs per level k (the arena-C layout of b31_arms is rebuilt for 2k pairs).

  python b44_name.py --cell std:8000 --arm name --out out/name/std_n8000.json
  python b44_name.py --cell k8 --enc phi2 --arm name --out out/k/phi2_k8_name.json
"""
import argparse, json, os, sys, time
from types import SimpleNamespace
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import b31_arms as B                                                        # noqa: E402
import b31_probe as BPR                                                     # noqa: E402


def set_k(k):
    """Arena-C layout with k pairs per level (k = 32: B31's layout, unchanged)."""
    orig = B.layout

    def layout(arena):
        if arena != "C" or k == 32:
            return orig(arena)
        kinds = ["R", "F"] * (2 * k) + ["R", "F"]
        hops = [(list(range(0, 2 * k, 2)), list(range(1, 2 * k, 2))),
                (list(range(2 * k, 4 * k, 2)), list(range(2 * k + 1, 4 * k, 2)))]
        q, c = 4 * k, 4 * k + 1
        return kinds, hops, q, c, [q, c]
    B.layout = layout
    BPR.layout = layout


class FView:
    """fp16 memmap(s) (N, M, d) read back as fp32 torch tensors; supports the indexing b31_arms.run uses. The split
    may be stored whole (<tag>.F16.npy) or in contiguous chunks (<tag>.c<i>of<k>.F16.npy, b44_feats --nchunks)."""
    def __init__(self, paths, n=None):
        self.arrs = [np.load(p, mmap_mode="r") for p in paths]
        self.off = np.cumsum([0] + [x.shape[0] for x in self.arrs])
        self.n = n or int(self.off[-1])
        self.arr = self.arrs[0] if len(self.arrs) == 1 else None

    def _take(self, idx):
        idx = np.asarray(idx)
        out = np.empty((len(idx),) + self.arrs[0].shape[1:], dtype=np.float16)
        ch = np.searchsorted(self.off, idx, side="right") - 1
        for c in np.unique(ch):
            m = ch == c
            out[m] = self.arrs[c][idx[m] - self.off[c]]
        return out

    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.numpy()
        if isinstance(idx, slice):
            st, sp, _ = idx.indices(self.n)
            if self.arr is not None:
                return torch.from_numpy(np.ascontiguousarray(self.arr[st:sp])).float()
            idx = np.arange(st, sp)
        return torch.from_numpy(self._take(idx)).float()

    def cols(self, j):
        """All rows of mention column j (the probe's query / candidate spans)."""
        out, need = [], self.n
        for a in self.arrs:
            if need <= 0:
                break
            t = min(need, a.shape[0]); out.append(np.asarray(a[:t, j])); need -= t
        return np.concatenate(out)

    def size(self, d):
        return self.arrs[0].shape[d] if d != 0 else self.n


def _parts(feats, tag):
    whole = os.path.join(feats, tag + ".pt")
    if os.path.exists(whole):
        return [whole]
    import glob as G
    ps = sorted(G.glob(os.path.join(feats, tag + ".c*of*.pt")), key=lambda p: int(p.rsplit(".c", 1)[1].split("of")[0]))
    assert ps, "no features for %s" % tag
    k = int(ps[0].rsplit("of", 1)[1].split(".")[0])
    assert len(ps) == k, "missing chunks for %s: %d of %d" % (tag, len(ps), k)
    return ps


class Split:
    def __init__(self, feats, enc, item, n=None):
        tag = "%s_%s" % (enc, item.replace(":", "_"))
        parts = _parts(feats, tag)
        objs = [torch.load(p, weights_only=False) for p in parts]
        N = n or sum(o["n"] for o in objs)
        self.F = FView([p[:-3] + ".F16.npy" for p in parts], N)
        cat = lambda k: torch.cat([o[k] for o in objs])[:N]
        self.y = cat("labels").long()
        self.sym = cat("sym").long()
        self.keys = None
        self.n = N
        self.small = {"last": cat("last"), "mean": cat("mean"), "y": self.y}
        self.text_md5 = objs[0]["text_md5"]


def probe_feats(sp, q, c):
    d = dict(sp.small)
    d["q"] = torch.from_numpy(np.ascontiguousarray(sp.F.cols(q))).float()
    d["c"] = torch.from_numpy(np.ascontiguousarray(sp.F.cols(c))).float()
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True, help="std:<n> | ho:<n> | k<k>")
    ap.add_argument("--enc", default="qwen8", choices=["qwen8", "phi2"])
    ap.add_argument("--arm", required=True, choices=["name", "exact", "probe"])
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    ap.add_argument("--feats", default=os.path.join(HERE, "feats"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if os.path.exists(a.out):
        print("exists, skip", a.out); return
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    if a.cell.startswith("k"):
        k = int(a.cell[1:]); set_k(k)
        D_ = {sp: Split(a.feats, a.enc, "k%d:%s" % (k, sp)) for sp in ("train", "dev", "test")}
    else:
        v, n = a.cell.split(":")
        D_ = {"train": Split(a.feats, a.enc, "%s:train" % v, int(n)),
              "dev": Split(a.feats, a.enc, "%s:dev" % v), "test": Split(a.feats, a.enc, "%s:test" % v)}
    out = {"args": vars(a), "host": os.uname()[1], "start": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "n": {k_: v_.n for k_, v_ in D_.items()}, "text_md5": {k_: v_.text_md5 for k_, v_ in D_.items()}}
    if a.arm == "probe":
        _, _, q, c, _ = B.layout("C")
        tr, dv, te = (probe_feats(D_[s], q, c) for s in ("train", "dev", "test"))
        stats = BPR.make_stats(tr)
        grid = []
        for fs in BPR.FSETS:
            X, Xd = BPR.design(fs, tr, stats), BPR.design(fs, dv, stats)
            for model, hs in (("logistic", BPR.LAMS), ("mlp", BPR.WDS)):
                for hyp in hs:
                    curve, _ = BPR.train_eval(model, hyp, X, tr["y"], Xd, dv["y"], None, 42, a.device)
                    bst = max(curve, key=lambda t: (t[2], -t[0]))
                    grid.append({"fset": fs, "model": model, "hyper": hyp, "best_step": bst[0], "best_dev": bst[2],
                                 "train_at_best": bst[1]})
                    print("  probe %s %s %s %g dev %.4f @%d" % (a.cell, fs, model, hyp, bst[2], bst[0]), flush=True)
        rank = {"logistic": 0, "mlp": 1}
        sel = max(grid, key=lambda g: (round(g["best_dev"], 6), -rank[g["model"]], -BPR.FSETS.index(g["fset"]),
                                       g["hyper"], -g["best_step"]))
        out["grid"], out["selected"], out["runs"] = grid, sel, []
        X, Xd, Xt = (BPR.design(sel["fset"], d, stats) for d in (tr, dv, te))
        for seed in a.seeds:
            curve, ts = BPR.train_eval(sel["model"], sel["hyper"], X, tr["y"], Xd, dv["y"], Xt, seed, a.device,
                                       steps_at=sel["best_step"])
            acc = float(((ts > 0).astype(int) == te["y"].numpy()).mean())
            out["runs"].append({"seed": seed, "test_acc": acc, "test_scores": [round(float(x), 5) for x in ts]})
            print("  TEST probe %s seed %d %.4f" % (a.cell, seed, acc), flush=True)
    else:
        args = SimpleNamespace(arena="C", arm=a.arm, mode="final", enc="perm", conv="named",
                               readings=["comm/named"] if a.arm == "exact" else [], idset="all", readout="thr",
                               lr=3e-4, lam=0.0, tau=0.1, epochs=a.epochs, fit_epochs=6, save_dev_scores=False,
                               out_root=None)
        out["runs"] = []
        for s in a.seeds:
            rec = B.run(args, s, D_, a.device)
            for e in rec["epochs"]:
                for kk in list(e):
                    if kk.endswith(":dev") and "scores" in e[kk]:
                        del e[kk]["scores"]
            out["runs"].append(rec)
            json.dump(out, open(a.out + ".part", "w"))
    out["end"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    json.dump(out, open(a.out + ".tmp", "w")); os.replace(a.out + ".tmp", a.out)
    if os.path.exists(a.out + ".part"):
        os.remove(a.out + ".part")


if __name__ == "__main__":
    main()
