"""B42 check made before registration (training split only): which mention codes the answer loss reaches through the
published chain (hard clean-up) and through the straight-through clean-up. Fraction of mentions, per role in the chain,
whose code receives a non-zero gradient, over the first 64 training instances of each arena (binding-loss map at its
initialisation, seed 42).

  python b42_gradpath.py --enc_model phi2 --out out/gradpath_phi2.json
"""
import argparse, json, os, sys
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import b31_arms as B                                                       # noqa: E402
import b42_arms as A42                                                     # noqa: E402


def slots(arena):
    kinds, hops, q, c, _ = B.layout(arena)
    out = {"query": [q], "candidate": [c]}
    for h, (kp, vp) in enumerate(hops):
        out["hop%d keys (%s)" % (h + 1, kinds[kp[0]])] = kp
        out["hop%d values (%s)" % (h + 1, kinds[vp[0]])] = vp
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enc_model", default="phi2")
    ap.add_argument("--feats", default="/path/to/work/b31/feats")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = "cuda:0"
    res = {}
    for arena in "ABC":
        tr = B.Split(a.enc_model, arena, "train", a.feats)
        tr.F = tr.F[:64]; tr.sym = tr.sym[:64]; tr.y = tr.y[:64]; tr.n = 64
        ns = argparse.Namespace(arena=arena, fam="bind", inw=0, code="none", idlam=0.0)
        torch.manual_seed(42)
        m = A42.Arm42(ns, 42, tr.F.size(-1), tr, dev)
        for cl in ("hard", "st"):
            Fb = tr.F.to(dev)
            K, _ = m.codes(Fb, tr.sym.to(dev), train=True)
            K.retain_grad()
            s_, _, _ = m.chain42(K, "perm", cl)
            loss = F.cross_entropy(m.answer_logits(s_), tr.y.to(dev))
            loss.backward()
            g = K.grad.norm(dim=-1)                                             # (64, M)
            res["%s/%s" % (arena, cl)] = {k: float((g[:, v] > 0).float().mean()) for k, v in slots(arena).items()}
            print(arena, cl, json.dumps(res["%s/%s" % (arena, cl)]), flush=True)
            m.zero_grad()
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
