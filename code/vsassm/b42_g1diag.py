"""B42 diagnostic after the A/B test run (training split only; changes no number): why the gumbel / st code path with
the identity loss (gate G1) does not reach the naming head in arena A. Gradient norms on the head from the answer loss
and from the identity loss, at initialisation and after 200 steps of the G1 configuration, for arenas A and C.

  python b42_g1diag.py --enc_model phi2 --fam gumbel --arena A
"""
import argparse, os, sys
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import b31_arms as B                                                       # noqa: E402
import b42_arms as A42                                                     # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enc_model", default="phi2")
    ap.add_argument("--arena", default="A")
    ap.add_argument("--fam", default="gumbel")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    tr = B.Split(a.enc_model, a.arena, "train", "/path/to/work/b31/feats")
    ns = argparse.Namespace(arena=a.arena, fam=a.fam, idlam=1.0, inw=0, code="none")
    torch.manual_seed(42)
    m = A42.Arm42(ns, 42, tr.F.size(-1), tr, a.device)
    rest = [p for n, p in m.named_parameters() if p.requires_grad and n not in ("scale", "bias")]
    sc = [p for n, p in m.named_parameters() if p.requires_grad and n in ("scale", "bias")]
    opt = torch.optim.AdamW(rest, lr=3e-4) if a.arena == "A" else torch.optim.AdamW(
        [{"params": rest, "lr": 3e-4}, {"params": sc, "lr": 0.1}])
    params = [p for p in m.parameters() if p.requires_grad]
    bs = B.BS[a.arena]
    g = torch.Generator().manual_seed(42)
    perm = torch.randperm(tr.n, generator=g)
    for step in range(a.steps + 1):
        i = perm[(step * bs) % tr.n:(step * bs) % tr.n + bs]
        Fb, sy, y = tr.F[i].to(a.device), tr.sym[i].to(a.device), tr.y[i].to(a.device)
        K, aux = m.codes(Fb, sy, train=True, tau=2.0 if step < 50 else 1.0)
        s_, _, _ = m.chain42(K, "perm", "hard")
        lt = F.cross_entropy(m.answer_logits(s_), y)
        li = aux["id"]
        if step in (0, a.steps):
            gt = torch.autograd.grad(lt, m.head.weight, retain_graph=True)[0].norm().item()
            gi = torch.autograd.grad(li, m.head.weight, retain_graph=True)[0].norm().item()
            tot = torch.autograd.grad(lt + li, params, retain_graph=True)
            tn = torch.sqrt(sum((x ** 2).sum() for x in tot if x is not None)).item()
            scale = (m.sim_scale.item(), m.sim_bias.item()) if a.arena == "A" else (m.scale.item(), m.bias.item())
            print("%s %s %s step %d: |grad head| answer %.4g identity %.4g; total grad norm %.4g (clip 1.0 -> x%.3g); "
                  "answer loss %.3f identity loss %.3f; scale/bias %.2f/%.2f" % (
                      a.enc_model, a.arena, a.fam, step, gt, gi, tn, min(1.0, 1.0 / max(tn, 1e-12)), lt.item(), li.item(),
                      *scale), flush=True)
        opt.zero_grad(); (lt + li).backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()


if __name__ == "__main__":
    main()
