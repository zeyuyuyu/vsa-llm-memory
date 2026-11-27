"""B31 gate G0-unit (prereg/B31_encoder_equalised_baselines_prereg.md): the new vectorised arms equal the ORIGINAL arm
classes on identical weights, on the Phi-2 training split (no dev or test instance is read).

For every arena and arm type (exact rows / naming head / binding-loss map) the original class (v13.BindArm,
v11.BindArm, v9.ExtractThenBind, v8.SpanVSA, SymbolicVSAChainClassifier) and b31_arms.Arm are built after the same
torch.manual_seed; the check reports (i) that the global-RNG draws give identical parameters (same initial head /
projection weights), (ii) the largest absolute difference of the final cosine (B, C) or of the positive logit (A) over
the first 200 training instances, and (iii) the share of instances with the same clean-up winners / decisions. Arena A
uses the published token scanner's view (b31_arms.scanner_struct) on both sides. Afterwards the original classes'
weights are perturbed (identical noise on both sides) to check equality away from the initialisation too.
"""
import json, os, sys
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import b27_common as BC                                                     # noqa: E402
import b31_arms as A31                                                      # noqa: E402

M = BC.load_arena_modules()
dev = "cuda:0"
N = 200
out = {}
tok = BC.tokenizer()
vis = torch.load("/path/to/work/b31/out/scanner_visible_A.pt")["train"]
for arena in ("A", "C", "B"):
    Dsp = A31.Split("phi2", arena, "train", "/path/to/work/b31/feats")
    obj = torch.load(BC.feats_path("/path/to/work/b27/feats", arena, "train"), weights_only=False)
    if arena == "A":
        ids_all, am_all, H_all, _ = obj
    else:
        H_all = obj
        rows = BC.build_rows(M, tok, arena, "train")
    for arm, pub in (("exact", "parser"), ("name", "probe_argmax"), ("bind", "spans")):
        for perturb in (False, True):
            torch.manual_seed(42)
            if arena == "C":
                P = M["v13"].BindArm(pub, 42).to(dev)
            elif arena == "B":
                P = M["v11"].BindArm(pub, M["systematic_role_binding_dataset"].SystematicConfig(), 42).to(dev)
            else:
                cls = {"exact": M["explicit_chain_models"].SymbolicVSAChainClassifier,
                       "name": M["v9"].ExtractThenBind, "bind": M["v8"].SpanVSA}[arm]
                P = cls(vsa_dim=2048, num_roles=64, vocab_size=256, seed=42).to(dev)
            torch.manual_seed(42)
            Q = A31.Arm(arena, arm, 42, Dsp.F.size(-1)).to(dev)
            same_init = None
            lin_p = getattr(P, "id_head", None) if arm == "name" else getattr(P, "proj", None)
            lin_q = getattr(Q, "head", None) if arm == "name" else getattr(Q, "proj", None)
            if lin_p is not None:
                same_init = bool(torch.equal(lin_p.weight, lin_q.weight) and torch.equal(lin_p.bias, lin_q.bias))
            if arena == "A":
                same_init = (same_init is not False) and bool(torch.equal(P.vsa.role_vectors.data, Q.table[:64]))
            if perturb and lin_p is not None:
                g = torch.Generator(device=dev); g.manual_seed(7)
                with torch.no_grad():
                    nz = torch.randn(lin_p.weight.shape, generator=g, device=dev) * lin_p.weight.std()
                    lin_p.weight.add_(nz); lin_q.weight.add_(nz)
            P.eval(); Q.eval()
            with torch.no_grad():
                if arena == "A":
                    lp_p, lp_q = [], []
                    for k in range(0, N, 8):
                        Hb = H_all[k:k + 8].to(dev)
                        if arm == "exact":       # the published parser arm reads no state
                            lg = P(ids_all[k:k + 8].to(dev), am_all[k:k + 8].to(dev))
                        else:
                            lg = P(ids_all[k:k + 8].to(dev), am_all[k:k + 8].to(dev), hidden=Hb)
                        lp_p.append(lg[:, 1].cpu())
                        st = A31.scanner_struct(vis[k:k + 8], dev)
                        K = Q.kcodes(Dsp.F[k:k + 8].to(dev), Dsp.sym[k:k + 8].to(dev), "index")
                        s_, w, ok = Q.chain(K, "comm", st)
                        l2 = Q.answer_logits(s_)[:, 1]
                        lp_q.append(torch.where(ok, l2, torch.zeros_like(l2)).cpu())
                    a_, b_ = torch.cat(lp_p), torch.cat(lp_q)
                    rep = {"max_abs_diff_logit": float((a_ - b_).abs().max()),
                           "same_decision": float(((a_ > 0) == (b_ > 0)).float().mean())}
                else:
                    sp_, sq_ = [], []
                    for k in range(0, N, 4):
                        sp_.append(P.sims(H_all[k:k + 4].to(dev), rows[k:k + 4], dev).cpu())
                        K = Q.kcodes(Dsp.F[k:k + 4].to(dev), Dsp.sym[k:k + 4].to(dev), "index")
                        sq_.append(Q.chain(K, "comm")[0].cpu())
                    a_, b_ = torch.cat(sp_), torch.cat(sq_)
                    rep = {"max_abs_diff_cos": float((a_ - b_).abs().max()),
                           "same_sign_at_0.1": float(((a_ > 0.1) == (b_ > 0.1)).float().mean())}
            rep["same_initial_parameters"] = same_init
            key = "%s/%s%s" % (arena, arm, "/perturbed" if perturb else "")
            out[key] = rep
            print(key, json.dumps(rep), flush=True)
json.dump(out, open("/path/to/work/b31/out/g0_unit.json", "w"), indent=1)
