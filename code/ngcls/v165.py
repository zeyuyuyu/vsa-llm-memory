"""V-165 (prereg in ledger.md, committed before this file): cross-talk of the projection trained through the binding
loss, measured on the arm exactly as it is trained for the three-arena table (v13.run, arm "spans", arena C).

The manuscript plotted this arm at cross-talk 0.699, which is the cross-talk of an UNTRAINED projection
(crosstalk_audit2, fit0). Here the arm is trained as in v13.run (train arena seed 42, 800 instances; test arena seed
4242, 500 instances; 2 epochs, batch 4, AdamW 3e-4, gradient clipping 1.0; seeds 42/123/456) and the cosine between
the vectors different symbols feed the store is measured before training and after each epoch, with
crosstalk_audit2's protocol (first 120 test instances, up to 4 occurrences per symbol, first 40 symbols).
"""
import argparse, json, os, sys
import numpy as np
import torch
import torch.nn.functional as F

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
sys.path.insert(0, "/path/to/work/ngcls")
import v13                                                            # noqa: E402

BANKED = {42: 0.472, 123: 0.502, 456: 0.476}                          # v13_logs/v13_fixed.json, arm "spans"


def geometry(m, H, rows, dev):
    by = {}
    with torch.no_grad():
        for b, r in enumerate(rows[:120]):
            if any(len(l) == 0 for l in r["ctx"]):
                continue
            for L in range(v13.HOPS):
                for aa, bb, spr, spf in r["ctx"][L]:
                    for sid, sp in ((aa, spr), (bb, spf)):
                        if len(by.get(sid, [])) >= 4:
                            continue
                        by.setdefault(sid, []).append(m.vec(H[b].to(dev), sid, sp, dev))
    ks = [k for k in by if by[k]][:40]
    diff = [float(F.cosine_similarity(by[ks[i]][0].unsqueeze(0), by[ks[j]][0].unsqueeze(0)))
            for i in range(len(ks)) for j in range(i + 1, len(ks))]
    same = [float(F.cosine_similarity(by[k][i].unsqueeze(0), by[k][j].unsqueeze(0)))
            for k in ks for i in range(len(by[k])) for j in range(i + 1, len(by[k]))]
    return float(np.mean(diff)), float(np.mean(same)), len(ks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="v165_logs/v165.json")
    a = ap.parse_args()
    dev, bs, epochs, lr = a.device, 4, 2, 3e-4
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    llm = AutoModel.from_pretrained("microsoft/phi-2", trust_remote_code=True, torch_dtype=torch.float16).to(dev).eval()
    for p in llm.parameters():
        p.requires_grad_(False)
    print("===== V-165: cross-talk of the projection trained through the binding loss (Table configuration) =====",
          flush=True)
    tr, te = v13.Arena(tok, 800, 42), v13.Arena(tok, 500, 4242)
    Htr, Hte = v13.cache(tr.rows, llm, dev), v13.cache(te.rows, llm, dev)
    out = {"seeds": []}
    for seed in (42, 123, 456):
        torch.manual_seed(seed)
        m = v13.BindArm("spans", seed).to(dev)
        sc = [q for n, q in m.named_parameters() if q.requires_grad and n in ("scale", "bias")]
        rest = [q for n, q in m.named_parameters() if q.requires_grad and n not in ("scale", "bias")]
        opt = torch.optim.AdamW([{"params": rest, "lr": lr}, {"params": sc, "lr": 1e-1}])
        params = [p for p in m.parameters() if p.requires_grad]

        def collect(rows, Hall):
            m.eval(); S, Y = [], []
            with torch.no_grad():
                for k in range(0, len(rows), bs):
                    S.append(m.sims(Hall[k:k + bs].to(dev), rows[k:k + bs], dev).cpu())
                    Y += [r["label"] for r in rows[k:k + bs]]
            return torch.cat(S).numpy(), np.array(Y)

        def acc():
            st, yt = collect(tr.rows, Htr); se, ye = collect(te.rows, Hte)
            c = np.unique(st)
            thr = float(c[int(np.argmax([(((st >= t) == (yt == 1)).mean()) for t in c]))])
            return float((((se >= thr) == (ye == 1)).mean()))

        rec = {"seed": seed, "epochs": []}
        d0, s0, n0 = geometry(m, Hte, te.rows, dev)
        rec["init"] = {"different": d0, "same": s0, "symbols": n0, "acc": acc()}
        best = 0.0
        g = torch.Generator(); g.manual_seed(seed)
        for ep in range(epochs):
            m.train()
            perm = torch.randperm(len(tr.rows), generator=g).tolist()
            for k in range(0, len(perm), bs):
                sel = perm[k:k + bs]; rows = [tr.rows[i] for i in sel]
                H = Htr[sel].to(dev)
                am = torch.stack([r["am"] for r in rows]).to(dev)
                y = torch.tensor([r["label"] for r in rows], device=dev)
                loss = F.cross_entropy(m(H, am, rows, dev), y)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
            aep = acc(); best = max(best, aep)
            m.eval(); d, s, _ = geometry(m, Hte, te.rows, dev)
            rec["epochs"].append({"epoch": ep + 1, "acc": aep, "different": d, "same": s, "loss_last": float(loss)})
        rec["acc_best"] = best; rec["banked"] = BANKED[seed]
        out["seeds"].append(rec)
        print("  seed %d: init different %.4f same %.4f (acc %.4f) | after training different %.4f same %.4f | "
              "acc best %.4f (banked %.3f)"
              % (seed, d0, s0, rec["init"]["acc"], rec["epochs"][-1]["different"], rec["epochs"][-1]["same"], best,
                 BANKED[seed]), flush=True)
        json.dump(out, open(a.out, "w"), indent=1)

    S = out["seeds"]
    g0 = all(abs(r["acc_best"] - r["banked"]) <= 0.03 for r in S)
    b1 = all(r["epochs"][-1]["different"] >= 0.90 for r in S)
    b2 = all(r["epochs"][-1]["different"] > r["init"]["different"] for r in S)
    out.update(G0=g0, B1=b1, B2=b2,
               different_trained_mean=float(np.mean([r["epochs"][-1]["different"] for r in S])),
               same_trained_mean=float(np.mean([r["epochs"][-1]["same"] for r in S])),
               different_init_mean=float(np.mean([r["init"]["different"] for r in S])),
               same_init_mean=float(np.mean([r["init"]["same"] for r in S])),
               acc_best_mean=float(np.mean([r["acc_best"] for r in S])))
    json.dump(out, open(a.out, "w"), indent=1)
    print("V-165 VERDICT: G0 (accuracy within 0.03 of the banked arm) %s | B1 (trained cross-talk >= 0.90, 3/3) %s | "
          "B2 (higher after training than at init, 3/3) %s | trained different %.4f same %.4f; init different %.4f "
          "same %.4f; acc %.4f"
          % ("PASS" if g0 else "FAIL", "HOLDS" if b1 else "FAILS", "HOLDS" if b2 else "FAILS",
             out["different_trained_mean"], out["same_trained_mean"], out["different_init_mean"],
             out["same_init_mean"], out["acc_best_mean"]), flush=True)
    print("V165-RUN-DONE", flush=True)


if __name__ == "__main__":
    main()
