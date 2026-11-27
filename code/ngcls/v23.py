"""V-23: identity from unsupervised clustering. No symbol labels anywhere.

The objection this answers. After V-22 nothing in the pipeline reads the input string, but the identity
targets still come from the task generator, which keeps the remedy a diagnostic rather than a deployable
method. That is the load-bearing objection to the paper.

Why the paper's own result says this is attackable. The `consistent` arm shows that relabelling EVERY symbol
consistently costs under a point: the algebra needs a code that is stable across occurrences, not one that
matches the generator's names. An unsupervised clustering of the span representations returns exactly that --
a permutation of the true identities -- using no per-instance labels. So k-means at k = vocabulary size,
fitted on TRAIN spans only and applied to eval spans by nearest centroid, is a label-free identity mechanism
that the paper's existing measurement predicts should work. If it does, the remedy stops being a diagnostic.

Every span is relabelled by the same map: the context roles and fillers, AND the query role and the candidate
filler. Relabelling the context while leaving the query at its true id is the V-11 candidate bug, and it
breaks the comparison by construction rather than measuring anything.

What is still supplied: span boundaries (V-22's detector needs span labels) and the vocabulary size k. What
is not supplied: any symbol identity, at training or evaluation.

Reported alongside and NOT gated: cluster purity and adjusted Rand index against the true symbols. A failure
at high purity means the chain is the obstacle; a failure at low purity means the clustering is. Without both
the two are indistinguishable, which is how arena B's anomaly survived for days.
"""
import argparse, json, os, sys
import numpy as np
import torch
import torch.nn.functional as F

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
sys.path.insert(0, "/path/to/work/ngcls")
import v13                                                            # noqa: E402


def pooled(H, sp):
    return H[sp[0]:sp[1]].float().mean(0) if sp[1] > sp[0] else H[sp[0]].float()


def all_spans(H, rows):
    """Every span in the arena, with its true id (used only for the purity diagnostic) and a slot tag."""
    X, Y, slot = [], [], []
    for b, r in enumerate(rows):
        for L, lvl in enumerate(r["ctx"]):
            for k, (ai, bi, spr, spf) in enumerate(lvl):
                X.append(pooled(H[b], spr)); Y.append(ai); slot.append(("ctx", b, L, k, 0))
                X.append(pooled(H[b], spf)); Y.append(bi); slot.append(("ctx", b, L, k, 1))
        X.append(pooled(H[b], r["q_span"])); Y.append(r["q"]); slot.append(("q", b, 0, 0, 0))
        X.append(pooled(H[b], r["cand_span"])); Y.append(r["cand"]); slot.append(("cand", b, 0, 0, 0))
    return torch.stack(X), np.array(Y), slot


def kmeans(X, k, iters=80, seed=0):
    g = torch.Generator().manual_seed(seed)
    Xn = F.normalize(X, dim=1)
    C = Xn[torch.randperm(Xn.size(0), generator=g)[:k]].clone()
    for _ in range(iters):
        a = (Xn @ C.t()).argmax(1)
        for j in range(k):
            m = a == j
            if m.any():
                C[j] = F.normalize(Xn[m].mean(0), dim=0)
    return C


def assign(X, C):
    return (F.normalize(X, dim=1) @ C.t()).argmax(1).tolist()


def purity_ari(a, t, k):
    a, t = np.asarray(a), np.asarray(t)
    correct = sum(np.bincount(t[a == j]).max() for j in range(k) if (a == j).sum())
    pur = correct / len(t)
    ku, tu = np.unique(a), np.unique(t)
    cm = np.zeros((len(ku), len(tu)), dtype=np.int64)
    ki = {v: i for i, v in enumerate(ku)}; ti = {v: i for i, v in enumerate(tu)}
    for x, y in zip(a, t):
        cm[ki[x], ti[y]] += 1
    c2 = lambda z: z * (z - 1) // 2                                          # noqa: E731
    sij, sa, sb = c2(cm).sum(), c2(cm.sum(1)).sum(), c2(cm.sum(0)).sum()
    n2 = c2(np.int64(len(t)))
    exp, mx = (sa * sb / n2 if n2 else 0.0), (sa + sb) / 2
    return pur, ((sij - exp) / (mx - exp) if (mx - exp) else 0.0)


def relabel(rows, X, slot, C):
    ids = assign(X, C)
    ctx_new = {}
    for (kind, b, L, k, leg), cid in zip(slot, ids):
        if kind == "ctx":
            ctx_new.setdefault((b, L, k), {})[leg] = int(cid)
        elif kind == "q":
            rows[b]["q"] = int(cid)
        else:
            rows[b]["cand"] = int(cid)
    for b, r in enumerate(rows):
        r["ctx"] = [[(ctx_new[(b, L, k)][0], ctx_new[(b, L, k)][1], spr, spf)
                     for k, (_a, _b2, spr, spf) in enumerate(lvl)]
                    for L, lvl in enumerate(r["ctx"])]


def run_chain(tr, te, Htr, Hte, seed, a):
    torch.manual_seed(seed)
    m = v13.BindArm("parser", seed).to(a.device)
    sc = [q for n, q in m.named_parameters() if q.requires_grad and n in ("scale", "bias")]
    opt = torch.optim.AdamW([{"params": sc, "lr": 1e-1}]) if sc else None
    params = [p for p in m.parameters() if p.requires_grad]

    def collect(rows, Hall):
        m.eval(); S, Y = [], []
        with torch.no_grad():
            for k in range(0, len(rows), a.batch_size):
                S.append(m.sims(Hall[k:k + a.batch_size].to(a.device),
                                rows[k:k + a.batch_size], a.device).cpu())
                Y += [r["label"] for r in rows[k:k + a.batch_size]]
        return torch.cat(S).numpy(), np.array(Y)

    def acc():
        st, yt = collect(tr.rows, Htr); se, ye = collect(te.rows, Hte)
        c = np.unique(st)
        if c.size == 0:
            return 0.5
        thr = float(c[int(np.argmax([(((st >= t) == (yt == 1)).mean()) for t in c]))])
        return float((((se >= thr) == (ye == 1)).mean()))

    best = 0.0
    g = torch.Generator(); g.manual_seed(seed)
    for ep in range(a.epochs):
        m.train()
        perm = torch.randperm(len(tr.rows), generator=g).tolist()
        for k in range(0, len(perm), a.batch_size):
            sel = perm[k:k + a.batch_size]; rows = [tr.rows[i] for i in sel]
            H = Htr[sel].to(a.device)
            am = torch.stack([r["am"] for r in rows]).to(a.device)
            y = torch.tensor([r["label"] for r in rows], device=a.device)
            loss = F.cross_entropy(m(H, am, rows, a.device), y)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        best = max(best, acc())
        print("      clustered ep%d test=%.4f" % (ep + 1, best), flush=True)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hops", type=int, default=2)
    ap.add_argument("--n_ctx", type=int, default=32)
    ap.add_argument("--n_sym", type=int, default=64)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    ap.add_argument("--n_train", type=int, default=800)
    ap.add_argument("--n_test", type=int, default=500)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="v23_logs/v23.json")
    a = ap.parse_args()

    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    llm = AutoModel.from_pretrained("microsoft/phi-2", trust_remote_code=True,
                                    torch_dtype=torch.float16).to(a.device).eval()
    for p in llm.parameters():
        p.requires_grad_(False)

    v13.HOPS, v13.N_CTX, v13.N_SYM = a.hops, a.n_ctx, a.n_sym
    print("===== V-23  HOPS=%d N_CTX=%d N_SYM=%d =====" % (v13.HOPS, v13.N_CTX, v13.N_SYM), flush=True)

    out = {"hops": a.hops, "n_ctx": a.n_ctx, "n_sym": a.n_sym, "arms": {}, "cluster": [], "clustered": []}
    for arm in ["floor", "parser", "consistent"]:
        accs = [v13.run(arm, s, tok, llm, a.device, a.n_train, a.n_test,
                        a.epochs, a.batch_size, a.lr)["acc"] for s in a.seeds]
        out["arms"][arm] = accs
        print("  %-14s %s mean=%.4f" % (arm, accs, float(np.mean(accs))), flush=True)
    fl = float(np.mean(out["arms"]["floor"])); pa = float(np.mean(out["arms"]["parser"]))
    H = pa - fl
    print("  floor=%.4f parser=%.4f headroom=%.4f" % (fl, pa, H), flush=True)

    for seed in a.seeds:
        torch.manual_seed(seed)
        tr = v13.Arena(tok, a.n_train, 42, max_len=2048)
        te = v13.Arena(tok, a.n_test, 4242, max_len=2048)
        for nm, ar in (("train", tr), ("eval", te)):
            bad = sum(1 for r in ar.rows
                      if len(r["ctx"]) != v13.HOPS or any(len(l) != v13.N_CTX for l in r["ctx"])
                      or r["q_span"] == (0, 1) or r["cand_span"] == (0, 1))
            if bad:
                raise SystemExit("TRUNCATION or degenerate query span in %s: %d/%d rows. Aborting."
                                 % (nm, bad, len(ar.rows)))
        Htr = v13.cache(tr.rows, llm, a.device)
        Hte = v13.cache(te.rows, llm, a.device)

        Xtr, Ytr, str_ = all_spans(Htr, tr.rows)
        C = kmeans(Xtr, a.n_sym, seed=seed)
        pur, ari = purity_ari(assign(Xtr, C), Ytr, a.n_sym)
        Xte, Yte, ste = all_spans(Hte, te.rows)
        pur_e, ari_e = purity_ari(assign(Xte, C), Yte, a.n_sym)
        out["cluster"].append({"seed": seed, "train_purity": float(pur), "train_ari": float(ari),
                               "eval_purity": float(pur_e), "eval_ari": float(ari_e),
                               "n_train_spans": int(Xtr.size(0))})
        print("  seed %d  clustering: train purity=%.4f ARI=%.4f | eval purity=%.4f ARI=%.4f  (k=%d, %d spans)"
              % (seed, pur, ari, pur_e, ari_e, a.n_sym, Xtr.size(0)), flush=True)

        relabel(tr.rows, Xtr, str_, C)
        relabel(te.rows, Xte, ste, C)
        acc = run_chain(tr, te, Htr, Hte, seed, a)
        out["clustered"].append(acc)
        print("  seed %d  chain on UNSUPERVISED codes: acc=%.4f" % (seed, acc), flush=True)
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(out, open(a.out, "w"), indent=1)

    cm = float(np.mean(out["clustered"]))
    out["clustered_mean"] = cm
    json.dump(out, open(a.out, "w"), indent=1)
    print("\n===== V-23 SCORING (bars registered before this run) =====", flush=True)
    print("  floor=%.4f parser=%.4f consistent=%.4f  unsupervised=%.4f  headroom=%.4f"
          % (fl, pa, float(np.mean(out["arms"]["consistent"])), cm, H), flush=True)
    print("  purity (eval): %s" % [round(c["eval_purity"], 4) for c in out["cluster"]], flush=True)
    dep = all(x >= pa - 0.10 * H for x in out["clustered"])
    req = all(x <= fl + 0.20 * H for x in out["clustered"])
    print("  within 0.10H of the parser in 3/3: %s" % dep, flush=True)
    if dep:
        v = "DEPLOYABLE (identity needs no labels; consistency from clustering suffices)"
    elif req:
        v = "SUPERVISION REQUIRED (label-free codes put the chain at the floor)"
    else:
        v = "INTERMEDIATE (report the measured value; license neither reading)"
    print("  >>> V-23 VERDICT: %s" % v, flush=True)
    print("V23-DONE", flush=True)


if __name__ == "__main__":
    main()
