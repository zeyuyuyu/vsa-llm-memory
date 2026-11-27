"""V-42: V-41 with the grouping decision rule specified against the right objective (prereg 3a2854e).

V-41 picked the threshold by pairwise F1, which weights a false merge and a false split equally. V-36 prices one
merge at twelve points and V-35 found over-splitting cheaper at equal rate, so that criterion was chosen against
an objective the mechanism does not have. Two changes: balanced pair sampling in training (positives were 1.4%%
of pairs), and the threshold chosen to maximise CHAIN ACCURACY ON TRAINING INSTANCES. Bars are V-41's, unchanged.

V-41 header follows.
V-41: can instance-local GROUPING replace global identity? (prereg 4efecad, bars registered before this file)

Every identity head in this paper is a fixed-class classifier over a closed, recurring symbol set. Real text is
open-vocabulary and most entities occur once, so that design cannot reach it. V-36 showed any INJECTIVE row
assignment works, which means the head does not need to NAME a symbol -- only to decide which mentions co-refer.
Rows then follow order of first appearance inside the instance.

The matching head is trained OUTSIDE v13.run and frozen before evaluation, the way frozen_cont is, so v13.run's
training loop still only fits scale and bias and the evaluation path stays identical to the mechanism table's.
It is trained on PAIRWISE same/different labels and has no output space in which a global symbol id could appear.
"""
import argparse, json, os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
sys.path.insert(0, "/path/to/work/ngcls")
import v13                                                            # noqa: E402

ARM = "group_argmax"
CB = 256                     # rows are arbitrary (V-36), so the group arm gets a codebook large enough that
                             # the number of recovered groups can never exceed it
_HEAD = {}                   # seed -> (frozen matching head, threshold)
_Base = v13.BindArm


def spans_of(r):
    """Every span in an instance with its true symbol id. Ids are used for TRAINING PAIRS ONLY."""
    out = []
    for L in range(v13.HOPS):
        for aa, bb, spr, spf in r["ctx"][L]:
            out.append((spr, aa))
            out.append((spf, bb))
    out.append((r["q_span"], r["q"]))
    out.append((r["cand_span"], r["cand"]))
    return out


class Matcher(nn.Module):
    def __init__(self, d=256):
        super().__init__()
        self.g = nn.Linear(v13.HID, d)

    def forward(self, X):
        return F.normalize(self.g(X.float()), dim=-1)


def sf(H, sp):
    return H[sp[0]:sp[1]].float().mean(0) if sp[1] > sp[0] else H[sp[0]].float()


def train_matcher(tok, llm, device, seed, n_tr, epochs=6, bs=8):
    """Pairwise same/different training. No global class ever exists in this objective."""
    tr = v13.Arena(tok, n_tr, 42)                      # v13.run uses this exact arena for training
    H = v13.cache(tr.rows, llm, device)
    mt = Matcher().to(device)
    opt = torch.optim.AdamW(mt.parameters(), lr=1e-3)
    g = torch.Generator().manual_seed(seed)
    idx = list(range(len(tr.rows)))
    for ep in range(epochs):
        tot, nb = 0.0, 0
        perm = torch.randperm(len(idx), generator=g).tolist()
        for k in range(0, len(perm), bs):
            loss = 0.0
            for i in perm[k:k + bs]:
                r = tr.rows[i]
                sp = spans_of(r)
                X = torch.stack([sf(H[i].to(device), s) for s, _ in sp])
                Z = mt(X)
                y = torch.tensor([q for _, q in sp], device=device)
                same = (y.unsqueeze(0) == y.unsqueeze(1)).float()
                S = (Z @ Z.T) * 8.0
                mask = ~torch.eye(len(sp), dtype=torch.bool, device=device)
                pos, neg = same.bool() & mask, (~same.bool()) & mask
                if pos.sum() == 0 or neg.sum() == 0:
                    continue
                # V-42: SAMPLE the negatives down to the number of positives. V-41 averaged over all negatives,
                # which is unbiased per-pair but leaves the head facing a 1.4%-positive problem and collapsing
                # toward calling everything the same symbol.
                sp_, sn_ = S[pos], S[neg]
                if sn_.numel() > sp_.numel():
                    pick = torch.randperm(sn_.numel(), device=sn_.device)[:sp_.numel()]
                    sn_ = sn_[pick]
                loss = loss + F.binary_cross_entropy_with_logits(sp_, torch.ones_like(sp_)) \
                            + F.binary_cross_entropy_with_logits(sn_, torch.zeros_like(sn_))
            if not torch.is_tensor(loss):
                continue
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.item()); nb += 1
        print("      matcher ep%d loss=%.4f" % (ep + 1, tot / max(1, nb)), flush=True)

    # V-42: the threshold is chosen on TRAINING instances by CHAIN accuracy, not by pairwise F1. Merging is the
    # expensive failure (V-36: twelve points for one merge) and over-splitting is cheap (V-35), so a criterion
    # that weights the two equally picks the wrong operating point. Pairwise F1 is still recorded, for comparison
    # with V-41, but it no longer selects anything.
    cbk = F.normalize(torch.randn(CB, v13.VSA_DIM,
                                 generator=torch.Generator().manual_seed(seed + 31)), dim=1).to(device)

    def chain_at(thr, rows_idx):
        ok = []
        for i in rows_idx:
            r = tr.rows[i]
            sp = spans_of(r)
            if any(len(l) == 0 for l in r["ctx"]):
                continue
            Z = mt(torch.stack([sf(H[i].to(device), s_) for s_, _ in sp]))
            gid = group(Z, thr)
            row = {(s_[0], s_[1]): cbk[gid[j] % CB] for j, (s_, _) in enumerate(sp)}
            key = row[(r["q_span"][0], r["q_span"][1])]
            for L in range(v13.HOPS):
                mem = torch.stack([v13.bind(row[(spf[0], spf[1])], row[(spr[0], spr[1])])
                                   for aa, bb, spr, spf in r["ctx"][L]], 0).sum(0)
                hat = F.normalize(v13.unbind(mem, key), dim=-1)
                cands = torch.stack([row[(spf[0], spf[1])] for aa, bb, spr, spf in r["ctx"][L]], 0)
                key = cands[int(F.cosine_similarity(cands, hat.unsqueeze(0)).argmax().item())]
            sim = float(torch.dot(key, row[(r["cand_span"][0], r["cand_span"][1])]))
            ok.append((sim, r["label"]))
        if not ok:
            return 0.5
        sims = np.array([a for a, _ in ok]); ys = np.array([b for _, b in ok])
        cs = np.unique(sims)
        return float(max((((sims >= t) == (ys == 1)).mean()) for t in cs)) if cs.size else 0.5

    with torch.no_grad():
        cos, lab = [], []
        for i in range(min(120, len(tr.rows))):
            r = tr.rows[i]
            sp = spans_of(r)
            Z = mt(torch.stack([sf(H[i].to(device), s_) for s_, _ in sp]))
            y = torch.tensor([q for _, q in sp])
            C = (Z @ Z.T).cpu()
            n = len(sp)
            iu = torch.triu_indices(n, n, offset=1)
            cos.append(C[iu[0], iu[1]]); lab.append((y[iu[0]] == y[iu[1]]).float())
        cos, lab = torch.cat(cos), torch.cat(lab)
        sel = list(range(min(140, len(tr.rows))))
        best, bt = -1.0, 0.5
        for t in np.linspace(0.0, 0.95, 20):
            a_ = chain_at(float(t), sel)
            if a_ > best:
                best, bt = a_, float(t)
        p_ = (cos >= bt).float()
        tp_ = (p_ * lab).sum()
        f1_at = float(2 * tp_ / (p_.sum() + lab.sum() + 1e-9))
    print("      matcher threshold=%.3f chosen by TRAIN CHAIN acc=%.4f (its pairwise F1=%.4f, positives %.3f%%)"
          % (bt, best, f1_at, 100.0 * float(lab.mean())), flush=True)
    best = f1_at
    for p in mt.parameters():
        p.requires_grad_(False)
    mt.eval()
    return mt, bt, best


def group(Z, thr):
    """Greedy single-link agglomeration in order of appearance; group index = order of first appearance."""
    gid = [-1] * Z.size(0)
    reps = []
    for i in range(Z.size(0)):
        placed = False
        for k, members in enumerate(reps):
            if float((Z[i:i + 1] @ Z[members].T).max()) >= thr:
                gid[i] = k; reps[k].append(i); placed = True; break
        if not placed:
            gid[i] = len(reps); reps.append([i])
    return gid


class GroupArm(_Base):
    def __init__(self, mode, seed):
        super().__init__("parser" if mode == ARM else mode, seed)
        self.mode = mode
        if mode == ARM:
            g = torch.Generator().manual_seed(seed + 31)
            self.register_buffer("cb", F.normalize(torch.randn(CB, v13.VSA_DIM, generator=g), dim=1))
            self.mt, self.thr, _ = _HEAD[seed]

    def q_vec(self, sid, device):
        if self.mode != ARM:
            return super().q_vec(sid, device)
        raise RuntimeError("q_vec must not be reached for the grouping arm: it would read a global id")

    def sims(self, H, rows, device):
        if self.mode != ARM:
            return super().sims(H, rows, device)
        out = torch.zeros(len(rows), device=device)
        for b, r in enumerate(rows):
            if any(len(l) == 0 for l in r["ctx"]):
                continue
            sp = spans_of(r)                                  # ids present but NEVER read below
            Z = self.mt(torch.stack([sf(H[b], s) for s, _ in sp]))
            gid = group(Z, self.thr)
            row = {}
            for j, (s, _) in enumerate(sp):
                row[(s[0], s[1])] = self.cb[gid[j] % CB].to(device)
            key = row[(r["q_span"][0], r["q_span"][1])]
            for L in range(v13.HOPS):
                mem = torch.stack([v13.bind(row[(spf[0], spf[1])], row[(spr[0], spr[1])])
                                   for aa, bb, spr, spf in r["ctx"][L]], 0).sum(0)
                hat = F.normalize(v13.unbind(mem, key), dim=-1)
                cands = torch.stack([row[(spf[0], spf[1])] for aa, bb, spr, spf in r["ctx"][L]], 0)
                key = cands[int(F.cosine_similarity(cands, hat.unsqueeze(0)).argmax().item())]
            out[b] = torch.dot(key, row[(r["cand_span"][0], r["cand_span"][1])]).clamp(-1, 1)
        return out


def eval_grouping(tok, llm, device, seed, n=150):
    mt, thr, _ = _HEAD[seed]
    ar = v13.Arena(tok, n, 4242)                              # v13.run's evaluation arena
    H = v13.cache(ar.rows, llm, device)
    tp = fp = fn = tn = 0
    ng, nt = [], []
    with torch.no_grad():
        for i, r in enumerate(ar.rows):
            sp = spans_of(r)
            Z = mt(torch.stack([sf(H[i].to(device), s) for s, _ in sp]))
            gid = torch.tensor(group(Z, thr))
            y = torch.tensor([q for _, q in sp])
            n_ = len(sp)
            iu = torch.triu_indices(n_, n_, offset=1)
            pred = (gid[iu[0]] == gid[iu[1]])
            true = (y[iu[0]] == y[iu[1]])
            tp += int((pred & true).sum()); fp += int((pred & ~true).sum())
            fn += int((~pred & true).sum()); tn += int((~pred & ~true).sum())
            ng.append(len(set(gid.tolist()))); nt.append(len(set(y.tolist())))
    f1 = 2 * tp / max(1, 2 * tp + fp + fn)
    # V-42: merge and split rates separately, which V-41 could not distinguish. merge_rate is the fraction of
    # DIFFERENT-symbol pairs wrongly joined (the expensive error); split_rate is the fraction of SAME-symbol
    # pairs wrongly separated (the cheap one).
    merge_rate = fp / max(1, fp + tn)
    split_rate = fn / max(1, fn + tp)
    return f1, float(np.mean(ng)), float(np.mean(nt)), merge_rate, split_rate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["floor", "parser", "probe_argmax", ARM])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    ap.add_argument("--n_train", type=int, default=800)
    ap.add_argument("--n_test", type=int, default=500)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--match_epochs", type=int, default=6)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="v42_logs/v42.json")
    a = ap.parse_args()

    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    llm = AutoModel.from_pretrained("microsoft/phi-2", trust_remote_code=True,
                                    torch_dtype=torch.float16).to(a.device).eval()
    for p in llm.parameters():
        p.requires_grad_(False)

    print("===== V-42  grouping with the decision rule specified by chain accuracy =====", flush=True)
    out = {"codebook": CB, "arms": {}, "matcher": {}, "grouping": {}}
    for s in a.seeds:
        print("    training matcher, seed %d (pairwise labels only)" % s, flush=True)
        _HEAD[s] = train_matcher(tok, llm, a.device, s, a.n_train, a.match_epochs)
        out["matcher"][str(s)] = {"threshold": _HEAD[s][1], "train_pair_f1": _HEAD[s][2]}
    v13.BindArm = GroupArm

    for arm in a.arms:
        accs = [v13.run(arm, s, tok, llm, a.device, a.n_train, a.n_test,
                        a.epochs, a.batch_size, a.lr)["acc"] for s in a.seeds]
        out["arms"][arm] = accs
        print("  %-13s %s mean=%.4f" % (arm, [round(x, 4) for x in accs], float(np.mean(accs))), flush=True)
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(out, open(a.out, "w"), indent=1)

    for s in a.seeds:
        f1, ng, nt, mr, sr = eval_grouping(tok, llm, a.device, s)
        out["grouping"][str(s)] = {"eval_pair_f1": f1, "groups": ng, "true_symbols": nt,
                                   "merge_rate": mr, "split_rate": sr}
        print("  DIAG seed %d  F1=%.4f groups=%.1f vs true=%.1f | merge_rate=%.4f split_rate=%.4f"
              % (s, f1, ng, nt, mr, sr), flush=True)

    m = {k: float(np.mean(v)) for k, v in out["arms"].items()}
    fl, pa = m["floor"], m["parser"]
    Hd = pa - fl
    out.update({"floor": fl, "parser": pa, "headroom": Hd})
    print("\n===== V-42 SCORING (bars = V-41's, registered at 4efecad) =====", flush=True)
    print("  floor=%.4f parser=%.4f H=%.4f  suffices >= %.4f partial >= %.4f fails <= %.4f"
          % (fl, pa, Hd, fl + 0.90 * Hd, fl + 0.50 * Hd, fl + 0.10 * Hd), flush=True)
    if Hd < 0.20:
        v = "VOID (admission failed: parser - floor = %.4f)" % Hd
    else:
        g_ = out["arms"][ARM]
        if all(x >= fl + 0.90 * Hd for x in g_):
            v = ("GROUPING SUFFICES (a closed recurring vocabulary is not a requirement of the mechanism; "
                 "open-vocabulary text is reachable in principle)")
        elif all(x >= fl + 0.50 * Hd for x in g_):
            v = "PARTIAL (grouping recovers part of the gap; the closed-vocabulary limit is narrowed)"
        elif all(x <= fl + 0.10 * Hd for x in g_):
            v = ("FAILS (global identity does work grouping cannot; the closed-vocabulary requirement becomes "
                 "an explicit scope limit)")
        else:
            v = "GRADED (report as measured; license neither reading)"
    out["verdict"] = v
    json.dump(out, open(a.out, "w"), indent=1)
    print("  >>> V-42 VERDICT: %s" % v, flush=True)
    print("V42-DONE", flush=True)


if __name__ == "__main__":
    main()
