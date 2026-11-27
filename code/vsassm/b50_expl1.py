"""B50 exploratory 1 (prereg/B50_exploratory1.md): a least-squares one-hot type readout (MSE-ONEHOT) as DISC's gate.

  python b50_expl1.py fit <b40_root> <b50_root>         (box; cached layer-20 states) -> <b50_root>/out/expl1_scores.json
  python b50_expl1.py read <b50_local> <b40_pred_root>  (local; composes accuracy)     -> results/b50_exploratory1.txt
"""
import json, sys
import numpy as np

SEEDS = (0, 1, 2)


def fit(b40, b50):
    import torch
    import b40_data as D
    H = torch.load(f"{b40}/cache/N_train_L20.pt").double()
    y = torch.tensor([D.OPS_TRAIN.index(o) for o in json.load(open(f"{b40}/cache/N_train_meta.json"))["op"]])
    X = torch.cat([H, torch.ones(len(H), 1, dtype=torch.float64)], 1)
    Y = torch.nn.functional.one_hot(y, len(D.OPS_TRAIN)).double()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    W = torch.linalg.lstsq(X.to(dev), Y.to(dev)).solution.cpu()  # ordinary least squares with intercept, no ridge
    score = lambda Z: (torch.cat([Z, torch.ones(len(Z), 1, dtype=torch.float64)], 1) @ W)
    tr_out = score(H)
    Zd = torch.load(f"{b40}/cache/N_dev_L20.pt").double()
    Zt = torch.load(f"{b50}/cache/N_test_L20.pt").double()
    od, ot = score(Zd), score(Zt)
    rows = D.gen("N", "test")
    res = {"train_fit_acc": float((tr_out.argmax(1) == y).double().mean()),
           "dev_scores": od.max(1).values.tolist(),
           "dev_type_acc": float((od.argmax(1) == torch.tensor([D.OPS_TRAIN.index(o) for o in
                                  json.load(open(f"{b40}/cache/N_dev_meta.json"))["op"]])).double().mean()),
           "test": {r["id"]: {"score": float(ot[i].max()), "type": D.OPS_TRAIN[int(ot[i].argmax())]}
                    for i, r in enumerate(rows)}}
    json.dump(res, open(f"{b50}/out/expl1_scores.json", "w"))
    torch.save(W, f"{b50}/out/lsq_W.pt")  # for the live check (b50_run.py --gate lsq --lsq out/lsq_W.pt)
    print("fit acc", res["train_fit_acc"], "dev type acc", res["dev_type_acc"])


def read(b50, b40p):
    import b40_data as D
    from b40_verdict import Arm, boot_diff, fmt
    from b50_verdict import auroc, Composed
    rng = np.random.default_rng(5051)
    sc = json.load(open(f"{b50}/out/expl1_scores.json"))
    rows = {r["id"]: r for r in D.gen("N", "test")}
    TR = [i for i, r in rows.items() if r["op"] in D.OPS_TRAIN]
    HO = [i for i, r in rows.items() if r["op"] in D.OPS_HELD]
    by = {op: [i for i, r in rows.items() if r["op"] == op] for op in D.OPS_HELD}
    vsa = Arm([f"{b40p}/N/vsa_s{s}.jsonl" for s in SEEDS])
    disc08 = Arm([f"{b40p}/N/disc_s{s}.jsonl" for s in SEEDS])
    base = Arm([f"{b40p}/N/base.jsonl"]).runs[0]
    opens = Arm([f"{b50}/preds/N/discopen_s{s}.jsonl" for s in SEEDS]).runs
    t = sc["test"]
    dv = np.sort(np.array(sc["dev_scores"]))[::-1]
    out = []
    P = lambda *a: out.append(" ".join(str(x) for x in a))
    P("B50 EXPLORATORY 1: least-squares one-hot type readout (MSE-ONEHOT) as DISC's gate (cached states; composed)")
    P(f"  fit: training type accuracy {sc['train_fit_acc']:.4f}, dev type accuracy {sc['dev_type_acc']:.4f}; dev scores min {dv.min():.4f} median "
      f"{np.median(dv):.4f} max {dv.max():.4f}")
    P(f"  AUROC trained vs held-out (score = max output): {auroc([t[i]['score'] for i in TR], [t[i]['score'] for i in HO]):.4f}")
    for op in D.OPS_TRAIN + D.OPS_HELD:
        ids = [i for i, r in rows.items() if r["op"] == op]
        s = np.array([t[i]["score"] for i in ids])
        P(f"     {op:15s} score q05 {np.quantile(s, .05):.3f} median {np.median(s):.3f} q95 {np.quantile(s, .95):.3f}")
    for name, tau, strict in (("> 0.8 (DE25's number)", 0.8, True), ("dev-matched (797th largest dev score)", float(dv[796]), False)):
        g = (lambda i, tau=tau, strict=strict: t[i]["score"] > tau if strict else t[i]["score"] >= tau)
        comp = Composed(opens, base, lambda s, i, g=g: g(i))
        otr = np.mean([g(i) for i in TR]); oad = np.mean([g(i) for i in by["addition"]]); odi = np.mean([g(i) for i in by["division"]])
        d = boot_diff(vsa, comp, HO, rng)
        dt = boot_diff(comp, disc08, TR, rng)
        P(f"  gate {name}: tau {tau:.4f}; open TR {otr:.4f} addition {oad:.4f} division {odi:.4f}")
        P(f"     composed TR {comp.acc(TR):.4f} (minus DISC-0.8 {fmt(dt)}), HO {comp.acc(HO):.4f} (add {comp.acc(by['addition']):.4f}, "
          f"div {comp.acc(by['division']):.4f}); HO(VSA) - HO {fmt(d)}; seeds HO {[round(x, 4) for x in comp.seed_accs(HO)]}")
    open("results/b50_exploratory1.txt", "w").write("\n".join(out) + "\n")
    print("\n".join(out))


def live(b50, b40p):
    """Live check: the 'lsq' gate run inside DISC's forward pass (preds/N/expl1_lsq{08,dev}_s*.jsonl)."""
    import os
    import b40_data as D
    from b40_verdict import Arm, boot_diff, fmt
    rng = np.random.default_rng(5052)
    sc = json.load(open(f"{b50}/out/expl1_scores.json"))["test"]
    rows = {r["id"]: r for r in D.gen("N", "test")}
    TR = [i for i, r in rows.items() if r["op"] in D.OPS_TRAIN]
    HO = [i for i, r in rows.items() if r["op"] in D.OPS_HELD]
    by = {op: [i for i, r in rows.items() if r["op"] == op] for op in D.OPS_HELD}
    vsa = Arm([f"{b40p}/N/vsa_s{s}.jsonl" for s in SEEDS])
    disc08 = Arm([f"{b40p}/N/disc_s{s}.jsonl" for s in SEEDS])
    base = Arm([f"{b40p}/N/base.jsonl"]).runs[0]
    opens = Arm([f"{b50}/preds/N/discopen_s{s}.jsonl" for s in SEEDS]).runs
    out = ["", "LIVE CHECK (the same least-squares gate run inside DISC's forward pass; full test split, seeds 0-2)"]
    for tag, tau in (("08", 0.8), ("dev", 0.8901834484174533)):
        fs = [f"{b50}/preds/N/expl1_lsq{tag}_s{s}.jsonl" for s in SEEDS]
        if not all(os.path.exists(f) for f in fs):
            out.append(f"  gate {tag}: MISSING"); continue
        a = Arm(fs)
        g = lambda ids: float(np.mean([[float(r[i]["mod"]["gate"]) for i in ids] for r in a.runs]))
        rel = sum(int(r[i]["mod"]["gate"] != (r[i]["mod"]["lsq"] >= tau)) for r in a.runs for i in r)
        cache = sum(int(r[i]["mod"]["gate"] != (sc[i]["score"] >= tau)) for r in a.runs for i in r)
        comp = sum(int(r[i]["correct"] != (opens[s][i]["correct"] if r[i]["mod"]["gate"] else base[i]["correct"]))
                   for s, r in enumerate(a.runs) for i in r)
        d = boot_diff(vsa, a, HO, rng); dt = boot_diff(a, disc08, TR, rng)
        out.append(f"  gate {tag} (tau {tau:.4f}): open TR {g(TR):.4f} addition {g(by['addition']):.4f} division "
                   f"{g(by['division']):.4f}; TR {a.acc(TR):.4f} (minus DISC-0.8 {fmt(dt)}), HO {a.acc(HO):.4f}; "
                   f"HO(VSA) - HO {fmt(d)}; seeds HO {[round(x, 4) for x in a.seed_accs(HO)]}")
        out.append(f"     checks: gate re-derivation mismatches {rel}/7500; live vs cached gate {cache}/7500; "
                   f"composition mismatches {comp}/7500")
    open("results/b50_exploratory1.txt", "a").write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    {"fit": fit, "read": read, "live": live}[sys.argv[1]](*sys.argv[2:4])
