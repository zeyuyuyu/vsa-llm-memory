"""B47 second scorer (prereg/B47_phi2_finetunes_hierboot_consistency_prereg.md), written separately from the main code
before any B47 test output is read.

(a) labels: an independent regex walk of the raw test text (b44_indep.walk) against b44_data's labels;
(b) fine-tunes: every saved test output re-scored with a second parser (answer-token arms: the sign of the saved margin
    read as text; scratchpad: a second yes/no reader, the last standalone yes/no token after stripping punctuation),
    per-instance agreement with the main scoring (b47_verdict's) and with the logged accuracy;
(c) sweep: a numpy float64 re-implementation of B31's arena-C chain (circular convolution by numpy.fft, the value
    permutation, clean-up to the instance's own value mentions, the winner's code as the next key, final cosine) and of
    the closed-form training threshold, run on the sweep's codes (seed 42, perm, mention model, rho 1.0 / 0.8 / 0.6 /
    0.55 / 0.4, c = 0), against the main sweep's saved test scores and decisions;
(d) bootstrap: a plain-loop H-nest for three margins with another random stream, against b47_hierboot.json.

  python b47_indep.py --out b47_out/indep.json
"""
import argparse, json, os, re, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
SEEDS = [42, 123, 456]


def yn2(txt):
    toks = [t for t in re.split(r"[^A-Za-z]+", txt.lower()) if t]
    for t in reversed(toks):
        if t in ("yes", "no"):
            return int(t == "yes")
    return -1


def chain_np(K, P_, hops, q, c, enc):
    """K (n, 130, d) float64 codes; returns final score per instance and the two winners (positions)."""
    n = K.shape[0]
    V = K[:, :, P_] if enc == "perm" else K
    key = K[:, q]
    wins = []
    for kp, vp in hops:
        FK = np.fft.rfft(K[:, kp], axis=-1); FV = np.fft.rfft(V[:, vp], axis=-1)
        mem = np.fft.irfft((FV * FK).sum(1), n=K.shape[-1], axis=-1)
        hat = np.fft.irfft(np.fft.rfft(mem, axis=-1) * np.conj(np.fft.rfft(key, axis=-1)), n=K.shape[-1], axis=-1)
        hat = hat / np.linalg.norm(hat, axis=-1, keepdims=True)
        cand = V[:, vp]
        cs = (cand * hat[:, None, :]).sum(-1) / np.linalg.norm(cand, axis=-1)
        w = cs.argmax(1)
        wins.append(np.asarray(vp)[w])
        key = K[np.arange(n), np.asarray(vp)[w]]
    score = np.clip((key * K[:, c]).sum(-1), -1, 1)
    return score, wins


def thr_np(st, yt):
    best, bt = -1.0, None
    for t in np.unique(st):
        acc = ((st >= t) == (yt == 1)).mean()
        if acc > best:
            best, bt = acc, t
    return bt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(HERE, "b47_out"))
    ap.add_argument("--out", default=os.path.join(HERE, "b47_out", "indep.json"))
    ap.add_argument("--skip_sweep", action="store_true")
    a = ap.parse_args()
    import b44_data as BD
    import b44_indep as I
    rep = {"cells": {}}
    st = BD.split("std", "test", struct=False)
    y1 = np.array([BD.dict_answer(s["text"])["label"] for s in st])
    y2 = np.array([I.walk(s["text"])[0] for s in st])
    rep["labels_agree"] = float((y1 == y2).mean())
    y = y2
    sel = json.load(open(os.path.join(a.root, "select.json"))) if os.path.exists(os.path.join(a.root, "select.json")) else {}
    import b44_lora as BL
    for key, s in sel.items():
        task, variant, n = key.split("_")[0], key.split("_")[1], int(key.split("_")[2][1:])
        for seed in SEEDS:
            p = os.path.join(a.root, "ft", "%s_%s_n%d_r%d_lr%s_s%d.json.test.json" % (task, variant, n, s["rank"],
                                                                                       "%g" % s["lr"], seed))
            if not os.path.exists(p):
                continue
            t = json.load(open(p))
            if task == "scratch":
                p2 = np.array([yn2(g) for g in t["test"]["gens"]])
                p1 = np.array([-1 if BL.parse_scratch(g)[0] is None else BL.parse_scratch(g)[0] for g in t["test"]["gens"]])
            else:
                p2 = np.array([1 if not str(x).startswith("-") and float(x) != 0.0 else 0 for x in t["test"]["scores"]])
                p1 = np.array([int(x > 0) for x in t["test"]["scores"]])
            rep["cells"]["%s_s%d" % (key, seed)] = {"agree": float((p1 == p2).mean()), "acc_second": float((p2 == y).mean()),
                                                    "acc_logged": t["test"]["acc"]}
    if not a.skip_sweep and os.path.exists(os.path.join(a.root, "sweep_consistency.json")):
        import torch
        import b31_arms as B
        import b47_sweep as W
        SW = json.load(open(os.path.join(a.root, "sweep_consistency.json")))
        data = {}
        for sp in ("train", "dev", "test"):
            st_, sym, yy = W.load_split(sp)
            data[sp] = (st_, sym, yy, W.gold_symbols(sym))
        torch.manual_seed(42)
        m = B.Arm("C", "exact", 42, 8)
        codes = W.Codes(m, 42, data)
        P_ = m.P.numpy()
        kinds, hops, q, c, _ = B.layout("C")
        T = m.table.double().numpy()
        for rho in (1.0, 0.8, 0.6, 0.55, 0.4):
            sc = {}
            for sp in ("train", "test"):
                sym = data[sp][1].numpy()
                t_ = T[sym]                                                    # (n, 130, d) canonical rows
                if rho < 1.0:
                    nz = codes.noise[sp].double().numpy()
                    nz = nz - (nz * t_).sum(-1, keepdims=True) * t_
                    nz = nz / np.linalg.norm(nz, axis=-1, keepdims=True)
                    K = rho * t_ + np.sqrt(1 - rho * rho) * nz
                else:
                    K = t_
                sc[sp] = np.concatenate([chain_np(K[i:i + 50], P_, hops, q, c, "perm")[0] for i in range(0, K.shape[0], 50)])
            thr = thr_np(sc["train"], data["train"][2])
            dec2 = (sc["test"] >= thr).astype(int)
            row = [x for x in SW["cells"] if x["structure"] == "mention" and x["enc"] == "perm" and abs(x["rho"] - rho) < 1e-9][0]
            r42 = [r for r in row["per_seed"] if r["seed"] == 42][0]
            dec1 = (np.array(r42["test_scores"]) >= r42["thr"]).astype(int)
            rep["cells"]["sweep_rho%.2f_seed42" % rho] = {
                "agree": float((dec1 == dec2).mean()), "acc_second": float((dec2 == data["test"][2]).mean()),
                "acc_main": r42["acc"], "max_abs_score_diff": float(np.abs(np.array(r42["test_scores"]) - sc["test"]).max()),
                "thr_main": r42["thr"], "thr_second": float(thr)}
            print("sweep rho", rho, rep["cells"]["sweep_rho%.2f_seed42" % rho], flush=True)
    hbp = os.path.join(a.root, "b47_hierboot.json")
    if os.path.exists(hbp):
        import b47_boot as H
        hb = json.load(open(hbp))
        R31 = os.path.join(HERE, "b47_out", "src_b31")
        chk = []
        for enc, ar, x, yk in (("qwen8", "C", "name", "lora"), ("phi2", "C", "name", "lora"), ("phi2", "A", "name", "lora")):
            yl = H.lab31(ar)
            X = H.mem(os.path.join(R31, "arms_final", "%s_%s_name.json" % (enc, ar)), "perm/named", yl)
            Y = H.b31_lora(R31, enc, ar, 0)
            rng = np.random.default_rng(987654)
            n = X["C"].shape[1]; vals = []
            for _ in range(10000):
                tot = 0.0
                for _j in range(3):
                    sx, sy = rng.integers(0, 3), rng.integers(0, 3)
                    qi = rng.integers(0, n, n)
                    tot += X["C"][sx][qi].mean() - Y["C"][sy][qi].mean()
                vals.append(tot / 3)
            lo, hi = float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))
            mid = [m for m in hb["margins"] if m["id"] in ("b31(b)|%s|%s|name-lora" % (enc, ar), "b31|%s|%s|name-lora" % (enc, ar))][0]
            chk.append({"margin": mid["id"], "second": [lo, hi], "main": mid["ci"]["hnest"],
                        "max_end_diff": max(abs(lo - mid["ci"]["hnest"][0]), abs(hi - mid["ci"]["hnest"][1]))})
            print(chk[-1], flush=True)
        rep["bootstrap_check"] = chk
    ag = [v["agree"] for k, v in rep["cells"].items()]
    rep["all_cells_ge_0.99"] = bool(ag) and min(ag) >= 0.99 and rep["labels_agree"] == 1.0
    rep["min_agree"] = min(ag) if ag else None
    json.dump(rep, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k != "cells"}, indent=1))


if __name__ == "__main__":
    main()
