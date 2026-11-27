"""B50 runner (prereg/B50_ood_gates_cot_budget_prereg.md). Extends B40 without changing it: b40_core.py, b40_data.py
and b40_run.py are imported unchanged (byte-identical copies on the box; md5 recorded in the prereg).

  fitmaha --out F        class means and tied covariance of B40's task-N TRAINING states at layer 20 (16,000 rows,
                         B40 cache, the states DISC's heads were trained on); two precision matrices:
                         'maha'   = Lee et al. 2018 (empirical covariance of the class-centred states, pseudo-inverse),
                         'mahalw' = Ledoit-Wolf shrinkage of the same covariance, inverse.
  eval --task N --arm discood --gate G --thr T --reader R --ckpt C --maha F ...
                         B40's trained DISC seed (reader + write-back, unchanged) with its type gate replaced by gate G
                         at threshold T; G in open | p08 | msp | energy | maha | mahalw | maxlogit. Every item logs all
                         five scores, so any gate can be re-derived from the file.
  eval --arm none|cot|vsa|disc ...  B40's eval; adds per-item generated-token count, EOS flag and batch wall time.
  cache --task N --split S --layers 20   B40's cache command, written under $B50_ROOT/cache.
"""
import argparse, json, os, sys, time
import numpy as np
import torch
import b40_data as D
import b40_core as C
import b40_run as R40

ROOT = os.environ.get("B50_ROOT", "/path/to/work/b50")
B40 = os.environ.get("B40_ROOT", "/path/to/work/b40")
GATES = ("open", "p08", "msp", "energy", "maha", "mahalw", "maxlogit", "lsq")  # lsq: exploratory 1 (live check)
SCORES = ("msp", "energy", "maha", "mahalw", "maxlogit")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def ledger(t0):
    os.makedirs(f"{ROOT}/logs", exist_ok=True)
    with open(f"{ROOT}/logs/gpu_hours.txt", "a") as f:
        f.write(json.dumps({"cmd": " ".join(sys.argv[1:]), "gpu_h": round((time.time() - t0) / 3600, 4),
                            "host": os.uname().nodename, "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
                            "end": time.strftime("%Y-%m-%dT%H:%M:%S")}) + "\n")


# ------------------------------------------------------------------------------------------------ Mahalanobis fit
def cmd_fitmaha(a):
    H = torch.load(f"{B40}/cache/N_train_L20.pt").double()  # [16000, 4096], fp16 on disk
    meta = json.load(open(f"{B40}/cache/N_train_meta.json"))
    y = torch.tensor([D.OPS_TRAIN.index(o) for o in meta["op"]])
    K = len(D.OPS_TRAIN)
    mu = torch.stack([H[y == c].mean(0) for c in range(K)])
    X = H - mu[y]
    # Lee et al. 2018 (deep_Mahalanobis_detector, lib_generation.sample_estimator): sklearn EmpiricalCovariance
    # (assume_centered=False) fit on the class-centred features; precision = pinvh(covariance).
    Xc = X - X.mean(0, keepdim=True)
    S = (Xc.T @ Xc) / len(Xc)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ev = torch.linalg.eigvalsh(S.to(dev)).cpu()
    P_emp = torch.linalg.pinv(S.to(dev), hermitian=True).cpu()
    # Ledoit-Wolf (sklearn.covariance.ledoit_wolf, assume_centered=False) on the same class-centred features
    from sklearn.covariance import ledoit_wolf
    S_lw, shrink = ledoit_wolf(X.numpy(), assume_centered=False)
    P_lw = torch.linalg.inv(torch.from_numpy(S_lw).to(dev)).cpu()
    tol = 4096 * torch.finfo(torch.float64).eps * float(ev.max())
    info = {"n": len(H), "classes": D.OPS_TRAIN, "eig_min": float(ev.min()), "eig_max": float(ev.max()),
            "eig_median": float(ev.median()), "n_eig_below_pinv_cutoff": int((ev <= tol).sum()),
            "lw_shrinkage": float(shrink)}
    torch.save({"mu": mu.float(), "P_emp": P_emp, "P_lw": P_lw, "info": info}, a.out)
    json.dump(info, open(a.out + ".json", "w"), indent=1)
    log("fitmaha", info)


# ------------------------------------------------------------------------------------------------ DISC with OOD gates
class DiscNOOD(C.DiscN):
    """B40's DiscN (same heads, same Python function, same write-back) with the type gate replaced.
    Scores (higher = more in-distribution), all from the same forward pass:
      msp      log-odds of the top type, log p_max - log(1 - p_max) (monotone in the max softmax probability;
               computed in log space so that saturation at p = 1.0 in floating point cannot tie items)
      energy   logsumexp of the 8 type logits (negative free energy, T = 1; Liu et al. 2020)
      maxlogit largest type logit (Hendrycks et al. 2022)
      maha     -min_c (h - mu_c)^T P_emp (h - mu_c) on the 4,096-d layer-20 state the heads read (Lee et al. 2018)
      mahalw   the same with the Ledoit-Wolf precision
    Gates: open (always), p08 (B40's: max softmax > 0.8), or score >= thr."""

    def __init__(self, gate, tau, maha):
        super().__init__(thr=0.8)
        assert gate in GATES
        self.gate_kind, self.tau = gate, tau
        self.mu = maha["mu"].to(C.DEV, torch.float64)
        self.Pm = {"maha": maha["P_emp"].to(C.DEV, torch.float64), "mahalw": maha["P_lw"].to(C.DEV, torch.float64)}
        self.lsq = None  # exploratory 1: least-squares one-hot readout [4097, 8] (b50_expl1.py fit -> out/lsq_W.pt)

    def scores(self, h_last):
        lo, _, _ = self.logits(h_last)
        z = lo.double()
        top = z.max(-1)
        rest = z.clone()
        rest[torch.arange(len(z), device=z.device), top.indices] = -float("inf")
        s = {"msp": top.values - torch.logsumexp(rest, -1), "energy": torch.logsumexp(z, -1), "maxlogit": top.values}
        x = h_last.double()
        diff = x.unsqueeze(1) - self.mu.unsqueeze(0)  # [B, K, d]
        for k, P in self.Pm.items():
            d2 = ((diff @ P) * diff).sum(-1)  # [B, K]
            s[k] = -d2.min(-1).values
        if self.lsq is not None:
            s["lsq"] = (torch.cat([x, torch.ones(len(x), 1, dtype=x.dtype, device=x.device)], 1) @ self.lsq).max(-1).values
        return s

    def forward(self, h, ctx):
        with torch.no_grad():
            hl = h[:, -1, :]
            op_i, op_p, x, y = self.read(hl)
            sc = self.scores(hl.float())
            ans = [C.safe_answer(D.OPS_TRAIN[o], a, b) for o, a, b in zip(op_i.tolist(), x.tolist(), y.tolist())]
            dig = torch.tensor([[(a // 10 ** k) % 10 for k in range(C.MAX_DIGITS)] for a in ans], device=h.device)
        w = self.emb[torch.arange(C.MAX_DIGITS, device=h.device).unsqueeze(0), dig].sum(1)
        if self.gate_kind == "open":
            gate = torch.ones(len(hl), dtype=torch.bool, device=h.device)
        elif self.gate_kind == "p08":
            gate = op_p > 0.8  # identical to DiscN (B40)
        else:
            gate = sc[self.gate_kind] >= self.tau
        scl = {k: v.tolist() for k, v in sc.items()}
        info = [{"op": D.OPS_TRAIN[o], "op_p": round(float(p_), 3), "x": int(a), "y": int(b), "ans": int(c),
                 "gate": bool(g), **{k: float(scl[k][j]) for k in scl}}
                for j, (o, p_, a, b, c, g) in enumerate(zip(op_i.tolist(), op_p.tolist(), x.tolist(), y.tolist(), ans,
                                                            gate.tolist()))]
        return w, gate, info


# ------------------------------------------------------------------------------------------------ eval
EOS = (128001, 128008, 128009)  # Llama-3.1 generation_config eos_token_id


@torch.no_grad()
def cmd_eval(a, tok, model):
    """B40's cmd_eval (b40_run.py) with the same batching, ordering, generation and parsing; adds the discood arm,
    per-item generated-token counts / EOS flags (n_gen, hit_limit) and per-batch wall times (sidecar .timing.json)."""
    rows = R40.rows_for(a.task, a.split)
    if a.limit:
        rows = rows[:a.limit]
    cot = a.arm == "cot"
    items = [C.build(tok, r, cot=cot) for r in rows]
    inj, mod = None, None
    if a.arm == "discood":
        mod = DiscNOOD(a.gate, a.thr, torch.load(a.maha)).to(C.DEV)
        if a.lsq:
            mod.lsq = torch.load(a.lsq).to(C.DEV, torch.float64)
        mod.load_state_dict(torch.load(f"{a.reader}/reader.pt"), strict=False)
        mod.load_state_dict(torch.load(f"{a.ckpt}/best.pt"), strict=False)
        mod.eval()
        inj = C.Injector(model, a.layer)
    elif a.arm not in ("none", "cot"):
        mod = R40.make_module(a, tok, model)
        if a.arm in ("vsa", "disc"):
            mod.load_state_dict(torch.load(f"{a.reader}/reader.pt"), strict=False)
        mod.load_state_dict(torch.load(f"{a.ckpt}/best.pt"), strict=False)
        mod.eval()
        inj = C.Injector(model, a.layer)
    torch.manual_seed(a.noise_seed)
    order = sorted(range(len(items)), key=lambda i: -len(items[i]["ids"]))
    bs = a.bs if not cot else a.cot_bs
    res = [None] * len(items)
    maxnew = a.max_new if not cot else a.cot_max_new
    timing, t_all = [], time.time()
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        bt = [items[i] for i in idx]
        ids, am, pos, ctx, gold = C.collate(tok, bt)
        if inj is not None:
            inj.arm(mod, ctx)
        torch.cuda.synchronize(); tb = time.time()
        out = model.generate(input_ids=ids, attention_mask=am, max_new_tokens=maxnew, do_sample=False,
                             temperature=None, top_p=None, pad_token_id=tok.pad_id,
                             return_dict_in_generate=True, output_logits=not cot)
        torch.cuda.synchronize(); dt = time.time() - tb
        info = inj.info if inj is not None else None
        if inj is not None:
            inj.armed = False
        first = out.logits[0].float() if not cot else None
        lp = torch.log_softmax(first, -1) if not cot else None
        gen = out.sequences[:, ids.shape[1]:]
        steps = gen.shape[1]
        timing.append({"batch": s // bs, "n": len(idx), "steps": int(steps), "sec": round(dt, 3),
                       "prompt_len": int(ids.shape[1])})
        for j, i in enumerate(idx):
            g = gen[j].tolist()
            e = [t for t, v in enumerate(g) if v in EOS]
            n_gen, hit = (e[0] + 1, False) if e else (len(g), True)
            txt = tok.decode(gen[j], skip_special_tokens=True)
            r = rows[i]
            pred = C.parse(a.task, txt, cot=cot)
            res[i] = {"id": r["id"], "gold": D.gold(r), "pred": pred, "correct": pred.strip().lower() == D.gold(r).lower(),
                      "out": txt if cot else txt[:40], "n_gen": n_gen, "hit_limit": hit,
                      "argmax_ok": (int(first[j].argmax()) == int(gold[j])) if not cot else None,
                      "ce": float(-lp[j, gold[j]]) if not cot else None,
                      **({"op": r["op"], "x": r["x"], "y": r["y"]} if a.task == "N" else
                         {"depth": r["depth"], "size": r["size"]}),
                      **({"mod": info[j]} if info else {})}
        log(f"{s + len(idx)}/{len(items)} acc so far {np.mean([x['correct'] for x in res if x]):.3f} "
            f"batch {dt:.1f}s steps {steps}")
    os.makedirs(os.path.dirname(a.pred) or ".", exist_ok=True)
    with open(a.pred, "w") as f:
        for x in res:
            f.write(json.dumps(x) + "\n")
    json.dump({"args": vars(a), "wall_s": round(time.time() - t_all, 2), "gen_s": round(sum(t["sec"] for t in timing), 2),
               "gpu": torch.cuda.get_device_name(0), "batches": timing}, open(a.pred + ".timing.json", "w"), indent=1)
    log("acc", np.mean([x["correct"] for x in res]), "hit_limit", sum(x["hit_limit"] for x in res))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd"); p.add_argument("--task", default="N"); p.add_argument("--arm", default="none")
    p.add_argument("--split", default="dev"); p.add_argument("--layers", default="20"); p.add_argument("--layer", type=int, default=20)
    p.add_argument("--bs", type=int, default=32); p.add_argument("--subset", type=int, default=0)
    p.add_argument("--reader", default=""); p.add_argument("--ckpt", default=""); p.add_argument("--out", default="")
    p.add_argument("--pred", default=""); p.add_argument("--noise", type=float, default=0.0)
    p.add_argument("--noise_seed", type=int, default=12345); p.add_argument("--soft_beta", type=float, default=0.0)
    p.add_argument("--limit", type=int, default=0); p.add_argument("--max_new", type=int, default=6)
    p.add_argument("--cot_max_new", type=int, default=2048); p.add_argument("--cot_bs", type=int, default=48)
    p.add_argument("--digit_thr", type=float, default=0.5); p.add_argument("--hops", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gate", default="open"); p.add_argument("--thr", type=float, default=0.0)
    p.add_argument("--maha", default=f"{ROOT}/maha.pt"); p.add_argument("--lsq", default="")
    a = p.parse_args()
    t0 = time.time()
    if a.cmd == "fitmaha":
        cmd_fitmaha(a)
    else:
        tok, model = C.load()
        if a.cmd == "eval":
            cmd_eval(a, tok, model)
        elif a.cmd == "cache":
            R40.CACHE = f"{ROOT}/cache"
            R40.cmd_cache(a, tok, model)
    ledger(t0)


if __name__ == "__main__":
    main()
