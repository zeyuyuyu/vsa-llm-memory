"""B40 runner (prereg/B40_inforward_vsa_prereg.md). One command per process; outputs under --out.

  cache   --task N|R --split S --layers 8,12,..  [--subset n]   frozen states at the read positions
  reader  --task N|R --arm vsa|disc --layer L --epochs E --seed s    supervised read (DE25 encoder / classifier heads)
  train   --task N|R --arm vsa|disc|lora1l|lora|e2e_vsa|e2e_tab|e2e_const --layer L --seed s ...   write-back / LoRA by CE
  eval    --task N|R --arm none|cot|<trained arm> --split dev|test [--ckpt dir] [--noise rho] [--soft_beta b]
"""
import argparse, json, os, sys, time, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import b40_data as D
import b40_core as C

ROOT = os.environ.get("B40_ROOT", "/path/to/work/b40")
CACHE = f"{ROOT}/cache"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def ledger(args, t0):
    os.makedirs(f"{ROOT}/logs", exist_ok=True)
    with open(f"{ROOT}/logs/gpu_hours.txt", "a") as f:
        f.write(json.dumps({"cmd": " ".join(sys.argv[1:]), "gpu_h": round((time.time() - t0) / 3600, 4),
                            "host": os.uname().nodename, "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
                            "end": time.strftime("%Y-%m-%dT%H:%M:%S")}) + "\n")


def rows_for(task, split):
    if task == "R" and split == "dev_e2e":
        return [r for r in D.gen_r("dev") if r["depth"] == 2]
    if task == "R" and split == "dev_e2e3":
        return [r for r in D.gen_r("dev") if r["depth"] == 3]
    return D.gen(task, split)


# ------------------------------------------------------------------------------------------------ cache
@torch.no_grad()
def cmd_cache(a, tok, model):
    rows = rows_for(a.task, a.split)
    if a.subset:
        rng = np.random.default_rng(77)
        rows = [rows[i] for i in sorted(rng.choice(len(rows), a.subset, replace=False))]
    layers = [int(x) for x in a.layers.split(",")]
    items = [C.build(tok, r) for r in rows]
    out = {L: [] for L in layers}
    lab = []
    for i in range(0, len(items), a.bs):
        bt = items[i:i + a.bs]
        ids, am, pos, ctx, gold = C.collate(tok, bt)
        hs = model(input_ids=ids, attention_mask=am, position_ids=pos, output_hidden_states=True,
                   logits_to_keep=1).hidden_states
        for L in layers:
            h = hs[L]
            if a.task == "N":
                out[L].append(h[:, -1, :].half().cpu())
            else:
                for j, it in enumerate(bt):
                    sh = ids.shape[1] - len(it["ids"])
                    idx = [t + sh for t in it["a_idx"] + it["b_idx"]] + [it["q_idx"] + sh]
                    out[L].append(h[j, idx, :].half().cpu())
        if a.task == "R":
            for it in bt:
                lab.append(it["a_ent"] + it["b_ent"] + [it["q_ent"]])
    os.makedirs(CACHE, exist_ok=True)
    tag = f"{a.task}_{a.split}" + (f"_sub{a.subset}" if a.subset else "")
    for L in layers:
        H = torch.cat(out[L])
        torch.save(H, f"{CACHE}/{tag}_L{L}.pt")
    if a.task == "N":
        meta = {"op": [r["op"] for r in rows], "x": [r["x"] for r in rows], "y": [r["y"] for r in rows]}
    else:
        meta = {"ent": [e for l in lab for e in l]}
    json.dump(meta, open(f"{CACHE}/{tag}_meta.json", "w"))
    log("cached", tag, layers, H.shape)


# ------------------------------------------------------------------------------------------------ readers
def load_cache(task, split, L, subset=0, dev=False):
    tag = f"{task}_{split}" + (f"_sub{subset}" if subset else "")
    H = torch.load(f"{CACHE}/{tag}_L{L}.pt")
    H = H.to(C.DEV) if dev else H  # fp16 on the GPU (readers), float per batch
    return H.float() if not dev else H, json.load(open(f"{CACHE}/{tag}_meta.json"))


def n_targets(meta, eng):
    op = torch.tensor([D.OPS_ALL.index(o) for o in meta["op"]])
    x, y = torch.tensor(meta["x"]), torch.tensor(meta["y"])
    return op, x, y, eng.problem(op.to(eng.dev), x.to(eng.dev), y.to(eng.dev))


def de25_lr(epoch, base):  # DE25 encoder schedule: x0.5 at 50, x0.5 at 100, x0.1 at 250, x0.4 at 500
    f = 1.0
    for e, m in ((50, 0.5), (100, 0.5), (250, 0.1), (500, 0.4)):
        if epoch >= e:
            f *= m
    return base * f


def read_acc_N(mod, arm, H, meta, eng):
    """Fraction of items whose op and both operands are read exactly; also per-digit error."""
    with torch.no_grad():
        outs = []
        for i in range(0, len(H), 2048):
            h = H[i:i + 2048].to(C.DEV)
            if arm == "vsa":
                v, op_i, op_s, x, y = mod.read(h)
                outs.append((op_i.cpu(), x.cpu(), y.cpu()))
            else:
                op_i, op_p, x, y = mod.read(h)
                outs.append((torch.tensor([D.OPS_ALL.index(D.OPS_TRAIN[o]) for o in op_i.tolist()]), x.cpu(), y.cpu()))
        op = torch.cat([o[0] for o in outs]); x = torch.cat([o[1] for o in outs]); y = torch.cat([o[2] for o in outs])
    gop = torch.tensor([D.OPS_ALL.index(o) for o in meta["op"]]); gx = torch.tensor(meta["x"]); gy = torch.tensor(meta["y"])
    exact = ((op == gop) & (x == gx) & (y == gy)).float().mean().item()
    dig_err = np.mean([((x // 10 ** k) % 10 != (gx // 10 ** k) % 10).float().mean().item() for k in range(3)] +
                      [((y // 10 ** k) % 10 != (gy // 10 ** k) % 10).float().mean().item() for k in range(3)])
    return {"exact": exact, "op": (op == gop).float().mean().item(), "x": (x == gx).float().mean().item(),
            "y": (y == gy).float().mean().item(), "digit_err": float(dig_err)}


def cmd_reader(a):
    torch.manual_seed(a.seed)
    sub = a.subset
    Htr, mtr = load_cache(a.task, "train", a.layer, sub, dev=True)
    Hdv, mdv = load_cache(a.task, a.dev_split, a.layer, a.dev_subset)
    if a.task == "N":
        eng = C.NumEngine().to(C.DEV)
        if a.arm == "vsa":
            mod = C.VSAN(eng).to(C.DEV)
            _, _, _, T = n_targets(mtr, eng)
            params = mod.enc.parameters()
        else:
            mod = C.DiscN().to(C.DEV)
            op = torch.tensor([D.OPS_TRAIN.index(o) for o in mtr["op"]]).to(C.DEV)
            xd = torch.stack([(torch.tensor(mtr["x"]) // 10 ** k) % 10 for k in range(C.MAX_DIGITS)], 1).to(C.DEV)
            yd = torch.stack([(torch.tensor(mtr["y"]) // 10 ** k) % 10 for k in range(C.MAX_DIGITS)], 1).to(C.DEV)
            params = mod.head.parameters()
    else:
        codes = C.EntCodes().to(C.DEV)
        ent = torch.tensor(mtr["ent"]).to(C.DEV)
        if a.arm == "vsa":
            mod = C.VSAR(codes).to(C.DEV)
            T = codes.K[ent.to(C.DEV)]
            params = mod.enc.parameters()
        else:
            mod = C.DiscR().to(C.DEV)
            params = mod.cls.parameters()
    opt = torch.optim.Adam(params, lr=a.lr)
    n = len(Htr)
    best, hist = None, []
    for ep in range(a.epochs):
        for g in opt.param_groups:
            g["lr"] = de25_lr(ep, a.lr)
        perm = torch.randperm(n, device=C.DEV)
        for i in range(0, n, 512):
            b = perm[i:i + 512]
            h = Htr[b].float()
            if a.task == "N" and a.arm == "vsa":
                loss = torch.sqrt(F.mse_loss(mod.enc(h), T[b]))
            elif a.task == "N":
                lo, lx, ly = mod.logits(h)
                loss = (F.cross_entropy(lo, op[b]) + F.cross_entropy(lx.reshape(-1, 10), xd[b].reshape(-1))
                        + F.cross_entropy(ly.reshape(-1, 10), yd[b].reshape(-1)))
            elif a.arm == "vsa":
                loss = torch.sqrt(F.mse_loss(mod.enc(h), T[b]))
            else:
                loss = F.cross_entropy(mod.cls(h), ent[b])
            opt.zero_grad(); loss.backward(); opt.step()
        if (ep + 1) % a.eval_every == 0 or ep + 1 == a.epochs:
            if a.task == "N":
                acc = read_acc_N(mod, a.arm, Hdv, mdv, eng)
            else:
                with torch.no_grad():
                    hd = Hdv.to(C.DEV)
                    pred = mod.read_codes(hd) if a.arm == "vsa" else mod.read_ids(hd)
                acc = {"exact": (pred.cpu() == torch.tensor(mdv["ent"])).float().mean().item()}
            acc.update(epoch=ep + 1, loss=float(loss))
            hist.append(acc)
            log(json.dumps(acc))
    os.makedirs(a.out, exist_ok=True)
    sd = {k: v.cpu() for k, v in mod.state_dict().items() if k.startswith(("enc", "head", "cls"))}
    torch.save(sd, f"{a.out}/reader.pt")
    json.dump({"args": vars(a), "hist": hist, "final": hist[-1]}, open(f"{a.out}/reader.json", "w"), indent=1)


# ------------------------------------------------------------------------------------------------ module construction
def make_module(a, tok, model):
    if a.arm in ("none", "cot", "lora"):
        return None
    if a.task == "N":
        eng = C.NumEngine().to(C.DEV)
        mod = {"vsa": lambda: C.VSAN(eng, digit_thr=a.digit_thr), "disc": lambda: C.DiscN(), "lora1l": lambda: C.Lora1L()}[a.arm]()
    else:
        codes = C.EntCodes().to(C.DEV)
        mod = {"vsa": lambda: C.VSAR(codes, soft_beta=a.soft_beta), "disc": lambda: C.DiscR(),
               "e2e_vsa": lambda: C.E2EVSA(a.seed, hops=a.hops), "e2e_tab": lambda: C.E2ETab(a.seed, hops=a.hops),
               "e2e_const": lambda: C.E2EConst(a.seed)}[a.arm]()
    return mod.to(C.DEV)


def trainable(mod, arm):
    if arm in ("vsa", "disc") and hasattr(mod, "dec"):
        return list(mod.dec.parameters())
    if arm == "disc":
        return [mod.emb]
    return list(mod.parameters())


def recon_init(mod, a):
    """DE25 decoder pre-training: linear map from the encoder's VSA vector back to the hidden state (RMSE)."""
    H, _ = load_cache(a.task, "train", a.layer)
    with torch.no_grad():
        V = torch.cat([mod.enc(H[i:i + 4096].to(C.DEV)).cpu() for i in range(0, len(H), 4096)])
    opt = torch.optim.Adam(mod.dec.parameters(), lr=1e-3)
    for ep in range(a.recon_epochs):
        f = 1.0
        for e, m in ((10, 0.1), (25, 0.5), (100, 0.5), (250, 0.4)):
            if ep >= e:
                f *= m
        for g in opt.param_groups:
            g["lr"] = 1e-3 * f
        perm = torch.randperm(len(H))
        for i in range(0, len(H), 512):
            b = perm[i:i + 512]
            loss = torch.sqrt(F.mse_loss(mod.dec(V[b].to(C.DEV)), H[b].to(C.DEV)))
            opt.zero_grad(); loss.backward(); opt.step()
    log("recon init rmse", float(loss))


# ------------------------------------------------------------------------------------------------ forward helpers
def forward_logits(model, inj, mod, tok, items, ctx_extra=None):
    ids, am, pos, ctx, gold = C.collate(tok, items)
    if ctx_extra:
        ctx.update(ctx_extra)
    if inj is not None:
        inj.arm(mod, ctx)
    out = model(input_ids=ids, attention_mask=am, position_ids=pos, use_cache=False, logits_to_keep=1)
    if inj is not None:
        inj.armed = False
    return out.logits[:, -1, :].float(), gold


@torch.no_grad()
def dev_score(model, inj, mod, tok, items, bs=32):
    ok = 0
    for i in range(0, len(items), bs):
        lg, gold = forward_logits(model, inj, mod, tok, items[i:i + bs])
        ok += (lg.argmax(-1) == gold).sum().item()
    return ok / len(items)


def lr_at(step, a):
    if a.arm == "lora" or a.arm.startswith("e2e"):
        if step < 50:
            return a.lr * (step + 1) / 50
        return a.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (step - 50) / max(1, a.steps - 50))))
    f = 1.0  # DE25 decoder fine-tuning schedule (x0.5 at 100, x0.5 at 500, x0.4 at 1000)
    for s, m in ((100, 0.5), (500, 0.5), (1000, 0.4)):
        if step >= s:
            f *= m
    return a.lr * f


def cmd_train(a, tok, model):
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    e2 = "" if a.hops == 2 else str(a.hops)
    tr_split = f"train_e2e{e2}" if a.arm.startswith("e2e") else "train"
    dv_split = a.dev_split if a.dev_split != "dev" else (f"dev_e2e{e2}" if a.arm.startswith("e2e") else "dev")
    tr = [C.build(tok, r) for r in rows_for(a.task, tr_split)]
    dv = [C.build(tok, r) for r in rows_for(a.task, dv_split)[:(a.dev_limit or None)]]
    os.makedirs(a.out, exist_ok=True)
    inj, mod = None, None
    if a.arm == "lora":
        from peft import LoraConfig, get_peft_model
        cfg = LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05, bias="none",
                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
        model = get_peft_model(model, cfg)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        params = [p for p in model.parameters() if p.requires_grad]
        base_model = model.base_model.model
    else:
        mod = make_module(a, tok, model)
        if a.arm in ("vsa", "disc"):
            sd = torch.load(f"{a.reader}/reader.pt")
            mod.load_state_dict(sd, strict=False)
            for n_, p in mod.named_parameters():
                if n_.startswith(("enc", "head", "cls")):
                    p.requires_grad_(False)
            if a.arm == "vsa" and a.task == "N" and a.dec_init == "recon":
                recon_init(mod, a)
        params = [p for p in mod.parameters() if p.requires_grad]
        inj = C.Injector(model, a.layer)
        base_model = model
    npar = sum(p.numel() for p in params)
    log("trainable params", npar)
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    hist, best = [], (-1, None)
    acc0 = dev_score(model, inj, mod, tok, dv)
    hist.append({"step": 0, "dev": acc0}); log("dev@0", acc0)
    t0 = time.time()
    for step in range(a.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, a)
        bt = [tr[i] for i in rng.integers(0, len(tr), a.bs)]
        if a.arm == "lora":
            model.train()
        lg, gold = forward_logits(model, inj, mod, tok, bt)
        loss = F.cross_entropy(lg, gold)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if (step + 1) % 50 == 0:
            log(f"step {step + 1} loss {loss.item():.4f} {(time.time() - t0) / (step + 1):.2f}s/step")
        if (step + 1) % a.eval_every == 0 or step + 1 == a.steps:
            if a.arm == "lora":
                model.eval()
            acc = dev_score(model, inj, mod, tok, dv)
            hist.append({"step": step + 1, "dev": acc, "loss": loss.item()}); log("dev", step + 1, acc)
            if acc > best[0]:
                best = (acc, step + 1)
                if a.arm == "lora":
                    model.save_pretrained(f"{a.out}/best")
                else:
                    torch.save({k: v.cpu() for k, v in mod.state_dict().items()}, f"{a.out}/best.pt")
    json.dump({"args": vars(a), "trainable_params": npar, "hist": hist, "best_dev": best[0], "best_step": best[1]},
              open(f"{a.out}/train.json", "w"), indent=1)


# ------------------------------------------------------------------------------------------------ eval
@torch.no_grad()
def cmd_eval(a, tok, model):
    rows = rows_for(a.task, a.split)
    if a.limit:
        rows = rows[:a.limit]
    cot = a.arm == "cot"
    items = [C.build(tok, r, cot=cot) for r in rows]
    inj, mod = None, None
    if a.arm == "lora":
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, f"{a.ckpt}/best")
        model.eval()
    elif a.arm not in ("none", "cot"):
        mod = make_module(a, tok, model)
        if a.arm in ("vsa", "disc"):
            mod.load_state_dict(torch.load(f"{a.reader}/reader.pt"), strict=False)
        mod.load_state_dict(torch.load(f"{a.ckpt}/best.pt"), strict=False)
        mod.eval()
        inj = C.Injector(model, a.layer)
        inj.read_noise = a.noise
    torch.manual_seed(a.noise_seed)
    order = sorted(range(len(items)), key=lambda i: -len(items[i]["ids"]))
    bs = a.bs if not cot else a.cot_bs
    res = [None] * len(items)
    maxnew = a.max_new if not cot else a.cot_max_new
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        bt = [items[i] for i in idx]
        ids, am, pos, ctx, gold = C.collate(tok, bt)
        if a.shuffle_facts:
            ctx["shuffle"] = True
        if inj is not None:
            inj.arm(mod, ctx)
        out = model.generate(input_ids=ids, attention_mask=am, max_new_tokens=maxnew, do_sample=False,
                             temperature=None, top_p=None, pad_token_id=tok.pad_id,
                             return_dict_in_generate=True, output_logits=not cot)  # CoT: 512 steps of logits OOM
        info = inj.info if inj is not None else None
        if inj is not None:
            inj.armed = False
        first = out.logits[0].float() if not cot else None
        lp = torch.log_softmax(first, -1) if not cot else None
        gen = out.sequences[:, ids.shape[1]:]
        for j, i in enumerate(idx):
            txt = tok.decode(gen[j], skip_special_tokens=True)
            r = rows[i]
            pred = C.parse(a.task, txt, cot=cot)
            res[i] = {"id": r["id"], "gold": D.gold(r), "pred": pred, "correct": pred.strip().lower() == D.gold(r).lower(),
                      "out": txt if cot else txt[:40],
                      "argmax_ok": (int(first[j].argmax()) == int(gold[j])) if not cot else None,
                      "ce": float(-lp[j, gold[j]]) if not cot else None,
                      **({"op": r["op"], "x": r["x"], "y": r["y"]} if a.task == "N" else
                         {"depth": r["depth"], "size": r["size"]}),
                      **({"mod": info[j]} if info else {})}
        log(f"{s + len(idx)}/{len(items)} acc so far {np.mean([x['correct'] for x in res if x]):.3f}")
    os.makedirs(os.path.dirname(a.pred) or ".", exist_ok=True)
    with open(a.pred, "w") as f:
        for x in res:
            f.write(json.dumps(x) + "\n")
    log("acc", np.mean([x["correct"] for x in res]), "argmax", np.mean([bool(x["argmax_ok"]) for x in res]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd"); p.add_argument("--task", default="N"); p.add_argument("--arm", default="none")
    p.add_argument("--split", default="dev"); p.add_argument("--layers", default="17"); p.add_argument("--layer", type=int, default=17)
    p.add_argument("--subset", type=int, default=0); p.add_argument("--bs", type=int, default=16)
    p.add_argument("--epochs", type=int, default=100); p.add_argument("--eval_every", type=int, default=250)
    p.add_argument("--lr", type=float, default=1e-3); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=1000); p.add_argument("--rank", type=int, default=6)
    p.add_argument("--reader", default=""); p.add_argument("--ckpt", default=""); p.add_argument("--out", default="")
    p.add_argument("--pred", default=""); p.add_argument("--noise", type=float, default=0.0)
    p.add_argument("--noise_seed", type=int, default=12345); p.add_argument("--soft_beta", type=float, default=0.0)
    p.add_argument("--dec_init", default="recon"); p.add_argument("--recon_epochs", type=int, default=100)
    p.add_argument("--limit", type=int, default=0); p.add_argument("--max_new", type=int, default=6)
    p.add_argument("--cot_max_new", type=int, default=512); p.add_argument("--cot_bs", type=int, default=48)
    p.add_argument("--shuffle_facts", type=int, default=0)
    p.add_argument("--dev_split", default="dev"); p.add_argument("--dev_subset", type=int, default=0)
    p.add_argument("--dev_limit", type=int, default=0); p.add_argument("--hops", type=int, default=2)
    p.add_argument("--digit_thr", type=float, default=0.5)
    a = p.parse_args()
    t0 = time.time()
    if a.cmd == "reader":
        cmd_reader(a)
    else:
        tok, model = C.load()
        if a.cmd == "pools":
            print("bad pools:", C.check_pools(tok))
        elif a.cmd == "cache":
            cmd_cache(a, tok, model)
        elif a.cmd == "train":
            cmd_train(a, tok, model)
        elif a.cmd == "eval":
            cmd_eval(a, tok, model)
    ledger(a, t0)


if __name__ == "__main__":
    main()
