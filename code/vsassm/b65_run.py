"""B65 runner (prereg/B65_tuned_lr_read_position_prereg.md): task N, Llama-3.1-8B-Instruct frozen; three in-forward-pass
modules (VSA with DE25's published thresholded read, VSA with best-match argmax read, the same-shape discrete module),
two read positions, write-back learning rate chosen on dev from {3e-4, 1e-3, 3e-3, 1e-2} (+ extensions) for EVERY module.

Read positions
  last : B40's design. Every module reads the state ENTERING layer L at the last prompt token (b40_core.VSAN / DiscN
         unchanged; B40's cache of those states is reused).
  ops  : the clean room's arm A. The module reads the states entering layer L at the two operand tokens of the target
         question (x and y; each 0-999 is one Llama-3 token). VSA: v = enc(h_x) + enc(h_y) (one shared linear
         4096 -> 2048 encoder, no bias, B40's shape; the states are SUMMED after the shared encoder). DISC: type head on
         h_x + h_y (4096 -> 8), digit heads on [h_x; h_y] CONCATENATED (8192 -> 2 x 5 x 10).
  Both positions write back where B40 writes: h_last <- 0.5 w + 0.5 h_last at layer L, last prompt token, first pass,
  where the gate is open (b40_core.Injector unchanged).
Arms: vsapub (VSAN digit_thr 0.5 = DE25), vsabm (digit_thr -inf = argmax over the ten digit atoms, B55's N-ARG), disc.
Each arm is trained with its own read (the VSA-BM write-back sees argmax reads in training).

Commands (one per process; outputs under $B65_ROOT):
  cache   --split train|dev --layers 8,12,..        operand-position states (fp16) -> cache/N_{split}_ops_L{L}.pt
  reader  --pos P --arm vsa|disc --layer L --lr r --epochs 100,300,1000 --seed s --out D   snapshots + dev read acc
  train   --pos P --arm vsapub|vsabm|disc --layer L --reader D/e{E} --lr r --seed s --out D   write-back by CE
  eval    --pos P --arm A --layer L --reader R --ckpt K --split test|train --pred F [--limit n]
  selftest (training rows only)
"""
import argparse, json, os, sys, time, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import b40_data as D
import b40_core as C
import b40_run as R40

ROOT = os.environ.get("B65_ROOT", "/path/to/work/b65")
B40 = os.environ.get("B40_ROOT", "/path/to/work/b40")
CACHE65, CACHE40 = f"{ROOT}/cache", f"{B40}/cache"
DIGIT_THR = {"vsapub": 0.5, "vsabm": float("-inf")}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def ledger(t0):
    os.makedirs(f"{ROOT}/logs", exist_ok=True)
    with open(f"{ROOT}/logs/gpu_hours.txt", "a") as f:
        f.write(json.dumps({"cmd": " ".join(sys.argv[1:]), "gpu_h": round((time.time() - t0) / 3600, 4),
                            "host": os.uname().nodename, "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
                            "end": time.strftime("%Y-%m-%dT%H:%M:%S")}) + "\n")


def rows(split, limit=0):
    r = D.gen("N", split)
    return r[:limit] if limit else r


# ------------------------------------------------------------------------------------------------ operand positions
def op_char_offsets(r):
    """Character offsets of x and y inside the target question (template-based: robust to x == y and to y == 2 in
    'x^2 mod y')."""
    t = D.op_question(r["op"], "\x00", "\x01")
    ix = t.index("\x00")
    iy = t.index("\x01") - 1 + len(str(r["x"]))
    q = D.op_question(r["op"], r["x"], r["y"])
    assert q[ix:ix + len(str(r["x"]))] == str(r["x"]) and q[iy:iy + len(str(r["y"]))] == str(r["y"]), (q, ix, iy)
    return q, ix, iy


def build65(tok, r):
    it = C.build(tok, r)
    text = C.chat_text(D.messages(r))
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    assert enc["input_ids"] == it["ids"]
    q, ix, iy = op_char_offsets(r)
    q0 = text.rfind(q)
    assert q0 >= 0

    def tidx(c, s):
        for t, (a, b) in enumerate(enc["offset_mapping"]):
            if a <= c < b:
                assert tok.decode([it["ids"][t]]).strip() == s, (tok.decode([it["ids"][t]]), s)
                return t
        raise AssertionError(c)
    it["xi"] = tidx(q0 + ix, str(r["x"]))
    it["yi"] = tidx(q0 + iy, str(r["y"]))
    assert it["xi"] < it["yi"] < len(it["ids"]) - 1
    return it


def collate65(tok, items):
    ids, am, pos, ctx, gold = C.collate(tok, items)
    T = ids.shape[1]
    ctx["xi"] = torch.tensor([it["xi"] + T - len(it["ids"]) for it in items], device=C.DEV)
    ctx["yi"] = torch.tensor([it["yi"] + T - len(it["ids"]) for it in items], device=C.DEV)
    return ids, am, pos, ctx, gold


def pair_states(h, ctx):  # [B, T, D] -> [B, 2, D] (x token, y token)
    ar = torch.arange(h.shape[0], device=h.device)
    return torch.stack([h[ar, ctx["xi"]], h[ar, ctx["yi"]]], 1)


# ------------------------------------------------------------------------------------------------ ops modules
class VSANOps(C.VSAN):
    """B40's VSAN reading the two operand tokens: v = enc(h_x) + enc(h_y); everything after the encoder unchanged."""

    def encode(self, hp):
        return self.enc(hp[:, 0].float()) + self.enc(hp[:, 1].float())

    def read(self, hp):
        v = self.encode(hp)
        op_i, op_s = self.eng.decode_op(v)
        x, _ = self.eng.decode_num(v, self.eng.n1, thr=self.digit_thr)
        y, _ = self.eng.decode_num(v, self.eng.n2, thr=self.digit_thr)
        return v, op_i, op_s, x, y

    def forward(self, h, ctx):
        with torch.no_grad():
            v, op_i, op_s, x, y = self.read(pair_states(h, ctx))
            ans = torch.tensor([C.safe_answer(D.OPS_ALL[o], a, b) for o, a, b in zip(op_i.tolist(), x.tolist(), y.tolist())],
                               device=h.device)
            code = self.eng.num(ans)
        gate = op_s > self.thr
        info = [{"op": D.OPS_ALL[o], "op_s": round(float(s), 3), "x": int(a), "y": int(b), "ans": int(c), "gate": bool(g)}
                for o, s, a, b, c, g in zip(op_i.tolist(), op_s.tolist(), x.tolist(), y.tolist(), ans.tolist(), gate.tolist())]
        return self.dec(code), gate, info


class DiscNOps(nn.Module):
    """B40's DiscN reading the two operand tokens: type head on h_x + h_y, digit heads on [h_x; h_y]; the same argmax,
    gate (softmax > 0.8), Python function and per-(place, digit) write-back embedding."""

    def __init__(self, thr=0.8):
        super().__init__()
        self.thr = thr
        self.head_op = nn.Linear(C.HID, len(D.OPS_TRAIN))
        self.head_dig = nn.Linear(2 * C.HID, 2 * C.MAX_DIGITS * 10)
        self.emb = nn.Parameter(torch.zeros(C.MAX_DIGITS, 10, C.HID))

    def logits(self, hp):
        hx, hy = hp[:, 0].float(), hp[:, 1].float()
        lo = self.head_op(hx + hy)
        z = self.head_dig(torch.cat([hx, hy], -1))
        return lo, z[:, :50].view(-1, C.MAX_DIGITS, 10), z[:, 50:].view(-1, C.MAX_DIGITS, 10)

    def read(self, hp):
        lo, lx, ly = self.logits(hp)
        pw = torch.tensor([10 ** k for k in range(C.MAX_DIGITS)], device=hp.device)
        pr = lo.softmax(-1)
        return pr.argmax(-1), pr.max(-1).values, (lx.argmax(-1) * pw).sum(1), (ly.argmax(-1) * pw).sum(1)

    def forward(self, h, ctx):
        with torch.no_grad():
            op_i, op_p, x, y = self.read(pair_states(h, ctx))
            ans = [C.safe_answer(D.OPS_TRAIN[o], a, b) for o, a, b in zip(op_i.tolist(), x.tolist(), y.tolist())]
            dig = torch.tensor([[(a // 10 ** k) % 10 for k in range(C.MAX_DIGITS)] for a in ans], device=h.device)
        w = self.emb[torch.arange(C.MAX_DIGITS, device=h.device).unsqueeze(0), dig].sum(1)
        gate = op_p > self.thr
        info = [{"op": D.OPS_TRAIN[o], "op_p": round(float(s), 3), "x": int(a), "y": int(b), "ans": int(c), "gate": bool(g)}
                for o, s, a, b, c, g in zip(op_i.tolist(), op_p.tolist(), x.tolist(), y.tolist(), ans, gate.tolist())]
        return w, gate, info


def make(pos, arm, eng=None):
    """arm in vsa (reader training; read rule irrelevant), vsapub, vsabm, disc."""
    if arm.startswith("vsa"):
        eng = eng or C.NumEngine().to(C.DEV)
        thr = DIGIT_THR.get(arm, 0.5)
        m = C.VSAN(eng, digit_thr=thr) if pos == "last" else VSANOps(eng, digit_thr=thr)
    else:
        m = C.DiscN() if pos == "last" else DiscNOps()
    return m.to(C.DEV)


# ------------------------------------------------------------------------------------------------ caches
@torch.no_grad()
def cmd_cache(a, tok, model):
    rs = rows(a.split, a.limit)
    layers = [int(x) for x in a.layers.split(",")]
    items = [build65(tok, r) for r in rs]
    out = {L: [] for L in layers}
    for i in range(0, len(items), a.bs):
        bt = items[i:i + a.bs]
        ids, am, pos, ctx, gold = collate65(tok, bt)
        hs = model(input_ids=ids, attention_mask=am, position_ids=pos, output_hidden_states=True,
                   logits_to_keep=1).hidden_states
        for L in layers:
            out[L].append(pair_states(hs[L], ctx).half().cpu())
    os.makedirs(CACHE65, exist_ok=True)
    tag = f"N_{a.split}" + (f"_lim{a.limit}" if a.limit else "")
    for L in layers:
        torch.save(torch.cat(out[L]), f"{CACHE65}/{tag}_ops_L{L}.pt")
    json.dump({"op": [r["op"] for r in rs], "x": [r["x"] for r in rs], "y": [r["y"] for r in rs]},
              open(f"{CACHE65}/{tag}_ops_meta.json", "w"))
    log("cached", tag, layers, torch.cat(out[layers[0]]).shape)


def load_states(pos, split, L, gpu=False):
    if pos == "last":
        H = torch.load(f"{CACHE40}/N_{split}_L{L}.pt")
        meta = json.load(open(f"{CACHE40}/N_{split}_meta.json"))
    else:
        H = torch.load(f"{CACHE65}/N_{split}_ops_L{L}.pt")
        meta = json.load(open(f"{CACHE65}/N_{split}_ops_meta.json"))
    return (H.to(C.DEV) if gpu else H), meta


def encode_rows(mod, pos, h):
    """The module's VSA vector for a batch of cached states."""
    return mod.enc(h.float()) if pos == "last" else mod.encode(h)


# ------------------------------------------------------------------------------------------------ readers
@torch.no_grad()
def read_acc(mod, arm, H, meta, thr=None):
    """Fraction of rows with type and both operands read exactly (b40_run.read_acc_N's metric), for a read rule."""
    old = getattr(mod, "digit_thr", None)
    if thr is not None:
        mod.digit_thr = thr
    ops, xs, ys = [], [], []
    for i in range(0, len(H), 2048):
        h = H[i:i + 2048].to(C.DEV)
        if arm == "vsa":
            v, op_i, op_s, x, y = mod.read(h)
            ops.append(op_i.cpu())
        else:
            op_i, op_p, x, y = mod.read(h)
            ops.append(torch.tensor([D.OPS_ALL.index(D.OPS_TRAIN[o]) for o in op_i.tolist()]))
        xs.append(x.cpu()); ys.append(y.cpu())
    if thr is not None:
        mod.digit_thr = old
    op, x, y = torch.cat(ops), torch.cat(xs), torch.cat(ys)
    gop = torch.tensor([D.OPS_ALL.index(o) for o in meta["op"]]); gx = torch.tensor(meta["x"]); gy = torch.tensor(meta["y"])
    return float(((op == gop) & (x == gx) & (y == gy)).float().mean())


def cmd_reader(a):
    """B40's cmd_reader (same RNG order, DE25 schedule, Adam, batch 512), with snapshots at the listed epochs. For
    pos=last and lr 1e-3 this is B40's reader; a snapshot at epoch E equals a separate E-epoch run (the schedule depends
    on the epoch only)."""
    torch.manual_seed(a.seed)
    Htr, mtr = load_states(a.pos, "train", a.layer, gpu=True)
    Hdv, mdv = load_states(a.pos, a.dev_split, a.layer)
    if a.dev_limit:
        Hdv = Hdv[:a.dev_limit]; mdv = {k: v[:a.dev_limit] for k, v in mdv.items()}
    eng = C.NumEngine().to(C.DEV)
    mod = make(a.pos, a.arm, eng)
    if a.arm == "vsa":
        _, _, _, T = R40.n_targets(mtr, eng)
        params = mod.enc.parameters()
    else:
        op = torch.tensor([D.OPS_TRAIN.index(o) for o in mtr["op"]]).to(C.DEV)
        xd = torch.stack([(torch.tensor(mtr["x"]) // 10 ** k) % 10 for k in range(C.MAX_DIGITS)], 1).to(C.DEV)
        yd = torch.stack([(torch.tensor(mtr["y"]) // 10 ** k) % 10 for k in range(C.MAX_DIGITS)], 1).to(C.DEV)
        params = [p for n_, p in mod.named_parameters() if n_.startswith("head")]
    opt = torch.optim.Adam(params, lr=a.lr)
    snaps = sorted(int(e) for e in a.epochs.split(","))
    n, hist = len(Htr), []
    os.makedirs(a.out, exist_ok=True)
    for ep in range(snaps[-1]):
        for g in opt.param_groups:
            g["lr"] = R40.de25_lr(ep, a.lr)
        perm = torch.randperm(n, device=C.DEV)
        for i in range(0, n, 512):
            b = perm[i:i + 512]
            h = Htr[b].float()
            if a.arm == "vsa":
                loss = torch.sqrt(F.mse_loss(encode_rows(mod, a.pos, h), T[b]))
            else:
                lo, lx, ly = mod.logits(h)
                loss = (F.cross_entropy(lo, op[b]) + F.cross_entropy(lx.reshape(-1, 10), xd[b].reshape(-1))
                        + F.cross_entropy(ly.reshape(-1, 10), yd[b].reshape(-1)))
            opt.zero_grad(); loss.backward(); opt.step()
        if ep + 1 in snaps:
            if a.arm == "vsa":
                acc = {"pub": read_acc(mod, "vsa", Hdv, mdv, 0.5), "bm": read_acc(mod, "vsa", Hdv, mdv, float("-inf"))}
            else:
                acc = {"disc": read_acc(mod, "disc", Hdv, mdv)}
            acc.update(epoch=ep + 1, loss=float(loss))
            hist.append(acc); log(json.dumps(acc))
            d = f"{a.out}/e{ep + 1}"
            os.makedirs(d, exist_ok=True)
            torch.save({k: v.cpu() for k, v in mod.state_dict().items() if k.startswith(("enc", "head"))}, f"{d}/reader.pt")
    json.dump({"args": vars(a), "hist": hist}, open(f"{a.out}/reader.json", "w"), indent=1)


# ------------------------------------------------------------------------------------------------ write-back
def recon_init_ops(mod, a):
    """DE25's decoder pre-training, ops analogue: the linear decoder maps the encoder's VSA vector (sum over the two
    operand tokens) to the state at the WRITE position (last prompt token, B40's cache), RMSE; b40_run.recon_init's
    schedule and batch."""
    Hp, _ = load_states("ops", "train", a.layer)
    H = torch.load(f"{CACHE40}/N_train_L{a.layer}.pt").float()
    with torch.no_grad():
        V = torch.cat([mod.encode(Hp[i:i + 4096].to(C.DEV)).cpu() for i in range(0, len(Hp), 4096)])
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
    log("recon init (ops) rmse", float(loss))


def forward_logits(model, inj, mod, tok, items):
    ids, am, pos, ctx, gold = collate65(tok, items)
    inj.arm(mod, ctx)
    out = model(input_ids=ids, attention_mask=am, position_ids=pos, use_cache=False, logits_to_keep=1)
    inj.armed = False
    return out.logits[:, -1, :].float(), gold


@torch.no_grad()
def dev_score(model, inj, mod, tok, items, bs=32):
    ok = 0
    for i in range(0, len(items), bs):
        lg, gold = forward_logits(model, inj, mod, tok, items[i:i + bs])
        ok += (lg.argmax(-1) == gold).sum().item()
    return ok / len(items)


def lr_at(step, base):  # b40_run.lr_at for the vsa / disc arms (DE25 decoder schedule)
    f = 1.0
    for s, m in ((100, 0.5), (500, 0.5), (1000, 0.4)):
        if step >= s:
            f *= m
    return base * f


def cmd_train(a, tok, model):
    """b40_run.cmd_train for the vsa / disc arms (same RNG order, AdamW wd 0, batch 16, clip 1, DE25 schedule, dev
    first-token argmax every eval_every steps, best step kept), with the read position and read rule of the arm."""
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    tr = [build65(tok, r) for r in rows(a.train_split, a.train_limit)]
    dv = [build65(tok, r) for r in rows(a.dev_split, a.dev_limit)]
    os.makedirs(a.out, exist_ok=True)
    mod = make(a.pos, a.arm)
    mod.load_state_dict(torch.load(f"{a.reader}/reader.pt"), strict=False)
    for n_, p in mod.named_parameters():
        if n_.startswith(("enc", "head")):
            p.requires_grad_(False)
    if a.arm.startswith("vsa") and a.dec_init == "recon":
        if a.pos == "last":
            R40.recon_init(mod, argparse.Namespace(task="N", layer=a.layer, recon_epochs=a.recon_epochs))
        else:
            recon_init_ops(mod, a)
    params = [p for p in mod.parameters() if p.requires_grad]
    inj = C.Injector(model, a.layer)
    npar = sum(p.numel() for p in params)
    log("trainable params", npar)
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    hist, best = [], (-1, None)
    acc0 = dev_score(model, inj, mod, tok, dv)
    hist.append({"step": 0, "dev": acc0}); log("dev@0", acc0)
    t0 = time.time()
    for step in range(a.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, a.lr)
        bt = [tr[i] for i in rng.integers(0, len(tr), a.bs)]
        lg, gold = forward_logits(model, inj, mod, tok, bt)
        loss = F.cross_entropy(lg, gold)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if (step + 1) % 50 == 0:
            log(f"step {step + 1} loss {loss.item():.4f} {(time.time() - t0) / (step + 1):.2f}s/step")
        if (step + 1) % a.eval_every == 0 or step + 1 == a.steps:
            acc = dev_score(model, inj, mod, tok, dv)
            hist.append({"step": step + 1, "dev": acc, "loss": loss.item()}); log("dev", step + 1, acc)
            if acc > best[0]:
                best = (acc, step + 1)
                torch.save({k: v.cpu() for k, v in mod.state_dict().items()}, f"{a.out}/best.pt")
    inj.handle.remove()
    json.dump({"args": vars(a), "trainable_params": npar, "hist": hist, "best_dev": best[0], "best_step": best[1]},
              open(f"{a.out}/train.json", "w"), indent=1)


# ------------------------------------------------------------------------------------------------ eval (B40's / B55's)
@torch.no_grad()
def cmd_eval(a, tok, model):
    rs = rows(a.split, a.limit)
    items = [build65(tok, r) for r in rs]
    mod = make(a.pos, a.arm)
    mod.load_state_dict(torch.load(f"{a.reader}/reader.pt"), strict=False)
    mod.load_state_dict(torch.load(f"{a.ckpt}/best.pt"), strict=False)
    mod.eval()
    inj = C.Injector(model, a.layer)
    torch.manual_seed(12345)
    order = sorted(range(len(items)), key=lambda i: -len(items[i]["ids"]))
    res = [None] * len(items)
    for s in range(0, len(order), a.bs):
        idx = order[s:s + a.bs]
        ids, am, pos, ctx, gold = collate65(tok, [items[i] for i in idx])
        inj.arm(mod, ctx)
        out = model.generate(input_ids=ids, attention_mask=am, max_new_tokens=6, do_sample=False,
                             temperature=None, top_p=None, pad_token_id=tok.pad_id,
                             return_dict_in_generate=True, output_logits=True)
        info = inj.info
        inj.armed = False
        first = out.logits[0].float()
        lp = torch.log_softmax(first, -1)
        gen = out.sequences[:, ids.shape[1]:]
        for j, i in enumerate(idx):
            txt = tok.decode(gen[j], skip_special_tokens=True)
            r = rs[i]
            pred = C.parse("N", txt, cot=False)
            res[i] = {"id": r["id"], "gold": D.gold(r), "pred": pred, "correct": pred.strip().lower() == D.gold(r).lower(),
                      "out": txt[:40], "argmax_ok": int(first[j].argmax()) == int(gold[j]), "ce": float(-lp[j, gold[j]]),
                      "op": r["op"], "x": r["x"], "y": r["y"], "mod": info[j]}
    inj.handle.remove()
    os.makedirs(os.path.dirname(a.pred) or ".", exist_ok=True)
    tmp = a.pred + ".part"
    with open(tmp, "w") as f:
        for x in res:
            f.write(json.dumps(x) + "\n")
    os.replace(tmp, a.pred)
    log("acc", round(float(np.mean([x["correct"] for x in res])), 4))


# ------------------------------------------------------------------------------------------------ self-test
def cmd_selftest(a, tok, model):
    """Training rows only: operand positions on every training row (asserted), module shapes, a forward with each
    module at each position, and that pos=last modules equal B40's classes."""
    tr = rows("train")
    bad = 0
    for r in tr:
        try:
            build65(tok, r)
        except AssertionError:
            bad += 1
    log("operand-position build on", len(tr), "training rows; failures", bad)
    assert bad == 0
    items = [build65(tok, r) for r in tr[:8]]
    ids, am, pos, ctx, gold = collate65(tok, items)
    for i, it in enumerate(items):  # padded operand indices point at the operand tokens
        sh = ids.shape[1] - len(it["ids"])
        assert int(ids[i, ctx["xi"][i]]) == it["ids"][it["xi"]] and int(ids[i, ctx["yi"][i]]) == it["ids"][it["yi"]]
        assert tok.decode([int(ids[i, ctx["xi"][i]])]).strip() == str(tr[i]["x"])
    for p in ("last", "ops"):
        for arm in ("vsapub", "vsabm", "disc"):
            m = make(p, arm)
            inj = C.Injector(model, 20)
            lg, g = forward_logits(model, inj, m, tok, items)
            inj.handle.remove()
            log(p, arm, "params", C.n_params(m), "logits", tuple(lg.shape), "gate", inj.info[0]["gate"])
    log("selftest ok")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd"); p.add_argument("--pos", default="last"); p.add_argument("--arm", default="vsapub")
    p.add_argument("--split", default="train"); p.add_argument("--layers", default="20"); p.add_argument("--layer", type=int, default=20)
    p.add_argument("--bs", type=int, default=16); p.add_argument("--epochs", default="100,300,1000")
    p.add_argument("--lr", type=float, default=1e-3); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=1000); p.add_argument("--eval_every", type=int, default=250)
    p.add_argument("--reader", default=""); p.add_argument("--ckpt", default=""); p.add_argument("--out", default="")
    p.add_argument("--pred", default=""); p.add_argument("--limit", type=int, default=0)
    p.add_argument("--dec_init", default="recon"); p.add_argument("--recon_epochs", type=int, default=100)
    p.add_argument("--dev_split", default="dev"); p.add_argument("--dev_limit", type=int, default=0)
    p.add_argument("--train_split", default="train"); p.add_argument("--train_limit", type=int, default=0)
    a = p.parse_args()
    if a.cmd in ("eval", "selftest") and a.split not in ("train", "test"):
        raise SystemExit("eval runs on train (self-test) or test only")
    t0 = time.time()
    if a.cmd == "reader":
        cmd_reader(a)
    else:
        tok, model = C.load()
        {"cache": cmd_cache, "train": cmd_train, "eval": cmd_eval, "selftest": cmd_selftest}[a.cmd](a, tok, model)
    ledger(t0)


if __name__ == "__main__":
    main()
