"""B48 part B arms on the alias stories (prereg/B48_one_form_and_alias_naming_prereg.md).

Every arm gives each story mention a key; a key's code is a fixed random unit vector (d = 2048, generator seeded by the
codebook seed and the md5 of the key), so equal keys give equal codes and different keys near-orthogonal codes. The
codes go into arena C's two-hop HRR chain (perm encoding: value argument permuted, generator seed + 7001; clean-up by
arg-max over the level's object codes after each hop; score = cosine of the hop-2 winner's code and the candidate's
code); the answer is "yes" iff score >= 0.5 (every arm's codes are snapped unit vectors).

  oracle  key = the gold entity
  hash    key = norm(surface string): exact string hashing (an alias or a pronoun is a different key by design)
  lm      key = norm(the name Qwen3-8B wrote for the mention, b48_lmcanon.py); question mentions: norm(surface)
  head    naming head on frozen Qwen3-8B states, open-vocabulary form: a bilinear pointer from each mention's span-mean
          state to the span-mean states of the 16 names listed at the top of the story; arg-max slot -> key = that
          slot's entity. Trained with identity labels (the gold slot of every mention) on training stories (training
          entities only); dev and test stories contain only unseen entities.

  python b48_alias.py tune  --kind W --layer 18 --lr 3e-4 --out out/b/tune/W_L18_lr0.0003.json
  python b48_alias.py select --out out/b/select.json
  python b48_alias.py final --kind W --select out/b/select.json --out out/b/final/W_head.json
  python b48_alias.py score --kind W --split test --head out/b/final/W_head.json --lm out/lm/W_test.json \
      --out out/b/scores_W_test.json
"""
import argparse, hashlib, json, math, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import b31_prep as P31                                                      # noqa: E402
import b48_struct as S48                                                    # noqa: E402

D = 2048
K = S48.K
SEEDS = [42, 123, 456]
LAYERS = [9, 18, 27, 36]
LRS = [3e-4, 1e-3]
THR = 0.5


# ------------------------------------------------------------------ codes and chain (torch, float32)

def code(key, seed):
    h = int(hashlib.md5(("%d|%s" % (seed, key)).encode()).hexdigest()[:15], 16)
    v = np.random.default_rng(h).standard_normal(D)
    return v / np.linalg.norm(v)


def perm(seed):
    return torch.randperm(D, generator=torch.Generator().manual_seed(seed + 7001))


def chain(s, keys, seed, cache):
    """Arena C's chain on one story. keys: one key per mention. Returns score, (hop-1 winner, hop-2 winner) mention
    indices."""
    def c(k):
        if k not in cache:
            cache[k] = torch.tensor(code(k, seed), dtype=torch.float32)
        return cache[k]
    P = cache.setdefault("__P", perm(seed))
    Kc = lambda i: c(keys[i])
    key = Kc(s["q"])
    wins = []
    for L in range(2):
        Kr = torch.stack([Kc(r) for r, _ in s["lv"][L]])
        Vf = torch.stack([Kc(f)[P] for _, f in s["lv"][L]])
        mem = torch.fft.irfft(torch.fft.rfft(Vf, dim=-1) * torch.fft.rfft(Kr, dim=-1), n=D, dim=-1).sum(0)
        hat = F.normalize(torch.fft.irfft(torch.fft.rfft(mem) * torch.conj(torch.fft.rfft(key)), n=D), dim=-1)
        w = int(F.cosine_similarity(Vf, hat[None, :], dim=-1).argmax())
        wi = s["lv"][L][w][1]
        wins.append(wi)
        key = Kc(wi)
    score = float((key * Kc(s["cand"])).sum().clamp(-1, 1))
    return score, wins


def chain_mentions(s):
    return sorted({i for L in range(2) for p in s["lv"][L] for i in p} | {s["q"], s["cand"]})


# ------------------------------------------------------------------ keys per arm

def keys_oracle(s):
    return ["ent:" + s["ents"][m[1]]["id"] for m in s["mentions"]]


def keys_hash(s):
    return ["str:" + S48.norm(s["text"][m[2]:m[3]]) for m in s["mentions"]]


def keys_lm(s, rec, sid):
    """rec: b48_lmcanon output of the story. Binding statement n (1-based; level 0: 1-16, level 1: 17-28 in text
    order) -> (subject, object). Missing or unparsed: a key no other mention has."""
    keys = keys_hash(s)                                          # roster and question mentions: their own string
    parsed = {int(k): v for k, v in rec["parsed"].items()}
    written = {}
    for L in range(2):
        for j, (r, f) in enumerate(s["lv"][L]):
            n = 16 * L + j + 1
            for pos, mi in enumerate((r, f)):
                if n in parsed and S48.norm(parsed[n][pos]):
                    keys[mi] = "str:" + S48.norm(parsed[n][pos]); written[mi] = parsed[n][pos]
                else:
                    keys[mi] = "miss:%s:%d" % (sid, mi); written[mi] = None
    return keys, written


def named_ok(s, keys, arm):
    """Per mention: does the arm give it the key it gives the entity's listed name (roster mention)?"""
    ros = {m[1]: keys[i] for i, m in enumerate(s["mentions"]) if m[4] == "roster"}
    if arm in ("oracle", "head"):
        return [keys[i] == "ent:" + s["ents"][m[1]]["id"] for i, m in enumerate(s["mentions"])]
    return [keys[i] == "str:" + S48.norm(s["ents"][m[1]]["name"]) == ros[m[1]] for i, m in enumerate(s["mentions"])]


# ------------------------------------------------------------------ the naming head (pointer over the listed names)

class Pointer(nn.Module):
    def __init__(self, hid, mu, sd, r=256):
        super().__init__()
        self.register_buffer("mu", mu); self.register_buffer("sd", sd)
        self.q = nn.Linear(hid, r); self.k = nn.Linear(hid, r)
        self.r = r

    def forward(self, X, ros):
        """X (B, M, hid) states; ros: indices of the K roster mentions. Logits (B, M, K)."""
        Z = (X - self.mu) / self.sd
        return torch.einsum("bmr,bkr->bmk", self.q(Z), self.k(Z[:, ros])) / math.sqrt(self.r)


def load_split(kind, sp, layer, feats):
    st = S48.build_B(kind, sp)
    o = torch.load(os.path.join(feats, "qwen8L%d_%s_%s.pt" % (layer, kind, sp)), weights_only=False)
    assert o["text_md5"] == P31.h([s["text"] for s in st]), "feats/structure mismatch"
    y = torch.tensor([[m[1] for m in s["mentions"]] for s in st])
    return st, o["F"], y


def head_run(a, seed, data, dev):
    torch.manual_seed(seed)
    (st_tr, Ftr, ytr), (st_dv, Fdv, ydv) = data["train"], data["dev"]
    ros = [i for i, m in enumerate(st_tr[0]["mentions"]) if m[4] == "roster"]
    assert ros == list(range(K))
    flat = Ftr.float().reshape(-1, Ftr.size(-1))
    m = Pointer(Ftr.size(-1), flat.mean(0), flat.std(0) + 1e-6).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr)
    g = torch.Generator().manual_seed(seed)
    cm = chain_mentions(st_dv[0])

    def predict(Fs):
        m.eval(); out = []
        with torch.no_grad():
            for k in range(0, Fs.size(0), 64):
                out.append(m(Fs[k:k + 64].to(dev).float(), ros).argmax(-1).cpu())
        return torch.cat(out)

    def evaluate(ep):
        e = {"epoch": ep}
        for sp in [x for x in ("dev", "test") if x in data]:
            st, Fs, ys = data[sp]
            pr = predict(Fs)
            ok = (pr == ys)
            mt = [mm[4] for mm in st[0]["mentions"]]
            e[sp] = {"naming_chain": float(ok[:, cm].float().mean()), "naming_all": float(ok.float().mean())}
            for t in sorted(set(mt)):
                cols = [i for i, x in enumerate(mt) if x == t]
                e[sp]["naming_" + t] = float(ok[:, cols].float().mean())
            if sp == "test" or a.save_pred or a.cmd == "final":
                e[sp]["pred"] = pr.tolist()
        return e

    rec = {"seed": seed, "epochs": []}
    t0 = time.time()
    for ep in range(a.epochs):
        m.train()
        pm = torch.randperm(Ftr.size(0), generator=g)
        tl, nb = 0.0, 0
        for k in range(0, Ftr.size(0), 4):
            i = pm[k:k + 4]
            lg = m(Ftr[i].to(dev).float(), ros)
            loss = F.cross_entropy(lg.reshape(-1, K), ytr[i].to(dev).reshape(-1))
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
            tl += float(loss); nb += 1
        e = evaluate(ep + 1); e["loss"] = tl / nb; e["secs"] = round(time.time() - t0, 1)
        rec["epochs"].append(e)
        print("  %s L%d lr %g seed %d ep%d loss %.4f dev chain-naming %.4f (alias %s, pron %s)%s" % (
            a.kind, a.layer, a.lr, seed, ep + 1, e["loss"], e["dev"]["naming_chain"],
            "%.4f" % e["dev"]["naming_alias"] if "naming_alias" in e["dev"] else "-",
            "%.4f" % e["dev"]["naming_pron"] if "naming_pron" in e["dev"] else "-",
            " test chain-naming %.4f" % e["test"]["naming_chain"] if "test" in e else ""), flush=True)
    dv = [x["dev"]["naming_chain"] for x in rec["epochs"]]
    rec["dev_selected_epoch"] = max(range(len(dv)), key=lambda j: (dv[j], -j)) + 1
    return rec


def cmd_head(a):
    if os.path.exists(a.out):
        print("exists, skip", a.out); return
    if a.cmd == "final":
        sel = json.load(open(a.select))[a.kind]
        a.layer, a.lr = sel["layer"], sel["lr"]
    splits = ["train", "dev"] + (["test"] if a.cmd == "final" else [])
    data = {sp: load_split(a.kind, sp, a.layer, a.feats) for sp in splits}
    out = {"args": vars(a), "host": os.uname()[1],
           "gpu": torch.cuda.get_device_name(a.device) if a.device.startswith("cuda") else "cpu",
           "start": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "runs": []}
    for s in (a.seeds if a.cmd == "final" else [42]):
        out["runs"].append(head_run(a, s, data, a.device))
    out["end"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out + ".tmp", "w")); os.replace(a.out + ".tmp", a.out)


def cmd_select(a):
    """Per dataset: the (layer, lr) with the highest best-epoch dev chain-naming accuracy; ties within 0.002 go to the
    earlier entry of the grid (layers 9, 18, 27, 36; lr 3e-4 before 1e-3)."""
    sel = {}
    for kind in "WP":
        rows = []
        for L in LAYERS:
            for lr in LRS:
                p = os.path.join(a.root, "tune", "%s_L%d_lr%g.json" % (kind, L, lr))
                if not os.path.exists(p):
                    print("MISSING", p); continue
                d = json.load(open(p))
                dv = [e["dev"]["naming_chain"] for e in d["runs"][0]["epochs"]]
                rows.append({"layer": L, "lr": lr, "dev": max(dv), "epoch": int(np.argmax(dv)) + 1})
        if rows:
            top = max(r["dev"] for r in rows)
            sel[kind] = [r for r in rows if r["dev"] >= top - 0.002][0]
            sel[kind + "_grid"] = rows
            print(kind, json.dumps(sel[kind]))
    json.dump(sel, open(a.out, "w"), indent=1)


# ------------------------------------------------------------------ scoring of all arms on a split

def cmd_score(a):
    torch.set_num_threads(1)                    # small FFTs: thread start-up dominates otherwise
    st = S48.build_B(a.kind, a.split)
    y = np.array([s["label"] for s in st])
    arms = {"oracle": [keys_oracle(s) for s in st], "hash": [keys_hash(s) for s in st]}
    written = None
    if a.lm and os.path.exists(a.lm):
        lm = json.load(open(a.lm))["stories"]
        kw = [keys_lm(s, lm[str(i)], "%s%s%d" % (a.kind, a.split, i)) for i, s in enumerate(st)]
        arms["lm"] = [k for k, _ in kw]; written = [w for _, w in kw]
    heads = {}
    if a.head and os.path.exists(a.head):
        hd = json.load(open(a.head))
        for run in hd["runs"]:
            pr = run["epochs"][run["dev_selected_epoch"] - 1][a.split]["pred"]
            heads[run["seed"]] = [["ent:" + s["ents"][p]["id"] for p in pr[i]] for i, s in enumerate(st)]
    res = {"args": vars(a), "n": len(st), "labels": y.tolist(), "arms": {}}
    for arm in list(arms) + (["head"] if heads else []):
        per_seed = []
        for seed in SEEDS:
            ks = heads[seed] if arm == "head" else arms[arm]
            cache = {}
            sc, w1, w2, nm = [], [], [], []
            for i, s in enumerate(st):
                score, wins = chain(s, ks[i], seed, cache)
                sc.append(round(score, 6))
                ent = [m[1] for m in s["mentions"]]
                w1.append(int(ent[wins[0]] == s["chain"][1])); w2.append(int(ent[wins[1]] == s["chain"][2]))
                nm.append([int(x) for x in named_ok(s, ks[i], "head" if arm == "head" else arm)])
            pred = (np.array(sc) >= THR).astype(int)
            per_seed.append({"seed": seed, "scores": sc, "correct": (pred == y).astype(int).tolist(),
                             "acc": float((pred == y).mean()), "hop1_ok": w1, "hop2_ok": w2, "named": nm})
            print("  %s %s %s seed %d acc %.4f hop1 %.3f hop2 %.3f" % (a.kind, a.split, arm, seed,
                                                                       per_seed[-1]["acc"], np.mean(w1), np.mean(w2)),
                  flush=True)
        res["arms"][arm] = per_seed
    res["mtypes"] = [m[4] for m in st[0]["mentions"]]
    res["chain_mentions"] = chain_mentions(st[0])
    if written is not None:
        res["lm_written"] = [{str(k): v for k, v in w.items()} for w in written]
        res["lm_parsed_lines"] = [len(lm[str(i)]["parsed"]) for i in range(len(st))]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(res, open(a.out + ".tmp", "w")); os.replace(a.out + ".tmp", a.out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["tune", "final", "select", "score"])
    ap.add_argument("--kind", choices=["W", "P"])
    ap.add_argument("--layer", type=int, default=36)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--save_pred", action="store_true")
    ap.add_argument("--split", default="test")
    ap.add_argument("--select", default="out/b/select.json")
    ap.add_argument("--root", default="out/b")
    ap.add_argument("--head", default=None)
    ap.add_argument("--lm", default=None)
    ap.add_argument("--feats", default="feats")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd in ("tune", "final"):
        cmd_head(a)
    elif a.cmd == "select":
        cmd_select(a)
    else:
        cmd_score(a)


if __name__ == "__main__":
    main()
