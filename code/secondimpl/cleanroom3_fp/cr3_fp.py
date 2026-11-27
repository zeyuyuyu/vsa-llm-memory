# Clean-room re-implementation of the forward-pass comparison (task N, arithmetic):
# vector-symbolic (HRR) module vs a discrete module of the same shape, inside frozen
# Llama-3.1-8B-Instruct. Written from the manuscript text + Dhanraj & Eliasmith (2025)
# paper/code description only. No pipeline code was read.
import os, sys, json, math, time, argparse, random
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

OPS_TRAIN = ["modulo", "multiplication", "gcd", "lcm", "square_mod",
             "bitwise_and", "bitwise_xor", "bitwise_or"]
TEMPL = {
    "modulo": "What is {x} mod {y}?",
    "multiplication": "What is {x} times {y} mod 1000?",
    "gcd": "What is the GCD of {x} and {y}?",
    "lcm": "What is the LCM of {x} and {y} mod 1000?",
    "square_mod": "What is {x}^2 mod {y}?",
    "bitwise_and": "What is {x} AND {y}?",
    "bitwise_xor": "What is {x} XOR {y}?",
    "bitwise_or": "What is {x} OR {y}?",
}
NONCOMM = {"modulo", "square_mod"}
SYS = ("You are a math solving helper. Don't use any commas in your output, "
       "and always answer problems according to the format of previous answers.")


def solve(op, x, y):
    if op == "modulo":
        return x % y if y != 0 else 0
    if op == "multiplication":
        return (x * y) % 1000
    if op == "gcd":
        return math.gcd(x, y)
    if op == "lcm":
        return (x * y // math.gcd(x, y)) % 1000 if (x and y) else 0
    if op == "square_mod":
        return (x * x) % y if y != 0 else 0
    if op == "bitwise_and":
        return x & y
    if op == "bitwise_xor":
        return x ^ y
    if op == "bitwise_or":
        return x | y
    raise ValueError(op)


# ----------------------------------------------------------------- data
def gen_split(n_per_op, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for op in OPS_TRAIN:
        for _ in range(n_per_op):
            # pairs sorted in descending order, as in the deposited train/dev rows (x >= y in every row)
            x, y = sorted([int(v) for v in rng.integers(1, 1000, size=2)], reverse=True)
            ex = []
            for _ in range(2):
                a, b = sorted([int(v) for v in rng.integers(1, 1000, size=2)], reverse=True)
                ex.append([a, b])
            rows.append({"op": op, "x": x, "y": y, "ex": ex, "answer": solve(op, x, y)})
    perm = rng.permutation(len(rows))
    return [rows[i] for i in perm]


def build_prompt(r):
    op = r["op"]
    s = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n" + SYS + "<|eot_id|>"
    for a, b in r["ex"]:
        s += "<|start_header_id|>user<|end_header_id|>\n\n" + TEMPL[op].format(x=a, y=b) + "<|eot_id|>"
        s += "<|start_header_id|>assistant<|end_header_id|>\n\n" + str(solve(op, a, b)) + "<|eot_id|>"
    s += "<|start_header_id|>user<|end_header_id|>\n\n"
    t = TEMPL[op]
    i_x, i_y = t.index("{x}"), t.index("{y}")
    pre, mid, post = t[:i_x], t[i_x + 3:i_y], t[i_y + 3:]
    s += pre
    xs = (len(s), len(s) + len(str(r["x"])))
    s += str(r["x"]) + mid
    ys = (len(s), len(s) + len(str(r["y"])))
    s += str(r["y"]) + post
    s += "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    return s, xs, ys


def tokenize_rows(tok, rows):
    out = []
    for r in rows:
        s, xs, ys = build_prompt(r)
        enc = tok(s, add_special_tokens=False, return_offsets_mapping=True)
        ids, offs = enc["input_ids"], enc["offset_mapping"]
        def last_tok(span):
            idx = [i for i, (a, b) in enumerate(offs) if a < span[1] and b > span[0]]
            assert idx, (s, span)
            return idx[-1]
        ans_ids = tok(str(r["answer"]), add_special_tokens=False)["input_ids"]
        out.append({"ids": ids, "p1": last_tok(xs), "p2": last_tok(ys), "ans_ids": ans_ids})
    return out


# ----------------------------------------------------------------- HRR algebra
class HRR:
    def __init__(self, d=2048, seed=4, n_places=4, device="cpu"):
        g = torch.Generator().manual_seed(seed)
        self.d = d

        def unitary():
            # random unitary vector: |FFT| = 1 everywhere, real signal
            ph = torch.rand(d // 2 + 1, generator=g) * 2 * math.pi
            ph[0] = 0.0
            if d % 2 == 0:
                ph[-1] = 0.0
            return torch.fft.irfft(torch.polar(torch.ones_like(ph), ph), n=d)
        self.base_digit = unitary()
        self.n1 = unitary(); self.n2 = unitary(); self.ptype = unitary()
        self.ops = torch.stack([unitary() for _ in OPS_TRAIN])
        p = unitary()
        places = []
        for _ in range(n_places):          # ones = p, tens = p*p, hundreds = (p*p)*(p*p), ...
            places.append(p)
            p = self.bind(p, p)
        self.places = torch.stack(places)
        ident = torch.zeros(d); ident[0] = 1.0
        digs = [ident]
        for k in range(1, 10):
            digs.append(self.bind(digs[-1], self.base_digit))
        self.digits = torch.stack(digs)     # 0 = identity, k = base^k
        for k in ["base_digit", "n1", "n2", "ptype", "ops", "places", "digits"]:
            setattr(self, k, getattr(self, k).to(device))
        self.n_places = n_places

    @staticmethod
    def bind(a, b):
        n = a.shape[-1]
        return torch.fft.irfft(torch.fft.rfft(a) * torch.fft.rfft(b), n=n)

    @staticmethod
    def inv(a):  # involution (exact inverse for unitary vectors)
        return torch.cat([a[..., :1], a[..., 1:].flip(-1)], dim=-1)

    def number(self, nums, n_places):
        # nums: LongTensor (B,) -> (B, d)
        out = 0
        for i in range(n_places):
            dig = (nums // (10 ** i)) % 10
            out = out + self.bind(self.places[i].unsqueeze(0), self.digits[dig])
        return out

    def problem(self, a, b, op_idx):
        return (self.bind(self.n1.unsqueeze(0), self.number(a, 3))
                + self.bind(self.n2.unsqueeze(0), self.number(b, 3))
                + self.bind(self.ptype.unsqueeze(0), self.ops[op_idx]))

    def solution(self, sol):
        z = torch.zeros_like(sol)
        return (self.bind(self.n1.unsqueeze(0), self.number(sol, self.n_places))
                + self.bind(self.n2.unsqueeze(0), self.number(z, self.n_places)))

    def read(self, X, thr=0.5, mode="published"):
        """X: (B, d). Returns a (B,), b (B,), op_idx (B,), op_sim (B,), digit sims."""
        q_op = self.bind(X, self.inv(self.ptype).unsqueeze(0))
        op_s = q_op @ self.ops.T
        op_sim, op_idx = op_s.max(-1)
        nums = []
        for role in [self.n1, self.n2]:
            u = self.bind(X, self.inv(role).unsqueeze(0))
            val = torch.zeros(X.shape[0], dtype=torch.long, device=X.device)
            for i in range(3):
                q = self.bind(u, self.inv(self.places[i]).unsqueeze(0))
                s = q @ self.digits.T                  # (B, 10)
                if mode == "published":
                    # DE25: digits whose similarity exceeds the threshold contribute their value;
                    # none above threshold -> digit 0
                    dv = ((s > thr).long() * torch.arange(10, device=X.device)).sum(-1)
                else:
                    dv = s.argmax(-1)
                val = val + dv * (10 ** i)
            nums.append(val)
        return nums[0], nums[1], op_idx, op_sim


def compute(op_idx, a, b):
    return torch.tensor([solve(OPS_TRAIN[o], int(x), int(y)) for o, x, y in
                         zip(op_idx.tolist(), a.tolist(), b.tolist())], dtype=torch.long)


# ----------------------------------------------------------------- model helpers
def load_model(path, device):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return tok, model


class Stop(Exception):
    pass


@torch.no_grad()
def cache_states(model, toks, layer, device, bs=32):
    """Input to decoder layer `layer` (= output of layer-1) at p1, p2 and last prompt token."""
    store = {}
    def hook(mod, args, kwargs):
        hs = args[0] if args else kwargs["hidden_states"]
        store["h"] = hs
        raise Stop()
    hd = model.model.layers[layer].register_forward_pre_hook(hook, with_kwargs=True)
    H1, H2, HL = [], [], []
    try:
        for i in range(0, len(toks), bs):
            chunk = toks[i:i + bs]
            L = max(len(t["ids"]) for t in chunk)
            ids = torch.full((len(chunk), L), 0, dtype=torch.long)
            am = torch.zeros((len(chunk), L), dtype=torch.long)
            for j, t in enumerate(chunk):
                ids[j, :len(t["ids"])] = torch.tensor(t["ids"]); am[j, :len(t["ids"])] = 1
            try:
                model(input_ids=ids.to(device), attention_mask=am.to(device), use_cache=False)
            except Stop:
                pass
            h = store["h"].float()
            ar = torch.arange(len(chunk), device=device)
            H1.append(h[ar, torch.tensor([t["p1"] for t in chunk], device=device)].half().cpu())
            H2.append(h[ar, torch.tensor([t["p2"] for t in chunk], device=device)].half().cpu())
            HL.append(h[ar, torch.tensor([len(t["ids"]) - 1 for t in chunk], device=device)].half().cpu())
    finally:
        hd.remove()
    return torch.cat(H1), torch.cat(H2), torch.cat(HL)


class Mixer:
    """Pre-hook on decoder layer: at the chosen position, h <- (1-a) h + a w (where gate open)."""
    def __init__(self, model, layer, alpha=0.5):
        self.w = None; self.pos = None; self.gate = None; self.alpha = alpha
        self.hd = model.model.layers[layer].register_forward_pre_hook(self.hook, with_kwargs=True)

    def hook(self, mod, args, kwargs):
        if self.w is None:
            return None
        hs = args[0] if args else kwargs["hidden_states"]
        if hs.shape[1] == 1:   # decode steps of generation: no intervention
            return None
        B = hs.shape[0]
        ar = torch.arange(B, device=hs.device)
        old = hs[ar, self.pos]
        new = (1 - self.alpha) * old + self.alpha * self.w.to(hs.dtype)
        g = self.gate.to(hs.device).unsqueeze(-1)
        mixed = torch.where(g, new, old)
        hs = hs.clone()
        hs[ar, self.pos] = mixed
        if args:
            return (hs,) + tuple(args[1:]), kwargs
        kwargs["hidden_states"] = hs
        return args, kwargs

    def set(self, w, pos, gate):
        self.w, self.pos, self.gate = w, pos, gate

    def clear(self):
        self.w = None


@torch.no_grad()
def generate(model, tok, toks, mixer, writes, gates, device, bs=32, max_new=6):
    """Greedy decode. writes: (N, 4096) tensor or None. Left padding; intervention at last prompt pos."""
    eot = tok.convert_tokens_to_ids("<|eot_id|>")
    outs = []
    for i in range(0, len(toks), bs):
        chunk = toks[i:i + bs]
        L = max(len(t["ids"]) for t in chunk)
        ids = torch.full((len(chunk), L), eot, dtype=torch.long)
        am = torch.zeros((len(chunk), L), dtype=torch.long)
        for j, t in enumerate(chunk):
            n = len(t["ids"])
            ids[j, L - n:] = torch.tensor(t["ids"]); am[j, L - n:] = 1
        if writes is not None:
            mixer.set(writes[i:i + bs].to(device), torch.full((len(chunk),), L - 1, device=device),
                      gates[i:i + bs])
        else:
            mixer.clear()
        g = model.generate(input_ids=ids.to(device), attention_mask=am.to(device), do_sample=False,
                           max_new_tokens=max_new, eos_token_id=[eot, tok.eos_token_id],
                           pad_token_id=eot, temperature=None, top_p=None)
        for j in range(len(chunk)):
            new = g[j, L:].tolist()
            if eot in new:
                new = new[:new.index(eot)]
            outs.append(tok.decode(new, skip_special_tokens=True).strip())
    mixer.clear()
    return outs


# ----------------------------------------------------------------- modules
class VectorModule(nn.Module):
    def __init__(self, hdim=4096, d=2048):
        super().__init__()
        self.enc = nn.Linear(hdim, d)
        self.dec = nn.Linear(d, hdim)

    def encode(self, h1, h2):
        return self.enc(h1) + self.enc(h2)


class DiscreteModule(nn.Module):
    def __init__(self, hdim=4096, n_ops=8, n_places_out=4):
        super().__init__()
        self.digits = nn.Linear(2 * hdim, 6 * 10)   # 2 operands x 3 places x 10 classes
        self.op = nn.Linear(hdim, n_ops)
        self.emb = nn.Parameter(torch.zeros(n_places_out, 10, hdim))  # learned answer embedding
        self.n_places_out = n_places_out

    def logits(self, h1, h2):
        dl = self.digits(torch.cat([h1, h2], -1)).view(-1, 2, 3, 10)
        ol = self.op(h1 + h2)
        return dl, ol

    def write(self, sol):
        out = 0
        for i in range(self.n_places_out):
            out = out + self.emb[i][(sol // (10 ** i)) % 10]
        return out


def digits_of(n, places=3):
    return torch.stack([(n // (10 ** i)) % 10 for i in range(places)], -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default=os.path.expanduser("~/cr3_fp/out"))
    ap.add_argument("--layer", type=int, default=17)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_train_per_op", type=int, default=2500)
    ap.add_argument("--n_dev_per_op", type=int, default=100)
    ap.add_argument("--n_test_per_op", type=int, default=250)
    ap.add_argument("--stage", default="all")
    ap.add_argument("--ft_bs", type=int, default=16)
    ap.add_argument("--ft_accum", type=int, default=2)
    ap.add_argument("--ft_lr", type=float, default=1e-3)
    ap.add_argument("--kind", default="vec")
    ap.add_argument("--ft_max_steps", type=int, default=10**9)
    ap.add_argument("--eval_n", type=int, default=10**9)
    ap.add_argument("--tag", default="")
    ap.add_argument("--eval_splits", default="dev,test")
    ap.add_argument("--read_pos", default="operands", choices=["operands", "last"])
    ap.add_argument("--data_dir", default=os.path.expanduser("~/cr3_fp/data"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = "cuda"
    log = open(os.path.join(args.out, "log.txt"), "a")
    def P(*a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True); log.write(s + "\n"); log.flush()

    # ---------------- data (own generator; disjoint seeds per split)
    dpath = os.path.join(args.out, "data.json")
    if not os.path.exists(dpath):
        rng = np.random.default_rng(12345)
        data = {}
        for sp in ["train", "dev"]:
            m = json.load(open(os.path.join(args.data_dir, f"N_{sp}_meta.json")))
            rows = []
            for op, x, y in zip(m["op"], m["x"], m["y"]):
                ex = [sorted([int(v) for v in rng.integers(1, 1000, size=2)], reverse=True) for _ in range(2)]
                rows.append({"op": op, "x": int(x), "y": int(y), "ex": ex, "answer": solve(op, int(x), int(y))})
            data[sp] = rows
        # no deposited test rows outside the prediction files -> own test rows, same distribution,
        # disjoint from train and dev
        seen = {(r["op"], r["x"], r["y"]) for sp in data for r in data[sp]}
        test = [r for r in gen_split(args.n_test_per_op + 20, 3000) if (r["op"], r["x"], r["y"]) not in seen]
        keep, cnt = [], {o: 0 for o in OPS_TRAIN}
        for r in test:
            if cnt[r["op"]] < args.n_test_per_op:
                keep.append(r); cnt[r["op"]] += 1
        data["test"] = keep
        json.dump(data, open(dpath, "w"))
    data = json.load(open(dpath))
    P({k: len(v) for k, v in data.items()}, "max answer", max(r["answer"] for r in data["train"]))

    tok, model = load_model(args.model, dev)
    T = {sp: tokenize_rows(tok, data[sp]) for sp in data}
    P("prompt len", np.mean([len(t["ids"]) for t in T["test"]]), "answer tok lens",
      np.bincount([len(t["ans_ids"]) for t in T["train"]]).tolist())

    # ---------------- cache states
    cpath = os.path.join(args.out, f"states_L{args.layer}.pt")
    if not os.path.exists(cpath):
        t0 = time.time()
        C = {sp: cache_states(model, T[sp], args.layer, dev) for sp in T}
        torch.save(C, cpath)
        P("cached states", time.time() - t0, "s")
    C = torch.load(cpath)

    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    hrr = HRR(device=dev)

    def labels(sp):
        rows = data[sp]
        a = torch.tensor([r["x"] for r in rows]); b = torch.tensor([r["y"] for r in rows])
        o = torch.tensor([OPS_TRAIN.index(r["op"]) for r in rows])
        s = torch.tensor([r["answer"] for r in rows])
        return a, b, o, s
    LAB = {sp: labels(sp) for sp in data}
    feats = {sp: (C[sp][0].float().to(dev), C[sp][1].float().to(dev), C[sp][2].float().to(dev)) for sp in C}
    if args.read_pos == "last":   # DE25 default: read the most recent (last prompt) token only
        feats = {sp: (f[2], torch.zeros_like(f[2]), f[2]) for sp, f in feats.items()}

    # ---------------- vector encoder: MSE to the HRR of the problem (DE25 RMSE objective)
    def vec_read(vm, sp, mode, bs=2000):
        h1, h2, _ = feats[sp]
        A, B, O, S = [], [], [], []
        with torch.no_grad():
            for i in range(0, h1.shape[0], bs):
                X = vm.encode(h1[i:i + bs], h2[i:i + bs])
                a, b, o, osim = hrr.read(X, mode=mode)
                A.append(a); B.append(b); O.append(o); S.append(osim)
        a, b, o, osim = [torch.cat(z).cpu() for z in (A, B, O, S)]
        return a, b, o, osim

    def module_correct(a, b, o, sp):
        la, lb, lo, ls = LAB[sp]
        sol = compute(o, a, b)
        return (sol == ls), sol

    def train_vec_encoder(lr, epochs_list, seed):
        torch.manual_seed(seed)
        vm = VectorModule().to(dev)
        opt = torch.optim.Adam(vm.enc.parameters(), lr=lr)
        h1, h2, _ = feats["train"]
        la, lb, lo, _ = LAB["train"]
        tgt = hrr.problem(la.to(dev), lb.to(dev), lo.to(dev))
        n = h1.shape[0]; bs = 512; res = {}
        g = torch.Generator(device="cpu").manual_seed(seed)
        for ep in range(1, max(epochs_list) + 1):
            perm = torch.randperm(n, generator=g).to(dev)
            for i in range(0, n, bs):
                idx = perm[i:i + bs]
                loss = F.mse_loss(vm.encode(h1[idx], h2[idx]), tgt[idx])
                opt.zero_grad(); loss.backward(); opt.step()
            if ep in epochs_list:
                a, b, o, osim = vec_read(vm, "dev", "published")
                c, _ = module_correct(a, b, o, "dev")
                gate = (osim >= 0.8)
                res[ep] = ({k: v.detach().clone() for k, v in vm.enc.state_dict().items()},
                           float((c & gate).float().mean()), float(loss))
                P(f"  vec lr={lr} ep={ep} train_mse={float(loss):.5f} dev module-correct&gate={res[ep][1]:.4f}")
        return res

    def train_disc_heads(lr, epochs_list, seed):
        torch.manual_seed(seed)
        dm = DiscreteModule().to(dev)
        opt = torch.optim.Adam(list(dm.digits.parameters()) + list(dm.op.parameters()), lr=lr)
        h1, h2, _ = feats["train"]
        la, lb, lo, _ = LAB["train"]
        yd = torch.stack([digits_of(la), digits_of(lb)], 1).to(dev)   # (N,2,3)
        yo = lo.to(dev)
        n = h1.shape[0]; bs = 512; res = {}
        g = torch.Generator(device="cpu").manual_seed(seed)
        for ep in range(1, max(epochs_list) + 1):
            perm = torch.randperm(n, generator=g).to(dev)
            for i in range(0, n, bs):
                idx = perm[i:i + bs]
                dl, ol = dm.logits(h1[idx], h2[idx])
                loss = F.cross_entropy(dl.reshape(-1, 10), yd[idx].reshape(-1)) * 6 + F.cross_entropy(ol, yo[idx])
                opt.zero_grad(); loss.backward(); opt.step()
            if ep in epochs_list:
                a, b, o, conf = disc_read(dm, "dev")
                c, _ = module_correct(a, b, o, "dev")
                res[ep] = ({k: v.detach().clone() for k, v in dm.state_dict().items()},
                           float((c & (conf >= 0.8)).float().mean()), float(loss))
                P(f"  disc lr={lr} ep={ep} train_loss={float(loss):.4f} dev module-correct&gate={res[ep][1]:.4f}")
        return res

    def disc_read(dm, sp, bs=2000):
        h1, h2, _ = feats[sp]
        A, B, O, S = [], [], [], []
        with torch.no_grad():
            for i in range(0, h1.shape[0], bs):
                dl, ol = dm.logits(h1[i:i + bs], h2[i:i + bs])
                d = dl.argmax(-1)  # (b,2,3)
                pw = torch.tensor([1, 10, 100], device=dev)
                A.append((d[:, 0] * pw).sum(-1)); B.append((d[:, 1] * pw).sum(-1))
                p = ol.softmax(-1); conf, o = p.max(-1)
                O.append(o); S.append(conf)
        return [torch.cat(z).cpu() for z in (A, B, O, S)]

    sel_path = os.path.join(args.out, f"selected_s{args.seed}.pt")
    if not os.path.exists(sel_path):
        best_v = None
        for lr in [1e-3, 1e-4]:
            r = train_vec_encoder(lr, [30, 100, 300, 1000], args.seed)
            for ep, (sd, acc, l) in r.items():
                if best_v is None or acc > best_v[2] + 1e-9:
                    best_v = (lr, ep, acc, sd)
        P("VEC selected lr/ep/dev", best_v[:3])
        best_d = None
        for lr in [1e-3, 1e-4]:
            r = train_disc_heads(lr, [30, 100, 300], args.seed)
            for ep, (sd, acc, l) in r.items():
                if best_d is None or acc > best_d[2] + 1e-9:
                    best_d = (lr, ep, acc, sd)
        P("DISC selected lr/ep/dev", best_d[:3])
        torch.save({"vec": best_v, "disc": best_d}, sel_path)
    if args.stage == "prep":
        P("prep done"); return
    sel = torch.load(sel_path)
    vm = VectorModule().to(dev); vm.enc.load_state_dict(sel["vec"][3])
    dm = DiscreteModule().to(dev); dm.load_state_dict(sel["disc"][3])

    # ---------------- write-back init: least squares from the answer code to the state at the write position
    hL_tr = feats["train"][2]
    sol_tr = LAB["train"][3].to(dev)
    with torch.no_grad():
        Sv = hrr.solution(sol_tr)                           # (N, 2048)
        Xa = torch.cat([Sv, torch.ones(Sv.shape[0], 1, device=dev)], 1)
        lam = 1e-2
        W = torch.linalg.solve(Xa.T @ Xa + lam * torch.eye(Xa.shape[1], device=dev), Xa.T @ hL_tr)
        vm.dec.weight.copy_(W[:-1].T); vm.dec.bias.copy_(W[-1])
        oh = F.one_hot(digits_of(sol_tr, 4), 10).float().reshape(-1, 40)
        We = torch.linalg.solve(oh.T @ oh + lam * torch.eye(40, device=dev), oh.T @ hL_tr)
        dm.emb.copy_(We.view(4, 10, -1))
        P("init recon rel-err vec", float(((Xa @ W - hL_tr).norm() / hL_tr.norm())),
          "disc", float(((oh @ We - hL_tr).norm() / hL_tr.norm())))

    # ---------------- reads on every split (frozen readers)
    READ = {}
    for sp in ["train", "dev", "test"]:
        a, b, o, osim = vec_read(vm, sp, "published"); c, s = module_correct(a, b, o, sp)
        READ[("vec_pub", sp)] = dict(a=a, b=b, o=o, gate=(osim >= 0.8), sol=s, correct=c)
        a, b, o, osim = vec_read(vm, sp, "best"); c, s = module_correct(a, b, o, sp)
        READ[("vec_best", sp)] = dict(a=a, b=b, o=o, gate=(osim >= 0.8), sol=s, correct=c)
        a, b, o, conf = disc_read(dm, sp); c, s = module_correct(a, b, o, sp)
        READ[("disc", sp)] = dict(a=a, b=b, o=o, gate=(conf >= 0.8), sol=s, correct=c)
    for k, v in READ.items():
        la, lb, lo, _ = LAB[k[1]]
        P("READ", k, "operands+op right", float(((v["a"] == la) & (v["b"] == lb) & (v["o"] == lo)).float().mean()),
          "answer right", float(v["correct"].float().mean()), "gate open", float(v["gate"].float().mean()))

    mixer = Mixer(model, args.layer, 0.5)
    eot = tok.convert_tokens_to_ids("<|eot_id|>")

    # ---------------- CE fine-tuning of the write-back (decoder / answer embedding), same budget
    def finetune(kind):
        params = list(vm.dec.parameters()) if kind == "vec" else [dm.emb]
        opt = torch.optim.Adam(params, lr=args.ft_lr)
        sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[100, 500], gamma=0.5)
        rd = READ[("vec_pub" if kind == "vec" else "disc", "train")]
        n = len(T["train"]); g = torch.Generator().manual_seed(args.seed + 7)
        perm = torch.randperm(n, generator=g).tolist()
        step = 0; t0 = time.time(); run = []
        mb = args.ft_bs; acc = args.ft_accum
        for i0 in range(0, n - mb * acc + 1, mb * acc):
            opt.zero_grad()
            for k in range(acc):
                idx = perm[i0 + k * mb: i0 + (k + 1) * mb]
                seqs = [T["train"][j]["ids"] + T["train"][j]["ans_ids"] + [eot] for j in idx]
                L = max(len(s) for s in seqs)
                ids = torch.full((len(idx), L), eot, dtype=torch.long); am = torch.zeros_like(ids)
                lab = torch.full((len(idx), L), -100, dtype=torch.long)
                pos = []
                for r, j in enumerate(idx):
                    s = seqs[r]; ids[r, :len(s)] = torch.tensor(s); am[r, :len(s)] = 1
                    pl = len(T["train"][j]["ids"])
                    pos.append(pl - 1)
                    lab[r, pl:len(s)] = torch.tensor(s[pl:])
                sol = rd["sol"][idx].to(dev)
                w = vm.dec(hrr.solution(sol)) if kind == "vec" else dm.write(sol)
                mixer.set(w, torch.tensor(pos, device=dev), rd["gate"][idx].to(dev))
                out = model(input_ids=ids.to(dev), attention_mask=am.to(dev), use_cache=False)
                logits = out.logits[:, :-1].float()
                loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), lab[:, 1:].reshape(-1).to(dev),
                                       ignore_index=-100)
                (loss / acc).backward()
                run.append(float(loss))
            opt.step(); sched.step(); step += 1
            if step >= args.ft_max_steps:
                break
            if step % 50 == 0:
                P(f"  ft {kind} step {step} loss {np.mean(run[-50 * acc:]):.4f} {time.time() - t0:.0f}s")
        mixer.clear()
        return step

    ck = os.path.join(args.out, f"ft_{args.kind}_s{args.seed}{args.tag}.pt")
    if not os.path.exists(ck):
        st = finetune(args.kind)
        P("ft steps", args.kind, st)
        torch.save({"dec": vm.dec.state_dict(), "emb": dm.emb.detach().cpu()}, ck)
    ckd = torch.load(ck)
    if args.kind == "vec":
        vm.dec.load_state_dict(ckd["dec"])
    else:
        dm.emb.data.copy_(ckd["emb"].to(dev))

    # ---------------- evaluation (dev for sanity, then test once)
    def run_eval(sp):
        res = {}
        nE = min(args.eval_n, len(data[sp]))
        gold = [str(r["answer"]) for r in data[sp]][:nE]
        arms = ["vec_pub", "vec_best"] if args.kind == "vec" else ["disc"]
        if args.kind == "vec":
            res["base"] = generate(model, tok, T[sp][:nE], mixer, None, None, dev)
        with torch.no_grad():
            for arm in arms:
                rd = READ[(arm, sp)]
                sol = rd["sol"].to(dev)
                w = vm.dec(hrr.solution(sol)) if arm != "disc" else dm.write(sol)
                res[arm] = generate(model, tok, T[sp][:nE], mixer, w.detach()[:nE], rd["gate"][:nE].to(dev), dev)
        corr = {k: np.array([p == g_ for p, g_ in zip(v, gold)]) for k, v in res.items()}
        return res, corr, nE

    for sp in args.eval_splits.split(","):
        res, corr, nE = run_eval(sp)
        P(sp, {k: float(v.mean()) for k, v in corr.items()})
        with open(os.path.join(args.out, f"preds_{sp}_{args.kind}_s{args.seed}{args.tag}.jsonl"), "w") as f:
            for i, r in enumerate(data[sp][:nE]):
                rec = dict(op=r["op"], x=r["x"], y=r["y"], gold=r["answer"])
                for k in res:
                    rec[k] = res[k][i]
                for arm in [k for k in res if k != "base"]:
                    rd = READ[(arm, sp)]
                    rec[arm + "_modsol"] = int(rd["sol"][i]); rec[arm + "_gate"] = bool(rd["gate"][i])
                f.write(json.dumps(rec) + "\n")
    P("done")


if __name__ == "__main__":
    main()
