"""B40 core: model, prompts, batching, the residual-stream injector, the VSA engine and the arms' modules.

Layer convention: 'layer L' = the hidden state ENTERING decoder layer L (HF hidden_states[L]); DE25 reads and writes
there ("just before they are processed by the selected layer"). Every module reads the frozen LLM's state at layer L
(detached) and writes h_last <- 0.5 * write + 0.5 * h_last at layer L, last prompt position, on the first forward
pass only (DE25 intervenes at curr_token == 0), where its gate is open.
"""
import math, os, re, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import b40_data as D

MODEL = os.environ.get("B40_MODEL", "unsloth/Llama-3.1-8B-Instruct")
DEV = "cuda"
HID = 4096
VSA_DIM = 2048
MAX_DIGITS = 5  # DE25 config max_digits


# ------------------------------------------------------------------------------------------------ model
def load(dtype=torch.bfloat16):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=dtype, attn_implementation="sdpa").to(DEV)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    pad = tok.convert_tokens_to_ids("<|finetune_right_pad_id|>")
    tok.pad_id = pad if isinstance(pad, int) and pad >= 0 else tok.eos_token_id
    return tok, model


def chat_text(msgs):
    """Meta's Llama-3 ChatFormat (llama/tokenizer.py in DE25's code): no date lines, contents stripped."""
    s = "<|begin_of_text|>"
    for m in msgs:
        s += f"<|start_header_id|>{m['role']}<|end_header_id|>\n\n{m['content'].strip()}<|eot_id|>"
    return s + "<|start_header_id|>assistant<|end_header_id|>\n\n"


def check_pools(tok):
    bad = []
    for n in range(0, 1000):
        if len(tok.encode(str(n), add_special_tokens=False)) != 1:
            bad.append(str(n))
    for e in D.ENTS:
        if len(tok.encode(" " + e, add_special_tokens=False)) != 1:
            bad.append(" " + e)
    for e in D.ROOMS:
        if len(tok.encode(e, add_special_tokens=False)) != 1:
            bad.append(e)
    return bad


def first_tok(tok, s):
    return tok.encode(s, add_special_tokens=False)[0]


def build(tok, r, cot=False):
    """Token ids of the prompt; for task R also the mention token indices of the target story and question."""
    text = chat_text(D.messages(r, cot))
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    out = {"ids": ids, "gold_tok": first_tok(tok, D.gold(r))}
    if r["task"] == "R" and not cot:
        story, qtext, ments = D.story_text(r)
        s0 = text.rfind("Story: " + story)
        assert s0 >= 0
        s0 += len("Story: ")
        q0 = text.rfind("Question: " + qtext) + len("Question: ")

        def tidx(c, ent):
            for t, (a, b) in enumerate(offs):
                if a <= c < b:
                    assert tok.decode([ids[t]]).strip() == ent, (tok.decode([ids[t]]), ent)
                    return t
            raise AssertionError(c)
        nf = len(r["facts"])
        a_idx, b_idx = [None] * nf, [None] * nf
        for c, ent, fi, role in ments:
            (a_idx if role == "a" else b_idx)[fi] = tidx(s0 + c, ent)
        qc = q0 + qtext.index(r["q"])
        out.update(a_idx=a_idx, b_idx=b_idx, q_idx=tidx(qc, r["q"]),
                   a_ent=[D.EID[f[1]] for f in r["facts"]], b_ent=[D.EID[f[2]] for f in r["facts"]],
                   q_ent=D.EID[r["q"]])
    return out


def collate(tok, items):
    """Left-pad; mention indices are shifted by the pad length."""
    T = max(len(it["ids"]) for it in items)
    B = len(items)
    ids = torch.full((B, T), tok.pad_id, dtype=torch.long)
    am = torch.zeros((B, T), dtype=torch.long)
    ctx = {}
    for i, it in enumerate(items):
        n = len(it["ids"])
        ids[i, T - n:] = torch.tensor(it["ids"])
        am[i, T - n:] = 1
    pos = (am.cumsum(1) - 1).clamp(min=0)
    if "a_idx" in items[0]:
        Fm = max(len(it["a_idx"]) for it in items)
        a = torch.zeros((B, Fm), dtype=torch.long); b = torch.zeros_like(a); fm = torch.zeros((B, Fm), dtype=torch.bool)
        ae = torch.zeros_like(a); be = torch.zeros_like(a)
        q = torch.zeros(B, dtype=torch.long); qe = torch.zeros(B, dtype=torch.long)
        for i, it in enumerate(items):
            sh = T - len(it["ids"]); nf = len(it["a_idx"])
            a[i, :nf] = torch.tensor(it["a_idx"]) + sh; b[i, :nf] = torch.tensor(it["b_idx"]) + sh
            ae[i, :nf] = torch.tensor(it["a_ent"]); be[i, :nf] = torch.tensor(it["b_ent"])
            fm[i, :nf] = True; q[i] = it["q_idx"] + sh; qe[i] = it["q_ent"]
        ctx = {"a": a.to(DEV), "b": b.to(DEV), "fm": fm.to(DEV), "q": q.to(DEV), "ae": ae, "be": be, "qe": qe}
    gold = torch.tensor([it["gold_tok"] for it in items])
    return ids.to(DEV), am.to(DEV), pos.to(DEV), ctx, gold.to(DEV)


# ------------------------------------------------------------------------------------------------ injector
class Injector:
    """Forward pre-hook on decoder layer L. When armed, calls module(h, ctx) on the first (prefill) pass and mixes
    its write-back into the last position: h_last <- where(gate, 0.5*w + 0.5*h_last, h_last)."""

    def __init__(self, model, layer):
        self.layer, self.module, self.ctx, self.armed, self.info = layer, None, None, False, None
        self.read_noise = 0.0
        self.handle = model.model.layers[layer].register_forward_pre_hook(self.hook, with_kwargs=True)

    def arm(self, module, ctx):
        self.module, self.ctx, self.armed = module, ctx, module is not None

    def hook(self, mod, args, kwargs):
        if not self.armed:
            return None
        h = args[0] if args else kwargs["hidden_states"]
        if h.shape[1] == 1:
            return None
        self.armed = False  # first pass only
        hr = h.detach()
        if self.read_noise > 0:  # noise on the READ only (relative norm per position), not on the LLM's own stream
            eps = torch.randn_like(hr, dtype=torch.float32)
            eps = eps / eps.norm(dim=-1, keepdim=True) * hr.float().norm(dim=-1, keepdim=True) * self.read_noise
            hr = (hr.float() + eps).to(h.dtype)
        w, gate, info = self.module(hr, self.ctx)
        self.info = info
        last = h[:, -1, :]
        mixed = torch.where(gate.view(-1, 1), 0.5 * w.to(h.dtype) + 0.5 * last, last)
        h2 = torch.cat([h[:, :-1, :], mixed.unsqueeze(1)], dim=1)
        if args:
            return (h2,) + tuple(args[1:]), kwargs
        kwargs["hidden_states"] = h2
        return args, kwargs


# ------------------------------------------------------------------------------------------------ HRR algebra
def make_unitary(v):
    fv = torch.fft.fft(v.double(), dim=-1)
    fv = fv / fv.abs()
    return torch.fft.ifft(fv, dim=-1).real.float()


def bind(a, b):
    n = a.shape[-1]
    return torch.fft.irfft(torch.fft.rfft(a, dim=-1) * torch.fft.rfft(b, dim=-1), n=n, dim=-1)


def inv(a):
    n = a.shape[-1]
    return a[..., (-torch.arange(n, device=a.device)) % n]


class NumEngine:
    """Replica of DE25's SymbolicEngine (llama/vsa_engine.py): unitary orthogonalised atoms for x, n1, n2, the ten
    problem types and the digit base; place 10^k = base bound with itself 2^k times (as in the code); digit i = x^i;
    a number = sum_k digit_k (*) place_k over MAX_DIGITS places; a problem = n1 (*) num(x) + n2 (*) num(y) + type."""

    def __init__(self, seed=4, dim=VSA_DIM):
        g = torch.Generator().manual_seed(seed)
        n_atoms = 4 + len(D.OPS_ALL)
        v = torch.randn(n_atoms, dim, generator=g)
        v = v / v.norm(dim=1, keepdim=True)
        v = make_unitary(v)
        for j in range(n_atoms):  # Gram-Schmidt as in the code, then unitary again
            q = v[j] / v[j].norm()
            for k in range(j + 1, n_atoms):
                v[k] = v[k] - (q @ v[k]) * q
        v = make_unitary(v)
        self.x, self.n1, self.n2 = v[0], v[1], v[2]
        self.ops = {op: v[3 + i] for i, op in enumerate(D.OPS_ALL)}
        base = v[3 + len(D.OPS_ALL)]
        places, p = [], base
        for k in range(MAX_DIGITS):
            if k > 0:
                p = bind(p, p)
            places.append(p)
        self.places = torch.stack(places)
        nums, cur = [torch.zeros(dim)], None
        nums[0][0] = 1.0
        cur = nums[0]
        for i in range(1, 10):
            cur = bind(cur, self.x)
            nums.append(cur)
        self.digits = torch.stack(nums)
        self.op_mat = torch.stack([self.ops[o] for o in D.OPS_ALL])
        self.dev = "cpu"

    def to(self, dev):
        for k in ("x", "n1", "n2", "places", "digits", "op_mat"):
            setattr(self, k, getattr(self, k).to(dev))
        self.dev = dev
        return self

    def num(self, n):  # n: LongTensor [B]
        dig = torch.stack([(n // 10 ** k) % 10 for k in range(MAX_DIGITS)], 1)  # [B, P]
        return bind(self.digits[dig], self.places.unsqueeze(0).expand(len(n), -1, -1)).sum(1)

    def problem(self, op_idx, x, y):  # op_idx into OPS_ALL
        return bind(self.num(x), self.n1) + bind(self.num(y), self.n2) + self.op_mat[op_idx]

    def decode_num(self, v, role, thr=0.5):
        n = bind(v, inv(role).unsqueeze(0))
        q = bind(n.unsqueeze(1), inv(self.places).unsqueeze(0))  # [B, P, d]
        sims = torch.einsum("bpd,kd->bpk", q, self.digits)  # [B, P, 10]
        mx, am = sims.max(-1)
        dig = torch.where(mx > thr, am, torch.zeros_like(am))
        return (dig * torch.tensor([10 ** k for k in range(MAX_DIGITS)], device=v.device)).sum(1), dig

    def decode_op(self, v):
        s = v @ self.op_mat.T  # dot products, no normalisation (normalize_VSA_before_dot = False)
        mx, am = s.max(-1)
        return am, mx


def safe_answer(op, x, y):
    try:
        if op in ("modulo", "division", "square_mod") and y == 0:
            return 0
        return D.op_answer(op, int(x), int(y))
    except Exception:
        return 0


# ------------------------------------------------------------------------------------------------ task-N modules
class VSAN(nn.Module):
    """DE25's NS-LLM block: linear encoder -> HRR problem vector -> type gate (> 0.8) + digit clean-up -> Python
    function -> HRR answer vector (single_number_generation: the bare number) -> linear decoder."""

    def __init__(self, eng, thr=0.8, digit_thr=0.5):
        super().__init__()
        self.eng, self.thr, self.digit_thr = eng, thr, digit_thr  # digit_thr 0.5 = DE25; exploratory 1: argmax
        self.enc = nn.Linear(HID, VSA_DIM, bias=False)
        self.dec = nn.Linear(VSA_DIM, HID, bias=False)

    def read(self, h_last):
        v = self.enc(h_last.float())
        op_i, op_s = self.eng.decode_op(v)
        x, _ = self.eng.decode_num(v, self.eng.n1, thr=self.digit_thr)
        y, _ = self.eng.decode_num(v, self.eng.n2, thr=self.digit_thr)
        return v, op_i, op_s, x, y

    def forward(self, h, ctx):
        with torch.no_grad():
            v, op_i, op_s, x, y = self.read(h[:, -1, :])
            ans = torch.tensor([safe_answer(D.OPS_ALL[o], a, b) for o, a, b in zip(op_i.tolist(), x.tolist(), y.tolist())],
                               device=h.device)
            code = self.eng.num(ans)
        gate = op_s > self.thr
        info = [{"op": D.OPS_ALL[o], "op_s": round(float(s), 3), "x": int(a), "y": int(b), "ans": int(c), "gate": bool(g)}
                for o, s, a, b, c, g in zip(op_i.tolist(), op_s.tolist(), x.tolist(), y.tolist(), ans.tolist(), gate.tolist())]
        return self.dec(code), gate, info


class DiscN(nn.Module):
    """Same shape, no VSA: linear heads (8 trained types; 5 places x 10 digits for each operand) -> argmax -> the same
    Python function -> sum of learned per-(place, digit) embeddings of the answer."""

    def __init__(self, thr=0.8):
        super().__init__()
        self.thr = thr
        self.head = nn.Linear(HID, len(D.OPS_TRAIN) + 2 * MAX_DIGITS * 10)
        self.emb = nn.Parameter(torch.zeros(MAX_DIGITS, 10, HID))

    def logits(self, h_last):
        z = self.head(h_last.float())
        no = len(D.OPS_TRAIN)
        return z[:, :no], z[:, no:no + 50].view(-1, MAX_DIGITS, 10), z[:, no + 50:].view(-1, MAX_DIGITS, 10)

    def read(self, h_last):
        lo, lx, ly = self.logits(h_last)
        pw = torch.tensor([10 ** k for k in range(MAX_DIGITS)], device=h_last.device)
        pr = lo.softmax(-1)
        return pr.argmax(-1), pr.max(-1).values, (lx.argmax(-1) * pw).sum(1), (ly.argmax(-1) * pw).sum(1)

    def forward(self, h, ctx):
        with torch.no_grad():
            op_i, op_p, x, y = self.read(h[:, -1, :])
            ans = [safe_answer(D.OPS_TRAIN[o], a, b) for o, a, b in zip(op_i.tolist(), x.tolist(), y.tolist())]
            dig = torch.tensor([[(a // 10 ** k) % 10 for k in range(MAX_DIGITS)] for a in ans], device=h.device)
        w = self.emb[torch.arange(MAX_DIGITS, device=h.device).unsqueeze(0), dig].sum(1)
        gate = op_p > self.thr
        info = [{"op": D.OPS_TRAIN[o], "op_p": round(float(s), 3), "x": int(a), "y": int(b), "ans": int(c), "gate": bool(g)}
                for o, s, a, b, c, g in zip(op_i.tolist(), op_p.tolist(), x.tolist(), y.tolist(), ans, gate.tolist())]
        return w, gate, info


class Lora1L(nn.Module):
    """DE25's 'LoRA' baseline: the same encoder/decoder shapes at the same layer, no symbolic step, always on."""

    def __init__(self):
        super().__init__()
        self.enc = nn.Linear(HID, VSA_DIM, bias=False)
        self.dec = nn.Linear(VSA_DIM, HID, bias=False)

    def forward(self, h, ctx):
        last = h[:, -1, :].float()
        return self.dec(self.enc(last)), torch.ones(len(last), dtype=torch.bool, device=h.device), None


# ------------------------------------------------------------------------------------------------ task-R modules
class EntCodes:
    def __init__(self, seed=40, dim=VSA_DIM, n=len(D.ENTS)):
        g = torch.Generator().manual_seed(seed)
        self.K = make_unitary(torch.randn(n, dim, generator=g))
        self.perm = torch.randperm(dim, generator=g)
        self.iperm = torch.argsort(self.perm)
        self.room0 = len(D.PEOPLE) + len(D.CONTAINERS)

    def to(self, dev):
        self.K, self.perm, self.iperm = self.K.to(dev), self.perm.to(dev), self.iperm.to(dev)
        return self

    def P(self, v):
        return v[..., self.perm]

    def Pinv(self, v):
        return v[..., self.iperm]


MAX_HOPS = 8


def gather(h, idx):  # h [B,T,D], idx [B,F] -> [B,F,D]
    return h[torch.arange(h.shape[0], device=h.device).unsqueeze(1), idx]


def read_ok(pa, pb, pq, ctx):
    """Per item: fraction of the story's mentions and the question mention read as the right entity."""
    fm = ctx["fm"].cpu()
    ok = ((pa == ctx["ae"]) & fm).sum(1) + ((pb == ctx["be"]) & fm).sum(1) + (pq == ctx["qe"]).long()
    return [round(float(o) / float(2 * n + 1), 4) for o, n in zip(ok.tolist(), fm.sum(1).tolist())]


class VSAR(nn.Module):
    """In-forward-pass VSA for 'where is X': linear encoder per mention -> codes; memory M = sum_f a_f (*) P(b_f);
    query = code of the question mention; hop: P^-1(q^dagger (*) M) -> clean-up against the entity codebook (hard
    argmax, as DE25 cleans up digits; soft = softmax with temperature 1/beta, eval variant) until a room; the room's
    code -> linear decoder."""

    def __init__(self, codes, soft_beta=0.0):
        super().__init__()
        self.c, self.soft_beta = codes, soft_beta
        self.enc = nn.Linear(HID, VSA_DIM, bias=False)
        self.dec = nn.Linear(VSA_DIM, HID, bias=False)

    def forward(self, h, ctx):
        with torch.no_grad():
            A = self.enc(gather(h, ctx["a"]).float()) * ctx["fm"].unsqueeze(-1)
            Bv = self.enc(gather(h, ctx["b"]).float())
            M = bind(A, self.c.P(Bv)).sum(1)  # [B, d]
            q = self.enc(h[torch.arange(h.shape[0], device=h.device), ctx["q"]].float())
            K, r0 = self.c.K, self.c.room0
            reads = read_ok(((A @ K.T).argmax(-1)).cpu(), ((Bv @ K.T).argmax(-1)).cpu(), ((q @ K.T).argmax(-1)).cpu(), ctx)
            B = h.shape[0]
            done = torch.zeros(B, dtype=torch.bool, device=h.device)
            ans_code = torch.zeros(B, VSA_DIM, device=h.device)
            hops = torch.zeros(B, dtype=torch.long, device=h.device)
            ans_ent = torch.full((B,), -1, dtype=torch.long, device=h.device)
            for t in range(MAX_HOPS):
                r = self.c.Pinv(bind(inv(q), M))
                s = r @ K.T
                k = s.argmax(-1)
                if self.soft_beta > 0:
                    w = torch.softmax(self.soft_beta * s / (r.norm(dim=-1, keepdim=True) + 1e-6), -1)
                    room_mass = w[:, r0:].sum(-1)
                    nxt_soft = w @ K
                    is_room = room_mass > 0.5
                    code_room = (w[:, r0:] / room_mass.unsqueeze(-1).clamp(min=1e-6)) @ K[r0:]
                    new = (~done) & is_room
                    ans_code[new] = code_room[new]
                    ans_ent[new] = k[new]
                    hops[new] = t + 1
                    done = done | new
                    q = torch.where(done.unsqueeze(-1), q, nxt_soft)
                else:
                    new = (~done) & (k >= r0)
                    ans_code[new] = K[k[new]]
                    ans_ent[new] = k[new]
                    hops[new] = t + 1
                    done = done | new
                    q = torch.where(done.unsqueeze(-1), q, K[k])
                if bool(done.all()):
                    break
        info = [{"ans": (D.ENTS[e] if e >= 0 else None), "hops": int(n), "gate": bool(g), "reads_ok": ro}
                for e, n, g, ro in zip(ans_ent.tolist(), hops.tolist(), done.tolist(), reads)]
        return self.dec(ans_code), done, info

    def read_codes(self, hm):  # for read-accuracy checks: [N, D] -> entity ids by clean-up
        return (self.enc(hm.float()) @ self.c.K.T).argmax(-1)


class DiscR(nn.Module):
    """Same shape, no VSA: entity classifier per mention -> table a -> b from the argmax reads; the query is the
    present key with the highest query logit (closed world); follow the table until a room; learned room embedding."""

    def __init__(self):
        super().__init__()
        self.cls = nn.Linear(HID, len(D.ENTS))
        self.emb = nn.Parameter(torch.zeros(len(D.ROOMS), HID))

    def forward(self, h, ctx):
        r0 = len(D.PEOPLE) + len(D.CONTAINERS)
        with torch.no_grad():
            la = self.cls(gather(h, ctx["a"]).float()).argmax(-1).tolist()
            lb = self.cls(gather(h, ctx["b"]).float()).argmax(-1).tolist()
            lq = self.cls(h[torch.arange(h.shape[0], device=h.device), ctx["q"]].float())
            fm = ctx["fm"].tolist()
            reads = read_ok(torch.tensor(la), torch.tensor(lb), lq.argmax(-1).cpu(), ctx)
            rooms, gates, info = [], [], []
            for i in range(h.shape[0]):
                tab = {}
                for f, ok in enumerate(fm[i]):
                    if ok:
                        tab[la[i][f]] = lb[i][f]
                keys = torch.tensor(sorted(tab), device=h.device)
                x = int(keys[lq[i, keys].argmax()]) if len(keys) else -1
                hops, room = 0, -1
                while x >= 0 and hops < MAX_HOPS:
                    if x >= r0:
                        room = x
                        break
                    if x not in tab:
                        break
                    x = tab[x]
                    hops += 1
                rooms.append(max(room - r0, 0)); gates.append(room >= 0)
                info.append({"ans": D.ENTS[room] if room >= 0 else None, "hops": hops, "gate": room >= 0,
                             "reads_ok": reads[i]})
        gate = torch.tensor(gates, device=h.device)
        return self.emb[torch.tensor(rooms, device=h.device)], gate, info

    def read_ids(self, hm):
        return self.cls(hm.float()).argmax(-1)


class E2EVSA(nn.Module):
    """Label-free: encoder codes (unit norm), M = sum a (*) P(b); two hops with SOFT clean-up against a fixed random
    unitary codebook of 64 codes (learnable inverse temperature); decoder. Trained by the answer's cross-entropy only."""

    def __init__(self, seed, n_codes=64, hops=2):
        super().__init__()
        self.hops = hops
        g = torch.Generator().manual_seed(seed + 5000)
        self.register_buffer("K", make_unitary(torch.randn(n_codes, VSA_DIM, generator=g)))
        p = torch.randperm(VSA_DIM, generator=g)
        self.register_buffer("perm", p)
        self.register_buffer("iperm", torch.argsort(p))
        self.enc = nn.Linear(HID, VSA_DIM, bias=False)
        self.dec = nn.Linear(VSA_DIM, HID, bias=False)
        nn.init.normal_(self.dec.weight, std=1e-3)
        self.beta = nn.Parameter(torch.tensor(20.0))

    def code(self, x):
        return F.normalize(self.enc(x.float()), dim=-1)

    def clean(self, r):
        return torch.softmax(self.beta * (r @ self.K.T), -1) @ self.K

    def forward(self, h, ctx, shuffle=False):
        A = self.code(gather(h, ctx["a"])) * ctx["fm"].unsqueeze(-1)
        Bv = self.code(gather(h, ctx["b"]))
        if ctx.get("shuffle"):  # ablation: pairings destroyed (b's permuted within the instance)
            Bv = Bv[:, torch.randperm(Bv.shape[1], device=h.device)]
        M = bind(A, Bv[..., self.perm]).sum(1)
        z = self.code(h[torch.arange(h.shape[0], device=h.device), ctx["q"]])
        for _ in range(self.hops):
            z = self.clean(bind(inv(z), M)[..., self.iperm])
        return self.dec(z), torch.ones(h.shape[0], dtype=torch.bool, device=h.device), None


class E2ETab(nn.Module):
    """Label-free, no VSA: soft one-hot reads over 64 slots, an exact outer-product table R = sum p_a p_b^T, two hops
    y = p_q R, z = y R; decoder from the 64-slot distribution. Trained by the answer's cross-entropy only."""

    def __init__(self, seed, n_slots=64, hops=2):
        super().__init__()
        self.hops = hops
        self.enc = nn.Linear(HID, n_slots)
        self.dec = nn.Linear(n_slots, HID, bias=False)
        nn.init.normal_(self.dec.weight, std=1e-3)

    def forward(self, h, ctx):
        pa = self.enc(gather(h, ctx["a"]).float()).softmax(-1) * ctx["fm"].unsqueeze(-1)
        pb = self.enc(gather(h, ctx["b"]).float()).softmax(-1)
        if ctx.get("shuffle"):
            pb = pb[:, torch.randperm(pb.shape[1], device=h.device)]
        R = torch.einsum("bfi,bfj->bij", pa, pb)
        z = self.enc(h[torch.arange(h.shape[0], device=h.device), ctx["q"]].float()).softmax(-1)
        for _ in range(self.hops):
            z = torch.einsum("bi,bij->bj", z, R)
        return self.dec(z), torch.ones(h.shape[0], dtype=torch.bool, device=h.device), None


class E2EConst(nn.Module):
    """Floor for the label-free arms: one learned vector mixed in the same way (no module information)."""

    def __init__(self, seed):
        super().__init__()
        self.c = nn.Parameter(torch.zeros(HID))

    def forward(self, h, ctx):
        return self.c.unsqueeze(0).expand(h.shape[0], -1), torch.ones(h.shape[0], dtype=torch.bool, device=h.device), None


def n_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


# ------------------------------------------------------------------------------------------------ parsing
def parse(task, text, cot=False):
    t = text.replace("−", "-")
    if task == "N":
        t2 = re.sub(r"(?<=\d),(?=\d{3})", "", t)
        if cot:
            k = t2.rfind("Final Answer")
            if k >= 0:
                m = re.search(r"-?\d+", t2[k:])
                if m:
                    return m.group(0)
            nums = re.findall(r"-?\d+", t2)
            return nums[-1] if nums else ""
        m = re.search(r"-?\d+", t2)
        return m.group(0) if m else ""
    low = t.lower()
    rooms = "|".join(D.ROOMS)
    if cot:
        k = low.rfind("final answer")
        if k >= 0:
            m = re.search(r"\b(" + rooms + r")\b", low[k:])
            if m:
                return m.group(1)
        ms = re.findall(r"\b(" + rooms + r")\b", low)
        return ms[-1] if ms else ""
    m = re.search(r"[a-z]+", low)
    return m.group(0) if m else ""
