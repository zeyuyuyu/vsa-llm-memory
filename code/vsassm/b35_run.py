"""B35 trainer / evaluator (prereg/B35_soft_token_vsa_prereg.md). One run = one arm at one budget B and one seed.

Arms (memory given to the LM between "Memory:" and the question; all trained with the same LoRA, optimiser, steps,
batch and training stream; the LM is Qwen3-1.7B):
  vsa     B soft positions = chunks of the instance's MAP bundle (b35_common), through a projector (LN -> MLP or
          linear -> LN x gain + position vector). Question-agnostic.
  lmt     B learned memory tokens without VSA structure: each parsed fact (e, r, v) -> MLP of the three symbols'
          standardised mean LM input embeddings -> Perceiver (B learned latents, cross-attention to the facts,
          self-attention, MLP; `--lmt_layers` blocks) -> LN x gain + position vector. Question-agnostic.
  vsaq    secondary, single-hop only: the bundle unbound OUTSIDE the LM with the parsed question key
          (P^-1(m * C[e] * C[r])), chunks through the same projector; the LM only names the value.
  ttrunc  the facts' text cut to its first B tokens.
  tfull   all facts as text (reference; ~10 tokens per fact).
Output: <out>/summary.json (accuracy per split x N x question type, config, timings) and <out>/<split>.jsonl.gz
(one line per question: N, idx, qtype, question, gold, pred, correct).
"""
import argparse
import gzip
import json
import math
import os
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

import b35_common as Cm
import b35_data as D

p = argparse.ArgumentParser()
p.add_argument('--arm', required=True, choices=['vsa', 'lmt', 'vsaq', 'ttrunc', 'tfull', 'icae'])
p.add_argument('--B', type=int, default=4)
p.add_argument('--seed', type=int, default=0)
p.add_argument('--steps', type=int, default=2000)
p.add_argument('--bs', type=int, default=32)
p.add_argument('--lr', type=float, default=2e-4)
p.add_argument('--lora_r', type=int, default=64)
p.add_argument('--codes', default='rand', choices=['rand', 'emb'])
p.add_argument('--proj', default='mlp', choices=['mlp', 'lin'])
p.add_argument('--lmt_layers', type=int, default=2)
p.add_argument('--eval_splits', default='dev')
p.add_argument('--eval_every', type=int, default=0)
p.add_argument('--eval_sub', type=int, default=100, help='instances per N for intermediate dev evals')
p.add_argument('--tok_budget', type=int, default=24000, help='tokens per micro-batch (text arms)')
p.add_argument('--device', default='cuda')
p.add_argument('--qfirst', type=int, default=0)
p.add_argument('--max_new', type=int, default=6, help='generated tokens (6; the zero-shot reference uses 16)')
p.add_argument('--mem_lr', type=float, default=0, help='learning rate of the memory module (0 = --lr)')
p.add_argument('--out', required=True)
args = p.parse_args()
os.makedirs(args.out, exist_ok=True)
_done = os.path.join(args.out, 'summary.json')
if os.path.exists(_done) and all(k in json.load(open(_done)) for k in args.eval_splits.split(',') if k and k != 'none'):
    print('already done:', args.out)
    raise SystemExit(0)
torch.manual_seed(Cm.seed_of('b35', 'init', args.arm, args.B, args.seed) % (2 ** 63))
dev = args.device
DT = 'cuda' if dev.startswith('cuda') else 'cpu'
T0 = time.time()

tok = AutoTokenizer.from_pretrained(Cm.MODEL)
model = AutoModelForCausalLM.from_pretrained(Cm.MODEL, torch_dtype=torch.bfloat16).to(dev)
model.config.use_cache = False
h = model.config.hidden_size
EMB = model.get_input_embeddings()
EW0 = EMB.weight.detach()
emb_rms = EW0.float().pow(2).mean().sqrt().item()
for q in model.parameters():
    q.requires_grad_(False)
if args.lora_r > 0:
    from peft import LoraConfig, get_peft_model
    cfg = LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.0, bias='none',
                     target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'])
    model = get_peft_model(model, cfg)
    base = model.get_base_model()
else:
    base = model
if args.arm in ('tfull', 'icae'):
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    base.enable_input_require_grads()
inner, lm_head = base.model, base.lm_head
NL = tok('\n', add_special_tokens=False)['input_ids'][0]
PRE = tok('Memory:', add_special_tokens=False)['input_ids']
ENC_PRE = tok('Facts:', add_special_tokens=False)['input_ids']


def ids(s):
    return tok(s, add_special_tokens=False)['input_ids']


# ------------------------------------------------------------------ memory modules
class Proj(nn.Module):
    def __init__(self, B, kind):
        super().__init__()
        self.ln = nn.LayerNorm(h)
        self.f = nn.Linear(h, h) if kind == 'lin' else nn.Sequential(nn.Linear(h, 2 * h), nn.GELU(),
                                                                     nn.Linear(2 * h, h))
        self.out = nn.LayerNorm(h, elementwise_affine=False)
        self.g = nn.Parameter(torch.tensor(emb_rms))
        self.pos = nn.Parameter(torch.zeros(B, h))

    def forward(self, x):  # (bs, B, h)
        return self.g * self.out(self.f(self.ln(x))) + self.pos


class Block(nn.Module):
    def __init__(self, heads=16):
        super().__init__()
        self.lq, self.lk, self.l2, self.l3 = nn.LayerNorm(h), nn.LayerNorm(h), nn.LayerNorm(h), nn.LayerNorm(h)
        self.ca = nn.MultiheadAttention(h, heads, batch_first=True)
        self.sa = nn.MultiheadAttention(h, heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(h, 2 * h), nn.GELU(), nn.Linear(2 * h, h))

    def forward(self, z, f, pad):
        q, kv = self.lq(z), self.lk(f)
        z = z + self.ca(q, kv, kv, key_padding_mask=pad, need_weights=False)[0]
        y = self.l2(z)
        z = z + self.sa(y, y, y, need_weights=False)[0]
        return z + self.mlp(self.l3(z))


class ICAE(nn.Module):
    """ICAE-style memory slots: B learned slot embeddings appended to the facts' text; the same LoRA-tuned LM encodes
    them and its last hidden states at the slots, through LN -> linear -> LN x gain + position vector, are the memory."""
    def __init__(self, B):
        super().__init__()
        self.slots = nn.Parameter(torch.randn(B, h) * emb_rms)
        self.ln = nn.LayerNorm(h)
        self.f = nn.Linear(h, h)
        self.out = nn.LayerNorm(h, elementwise_affine=False)
        self.g = nn.Parameter(torch.tensor(emb_rms))
        self.pos = nn.Parameter(torch.zeros(B, h))

    def forward(self, H):
        return self.g * self.out(self.f(self.ln(H))) + self.pos


class LMT(nn.Module):
    def __init__(self, B, layers):
        super().__init__()
        self.fact = nn.Sequential(nn.Linear(3 * h, h), nn.GELU(), nn.Linear(h, h))
        self.lat = nn.Parameter(torch.randn(B, h) * 0.02)
        self.blocks = nn.ModuleList([Block() for _ in range(layers)])
        self.out = nn.LayerNorm(h, elementwise_affine=False)
        self.g = nn.Parameter(torch.tensor(emb_rms))
        self.pos = nn.Parameter(torch.zeros(B, h))

    def forward(self, F3, pad):  # F3 (bs, n, 3h), pad (bs, n) True = padding
        f = self.fact(F3)
        z = self.lat.unsqueeze(0).expand(F3.shape[0], -1, -1)
        for b in self.blocks:
            z = b(z, f, pad)
        return self.g * self.out(z) + self.pos


ebar = Cm.mean_embeddings(EW0, tok)
mem = None
if args.arm in ('vsa', 'vsaq'):
    C, perm = Cm.codebook(args.codes, h, ebar, args.B)
    C = C.to(dev)
    PC = Cm.permute_chunks(C, perm)
    mem = Proj(args.B, args.proj).to(dev)
elif args.arm == 'lmt':
    Z = ((ebar - ebar.mean(0)) / (ebar.std(0) + 1e-6)).to(dev)
    mem = LMT(args.B, args.lmt_layers).to(dev)
elif args.arm == 'icae':
    mem = ICAE(args.B).to(dev)


def soft_memory(insts, qs):
    """(bs, B, h) bf16 memory embeddings for soft arms."""
    if args.arm == 'vsa':
        M = Cm.bundles(insts, C, PC)
        x = M.view(len(insts), args.B, h)
    elif args.arm == 'vsaq':
        M = Cm.bundles(insts, C, PC)
        e = torch.tensor([Cm.SID[q['chain'][0][0]] for q in qs], device=dev)
        r = torch.tensor([Cm.SID[q['chain'][0][1]] for q in qs], device=dev)
        x = Cm.unbind(M, e, r, C, perm).view(len(insts), args.B, h)
    elif args.arm == 'icae':
        seqs = [EMB(torch.tensor(ENC_PRE + ids(x['text']), device=dev)) for x in insts]
        L = max(t.shape[0] for t in seqs) + args.B
        embs = torch.zeros(len(insts), L, h, device=dev, dtype=torch.bfloat16)
        am = torch.zeros(len(insts), L, device=dev, dtype=torch.long)
        slots = mem.slots.to(torch.bfloat16)
        for k, t in enumerate(seqs):
            n = t.shape[0] + args.B
            embs[k, L - n:L - args.B] = t
            embs[k, L - args.B:] = slots
            am[k, L - n:] = 1
        pos = (am.cumsum(1) - 1).clamp(min=0)
        with torch.autocast(DT, dtype=torch.bfloat16):
            H = forward_hidden(embs, am, pos)[:, -args.B:]
        return mem(H.float()).to(torch.bfloat16)
    else:
        n = max(len(x['facts']) for x in insts)
        F3 = torch.zeros(len(insts), n, 3 * h, device=dev)
        pad = torch.ones(len(insts), n, dtype=torch.bool, device=dev)
        for k, x in enumerate(insts):
            t = torch.tensor([[Cm.SID[e], Cm.SID[r], Cm.SID[v]] for e, r, v in x['facts']], device=dev)
            F3[k, :len(t)] = Z[t].reshape(len(t), 3 * h)
            pad[k, :len(t)] = False
        x = None
        return mem(F3, pad).to(torch.bfloat16)
    return mem(x).to(torch.bfloat16)


def text_memory_ids(inst):
    t = ids(inst['text'])
    return t[:args.B] if args.arm == 'ttrunc' else t


def build(insts, qs, with_answer):
    """List of (embeddings (L, h) bf16, n_label) per item; labels are the last n_label positions' next tokens."""
    if args.arm in ('vsa', 'vsaq', 'lmt', 'icae'):
        S = soft_memory(insts, qs)
    out = []
    for k, (x, q) in enumerate(zip(insts, qs)):
        if args.qfirst:  # question before the memory: the memory positions can attend to the question
            pre_ids, suf = ids('Question: ' + q['text'] + '\nMemory:'), ids('\nAnswer:')
        else:
            pre_ids, suf = PRE, ids('\nQuestion: ' + q['text'] + '\nAnswer:')
        ans = ids(' ' + q['answer']) + [NL] if with_answer else []
        if args.arm in ('vsa', 'vsaq', 'lmt', 'icae'):
            pre = EMB(torch.tensor(pre_ids, device=dev))
            rest = EMB(torch.tensor(suf + ans, device=dev))
            e = torch.cat([pre, S[k], rest], 0)
            seq_ids = None
        else:
            seq = pre_ids + text_memory_ids(x) + suf + ans
            e = None
            seq_ids = seq
        out.append((e, seq_ids, ans))
    return out


def forward_hidden(embs, am, pos):
    return inner(inputs_embeds=embs, attention_mask=am, position_ids=pos, use_cache=False).last_hidden_state


def loss_on(items):
    """Sum of answer-token CE and number of answer tokens for a list of built items (right padding)."""
    L = max(len(i[1]) if i[0] is None else i[0].shape[0] for i in items)
    bs = len(items)
    embs = torch.zeros(bs, L, h, device=dev, dtype=torch.bfloat16)
    am = torch.zeros(bs, L, device=dev, dtype=torch.long)
    tgt_pos, tgt = [], []
    for k, (e, s, ans) in enumerate(items):
        if e is None:
            e = EMB(torch.tensor(s, device=dev))
        n = e.shape[0]
        embs[k, :n] = e
        am[k, :n] = 1
        for j, a in enumerate(ans):
            tgt_pos.append((k, n - len(ans) - 1 + j))
            tgt.append(a)
    pos = (am.cumsum(1) - 1).clamp(min=0)
    with torch.autocast(DT, dtype=torch.bfloat16):
        H = forward_hidden(embs, am, pos)
        kk = torch.tensor([t[0] for t in tgt_pos], device=dev)
        jj = torch.tensor([t[1] for t in tgt_pos], device=dev)
        logits = lm_head(H[kk, jj]).float()
    return F.cross_entropy(logits, torch.tensor(tgt, device=dev), reduction='sum'), len(tgt)


@torch.no_grad()
def generate(items, max_new=None):
    max_new = max_new or args.max_new
    bs = len(items)
    seqs = [(e if e is not None else EMB(torch.tensor(s, device=dev))) for e, s, _ in items]
    L = max(x.shape[0] for x in seqs)
    embs = torch.zeros(bs, L, h, device=dev, dtype=torch.bfloat16)
    am = torch.zeros(bs, L, device=dev, dtype=torch.long)
    for k, x in enumerate(seqs):
        embs[k, L - x.shape[0]:] = x
        am[k, L - x.shape[0]:] = 1
    pos = (am.cumsum(1) - 1).clamp(min=0)
    with torch.autocast(DT, dtype=torch.bfloat16):
        o = base(inputs_embeds=embs, attention_mask=am, position_ids=pos, use_cache=True, logits_to_keep=1)
        past, nxt = o.past_key_values, o.logits[:, -1].argmax(-1)
        toks, cur = [nxt], pos[:, -1]
        for _ in range(max_new - 1):
            am = torch.cat([am, torch.ones(bs, 1, device=dev, dtype=torch.long)], 1)
            cur = cur + 1
            o = base(input_ids=nxt[:, None], attention_mask=am, position_ids=cur[:, None], past_key_values=past,
                     use_cache=True, logits_to_keep=1)
            past, nxt = o.past_key_values, o.logits[:, -1].argmax(-1)
            toks.append(nxt)
    T = torch.stack(toks, 1).tolist()
    outs = []
    for row in T:
        if NL in row:
            row = row[:row.index(NL)]
        outs.append(tok.decode(row, skip_special_tokens=True).strip())
    return outs


def norm(s):
    return s.strip().rstrip('.').strip()


def evaluate(split, sub=None):
    model.eval()
    if mem is not None:
        mem.eval()
    es = D.eval_set(split)
    if sub:
        es = [x for x in es if x['idx'] < sub]
    pairs = [(x, x[q]) for x in es for q in (('q1',) if args.arm == 'vsaq' else ('q1', 'q2'))]
    rows = []
    gbs = 32 if args.arm in ('tfull', 'icae') else 128
    for i in range(0, len(pairs), gbs):
        chunk = pairs[i:i + gbs]
        with torch.no_grad():
            items = build([c[0] for c in chunk], [c[1] for c in chunk], False)
        preds = generate(items)
        for (x, q), pr in zip(chunk, preds):
            rows.append(dict(N=x['N'], idx=x['idx'], qtype=q['qtype'], question=q['text'], gold=q['answer'],
                             pred=pr, correct=int(norm(pr) == q['answer'])))
    acc = {}
    for N in D.NS:
        for qt in ('hop1', 'hop2'):
            r = [z['correct'] for z in rows if z['N'] == N and z['qtype'] == qt]
            if r:
                acc[f'N{N}_{qt}'] = sum(r) / len(r)
    model.train()
    if mem is not None:
        mem.train()
    return acc, rows


# ------------------------------------------------------------------ training
params = [q for q in model.parameters() if q.requires_grad] + (list(mem.parameters()) if mem is not None else [])
n_lora = sum(q.numel() for q in model.parameters() if q.requires_grad)
n_mem = sum(q.numel() for q in mem.parameters()) if mem is not None else 0
MEM_LR = args.mem_lr if args.mem_lr > 0 else args.lr
groups = [dict(params=[q for q in model.parameters() if q.requires_grad], base_lr=args.lr),
          dict(params=list(mem.parameters()) if mem is not None else [], base_lr=MEM_LR)]
groups = [g for g in groups if g['params']]
opt = torch.optim.AdamW(groups, lr=args.lr, betas=(0.9, 0.99), weight_decay=0.0) if groups else None
warm = max(1, min(100, args.steps // 20))


def lr_at(s):
    if s < warm:
        return (s + 1) / warm
    t = (s - warm) / max(1, args.steps - warm)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t))


rng = random.Random(Cm.seed_of('b35', 'stream', args.seed))
log = open(os.path.join(args.out, 'train.log'), 'a')
curve = []
model.train()
t_train = time.time()
for step in range(args.steps):
    for gq in opt.param_groups:
        gq['lr'] = gq['base_lr'] * lr_at(step)
    insts, qs = [], []
    for i in range(args.bs):
        N = rng.choice(D.NS)
        x = D.instance('train', N, rng.randrange(10 ** 12))
        q = x['q1'] if (args.arm == 'vsaq' or rng.random() < 0.5) else x['q2']
        insts.append(x); qs.append(q)
    opt.zero_grad(set_to_none=True)
    tot_tok = 0
    tot_loss = 0.0
    if args.arm in ('ttrunc', 'tfull', 'icae'):
        order = sorted(range(args.bs), key=lambda k: len(insts[k]['facts']))
        groups, cur, cur_max = [], [], 0
        for k in order:
            Lk = 15 * len(insts[k]['facts']) if args.arm in ('tfull', 'icae') else 40
            if cur and max(cur_max, Lk) * (len(cur) + 1) > args.tok_budget:
                groups.append(cur); cur, cur_max = [], 0
            cur.append(k); cur_max = max(cur_max, Lk)
        groups.append(cur)
        built = [None] * args.bs
        n_tok_all = sum(len(ids(' ' + q['answer'])) + 1 for q in qs)
        for g in groups:
            items = build([insts[k] for k in g], [qs[k] for k in g], True)
            ls, nt = loss_on(items)
            (ls / n_tok_all).backward()
            tot_loss += ls.item(); tot_tok += nt
    else:
        items = build(insts, qs, True)
        ls, nt = loss_on(items)
        (ls / nt).backward()
        tot_loss, tot_tok = ls.item(), nt
    if step % 50 == 0 or step == args.steps - 1:
        gm = math.sqrt(sum(float(q.grad.float().pow(2).sum()) for q in (mem.parameters() if mem is not None else [])
                           if q.grad is not None))
        gl = math.sqrt(sum(float(q.grad.float().pow(2).sum()) for q in model.parameters()
                           if q.requires_grad and q.grad is not None))
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step()
    if step % 50 == 0 or step == args.steps - 1:
        print(json.dumps(dict(step=step, loss=tot_loss / tot_tok, lr=args.lr * lr_at(step), gmem=gm, glora=gl,
                              t=round(time.time() - t_train, 1))), file=log, flush=True)
    if args.eval_every and (step + 1) % args.eval_every == 0 and step + 1 < args.steps:
        acc, _ = evaluate('dev', args.eval_sub)
        curve.append(dict(step=step + 1, **acc))
        print(json.dumps(dict(step=step + 1, dev_sub=acc)), file=log, flush=True)
train_s = time.time() - t_train

import socket
summary = dict(host=socket.gethostname(), gpu=torch.cuda.get_device_name() if DT == 'cuda' else 'cpu',
               config=vars(args), n_trainable_lm=n_lora, n_trainable_mem=n_mem, emb_rms=emb_rms, curve=curve,
               train_seconds=train_s)
SPLITS = [s for s in args.eval_splits.split(',') if s and s != 'none']
for sp in SPLITS:
    t1 = time.time()
    acc, rows = evaluate(sp)
    summary[sp] = acc
    summary[f'{sp}_seconds'] = time.time() - t1
    with gzip.open(os.path.join(args.out, f'{sp}.jsonl.gz'), 'wt') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
summary['total_seconds'] = time.time() - T0
json.dump(summary, open(os.path.join(args.out, 'summary.json'), 'w'), indent=1)
print(json.dumps({k: summary[k] for k in summary if k in SPLITS}), file=log, flush=True)
