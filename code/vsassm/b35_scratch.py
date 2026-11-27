"""B35 secondary (CPU): the same question with a small transformer reader trained FROM SCRATCH instead of a pretrained LM.

Separates "a transformer cannot learn to read a MAP bundle from a few input vectors" from "a pretrained LM with LoRA
does not learn it in the budget". Same generator, splits, facts and questions as the LM arms (b35_data); N in {8, 32}.
  vsa  B memory vectors = the chunks of the instance's MAP bundle (random +-1 codes, d coordinates per chunk, the
       order-preserving permutation of b35_common), each through a shared linear map + a position vector.
  vsakey  the same bundle chunks plus the parsed first-hop key C[e] * C[r] as B more vectors (the reader still has
       to multiply and clean up). vsaq  the bundle unbound outside the reader with that key (positive control).
  lmt  B learned memory vectors: each fact (e, r, v) -> sum of three learned symbol embeddings -> MLP -> a Perceiver
       (B learned latents, cross-attention to the facts, self-attention, MLP) -> B vectors.
Reader: a 4-layer transformer (width 256, 4 heads) over [memory vectors; question tokens], the answer read out at the
last position as a 192-way classification over the attribute values. Question tokens: (attribute, name) for one hop,
(attribute, manager, name) for two hops.
usage: python b35_scratch.py --arm vsa|lmt --B 4 --seed 0 --steps 6000 --out b35_out/scratch_<arm>_B4_s0.json
"""
import argparse
import json
import math
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import b35_common as Cm
import b35_data as D

ap = argparse.ArgumentParser()
ap.add_argument('--arm', required=True, choices=['vsa', 'vsakey', 'vsaq', 'lmt'])
ap.add_argument('--B', type=int, default=4)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--steps', type=int, default=6000)
ap.add_argument('--bs', type=int, default=128)
ap.add_argument('--lr', type=float, default=1e-3)
ap.add_argument('--d', type=int, default=256, help='VSA coordinates per chunk')
ap.add_argument('--dm', type=int, default=256)
ap.add_argument('--Ns', default='8,32')
ap.add_argument('--splits', default='dev')
ap.add_argument('--eval_every', type=int, default=1000)
ap.add_argument('--threads', type=int, default=8)
ap.add_argument('--out', required=True)
a = ap.parse_args()
torch.set_num_threads(a.threads)
torch.manual_seed(Cm.seed_of('b35', 'scratch', a.arm, a.B, a.seed) % (2 ** 63))
NS = tuple(int(x) for x in a.Ns.split(','))
S = len(Cm.SYMS)
VAL0 = len(D.NAMES) + len(D.RELATIONS)  # SYMS = names, relations, values
dm = a.dm

if a.arm in ('vsa', 'vsakey', 'vsaq'):
    C, perm = Cm.codebook('rand', a.d, None, a.B)
    PC = Cm.permute_chunks(C, perm)


class Perceiver(nn.Module):
    def __init__(self):
        super().__init__()
        self.e = nn.Embedding(S, dm); self.r = nn.Embedding(S, dm); self.v = nn.Embedding(S, dm)
        self.f = nn.Sequential(nn.Linear(dm, dm), nn.GELU(), nn.Linear(dm, dm))
        self.lat = nn.Parameter(torch.randn(a.B, dm) * 0.02)
        self.ca = nn.MultiheadAttention(dm, 4, batch_first=True)
        self.sa = nn.MultiheadAttention(dm, 4, batch_first=True)
        self.ln = nn.ModuleList([nn.LayerNorm(dm) for _ in range(4)])
        self.mlp = nn.Sequential(nn.Linear(dm, 4 * dm), nn.GELU(), nn.Linear(4 * dm, dm))

    def forward(self, E, R, V, pad):
        f = self.f(self.e(E) + self.r(R) + self.v(V))
        z = self.lat.unsqueeze(0).expand(E.shape[0], -1, -1)
        kv = self.ln[0](f)
        z = z + self.ca(self.ln[1](z), kv, kv, key_padding_mask=pad, need_weights=False)[0]
        y = self.ln[2](z)
        z = z + self.sa(y, y, y, need_weights=False)[0]
        return z + self.mlp(self.ln[3](z))


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(S, dm)
        self.nm = 2 * a.B if a.arm == 'vsakey' else a.B
        self.pos = nn.Parameter(torch.randn(self.nm + 3, dm) * 0.02)
        if a.arm in ('vsa', 'vsakey', 'vsaq'):
            self.mem = nn.Sequential(nn.LayerNorm(a.d), nn.Linear(a.d, dm))
        else:
            self.mem = Perceiver()
        layer = nn.TransformerEncoderLayer(dm, 4, 4 * dm, dropout=0.0, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, 4)
        self.head = nn.Sequential(nn.LayerNorm(dm), nn.Linear(dm, len(Cm.VALUES)))

    def forward(self, insts, qs):
        n = len(insts)
        if a.arm in ('vsa', 'vsakey', 'vsaq'):
            M = Cm.bundles(insts, C, PC)
            e1 = torch.tensor([Cm.SID[q['chain'][0][0]] for q in qs])
            r1 = torch.tensor([Cm.SID[q['chain'][0][1]] for q in qs])
            if a.arm == 'vsaq':  # unbound outside the reader with the first-hop key
                M = Cm.unbind(M, e1, r1, C, perm)
            m = self.mem(M.view(n, a.B, a.d))
            if a.arm == 'vsakey':  # the parsed first-hop key (C[e] * C[r]) as B more input vectors
                m = torch.cat([m, self.mem((C[e1] * C[r1]).view(n, a.B, a.d))], 1)
        else:
            L = max(len(x['facts']) for x in insts)
            E = torch.zeros(n, L, dtype=torch.long); R = torch.zeros_like(E); V = torch.zeros_like(E)
            pad = torch.ones(n, L, dtype=torch.bool)
            for k, x in enumerate(insts):
                for j, (e, r, v) in enumerate(x['facts']):
                    E[k, j], R[k, j], V[k, j] = Cm.SID[e], Cm.SID[r], Cm.SID[v]
                pad[k, :len(x['facts'])] = False
            m = self.mem(E, R, V, pad)
        Q = torch.zeros(n, 3, dtype=torch.long)
        qpad = torch.zeros(n, self.nm + 3, dtype=torch.bool)
        for k, q in enumerate(qs):
            if q['qtype'] == 'hop1':
                (x, at), = q['chain']
                t = [Cm.SID[at], Cm.SID[x]]
            else:
                (x, r), (_, at) = q['chain']
                t = [Cm.SID[at], Cm.SID[r], Cm.SID[x]]
            Q[k, 3 - len(t):] = torch.tensor(t)
            qpad[k, self.nm:self.nm + 3 - len(t)] = True
        h = torch.cat([m, self.tok(Q)], 1) + self.pos
        h = self.enc(h, src_key_padding_mask=qpad)
        return self.head(h[:, -1])


def target(qs):
    return torch.tensor([Cm.SID[q['answer']] - VAL0 for q in qs])


model = Model()
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.0)
warm = 200


def lr_at(s):
    return (s + 1) / warm if s < warm else 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, a.steps - warm)))


def evaluate(split):
    model.eval()
    es = [x for x in D.eval_set(split) if x['N'] in NS]
    res = {}
    with torch.no_grad():
        for N in NS:
            for qn in ('q1', 'q2'):
                xs = [x for x in es if x['N'] == N]
                qs = [x[qn] for x in xs]
                pred = torch.cat([model(xs[i:i + 256], qs[i:i + 256]).argmax(-1) for i in range(0, len(xs), 256)])
                res[f'N{N}_{qs[0]["qtype"]}'] = float((pred == target(qs)).float().mean())
    model.train()
    return res


rng = random.Random(Cm.seed_of('b35', 'scratch-stream', a.seed))
curve, t0 = [], time.time()
for step in range(a.steps):
    for g in opt.param_groups:
        g['lr'] = a.lr * lr_at(step)
    insts = [D.instance('train', rng.choice(NS), rng.randrange(10 ** 12)) for _ in range(a.bs)]
    qs = [x['q1'] if rng.random() < 0.5 else x['q2'] for x in insts]
    loss = F.cross_entropy(model(insts, qs), target(qs))
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    if (step + 1) % a.eval_every == 0 and step + 1 < a.steps:
        curve.append(dict(step=step + 1, loss=float(loss), t=round(time.time() - t0), **evaluate('dev')))
        print(json.dumps(curve[-1]), flush=True)
out = dict(config=vars(a), curve=curve, seconds=time.time() - t0)
SPL = [x for x in a.splits.split(',') if x and x != 'none']
for sp in SPL:
    out[sp] = evaluate(sp)
print(json.dumps({k: out[k] for k in SPL}), flush=True)
json.dump(out, open(a.out, 'w'), indent=1)
