"""B35 shared pieces: symbols, token ids, the LM's mean input embedding per symbol, VSA codebooks and bundles.

VSA encoding (MAP, bipolar). Every symbol s (600 names, 7 relations, 192 values) has a code C[s] in {-1,+1}^D,
D = B * h (h = the LM's hidden width), read as B chunks of h coordinates. A fact (e, r, v) is stored as
C[e] * C[r] * P(C[v]) (elementwise products; P = a fixed random permutation inside each chunk, the order-preserving
binding of B28, so that (x, manager, y) and (y, manager, x) do not alias). The memory of an instance is the sum over
its N facts divided by sqrt(N); chunk b becomes soft position b. Unbinding a key (e, r): P^-1(m * C[e] * C[r]).

Codebooks:
  rand  i.i.d. random signs (seeded; the first B*h coordinates of one 8*h-coordinate code, so codes are nested in B)
  emb   chunk b = sign(s_b * z(s)), z(s) = the symbol's mean LM input embedding standardised per coordinate over the
        799 symbols, s_b a seeded random sign vector per chunk: codes the LM could in principle compute from its own
        embedding of the question's words.
"""
import hashlib
import json
import os

import torch

import b35_data as D

MODEL = os.environ.get('B35_MODEL',
                       '/path/to/work/hf/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e')
VALUES = [v for a in D.ATTR_NAMES for v in D.ATTRS[a]]
SYMS = D.NAMES + D.RELATIONS + VALUES
SID = {s: i for i, s in enumerate(SYMS)}
ATTR_CAND = {a: [SID[v] for v in D.ATTRS[a]] for a in D.ATTR_NAMES}
NAME_IDS = [SID[n] for n in D.NAMES]
BMAX = 8


def seed_of(*parts):
    return int(hashlib.sha256('|'.join(map(str, parts)).encode()).hexdigest()[:15], 16)


def sym_token_ids(tok):
    return [tok(' ' + s, add_special_tokens=False)['input_ids'] for s in SYMS]


def mean_embeddings(emb_weight, tok):
    """(S, h) float32: mean input-embedding row of each symbol's tokens (with its leading space)."""
    W = emb_weight.detach().float().cpu()
    return torch.stack([W[ids].mean(0) for ids in sym_token_ids(tok)])


def load_embedding_matrix(model_dir=MODEL):
    from safetensors import safe_open
    idx = os.path.join(model_dir, 'model.safetensors.index.json')
    if os.path.exists(idx):
        wm = json.load(open(idx))['weight_map']
        f = wm['model.embed_tokens.weight']
    else:
        f = 'model.safetensors'
    with safe_open(os.path.join(model_dir, f), 'pt') as g:
        return g.get_tensor('model.embed_tokens.weight')


def codebook(kind, h, ebar=None, B=BMAX):
    """(S, B*h) float32 codes in {-1,+1} and the per-chunk permutations perm (B, h)."""
    S = len(SYMS)
    g = torch.Generator().manual_seed(seed_of('b35', 'codes', kind) % (2 ** 63))
    if kind == 'rand':
        C = torch.randint(0, 2, (S, BMAX * h), generator=g).float() * 2 - 1
    elif kind == 'emb':
        z = (ebar - ebar.mean(0)) / (ebar.std(0) + 1e-6)
        sb = torch.randint(0, 2, (BMAX, h), generator=g).float() * 2 - 1
        C = torch.cat([torch.where(sb[b] * z >= 0, 1.0, -1.0) for b in range(BMAX)], 1)
    else:
        raise ValueError(kind)
    gp = torch.Generator().manual_seed(seed_of('b35', 'perm') % (2 ** 63))
    perm = torch.stack([torch.randperm(h, generator=gp) for _ in range(BMAX)])
    return C[:, :B * h].contiguous(), perm[:B].contiguous()


def permute_chunks(X, perm):
    """X (..., B*h) -> P(X): coordinate perm[b][j] of chunk b goes to position j."""
    B, h = perm.shape
    Xc = X.reshape(*X.shape[:-1], B, h)
    idx = perm.to(X.device).expand(*Xc.shape)
    return torch.gather(Xc, -1, idx).reshape(X.shape)


def unpermute_chunks(X, perm):
    B, h = perm.shape
    inv = torch.argsort(perm, dim=1)
    return permute_chunks(X, inv)


def fact_index(insts):
    """Flatten facts of a list of instances: (inst_id, e, r, v) long tensors."""
    ii, ee, rr, vv = [], [], [], []
    for k, x in enumerate(insts):
        for e, r, v in x['facts']:
            ii.append(k); ee.append(SID[e]); rr.append(SID[r]); vv.append(SID[v])
    return [torch.tensor(t, dtype=torch.long) for t in (ii, ee, rr, vv)]


def bundles(insts, C, PC):
    """(len(insts), D) memory vectors: sum_f C[e] C[r] PC[v] / sqrt(N). C, PC on the target device."""
    dev = C.device
    ii, ee, rr, vv = [t.to(dev) for t in fact_index(insts)]
    terms = C[ee] * C[rr] * PC[vv]
    M = torch.zeros(len(insts), C.shape[1], device=dev, dtype=C.dtype)
    M.index_add_(0, ii, terms)
    n = torch.tensor([len(x['facts']) for x in insts], device=dev, dtype=C.dtype)
    return M / n.sqrt()[:, None]


def unbind(M, e_ids, r_ids, C, perm):
    """P^-1(M * C[e] * C[r]) for rows of M."""
    return unpermute_chunks(M * C[e_ids] * C[r_ids], perm)
