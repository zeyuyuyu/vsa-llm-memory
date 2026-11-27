"""B35 capacity check: exact external decoding of the VSA arm's bundles (no language model) against HRR/MAP theory.

For every instance the bundle is built exactly as the vsa arm builds it (b35_common.bundles). Single hop: unbind the
true key (person, attribute) and clean up among the attribute's 32 values. Two hop: unbind (person, manager), clean up
among the 600 names, unbind (that name, attribute), clean up among the 32 values. Theory (Frady, Kleyko & Sommer 2018,
Gaussian approximation, winner-take-all among V candidates): p(V, s) = int phi(t) Phi(t + s)^(V-1) dt with
s = sqrt(D / N), D = B * h coordinates, N bindings in the bundle; two hop ~ p(600, s) * p(32, s).

usage: python b35_ext.py --split dev|test|cap --out results/b35_ext_<split>.json [--codes rand,emb] [--B 1,2,4,8]
  split cap = a capacity sweep N in 8..1024 on fresh instances (split name 'cap', 200 per N), dev/test = the eval sets.
"""
import argparse
import json
import math

import numpy as np
import torch
from scipy.stats import norm as _norm

import b35_common as Cm
import b35_data as D


def p_theory(V, s):
    t = np.linspace(-10, 10, 4001)
    trap = getattr(np, "trapezoid", None) or np.trapz
    return float(trap(_norm.pdf(t) * _norm.cdf(t + s) ** (V - 1), t))


def decode(insts, C, perm, PC):
    M = Cm.bundles(insts, C, PC)
    out = []
    for k, x in enumerate(insts):
        row = {}
        for qn in ('q1', 'q2'):
            q = x[qn]
            m = M[k:k + 1]
            if q['qtype'] == 'hop1':
                (e, a), = q['chain']
            else:
                (x2, r), (_, a) = q['chain']
                u = Cm.unbind(m, torch.tensor([Cm.SID[x2]]), torch.tensor([Cm.SID[r]]), C, perm)[0]
                sc = C[Cm.NAME_IDS] @ u
                e = D.NAMES[int(sc.argmax())]
                row['hop2_mid_ok'] = int(e == q['chain'][1][0])
            u = Cm.unbind(m, torch.tensor([Cm.SID[e]]), torch.tensor([Cm.SID[a]]), C, perm)[0]
            cand = Cm.ATTR_CAND[a]
            pred = Cm.SYMS[cand[int((C[cand] @ u).argmax())]]
            row[q['qtype']] = int(pred == q['answer'])
        out.append(row)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--codes', default='rand,emb')
    ap.add_argument('--B', default='1,2,4,8')
    ap.add_argument('--h', type=int, default=2048)
    a = ap.parse_args()
    ebar = None
    if 'emb' in a.codes:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(Cm.MODEL)
        ebar = Cm.mean_embeddings(Cm.load_embedding_matrix(), tok)
    if a.split == 'cap':
        Ns = (8, 16, 32, 64, 128, 256, 512, 1024)
        sets = {N: [D.instance('cap', N, i) for i in range(200)] for N in Ns}
    else:
        es = D.eval_set(a.split)
        sets = {N: [x for x in es if x['N'] == N] for N in D.NS}
    res = {}
    for kind in a.codes.split(','):
        Cfull, pfull = Cm.codebook(kind, a.h, ebar, Cm.BMAX)
        off = (Cfull @ Cfull.T / Cfull.shape[1])
        off = off[~torch.eye(len(off), dtype=torch.bool)]
        res[f'{kind}_code_cos'] = dict(mean=float(off.mean()), mean_abs=float(off.abs().mean()), max=float(off.max()))
        for B in map(int, a.B.split(',')):
            C, perm = Cfull[:, :B * a.h].contiguous(), pfull[:B]
            PC = Cm.permute_chunks(C, perm)
            for N, xs in sets.items():
                rows = decode(xs, C, perm, PC)
                s = math.sqrt(B * a.h / N)
                r = dict(n=len(rows), hop1=float(np.mean([z['hop1'] for z in rows])),
                         hop2=float(np.mean([z['hop2'] for z in rows])),
                         hop2_mid=float(np.mean([z['hop2_mid_ok'] for z in rows])),
                         theory_hop1=p_theory(32, s), theory_hop2=p_theory(600, s) * p_theory(32, s),
                         facts_per_position=N / B)
                res[f'{kind}_B{B}_N{N}'] = r
                print(kind, B, N, json.dumps(r), flush=True)
    json.dump(res, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
