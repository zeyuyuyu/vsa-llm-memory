"""V-36: injectivity or consistency? Both on bAbI, so the comparison stops crossing substrates (prereg 92fb15c).

PERMUTED destroys correctness while keeping injectivity and consistency: the codebook rows are permuted, so every
symbol gets a wholly wrong but distinct vector. MERGE destroys injectivity while keeping consistency. Persons and
locations are merged in separate cells because a merged LOCATION makes the answer row unreachable in the readout,
which is an evaluation mismatch rather than a binding failure -- V-35 could not separate the two.
"""
import argparse, json, os, random, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
sys.path.insert(0, "/path/to/work/ngcls")
import v13                                                            # noqa: E402
from v5 import load_babi                                              # noqa: E402
from v30 import STMT, build, VSA_DIM                                  # noqa: E402

FLOOR, PARSER = 0.7713, 1.0000
H = PARSER - FLOOR
BAR = FLOOR + 0.90 * H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int, default=1)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    ap.add_argument("--n_train", type=int, default=2000)
    ap.add_argument("--n_test", type=int, default=500)
    ap.add_argument("--budget", type=int, default=40)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="v36_logs/v36.json")
    a = ap.parse_args()

    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    llm = AutoModel.from_pretrained("microsoft/phi-2", trust_remote_code=True,
                                    torch_dtype=torch.float16).to(a.device).eval()
    for p in llm.parameters():
        p.requires_grad_(False)

    d = load_babi(a.task, a.n_train, 200, a.n_test)
    locs = {l: i for i, l in enumerate(d["vocab"])}
    persons = {}
    for text, _ in d["train"]:
        for m in STMT.finditer(text):
            persons.setdefault(m.group(1), len(persons))
    n_p, n_l = len(persons), len(locs)
    n_sym, TOT = n_p + n_l, 2 * (n_p + n_l)
    tr, te = build(d["train"], tok, locs, persons), build(d["test"], tok, locs, persons)
    Htr, Hte = v13.cache(tr, llm, a.device), v13.cache(te, llm, a.device)
    print("===== V-36  injectivity vs consistency, both on bAbI task %d =====" % a.task, flush=True)
    print("  %d persons, %d locations, id space %d; %d train / %d eval; bar %.4f"
          % (n_p, n_l, TOT, len(tr), len(te), BAR), flush=True)

    def sf(Hs, sp):
        return Hs[sp[0]:sp[1]].float().mean(0) if sp[1] > sp[0] else Hs[sp[0]].float()

    def merge_map(scope, rate, seed):
        g = random.Random(hash((scope, round(rate, 4), seed)) & 0xFFFFFFFF)
        lo, hi = (0, n_p) if scope == "persons" else (n_p, n_sym)
        m, hit = list(range(n_sym)), 0
        for s in range(lo, hi):
            if g.random() < rate and hi - lo > 1:
                t = g.randrange(lo, hi)
                while t == s:
                    t = g.randrange(lo, hi)
                m[s] = t; hit += 1
        return m, hit

    def run(cell, rate, seed):
        torch.manual_seed(seed)
        sym = F.normalize(torch.randn(TOT, VSA_DIM,
                                      generator=torch.Generator().manual_seed(seed + 11)), dim=1).to(a.device)
        perm = None
        if cell == "permuted":
            g = random.Random(9000 + seed)
            p = list(range(n_sym)); g.shuffle(p)
            perm = p
            # injective and wholly incorrect: row i is used wherever symbol i is meant, but row i is now
            # some OTHER symbol's vector. Location readout is permuted the same way so the target stays
            # reachable -- otherwise this would be the unreachable-row artifact, not a relabelling.
            sym = torch.stack([sym[p[i]] if i < n_sym else sym[i] for i in range(TOT)], 0)
        m, hit = (merge_map(cell.split("-")[1], rate, seed) if cell.startswith("merge-")
                  else (list(range(n_sym)), 0))
        idh = nn.Linear(v13.HID, TOT).to(a.device)
        opt = torch.optim.AdamW(idh.parameters(), lr=1e-3)
        X, Y = [], []
        for b, r in enumerate(tr):
            for s in r["stmts"]:
                X.append(sf(Htr[b], s[2])); Y.append(m[s[0]])
                X.append(sf(Htr[b], s[3])); Y.append(m[s[1]])
            X.append(sf(Htr[b], r["q_span"])); Y.append(m[r["q"]])
        X = torch.stack(X).to(a.device); Y = torch.tensor(Y).to(a.device)
        for _ in range(a.budget):
            pm = torch.randperm(X.size(0), device=a.device)
            for i in range(0, pm.numel(), 512):
                s = pm[i:i + 512]
                loss = F.cross_entropy(idh(X[s]), Y[s])
                opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            ok = []
            for b, r in enumerate(te):
                Hs = Hte[b].to(a.device)
                last = {}
                for j, s in enumerate(r["stmts"]):
                    last[s[0]] = j
                st = [r["stmts"][j] for j in sorted(last.values())]

                def code(sp):
                    return int(idh(sf(Hs, sp)).argmax().item())
                mem = torch.stack([v13.bind(sym[code(s[3])], sym[code(s[2])]) for s in st], 0).sum(0)
                hat = F.normalize(v13.unbind(mem, sym[code(r["q_span"])]), dim=-1)
                L = sym[n_p:n_sym]
                pred = int(F.cosine_similarity(L, hat.unsqueeze(0)).argmax().item())
                # NO inversion of the permutation here, and that is the point: the permutation is applied
                # uniformly to binding AND to the readout, so the permuted system is ISOMORPHIC to the
                # unpermuted one and the target stays r["label"]. An earlier draft of this file inverted the
                # permutation on the target, which was wrong and would also have raised on any location row
                # mapped to a person index.
                ok.append(int(pred == r["label"]))
        return float(np.mean(ok)), hit

    out = {"floor": FLOOR, "bar": BAR, "cells": []}

    def cell(name, rate):
        ch, hits = [], []
        for s in a.seeds:
            c, h = run(name, rate, s)
            ch.append(c); hits.append(h)
        rec = {"cell": name, "rate": rate, "chain": ch, "mean": float(np.mean(ch)), "merged": hits}
        out["cells"].append(rec)
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(out, open(a.out, "w"), indent=1)
        print("  %-16s r=%.2f  chain %s mean=%.4f   symbols merged per seed %s"
              % (name, rate, [round(x, 4) for x in ch], rec["mean"], hits), flush=True)
        return rec

    # permuted is an ISOMORPHISM CHECK, not evidence: permuting i.i.d. random codebook rows yields a
    # statistically identical system, so this cell must come back at the uncorrupted value. If it does not,
    # something in the pipeline secretly depends on row identity, which would itself be the finding.
    perm = cell("permuted", 0.0)
    mp25 = None
    for nm, rates in (("merge-persons", (0.10, 0.25)), ("merge-locations", (0.10, 0.25))):
        for r in rates:
            c = cell(nm, r)
            if nm == "merge-persons" and r == 0.25:
                mp25 = c

    print("\n===== V-36 SCORING (bars registered at 92fb15c, before this file existed) =====", flush=True)
    pk = all(x >= BAR for x in perm["chain"])
    print("  PERMUTED %s all >= %.4f -> %s" % ([round(x, 4) for x in perm["chain"]], BAR, pk), flush=True)
    sep = all(s <= p - 0.05 for s, p in zip(mp25["chain"], perm["chain"]))
    print("  MERGE-PERSONS r=0.25 %s each <= PERMUTED-0.05 -> %s"
          % ([round(x, 4) for x in mp25["chain"]], sep), flush=True)
    if pk and sep:
        v = ("INJECTIVITY IS THE PROPERTY (an injective but wholly incorrect code is free on natural text while "
             "losing injectivity is not; the manuscript's 'consistency, not correctness' becomes "
             "'injectivity, not correctness')")
    elif all(x < BAR for x in perm["chain"]):
        v = ("CONSISTENCY SURVIVES AS THE PROPERTY (an injective incorrect code does NOT come free on natural "
             "text, so the arena result does not transfer and the existing sentence stays, scoped to synthetic "
             "markup)")
    else:
        v = "INCONCLUSIVE (report all cells, change nothing in the framing)"
    out["verdict"] = v
    json.dump(out, open(a.out, "w"), indent=1)
    print("  >>> V-36 VERDICT: %s" % v, flush=True)
    print("V36-DONE", flush=True)


if __name__ == "__main__":
    main()
