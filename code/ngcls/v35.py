"""V-35: how good must the mention partition be? (prereg cab4405, bars registered before this file existed)

On bAbI the identity labels are persons[word] / locs[word] -- indices assigned by surface form, not values handed
over by a generator -- so the supervision the algebra needs on natural text is a mention partition obtainable from
the text. Untested until now: how good that partition has to be. Two corruption families applied to the TRAINING
partition only, evaluation always against the true label:

  MERGE (under-splitting, within type so it is an aliasing error a real system would make): two symbols share one
    code. The mention-to-code map stays a function of the symbol -- consistent, not injective.
  SPLIT (over-splitting): a mention of one symbol is given a shadow code. The map is no longer a function of the
    symbol -- consistency is broken, and a key bound with one code and unbound with another fails outright.

"Consistency, not correctness" predicts SPLIT hurts more than MERGE at equal rate. If it does not, the paper's
framing is challenged and that is what gets written up.

Capacity is held constant across every condition: the head always has 2*n_sym outputs and the codebook always has
2*n_sym rows, so the shadow ids the SPLIT family needs are not an extra-capacity confound for MERGE or for r=0.
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

FLOOR, PARSER = 0.7713, 1.0000                  # banked reference points, V-30/V-33
H = PARSER - FLOOR
BAR_HI, BAR_LO = FLOOR + 0.90 * H, FLOOR + 0.10 * H


def corrupt(family, rate, n_p, n_l, seed):
    """Return f(symbol_id, mention_index) -> training label, in an id space of size 2*(n_p+n_l)."""
    n_sym = n_p + n_l
    if rate == 0.0:
        return lambda s, k: s
    g = random.Random(hash((family, round(rate, 4), seed)) & 0xFFFFFFFF)
    if family == "merge":
        # within type, so persons alias to persons and locations to locations
        m = list(range(n_sym))
        for s in range(n_sym):
            if g.random() < rate:
                lo, hi = (0, n_p) if s < n_p else (n_p, n_sym)
                if hi - lo > 1:
                    t = g.randrange(lo, hi)
                    while t == s:
                        t = g.randrange(lo, hi)
                    m[s] = t
        return lambda s, k: m[s]
    if family == "split":
        # per-MENTION, so the same symbol receives two different codes across its own mentions
        g2 = random.Random(hash(("split", round(rate, 4), seed)) & 0xFFFFFFFF)
        draws = [g2.random() for _ in range(200000)]
        return lambda s, k: (n_sym + s) if draws[k % len(draws)] < rate else s
    raise ValueError(family)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int, default=1)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    ap.add_argument("--n_train", type=int, default=2000)
    ap.add_argument("--n_test", type=int, default=500)
    ap.add_argument("--budget", type=int, default=40)
    ap.add_argument("--rates", type=float, nargs="+", default=[0.05, 0.10, 0.20, 0.40])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="v35_logs/v35.json")
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
    print("===== V-35  partition quality: MERGE vs SPLIT on bAbI task %d =====" % a.task, flush=True)
    print("  %d persons, %d locations, id space %d (capacity held constant); %d train / %d eval instances"
          % (n_p, n_l, TOT, len(tr), len(te)), flush=True)
    print("  banked references: floor %.4f, uncorrupted chain %.4f, H %.4f -> ROBUST bar %.4f, BRITTLE bar %.4f"
          % (FLOOR, PARSER, H, BAR_HI, BAR_LO), flush=True)

    def sf(Hs, sp):
        return Hs[sp[0]:sp[1]].float().mean(0) if sp[1] > sp[0] else Hs[sp[0]].float()

    def run(family, rate, seed):
        f = corrupt(family, rate, n_p, n_l, seed)
        torch.manual_seed(seed)
        sym = F.normalize(torch.randn(TOT, VSA_DIM,
                                      generator=torch.Generator().manual_seed(seed + 11)), dim=1).to(a.device)
        idh = nn.Linear(v13.HID, TOT).to(a.device)
        opt = torch.optim.AdamW(idh.parameters(), lr=1e-3)
        X, Y, k = [], [], 0
        for b, r in enumerate(tr):
            for s in r["stmts"]:
                X.append(sf(Htr[b], s[2])); Y.append(f(s[0], k)); k += 1
                X.append(sf(Htr[b], s[3])); Y.append(f(s[1], k)); k += 1
            X.append(sf(Htr[b], r["q_span"])); Y.append(f(r["q"], k)); k += 1
        X = torch.stack(X).to(a.device); Y = torch.tensor(Y).to(a.device)
        for _ in range(a.budget):
            p = torch.randperm(X.size(0), device=a.device)
            for i in range(0, p.numel(), 512):
                s = p[i:i + 512]
                loss = F.cross_entropy(idh(X[s]), Y[s])
                opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            ok, cons = [], []
            for b, r in enumerate(te):
                Hs = Hte[b].to(a.device)
                last = {}
                for j, s in enumerate(r["stmts"]):
                    last[s[0]] = j
                st = [r["stmts"][j] for j in sorted(last.values())]

                def code(sp):
                    return int(idh(sf(Hs, sp)).argmax().item())
                cq = code(r["q_span"])
                same = [s for s in st if s[0] == r["q"]]
                if same:
                    # the requirement the algebra actually imposes: the query mention and the statement
                    # mention of the SAME symbol must receive the SAME code, correct or not
                    cons.append(int(code(same[-1][2]) == cq))
                mem = torch.stack([v13.bind(sym[code(s[3])], sym[code(s[2])]) for s in st], 0).sum(0)
                hat = F.normalize(v13.unbind(mem, sym[cq]), dim=-1)
                L = sym[n_p:n_sym]                     # readout over TRUE location rows only
                ok.append(int(F.cosine_similarity(L, hat.unsqueeze(0)).argmax().item()) == r["label"])
        return float(np.mean(ok)), (float(np.mean(cons)) if cons else float("nan"))

    out = {"n_sym": n_sym, "id_space": TOT, "floor": FLOOR, "parser": PARSER,
           "bar_robust": BAR_HI, "bar_brittle": BAR_LO, "cells": []}

    def cell(family, rate):
        ch, cs = [], []
        for s in a.seeds:
            c, k = run(family, rate, s)
            ch.append(c); cs.append(k)
        rec = {"family": family, "rate": rate, "chain": ch, "consistency": cs,
               "chain_mean": float(np.mean(ch)), "cons_mean": float(np.mean(cs))}
        out["cells"].append(rec)
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(out, open(a.out, "w"), indent=1)
        print("  %-6s r=%.2f  chain %s mean=%.4f   code-consistency mean=%.4f"
              % (family, rate, [round(x, 4) for x in ch], rec["chain_mean"], rec["cons_mean"]), flush=True)
        return rec

    base = cell("none", 0.0)
    for fam in ("merge", "split"):
        for r in a.rates:
            cell(fam, r)

    def get(fam, rate):
        return next(c for c in out["cells"] if c["family"] == fam and abs(c["rate"] - rate) < 1e-9)

    print("\n===== V-35 SCORING (bars registered at cab4405, before this file existed) =====", flush=True)
    if not all(x >= BAR_HI for x in base["chain"]):
        out["verdict"] = ("VOID (validity condition failed: uncorrupted chain %s does not reproduce V-33 at the "
                          "%.4f bar in 3/3, so the curve is not readable)" % ([round(x, 4) for x in base["chain"]], BAR_HI))
        json.dump(out, open(a.out, "w"), indent=1)
        print("  >>> V-35 VERDICT: %s" % out["verdict"], flush=True)
        print("V35-DONE", flush=True)
        return
    print("  validity OK: uncorrupted chain %s all >= %.4f" % ([round(x, 4) for x in base["chain"]], BAR_HI),
          flush=True)
    verd = {}
    for fam in ("merge", "split"):
        c = get(fam, 0.10)["chain"]
        verd[fam] = ("ROBUST" if all(x >= BAR_HI for x in c) else
                     "BRITTLE" if all(x <= BAR_LO for x in c) else "GRADED")
        print("  %-6s at r=0.10: %s -> %s" % (fam, [round(x, 4) for x in c], verd[fam]), flush=True)
    ms, ss = get("merge", 0.10)["chain"], get("split", 0.10)["chain"]
    if all(s < m for s, m in zip(ss, ms)):
        asym = "ASYMMETRY CONFIRMED (split below merge in 3/3, as consistency-not-correctness predicts)"
    elif all(s >= m for s, m in zip(ss, ms)):
        asym = ("ASYMMETRY REFUTED (split at or above merge in 3/3, which CONTRADICTS "
                "consistency-not-correctness and is written up as contradicting it)")
    else:
        asym = "ASYMMETRY INCONCLUSIVE (seeds disagree; report the numbers, license neither reading)"
    mono = {fam: all(get(fam, a.rates[i])["chain_mean"] >= get(fam, a.rates[i + 1])["chain_mean"]
                     for i in range(len(a.rates) - 1)) for fam in ("merge", "split")}
    out["verdict_merge"], out["verdict_split"] = verd["merge"], verd["split"]
    out["asymmetry"], out["monotone"] = asym, mono
    json.dump(out, open(a.out, "w"), indent=1)
    print("  monotone in r (expectation, not a bar): %s" % mono, flush=True)
    print("  >>> V-35 ASYMMETRY: %s" % asym, flush=True)
    print("V35-DONE", flush=True)


if __name__ == "__main__":
    main()
