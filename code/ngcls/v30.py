"""V-30: the mechanism on NATURAL TEXT, and one thing the algebra cannot express.

Why bAbI task 1 is the right first natural-text arena. It is literally role-filler binding in English:
"Mary moved to the bathroom" is the pair (Mary, bathroom), and "Where is Mary?" is an unbinding by Mary. The
symbols are four persons and six locations, so a codebook of ten covers the vocabulary, and the spans are
name and noun mentions rather than bracketed markers -- which is the whole point, since every mechanism result
in this paper so far has located symbols in synthetic markup.

FIRST FINDING, measured before any arm runs: the bundle is order-insensitive and this task is not. In 66.4% of
instances the queried person appears more than once, so a bundle of all statements holds several locations for
that person and cleanup cannot prefer the last. Recency is not expressible in the superposition. This is
reported as a property of the algebra on natural language, and it is quantified rather than asserted:
  parser_plain     binds every statement -- the faithful bundle, expected to fail where recency matters
  parser_recency   binds only the last statement per person -- the same algebra with an ORDERING RULE supplied
                   from outside it
The gap between those two arms is the cost of what the bundle cannot say.

SECOND FINDING, asked only where the algebra can express the task: given the recency rule, does the pattern
replicate on natural text? Exact codebook rows work, a projection trained through the binding loss fails, and a
supervised discrete handover recovers the gap. Arms parser_recency, probe_argmax and spans answer that, with a
6-way linear probe on the pooled frozen state as the floor.
"""
import argparse, json, os, re, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
sys.path.insert(0, "/path/to/work/ngcls")
import v13                                                            # noqa: E402
from v5 import load_babi                                              # noqa: E402

STMT = re.compile(r"(\w+) (?:moved|went|journeyed|travelled|traveled|went back) to the (\w+)\.")
QUERY = re.compile(r"Where is (\w+)\?")
VSA_DIM = 2048


def tok_span(offs, cs, ce):
    lo = hi = None
    for i, (x, y) in enumerate(offs):
        if y <= x:
            continue
        if x < ce and y > cs:
            lo = i if lo is None else lo
            hi = i + 1
    return (lo, hi) if lo is not None else None


def build(split, tok, locs, persons, max_len=512):
    rows = []
    for text, y in split:
        enc = tok(text, truncation=True, max_length=max_len, padding="max_length",
                  return_tensors="pt", return_offsets_mapping=True)
        offs = enc["offset_mapping"].squeeze(0).tolist()
        stmts = []
        ok = True
        for m in STMT.finditer(text):
            p, l = m.group(1), m.group(2)
            if p not in persons or l not in locs:
                ok = False; break
            sp_p, sp_l = tok_span(offs, *m.span(1)), tok_span(offs, *m.span(2))
            if sp_p is None or sp_l is None:
                ok = False; break
            stmts.append((persons[p], len(persons) + locs[l], sp_p, sp_l))
        qm = QUERY.search(text)
        if not ok or not stmts or qm is None or qm.group(1) not in persons:
            continue
        q_sp = tok_span(offs, *qm.span(1))
        if q_sp is None:
            continue
        rows.append({"ids": enc["input_ids"].squeeze(0), "am": enc["attention_mask"].squeeze(0),
                     "stmts": stmts, "q": persons[qm.group(1)], "q_span": q_sp, "label": int(y)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int, default=1)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    ap.add_argument("--n_train", type=int, default=2000)
    ap.add_argument("--n_test", type=int, default=500)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--probe_epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="v30_logs/v30.json")
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
    n_sym = len(persons) + len(locs)
    tr = build(d["train"], tok, locs, persons)
    te = build(d["test"], tok, locs, persons)
    rep = float(np.mean([sum(1 for s in r["stmts"] if s[0] == r["q"]) > 1 for r in te]))
    print("===== V-30  bAbI task %d, natural text =====" % a.task, flush=True)
    print("  %d persons, %d locations, codebook %d; kept %d/%d train and %d/%d eval instances"
          % (len(persons), len(locs), n_sym, len(tr), len(d["train"]), len(te), len(d["test"])), flush=True)
    print("  queried person appears more than once in %.3f of eval instances -- the recency fraction"
          % rep, flush=True)

    Htr = v13.cache(tr, llm, a.device)
    Hte = v13.cache(te, llm, a.device)
    out = {"task": a.task, "n_sym": n_sym, "recency_frac": rep, "n_train": len(tr), "n_eval": len(te),
           "arms": {}}

    def pooled(H, am):
        idx = am.sum(-1).long().clamp(min=1) - 1
        return H[torch.arange(H.size(0)), idx].float()

    # ---------------- floor: 6-way probe on the pooled frozen state ----------------
    amtr = torch.stack([r["am"] for r in tr]); amte = torch.stack([r["am"] for r in te])
    Xtr, Xte = pooled(Htr, amtr), pooled(Hte, amte)
    Ytr = torch.tensor([r["label"] for r in tr]); Yte = torch.tensor([r["label"] for r in te])
    accs = []
    for seed in a.seeds:
        torch.manual_seed(seed)
        pr = nn.Linear(v13.HID, len(locs))
        op = torch.optim.AdamW(pr.parameters(), lr=1e-3)
        for _ in range(a.probe_epochs):
            p = torch.randperm(Xtr.size(0))
            for i in range(0, p.numel(), 64):
                s = p[i:i + 64]
                loss = F.cross_entropy(pr(Xtr[s]), Ytr[s])
                op.zero_grad(); loss.backward(); op.step()
        with torch.no_grad():
            accs.append(float((pr(Xte).argmax(1) == Yte).double().mean()))
    out["arms"]["floor"] = accs
    print("  %-16s %s mean=%.4f" % ("floor", [round(x, 4) for x in accs], float(np.mean(accs))), flush=True)

    # ---------------- the binding arms ----------------
    def sf(H, sp):
        return H[sp[0]:sp[1]].float().mean(0) if sp[1] > sp[0] else H[sp[0]].float()

    def run_bind(mode, recency, seed):
        torch.manual_seed(seed)
        sym = F.normalize(torch.randn(n_sym, VSA_DIM, generator=torch.Generator().manual_seed(seed + 11)),
                          dim=1).to(a.device)
        proj = nn.Linear(v13.HID, VSA_DIM).to(a.device) if mode in ("spans",) else None
        idh = nn.Linear(v13.HID, n_sym).to(a.device) if mode == "probe_argmax" else None
        params = [p for m_ in (proj, idh) if m_ is not None for p in m_.parameters()]
        opt = torch.optim.AdamW(params, lr=a.lr) if params else None

        def vec(H, sid, sp):
            if mode == "parser":
                return sym[sid]
            v = sf(H, sp)
            if mode == "spans":
                return F.normalize(proj(v), dim=-1)
            return sym[int(idh(v).argmax().item())]

        def predict(H, r):
            st = r["stmts"]
            if recency:
                last = {}
                for k, s in enumerate(st):
                    last[s[0]] = k
                st = [st[k] for k in sorted(last.values())]
            mem = torch.stack([v13.bind(vec(H, s[1], s[3]), vec(H, s[0], s[2])) for s in st], 0).sum(0)
            hat = F.normalize(v13.unbind(mem, vec(H, r["q"], r["q_span"])), dim=-1)
            L = sym[len(persons):]
            return int(F.cosine_similarity(L, hat.unsqueeze(0)).argmax().item())

        if opt is not None:                      # train the identity head / projection on the task
            for _ in range(a.epochs):
                perm = torch.randperm(len(tr)).tolist()
                for k in range(0, len(perm), a.batch_size):
                    loss = torch.zeros((), device=a.device)
                    for i in perm[k:k + a.batch_size]:
                        r = tr[i]; H = Htr[i].to(a.device)
                        if mode == "probe_argmax":
                            for s in r["stmts"]:
                                for sid, sp in ((s[0], s[2]), (s[1], s[3])):
                                    loss = loss + F.cross_entropy(idh(sf(H, sp)).unsqueeze(0),
                                                                  torch.tensor([sid], device=a.device))
                        else:
                            for s in r["stmts"]:
                                for sid, sp in ((s[0], s[2]), (s[1], s[3])):
                                    v = F.normalize(proj(sf(H, sp)), dim=-1)
                                    loss = loss + (1.0 - torch.dot(v, sym[sid]))
                    opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            ok = [predict(Hte[i].to(a.device), te[i]) == te[i]["label"] for i in range(len(te))]
        return float(np.mean(ok))

    for name, mode, rec in [("parser_plain", "parser", False), ("parser_recency", "parser", True),
                            ("probe_argmax", "probe_argmax", True), ("spans", "spans", True)]:
        accs = [run_bind(mode, rec, s) for s in a.seeds]
        out["arms"][name] = accs
        print("  %-16s %s mean=%.4f" % (name, [round(x, 4) for x in accs], float(np.mean(accs))), flush=True)
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(out, open(a.out, "w"), indent=1)

    m = {k: float(np.mean(v)) for k, v in out["arms"].items()}
    fl, pa = m["floor"], m["parser_recency"]
    Hd = pa - fl
    out["floor"], out["parser"], out["headroom"] = fl, pa, Hd
    print("\n===== V-30 SCORING (bars registered before this run) =====", flush=True)
    print("  floor=%.4f parser_recency=%.4f headroom=%.4f  |  parser_plain=%.4f (recency cost %.4f)"
          % (fl, pa, Hd, m["parser_plain"], pa - m["parser_plain"]), flush=True)
    b = {"(a) parser_recency - floor >= 0.10": Hd >= 0.10,
         "(b) plain bundle loses >= 0.15 to recency": (pa - m["parser_plain"]) >= 0.15,
         "(c) spans <= floor + 0.10H": all(x <= fl + 0.10 * Hd for x in out["arms"]["spans"]),
         "(d) probe_argmax >= parser - 0.10H": all(x >= pa - 0.10 * Hd for x in out["arms"]["probe_argmax"])}
    for k, v in b.items():
        print("  %-42s %s" % (k, v), flush=True)
    if b["(a) parser_recency - floor >= 0.10"] and b["(c) spans <= floor + 0.10H"] and \
            b["(d) probe_argmax >= parser - 0.10H"]:
        v = "MECHANISM REPLICATES ON NATURAL TEXT"
    elif not b["(d) probe_argmax >= parser - 0.10H"]:
        v = "MECHANISM FAILS ON NATURAL TEXT (the discrete handover does not transfer off synthetic markup)"
    else:
        v = "INTERMEDIATE (report which bars held; license neither reading)"
    out["verdict"] = v
    json.dump(out, open(a.out, "w"), indent=1)
    print("  >>> V-30 VERDICT: %s" % v, flush=True)
    print("V30-DONE", flush=True)


if __name__ == "__main__":
    main()
