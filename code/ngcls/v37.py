"""V-37: does the discrete handover compose at DEPTH TWO on natural text? (prereg d7e2bde, bars registered first)

bAbI task 2 in English is object -> holder -> location. Two bundles, two unbindings, exactly arena C's shape:
  mem1 = sum bind(code(holder), code(object))     unbound by the queried object  -> holder
  mem2 = sum bind(code(location), code(person))   unbound by that holder         -> location

The ordering rule is supplied from OUTSIDE the algebra, as in V-30, and for the same reason: a superposition
cannot express recency, and task 2 adds a second thing it cannot express -- "Mary dropped the football" revokes
a binding, and a bundle has no retraction. So the selected events are chosen by an oracle rule and what is
measured is the ENCODING question only.
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
from v30 import VSA_DIM, tok_span                                     # noqa: E402

MOVE = re.compile(r"(\w+) (?:moved|went|journeyed|travelled|traveled|went back) to the (\w+)\.")
GET = re.compile(r"(\w+) (?:got|took|picked up) the (\w+)(?: there)?\.")
DROP = re.compile(r"(\w+) (?:dropped|left|put down) the (\w+)(?: there)?\.")
QUERY = re.compile(r"Where is the (\w+)\?")


def events(text):
    """Every statement in order, as (kind, subject, object_or_location, subj_span_chars, obj_span_chars)."""
    ev = []
    for pat, kind in ((MOVE, "move"), (GET, "get"), (DROP, "drop")):
        for m in pat.finditer(text):
            ev.append((m.start(), kind, m.group(1), m.group(2), m.span(1), m.span(2)))
    ev.sort()
    return ev


def build(split, tok, persons, locs, objs, max_len=768):
    """One row per instance. mem1 carries the LAST event per object and mem2 the LAST movement per person, so
    both bundles have distractors -- a single-term bundle would be solved exactly by any injective code and the
    arena would be degenerate. The oracle ordering rule is stated in the prereg and supplied from outside the
    algebra. Instances the bundle CANNOT express (the queried object was dropped and its holder moved
    afterwards, so the object is not where the holder now is) are marked, counted, and excluded from the arm
    comparison rather than silently absorbed into it."""
    rows, inexpressible = [], 0
    for text, y in split:
        qm = QUERY.search(text)
        if qm is None or qm.group(1) not in objs:
            continue
        ev = events(text)
        obj_last, mv_last = {}, {}
        for e in ev:
            if e[1] == "move":
                mv_last[e[2]] = e
            else:
                obj_last[e[3]] = e
        q = qm.group(1)
        if q not in obj_last:
            continue
        if any(o not in objs or e[2] not in persons for o, e in obj_last.items()):
            continue
        if any(p not in persons or e[3] not in locs for p, e in mv_last.items()):
            continue
        holder = obj_last[q][2]
        if holder not in mv_last:
            continue
        enc = tok(text, truncation=True, max_length=max_len, padding="max_length",
                  return_tensors="pt", return_offsets_mapping=True)
        offs = enc["offset_mapping"].squeeze(0).tolist()

        def sp(pair):
            return tok_span(offs, *pair)

        oe, mv, bad = [], [], False
        for o, e in obj_last.items():
            a, b = sp(e[4]), sp(e[5])
            if a is None or b is None:
                bad = True; break
            oe.append((objs[o], persons[e[2]], b, a))
        for pn, e in mv_last.items():
            a, b = sp(e[4]), sp(e[5])
            if a is None or b is None:
                bad = True; break
            mv.append((persons[pn], locs[e[3]], a, b))
        qsp = sp(qm.span(1))
        if bad or qsp is None or not oe or not mv:
            continue
        # what the bundle can say, under the oracle rule, versus the dataset answer
        bundle_loc = locs[mv_last[holder][3]]
        exp = int(bundle_loc == int(y))
        inexpressible += 1 - exp
        rows.append({"ids": enc["input_ids"].squeeze(0), "am": enc["attention_mask"].squeeze(0),
                     "obj_ev": oe, "moves": mv, "q_obj": objs[q], "q_span": qsp,
                     "holder": persons[holder], "label": int(y), "bundle_loc": bundle_loc,
                     "expressible": exp})
    return rows, inexpressible


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    ap.add_argument("--n_train", type=int, default=2000)
    ap.add_argument("--n_test", type=int, default=500)
    ap.add_argument("--budget", type=int, default=40)
    ap.add_argument("--probe_epochs", type=int, default=40)
    ap.add_argument("--span_epochs", type=int, default=3)
    ap.add_argument("--arms", nargs="+", default=["floor", "parser", "probe_argmax", "spans"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="v37_logs/v37.json")
    a = ap.parse_args()

    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    llm = AutoModel.from_pretrained("microsoft/phi-2", trust_remote_code=True,
                                    torch_dtype=torch.float16).to(a.device).eval()
    for p in llm.parameters():
        p.requires_grad_(False)

    d = load_babi(2, a.n_train, 200, a.n_test)
    # LABEL-SPACE FIX (the fault that voided the first V-37 runs): load_babi returns the answer as an index
    # into d["vocab"], so locs MUST be that same enumeration -- v30 does exactly this. Building locs from
    # first-seen movement order instead made "expressible" a filter for rows where my index happened to equal
    # the dataset's, on which the parser is 1.0 BY CONSTRUCTION rather than by measurement.
    locs = {l: i for i, l in enumerate(d["vocab"])}
    persons, objs = {}, {}
    for text, _ in d["train"]:
        for _, k, s_, o, _, _ in events(text):
            persons.setdefault(s_, len(persons))
            if k != "move":
                objs.setdefault(o, len(objs))
    n_p, n_l, n_o = len(persons), len(locs), len(objs)
    P, L, O = 0, n_p, n_p + n_l
    TOT = n_p + n_l + n_o
    tr, itr = build(d["train"], tok, persons, locs, objs)
    te, ite = build(d["test"], tok, persons, locs, objs)
    trE = [r for r in tr if r["expressible"]]
    teE = [r for r in te if r["expressible"]]
    Htr, Hte = v13.cache(trE, llm, a.device), v13.cache(teE, llm, a.device)
    print("===== V-37  depth two on natural text: bAbI task 2 =====", flush=True)
    print("  %d persons, %d locations, %d objects, codebook %d; kept %d/%d train and %d/%d eval"
          % (n_p, n_l, n_o, TOT, len(tr), len(d["train"]), len(te), len(d["test"])), flush=True)
    print("  MEASURED EXPRESSIVENESS CEILING: the oracle bundle answers %.1f%% of eval instances; the other "
          "%.1f%% are dropped-then-moved, which a superposition cannot retract. Arms are compared on the "
          "expressible subset only (%d train / %d eval) and this ceiling is reported, not absorbed."
          % (100.0 * (1 - ite / max(1, len(te))), 100.0 * ite / max(1, len(te)), len(trE), len(teE)), flush=True)
    print("  bundle terms per instance (eval): mem1 %.1f objects, mem2 %.1f persons -- distractors present"
          % (float(np.mean([len(r["obj_ev"]) for r in teE])),
             float(np.mean([len(r["moves"]) for r in teE]))), flush=True)

    def sf(H, sp):
        return H[sp[0]:sp[1]].float().mean(0) if sp[1] > sp[0] else H[sp[0]].float()

    def pooled(H, r):
        m = r["am"].to(H.device).unsqueeze(-1).float()
        return (H.float() * m).sum(0) / m.sum().clamp(min=1)

    out = {"n_sym": TOT, "ceiling_eval": 1 - ite / max(1, len(te)),
           "n_train_expressible": len(trE), "n_eval_expressible": len(teE), "arms": {}}

    def run(arm, seed):
        torch.manual_seed(seed)
        sym = F.normalize(torch.randn(TOT, VSA_DIM,
                                      generator=torch.Generator().manual_seed(seed + 11)), dim=1).to(a.device)
        if arm == "floor":
            pr = nn.Linear(v13.HID, n_l).to(a.device)
            opt = torch.optim.AdamW(pr.parameters(), lr=1e-3)
            X = torch.stack([pooled(Htr[b].to(a.device), r) for b, r in enumerate(trE)])
            Y = torch.tensor([r["label"] for r in trE]).to(a.device)
            for _ in range(a.probe_epochs):
                pm = torch.randperm(X.size(0), device=a.device)
                for i in range(0, pm.numel(), 256):
                    sl = pm[i:i + 256]
                    loss = F.cross_entropy(pr(X[sl]), Y[sl])
                    opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                Xt = torch.stack([pooled(Hte[b].to(a.device), r) for b, r in enumerate(teE)])
                pred = pr(Xt).argmax(1).cpu()
            return float((pred == torch.tensor([r["label"] for r in teE])).double().mean())

        idh = proj = None
        if arm == "probe_argmax":
            idh = nn.Linear(v13.HID, TOT).to(a.device)
            opt = torch.optim.AdamW(idh.parameters(), lr=1e-3)
            X, Y = [], []
            for b, r in enumerate(trE):
                H = Htr[b].to(a.device)
                for oid, hid, osp, hsp in r["obj_ev"]:
                    X.append(sf(H, osp)); Y.append(O + oid)
                    X.append(sf(H, hsp)); Y.append(P + hid)
                for pid, lid, psp, lsp in r["moves"]:
                    X.append(sf(H, psp)); Y.append(P + pid)
                    X.append(sf(H, lsp)); Y.append(L + lid)
                X.append(sf(H, r["q_span"])); Y.append(O + r["q_obj"])
            X = torch.stack(X); Y = torch.tensor(Y).to(a.device)
            for _ in range(a.budget):
                pm = torch.randperm(X.size(0), device=a.device)
                for i in range(0, pm.numel(), 512):
                    sl = pm[i:i + 512]
                    loss = F.cross_entropy(idh(X[sl]), Y[sl])
                    opt.zero_grad(); loss.backward(); opt.step()
        if arm == "spans":
            proj = nn.Linear(v13.HID, VSA_DIM).to(a.device)
            opt = torch.optim.AdamW(proj.parameters(), lr=3e-4)

        def codes(H, r, arm_, sym_):
            """Return the vectors for every span the two hops need, per arm."""
            def one(sp, gid):
                if arm_ == "parser":
                    return sym_[gid]
                if arm_ == "probe_argmax":
                    return sym_[int(idh(sf(H, sp)).argmax().item())]
                return F.normalize(proj(sf(H, sp)), dim=-1)
            oe = [(one(osp, O + oid), one(hsp, P + hid)) for oid, hid, osp, hsp in r["obj_ev"]]
            mv = [(one(psp, P + pid), one(lsp, L + lid)) for pid, lid, psp, lsp in r["moves"]]
            qv = one(r["q_span"], O + r["q_obj"])
            return oe, mv, qv

        def forward(H, r, arm_, sym_):
            oe, mv, qv = codes(H, r, arm_, sym_)
            mem1 = torch.stack([v13.bind(hv, ov) for ov, hv in oe], 0).sum(0)
            h = F.normalize(v13.unbind(mem1, qv), dim=-1)
            hc = torch.stack([hv for _, hv in oe], 0)
            h = hc[int(F.cosine_similarity(hc, h.unsqueeze(0)).argmax().item())]   # cleanup between hops
            mem2 = torch.stack([v13.bind(lv, pv) for pv, lv in mv], 0).sum(0)
            got = F.normalize(v13.unbind(mem2, h), dim=-1)
            cand = torch.stack([lv for _, lv in mv], 0)
            return got, cand, [lid for _, lid, _, _ in r["moves"]]

        if arm == "spans":
            for _ in range(a.span_epochs):
                for b, r in enumerate(trE):
                    H = Htr[b].to(a.device)
                    got, cand, lids = forward(H, r, arm, sym)
                    tgt = lids.index(r["label"]) if r["label"] in lids else None
                    if tgt is None:
                        continue
                    logits = F.cosine_similarity(cand, got.unsqueeze(0)) * 10.0
                    loss = F.cross_entropy(logits.unsqueeze(0), torch.tensor([tgt], device=a.device))
                    opt.zero_grad(); loss.backward(); opt.step()

        with torch.no_grad():
            ok = []
            for b, r in enumerate(teE):
                H = Hte[b].to(a.device)
                got, cand, lids = forward(H, r, arm, sym)
                ok.append(int(lids[int(F.cosine_similarity(cand, got.unsqueeze(0)).argmax().item())]
                              == r["label"]))
        return float(np.mean(ok))

    for arm in a.arms:
        accs = [run(arm, s_) for s_ in a.seeds]
        out["arms"][arm] = accs
        print("  %-12s %s mean=%.4f" % (arm, [round(x, 4) for x in accs], float(np.mean(accs))), flush=True)
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(out, open(a.out, "w"), indent=1)

    m = {k: float(np.mean(v)) for k, v in out["arms"].items()}
    fl, pa = m["floor"], m["parser"]
    Hd = pa - fl
    out.update({"floor": fl, "parser": pa, "headroom": Hd})
    print("\n===== V-37 SCORING (bars registered at d7e2bde, before this file existed) =====", flush=True)
    print("  floor=%.4f parser=%.4f headroom=%.4f  bars: transfers >= %.4f, fails <= %.4f"
          % (fl, pa, Hd, fl + 0.90 * Hd, fl + 0.10 * Hd), flush=True)
    if Hd < 0.20:
        v = "VOID (admission failed: parser - floor = %.4f < 0.20, the task cannot separate arms)" % Hd
    else:
        pq = out["arms"]["probe_argmax"]
        if all(x >= fl + 0.90 * Hd for x in pq):
            v = ("TRANSFERS AT DEPTH TWO (the supervised discrete handover composes on natural text; the central "
                 "claim now covers two natural-text tasks and two hops)")
        elif all(x <= fl + 0.10 * Hd for x in pq):
            v = "FAILS AT DEPTH TWO (the claim narrows to depth one on natural text, against my own abstract)"
        else:
            v = "GRADED (report probe_argmax as measured; license neither reading)"
        sp = out["arms"].get("spans")
        if sp and all(x >= fl + 0.50 * Hd for x in sp):
            v += " | SPANS UNEXPECTEDLY WORKS (the through-the-loss route clears half the headroom here)"
    out["verdict"] = v
    json.dump(out, open(a.out, "w"), indent=1)
    print("  >>> V-37 VERDICT: %s" % v, flush=True)
    print("V37-DONE", flush=True)


if __name__ == "__main__":
    main()
