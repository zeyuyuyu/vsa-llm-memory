"""B28 bAbI arms (prereg/B28_noncommutative_binding_prereg.md).

  --task 4 / 15 : exact codebook rows (parser) and the naming head (probe_argmax) under the five encodings of
                  b28_common (comm = published HRR, perm, roles, rtag, shift), seeds 42/123/456. Rows, codebook
                  (seed + 11), naming-head training and clean-up candidates are those of the verbatim v58.py (task 4)
                  and v52.py (task 15). The naming head is trained once per seed exactly as published (it does not
                  depend on the encoding) and then read under every encoding.
  --task 2      : the memory route of v37b.py (parse + ordering rule + two unbindings) on ALL parsed test questions,
                  mapped onto the 500 test questions (a question the parse cannot read is recorded as unanswered).
                  --patterns published | completed (completed adds "grabbed" to the pick-up verbs and "discarded" to
                  the drop verbs; secondary arm).

Per-question outputs go to <out>/b28_task<T>[_<patterns>].json. --smoke scores the TRAINING split only (first
--smoke_n rows) and never builds the test split.
"""
import argparse, json, os, re, sys, time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import b28_common as C                                                # noqa: E402

DIRS = ["north", "south", "east", "west"]
GET_COMPLETED = re.compile(r"(\w+) (?:got|took|picked up|grabbed) the (\w+)(?: there)?\.")
DROP_COMPLETED = re.compile(r"(\w+) (?:dropped|left|put down|discarded) the (\w+)(?: there)?\.")


def sf(H, sp):
    return H[sp[0]:sp[1]].float().mean(0) if sp[1] > sp[0] else H[sp[0]].float()


def train_head(X, Y, n_out, epochs, device, HID):
    """Exactly the naming-head loop of v58/v52/v37b: AdamW 1e-3, randperm on the device, batches of 512.
    The caller has already called torch.manual_seed(seed); nn.Linear draws from the global RNG here."""
    idh = nn.Linear(HID, n_out).to(device)
    opt = torch.optim.AdamW(idh.parameters(), lr=1e-3)
    for _ in range(epochs):
        p = torch.randperm(X.size(0), device=device)
        for i in range(0, p.numel(), 512):
            s = p[i:i + 512]
            loss = F.cross_entropy(idh(X[s]), Y[s]); opt.zero_grad(); loss.backward(); opt.step()
    return idh


def index_rows(raw, rows, tok, max_len):
    """Map built rows (a filtered, order-preserving subset of raw) to raw indices by token ids."""
    out, j = [], 0
    for i, (text, y) in enumerate(raw):
        if j >= len(rows):
            break
        ids = tok(text, truncation=True, max_length=max_len, padding="max_length",
                  return_tensors="pt")["input_ids"].squeeze(0)
        if torch.equal(ids, rows[j]["ids"]):
            out.append(i); j += 1
    assert j == len(rows), "could not map every built row to the raw split (%d of %d)" % (j, len(rows))
    return out


# ------------------------------------------------------------------ task 4 (v58)

def eval4(r, H, idf, enc, dvec, v13):
    mem = torch.zeros(C.VSA_DIM, device=dvec["north"].device)
    mentioned = []
    for ra, dr, rb, sa, sb in r["stmts"]:
        ia, ib = idf(H, sa, ra), idf(H, sb, rb)
        mem = mem + v13.bind(enc.V(ia), v13.bind(dvec[dr], enc.K(ib)))
        mem = mem + v13.bind(enc.V(ib), v13.bind(dvec[C.INV[dr]], enc.K(ia)))
        mentioned += [(ra, ia), (rb, ib)]
    iq = idf(H, r["q_span"], r["qroom"])
    got = F.normalize(v13.unbind(mem, v13.bind(dvec[r["qdir"]], enc.K(iq))), dim=-1)
    cand = torch.stack([enc.V(i) for _, i in mentioned], 0)
    cs = F.cosine_similarity(cand, got.unsqueeze(0))
    w = int(cs.argmax().item())
    pred = mentioned[w][0]                          # published scoring: gold id of the winning mention
    other = [float(c) for (g, _), c in zip(mentioned, cs.tolist()) if g != pred]
    named_ok = all(g == n for g, n in mentioned) and iq == r["qroom"]
    return {"pred": int(pred), "named_pred": int(mentioned[w][1]), "cos_top": float(cs[w]),
            "cos_other": max(other) if other else None, "named_ok": bool(named_ok)}


# ------------------------------------------------------------------ task 15 (v52)

def eval15(r, H, idf, enc, n_an, v13):
    ge = lambda i: n_an + i
    m1 = torch.stack([v13.bind(enc.V(idf(H, csp, cid)), enc.K(idf(H, esp, ge(eid))))
                      for eid, cid, esp, csp in r["mem"]], 0).sum(0)
    iq = idf(H, r["q_span"], ge(r["q"]))
    h = F.normalize(v13.unbind(m1, enc.K(iq)), dim=-1)
    cc_ids = [idf(H, csp, cid) for _, cid, _, csp in r["mem"]]
    cc = torch.stack([enc.V(i) for i in cc_ids], 0)
    w1 = int(F.cosine_similarity(cc, h.unsqueeze(0)).argmax().item())
    key = enc.K(cc_ids[w1])
    m2 = torch.stack([v13.bind(enc.V(idf(H, psp, pid)), enc.K(idf(H, csp, cid)))
                      for cid, pid, csp, psp in r["cls"]], 0).sum(0)
    got = F.normalize(v13.unbind(m2, key), dim=-1)
    p_ids = [idf(H, psp, pid) for _, pid, _, psp in r["cls"]]
    cand = torch.stack([enc.V(i) for i in p_ids], 0)
    gold = [pid for _, pid, _, _ in r["cls"]]
    cs = F.cosine_similarity(cand, got.unsqueeze(0))
    w = int(cs.argmax().item())
    pred = gold[w]
    other = [float(c) for g, c in zip(gold, cs.tolist()) if g != pred]
    named_ok = (all(idf(H, csp, cid) == cid and idf(H, esp, ge(eid)) == ge(eid) for eid, cid, esp, csp in r["mem"])
                and all(idf(H, csp, cid) == cid and idf(H, psp, pid) == pid for cid, pid, csp, psp in r["cls"])
                and iq == ge(r["q"]))
    hop1_gold = [cid for _, cid, _, _ in r["mem"]][w1]
    return {"pred": int(pred), "hop1": int(hop1_gold), "cos_top": float(cs[w]),
            "cos_other": max(other) if other else None, "named_ok": bool(named_ok)}


# ------------------------------------------------------------------ task 2 (v37b)

def eval2(r, H, idf, enc, P, L, O, v13):
    oe = [(idf(H, osp, O + oid), idf(H, hsp, P + hid)) for oid, hid, osp, hsp in r["obj_ev"]]
    mv = [(idf(H, psp, P + pid), idf(H, lsp, L + lid)) for pid, lid, psp, lsp in r["moves"]]
    iq = idf(H, r["q_span"], O + r["q_obj"])
    mem1 = torch.stack([v13.bind(enc.V(h), enc.K(o)) for o, h in oe], 0).sum(0)
    hh = F.normalize(v13.unbind(mem1, enc.K(iq)), dim=-1)
    hc = torch.stack([enc.V(h) for _, h in oe], 0)
    w1 = int(F.cosine_similarity(hc, hh.unsqueeze(0)).argmax().item())
    key = enc.K(oe[w1][1])
    mem2 = torch.stack([v13.bind(enc.V(l), enc.K(p)) for p, l in mv], 0).sum(0)
    got = F.normalize(v13.unbind(mem2, key), dim=-1)
    cand = torch.stack([enc.V(l) for _, l in mv], 0)
    lids = [lid for _, lid, _, _ in r["moves"]]
    w = int(F.cosine_similarity(cand, got.unsqueeze(0)).argmax().item())
    named_ok = (all(a == O + oid and b == P + hid for (a, b), (oid, hid, _, _) in zip(oe, r["obj_ev"]))
                and all(a == P + pid and b == L + lid for (a, b), (pid, lid, _, _) in zip(mv, r["moves"]))
                and iq == O + r["q_obj"])
    return {"pred": int(lids[w]), "hop1_holder": int([hid for _, hid, _, _ in r["obj_ev"]][w1]),
            "named_ok": bool(named_ok)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int, required=True, choices=[2, 4, 15])
    ap.add_argument("--patterns", default="published", choices=["published", "completed"])
    ap.add_argument("--seeds", type=int, nargs="+", default=C.SEEDS)
    ap.add_argument("--encodings", nargs="+", default=C.ENCODINGS)
    ap.add_argument("--head_epochs", type=int, default=40)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="out")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke_n", type=int, default=100)
    a = ap.parse_args()
    t0 = time.time()
    M = C.load_ngcls()
    v13, v5 = M["v13"], M["v5"]
    tok = C.tokenizer()
    dev = a.device
    out = {"task": a.task, "patterns": a.patterns, "smoke": a.smoke, "seeds": a.seeds, "encodings": a.encodings,
           "split_scored": "train[:%d]" % a.smoke_n if a.smoke else "test", "runs": {}}

    if a.task == 4:
        v58 = M["v58"]
        d = v5.load_babi(4, 2000, 200, 500)
        rooms = {w: i for i, w in enumerate(d["vocab"])}
        tr = v58.build(d["train"], tok, rooms)
        te = tr[:a.smoke_n] if a.smoke else v58.build(d["test"], tok, rooms)
        n_r = len(rooms); TOT = n_r + len(DIRS); n_head = n_r
        out.update({"vocab": d["vocab"], "n_train": len(tr), "n_scored": len(te), "codebook": TOT})
    elif a.task == 15:
        v52 = M["v52"]
        d = v5.load_babi(15, 2000, 200, 500)
        animals = {w: i for i, w in enumerate(d["vocab"])}
        ents = {}
        tr = v52.build(d["train"], tok, animals, ents)
        te = tr[:a.smoke_n] if a.smoke else v52.build(d["test"], tok, animals, ents)
        n_an = len(animals); TOT = n_an + len(ents); n_head = TOT
        out.update({"vocab": d["vocab"], "n_train": len(tr), "n_scored": len(te), "codebook": TOT})
    else:
        v37b = M["v37b"]
        if a.patterns == "completed":
            v37b.GET, v37b.DROP = GET_COMPLETED, DROP_COMPLETED
        d = v5.load_babi(2, 2000, 200, 500)
        locs = {l: i for i, l in enumerate(d["vocab"])}
        persons, objs = {}, {}
        for text, _ in d["train"]:
            for _, k, s_, o, _, _ in v37b.events(text):
                persons.setdefault(s_, len(persons))
                if k != "move":
                    objs.setdefault(o, len(objs))
        n_p, n_l, n_o = len(persons), len(locs), len(objs)
        P, L, O = 0, n_p, n_p + n_l
        TOT = n_p + n_l + n_o; n_head = TOT
        tr_all, _ = v37b.build(d["train"], tok, persons, locs, objs)
        tr = [r for r in tr_all if r["expressible"]]
        if a.smoke:
            te = tr_all[:a.smoke_n]
            idx = list(range(len(te)))
            n_raw = len(te)
        else:
            te, _ = v37b.build(d["test"], tok, persons, locs, objs)
            idx = index_rows(d["test"], te, tok, 768)
            n_raw = len(d["test"])
        out.update({"vocab": d["vocab"], "n_train_parsed": len(tr_all), "n_train": len(tr), "n_scored": len(te),
                    "n_questions": n_raw, "codebook": TOT, "persons": persons, "objects": objs,
                    "row_index": idx,
                    "labels_all": [int(y) for _, y in (d["train"][:a.smoke_n] if a.smoke else d["test"])],
                    "expressible": [int(r["expressible"]) for r in te],
                    "bundle_loc": [int(r["bundle_loc"]) for r in te]})
    print("B28 task %d (%s): %d train rows, %d scored rows, codebook %d, smoke=%s"
          % (a.task, a.patterns, len(tr), len(te), TOT, a.smoke), flush=True)

    # gold structures and alias flags (needed by the independent implementation and the mechanism check)
    gold = []
    for r in te:
        if a.task == 4:
            st = [(ra, dr, rb) for ra, dr, rb, _, _ in r["stmts"]]
            gold.append({"stmts": st, "qdir": r["qdir"], "qroom": r["qroom"], "label": r["label"],
                         "alias": C.alias_task4(st, r["qdir"], r["qroom"], r["label"])})
        elif a.task == 15:
            cls = [(c, p) for c, p, _, _ in r["cls"]]
            mem = [(e, c) for e, c, _, _ in r["mem"]]
            gold.append({"cls": cls, "mem": mem, "q": r["q"], "label": r["label"],
                         "alias": C.alias_task15(cls, mem, r["q"], r["label"])})
        else:
            gold.append({"obj_ev": [(o, h) for o, h, _, _ in r["obj_ev"]],
                         "moves": [(p, l) for p, l, _, _ in r["moves"]], "q_obj": r["q_obj"],
                         "label": r["label"], "alias": 0})
    out["gold"] = gold

    from transformers import AutoModel
    llm = AutoModel.from_pretrained("microsoft/phi-2", trust_remote_code=True,
                                    torch_dtype=torch.float16).to(dev).eval()
    for p in llm.parameters():
        p.requires_grad_(False)
    Htr = v13.cache(tr, llm, dev)
    Hte = Htr[:len(te)] if (a.smoke and a.task != 2) else v13.cache(te, llm, dev)
    del llm
    torch.cuda.empty_cache()
    print("  states cached (%.0fs)" % (time.time() - t0), flush=True)

    for seed in a.seeds:
        torch.manual_seed(seed)
        sym = F.normalize(torch.randn(TOT, C.VSA_DIM,
                                      generator=torch.Generator().manual_seed(seed + 11)), dim=1).to(dev)
        # naming head exactly as published (v58 / v52 / v37b frozen arm 'probe_argmax')
        X, Y = [], []
        for b, r in enumerate(tr):
            H = Htr[b].to(dev)
            if a.task == 4:
                for ra, dr, rb, sa, sb in r["stmts"]:
                    X.append(sf(H, sa)); Y.append(ra)
                    X.append(sf(H, sb)); Y.append(rb)
                X.append(sf(H, r["q_span"])); Y.append(r["qroom"])
            elif a.task == 15:
                for cid, pid, csp, psp in r["cls"]:
                    X.append(sf(H, csp)); Y.append(cid)
                    X.append(sf(H, psp)); Y.append(pid)
                for eid, cid, esp, csp in r["mem"]:
                    X.append(sf(H, esp)); Y.append(n_an + eid)
                    X.append(sf(H, csp)); Y.append(cid)
                X.append(sf(H, r["q_span"])); Y.append(n_an + r["q"])
            else:
                for oid, hid, osp, hsp in r["obj_ev"]:
                    X.append(sf(H, osp)); Y.append(O + oid)
                    X.append(sf(H, hsp)); Y.append(P + hid)
                for pid, lid, psp, lsp in r["moves"]:
                    X.append(sf(H, psp)); Y.append(P + pid)
                    X.append(sf(H, lsp)); Y.append(L + lid)
                X.append(sf(H, r["q_span"])); Y.append(O + r["q_obj"])
        X = torch.stack(X).to(dev); Y = torch.tensor(Y).to(dev)
        idh = train_head(X, Y, n_head, a.head_epochs, dev, v13.HID)
        idh.eval()
        if a.task == 4:
            dvec = {dr: sym[n_r + i] for i, dr in enumerate(DIRS)}

        with torch.no_grad():
            cache = {}

            def named(H, sp, b):
                k = (b, sp[0], sp[1])
                if k not in cache:
                    cache[k] = int(idh(sf(H, sp)).argmax().item())
                return cache[k]

            for enc_name in a.encodings:
                enc = C.Encoding(enc_name, sym, seed)
                for arm in ["parser", "probe_argmax"]:
                    rec = []
                    for b, r in enumerate(te):
                        H = Hte[b].to(dev)
                        if arm == "parser":
                            idf = lambda H_, sp, gid: gid
                        else:
                            idf = lambda H_, sp, gid, b=b: named(H_, sp, b)
                        if a.task == 4:
                            o = eval4(r, H, idf, enc, dvec, v13)
                        elif a.task == 15:
                            o = eval15(r, H, idf, enc, n_an, v13)
                        else:
                            o = eval2(r, H, idf, enc, P, L, O, v13)
                        o["correct"] = int(o["pred"] == r["label"])
                        rec.append(o)
                    acc = float(np.mean([o["correct"] for o in rec]))
                    key = "%s/%s/%d" % (arm, enc_name, seed)
                    out["runs"][key] = {"acc_scored": acc, "per_q": rec}
                    extra = ""
                    if a.task != 2:
                        al = np.array([g["alias"] for g in gold]); cc = np.array([o["correct"] for o in rec])
                        extra = "  alias %d: %.3f | non-alias %d: %.3f" % (
                            al.sum(), cc[al == 1].mean() if al.sum() else float("nan"),
                            (1 - al).sum(), cc[al == 0].mean() if (1 - al).sum() else float("nan"))
                    else:
                        ex = np.array(out["expressible"]); cc = np.array([o["correct"] for o in rec])
                        extra = "  expressible %d: %.4f | parsed %d: %.4f | /%d questions: %.4f" % (
                            ex.sum(), cc[ex == 1].mean(), len(cc), cc.mean(), out["n_questions"],
                            cc.sum() / out["n_questions"])
                    print("  %-12s %-5s seed %3d  acc %.4f%s" % (arm, enc_name, seed, acc, extra), flush=True)
        os.makedirs(a.out, exist_ok=True)
        fn = os.path.join(a.out, "b28_task%d%s%s.json" % (a.task, "" if a.patterns == "published"
                                                          else "_" + a.patterns, "_smoke" if a.smoke else ""))
        json.dump(out, open(fn + ".tmp", "w"))
        os.replace(fn + ".tmp", fn)
    out["secs"] = round(time.time() - t0, 1)
    json.dump(out, open(fn + ".tmp", "w"))
    os.replace(fn + ".tmp", fn)
    print("B28-BABI-DONE task %d (%.0fs) -> %s" % (a.task, time.time() - t0, fn), flush=True)


if __name__ == "__main__":
    main()
