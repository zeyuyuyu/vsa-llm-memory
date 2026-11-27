"""V-52: bAbI task 15, entity -> class -> property, with the supervised opponent (prereg a6a18fc).

Memorisability was checked first and cleared: the question key determines the answer in 0.0000 of training
instances and there are 225 distinct statement blocks over 900 instances, so the knowledge base varies and a
fine-tune cannot simply memorise it.

Two genuine hops. Plural and singular mentions of one animal -- "Mice" and "mouse" -- must map to the SAME
symbol, which is a mention partition the identity head has to learn rather than be told. No outside ordering
rule is needed on this task.
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
from v30 import tok_span                                              # noqa: E402

VSA_DIM = 2048
CLASS = re.compile(r"([A-Za-z]+) are afraid of ([A-Za-z]+)\.")
MEMBER = re.compile(r"([A-Za-z]+) is a ([A-Za-z]+)\.")
QUERY = re.compile(r"What is ([A-Za-z]+) afraid of\?")
IRREG = {"mice": "mouse", "wolves": "wolf", "sheep": "sheep", "cats": "cat"}
PROMPT_BANKED = 0.3867          # V-51, better prompting arm on task 15


def sing(w):
    w = w.lower()
    if w in IRREG:
        return IRREG[w]
    if w.endswith("ves"):
        return w[:-3] + "f"
    if w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def build(split, tok, animals, ents, max_len=256):
    rows = []
    for text, y in split:
        enc = tok(text, truncation=True, max_length=max_len, padding="max_length",
                  return_tensors="pt", return_offsets_mapping=True)
        offs = enc["offset_mapping"].squeeze(0).tolist()
        cls, mem, ok = [], [], True
        for m in CLASS.finditer(text):
            c, p = sing(m.group(1)), sing(m.group(2))
            if c not in animals or p not in animals:
                ok = False; break
            a, b = tok_span(offs, *m.span(1)), tok_span(offs, *m.span(2))
            if a is None or b is None:
                ok = False; break
            cls.append((animals[c], animals[p], a, b))
        for m in MEMBER.finditer(text):
            e, c = m.group(1).lower(), sing(m.group(2))
            if c not in animals:
                ok = False; break
            ents.setdefault(e, len(ents))
            a, b = tok_span(offs, *m.span(1)), tok_span(offs, *m.span(2))
            if a is None or b is None:
                ok = False; break
            mem.append((ents[e], animals[c], a, b))
        qm = QUERY.search(text)
        if not ok or not cls or not mem or qm is None:
            continue
        e = qm.group(1).lower()
        if e not in ents:
            continue
        qsp = tok_span(offs, *qm.span(1))
        if qsp is None:
            continue
        rows.append({"ids": enc["input_ids"].squeeze(0), "am": enc["attention_mask"].squeeze(0),
                     "cls": cls, "mem": mem, "q": ents[e], "q_span": qsp, "label": int(y)})
    return rows


def sf(H, sp):
    return H[sp[0]:sp[1]].float().mean(0) if sp[1] > sp[0] else H[sp[0]].float()


def last_tok(H, r):
    return H[max(int(r["am"].sum()) - 1, 0)].float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+",
                    default=["floor", "parser", "probe_argmax", "spans", "lora_direct"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    ap.add_argument("--n_train", type=int, default=2000)
    ap.add_argument("--n_test", type=int, default=500)
    ap.add_argument("--head_epochs", type=int, default=40)
    ap.add_argument("--probe_epochs", type=int, default=40)
    ap.add_argument("--span_epochs", type=int, default=3)
    ap.add_argument("--lora_epochs", type=int, default=2)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="v52_logs/v52.json")
    a = ap.parse_args()

    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    d = load_babi(15, a.n_train, 200, a.n_test)
    animals = {w: i for i, w in enumerate(d["vocab"])}
    ents = {}
    tr = build(d["train"], tok, animals, ents)
    te = build(d["test"], tok, animals, ents)
    n_an, n_ent = len(animals), len(ents)
    TOT = n_an + n_ent
    print("===== V-52  bAbI task 15: entity -> class -> property =====", flush=True)
    print("  %d animals %s, %d entity names, codebook %d; %d train / %d eval"
          % (n_an, d["vocab"], n_ent, TOT, len(tr), len(te)), flush=True)
    print("  banked prompting (V-51): %.4f" % PROMPT_BANKED, flush=True)
    out = {"task": 15, "n_train": len(tr), "n_eval": len(te), "codebook": TOT,
           "prompting_banked": PROMPT_BANKED, "arms": {}}

    llm = AutoModel.from_pretrained("microsoft/phi-2", trust_remote_code=True,
                                    torch_dtype=torch.float16).to(a.device).eval()
    for p in llm.parameters():
        p.requires_grad_(False)
    Htr, Hte = v13.cache(tr, llm, a.device), v13.cache(te, llm, a.device)

    def gid_ent(i):
        return n_an + i

    def frozen_arm(arm, seed):
        torch.manual_seed(seed)
        sym = F.normalize(torch.randn(TOT, VSA_DIM,
                                      generator=torch.Generator().manual_seed(seed + 11)), dim=1).to(a.device)
        if arm == "floor":
            pr = nn.Linear(v13.HID, n_an).to(a.device)
            opt = torch.optim.AdamW(pr.parameters(), lr=1e-3)
            X = torch.stack([last_tok(Htr[b].to(a.device), r) for b, r in enumerate(tr)])
            Y = torch.tensor([r["label"] for r in tr]).to(a.device)
            for _ in range(a.probe_epochs):
                p = torch.randperm(X.size(0), device=a.device)
                for i in range(0, p.numel(), 256):
                    s = p[i:i + 256]
                    loss = F.cross_entropy(pr(X[s]), Y[s]); opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                Xt = torch.stack([last_tok(Hte[b].to(a.device), r) for b, r in enumerate(te)])
                return float((pr(Xt).argmax(1).cpu()
                              == torch.tensor([r["label"] for r in te])).double().mean())

        idh = proj = None
        if arm == "probe_argmax":
            idh = nn.Linear(v13.HID, TOT).to(a.device)
            opt = torch.optim.AdamW(idh.parameters(), lr=1e-3)
            X, Y = [], []
            for b, r in enumerate(tr):
                H = Htr[b].to(a.device)
                for cid, pid, csp, psp in r["cls"]:
                    X.append(sf(H, csp)); Y.append(cid)
                    X.append(sf(H, psp)); Y.append(pid)
                for eid, cid, esp, csp in r["mem"]:
                    X.append(sf(H, esp)); Y.append(gid_ent(eid))
                    X.append(sf(H, csp)); Y.append(cid)
                X.append(sf(H, r["q_span"])); Y.append(gid_ent(r["q"]))
            X = torch.stack(X).to(a.device); Y = torch.tensor(Y).to(a.device)
            for _ in range(a.head_epochs):
                p = torch.randperm(X.size(0), device=a.device)
                for i in range(0, p.numel(), 512):
                    s = p[i:i + 512]
                    loss = F.cross_entropy(idh(X[s]), Y[s]); opt.zero_grad(); loss.backward(); opt.step()
        if arm == "spans":
            proj = nn.Linear(v13.HID, VSA_DIM).to(a.device)
            popt = torch.optim.AdamW(proj.parameters(), lr=3e-4)

        def code(H, sp, gid):
            if arm == "parser":
                return sym[gid]
            if arm == "probe_argmax":
                return sym[int(idh(sf(H, sp)).argmax().item())]
            return F.normalize(proj(sf(H, sp)), dim=-1)

        def forward(H, r):
            m1 = torch.stack([v13.bind(code(H, csp, cid), code(H, esp, gid_ent(eid)))
                              for eid, cid, esp, csp in r["mem"]], 0).sum(0)
            h = F.normalize(v13.unbind(m1, code(H, r["q_span"], gid_ent(r["q"]))), dim=-1)
            cc = torch.stack([code(H, csp, cid) for _, cid, _, csp in r["mem"]], 0)
            h = cc[int(F.cosine_similarity(cc, h.unsqueeze(0)).argmax().item())]
            m2 = torch.stack([v13.bind(code(H, psp, pid), code(H, csp, cid))
                              for cid, pid, csp, psp in r["cls"]], 0).sum(0)
            got = F.normalize(v13.unbind(m2, h), dim=-1)
            cand = torch.stack([code(H, psp, pid) for _, pid, _, psp in r["cls"]], 0)
            return got, cand, [pid for _, pid, _, _ in r["cls"]]

        if arm == "spans":
            for _ in range(a.span_epochs):
                for b, r in enumerate(tr):
                    H = Htr[b].to(a.device)
                    got, cand, ids_ = forward(H, r)
                    if r["label"] not in ids_:
                        continue
                    lg = F.cosine_similarity(cand, got.unsqueeze(0)) * 10.0
                    loss = F.cross_entropy(lg.unsqueeze(0),
                                           torch.tensor([ids_.index(r["label"])], device=a.device))
                    popt.zero_grad(); loss.backward(); popt.step()

        with torch.no_grad():
            ok = []
            for b, r in enumerate(te):
                H = Hte[b].to(a.device)
                got, cand, ids_ = forward(H, r)
                ok.append(int(ids_[int(F.cosine_similarity(cand, got.unsqueeze(0)).argmax().item())]
                              == r["label"]))
        return float(np.mean(ok))

    for arm in [x for x in a.arms if x != "lora_direct"]:
        accs = [frozen_arm(arm, s) for s in a.seeds]
        out["arms"][arm] = accs
        print("  %-13s %s mean=%.4f" % (arm, [round(x, 4) for x in accs], float(np.mean(accs))), flush=True)
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(out, open(a.out, "w"), indent=1)
    del llm, Htr, Hte
    torch.cuda.empty_cache()

    if "lora_direct" in a.arms:
        from peft import LoraConfig, get_peft_model
        ans_ids = [tok(" " + w, add_special_tokens=False)["input_ids"][0] for w in d["vocab"]]
        accs = []
        for seed in a.seeds:
            torch.manual_seed(seed)
            m = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", trust_remote_code=True,
                                                     torch_dtype=torch.bfloat16).to(a.device)
            m = get_peft_model(m, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0, bias="none",
                                             target_modules=["q_proj", "k_proj", "v_proj", "dense"],
                                             task_type="CAUSAL_LM"))
            m.train()
            params = [p for p in m.parameters() if p.requires_grad]
            opt = torch.optim.AdamW(params, lr=1e-4)
            g = torch.Generator().manual_seed(seed)
            for ep in range(a.lora_epochs):
                perm = torch.randperm(len(tr), generator=g).tolist()
                tot, nb = 0.0, 0
                for k in range(0, len(perm), a.bs):
                    rows = [tr[i] for i in perm[k:k + a.bs]]
                    ids = torch.stack([r["ids"] for r in rows]).to(a.device)
                    am = torch.stack([r["am"] for r in rows]).to(a.device)
                    lg = m(input_ids=ids, attention_mask=am).logits
                    last = am.sum(-1) - 1
                    sel = lg[torch.arange(lg.size(0), device=a.device), last].float()[:, ans_ids]
                    y = torch.tensor([r["label"] for r in rows], device=a.device)
                    loss = F.cross_entropy(sel, y)
                    opt.zero_grad(); loss.backward()
                    torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
                    tot += float(loss.item()); nb += 1
                print("      lora_direct seed %d ep%d loss=%.4f" % (seed, ep + 1, tot / max(1, nb)), flush=True)
            m.eval()
            ok = []
            with torch.no_grad():
                for k in range(0, len(te), a.bs):
                    rows = te[k:k + a.bs]
                    ids = torch.stack([r["ids"] for r in rows]).to(a.device)
                    am = torch.stack([r["am"] for r in rows]).to(a.device)
                    lg = m(input_ids=ids, attention_mask=am).logits
                    last = am.sum(-1) - 1
                    sel = lg[torch.arange(lg.size(0), device=a.device), last].float()[:, ans_ids]
                    ok += [int(p == r["label"]) for p, r in zip(sel.argmax(-1).tolist(), rows)]
            accs.append(float(np.mean(ok)))
            del m; torch.cuda.empty_cache()
        out["arms"]["lora_direct"] = accs
        print("  %-13s %s mean=%.4f" % ("lora_direct", [round(x, 4) for x in accs],
                                        float(np.mean(accs))), flush=True)
        json.dump(out, open(a.out, "w"), indent=1)

    print("", flush=True)
    print("===== V-52 SCORING task 15 (bars registered at a6a18fc) =====", flush=True)
    m = {k: float(np.mean(v)) for k, v in out["arms"].items()}
    for k in ("floor", "parser", "probe_argmax", "spans", "lora_direct"):
        if k in m:
            print("  %-13s %.4f" % (k, m[k]), flush=True)
    print("  prompting (banked) %.4f" % PROMPT_BANKED, flush=True)
    adm = m.get("parser", 0) - PROMPT_BANKED
    print("  admission: parser - prompting = %.4f (needs >= 0.20)" % adm, flush=True)
    if adm < 0.20:
        v = "VOID (admission failed: parser - prompting = %.4f)" % adm
    elif "lora_direct" in out["arms"] and "probe_argmax" in out["arms"]:
        gaps = [h - l for h, l in zip(out["arms"]["probe_argmax"], out["arms"]["lora_direct"])]
        print("  handover - lora_direct per seed: %s" % [round(x, 4) for x in gaps], flush=True)
        if all(x >= 0.10 for x in gaps):
            v = ("STRUCTURE PAYS on task 15 (handover beats a supervised fine-tune by >= .10 in 3/3; "
                 "structure plus identity supervision against answer supervision alone)")
        elif all(x <= 0.02 for x in gaps):
            v = "NO GAIN on task 15 (the fine-tune matches the handover)"
        else:
            v = "GRADED on task 15 (report as measured)"
    else:
        v = "INCOMPLETE (lora_direct or probe_argmax missing)"
    out["verdict"] = v
    json.dump(out, open(a.out, "w"), indent=1)
    print("  >>> V-52 VERDICT: %s" % v, flush=True)
    print("V52-DONE", flush=True)


if __name__ == "__main__":
    main()
