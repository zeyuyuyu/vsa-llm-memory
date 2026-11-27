"""V-58 (V-56 Part A): bAbI task 4, which the withdrawn-then-refined condition predicts will CAP (bars at 86d5fec,
re-derived under the amendment at 0c9080f).

Task 4: "The A is DIR of B." asked both ways. Every argument is a room, so the spurious cross-term at cleanup is
also a room -- the same-type situation that capped task 15. The inverse mapping (north/south, east/west) is
supplied from OUTSIDE the algebra, and the four direction words get CODEBOOK ROWS rather than state-derived codes
(registered amendment: after inversion a direction may have no mention in the text). Rooms come from span states
via the handover. A hybrid, and the manuscript must say so.

Encoding: each statement "A is DIR of B" is stored twice in one bundle:
    bind(A, bind(DIR, B))          answers "What is DIR of B?"
    bind(B, bind(inv(DIR), A))     answers "What is A DIR-inverse of?" i.e. "What is the A DIR of?"
The query "What is DIR of B?" unbinds by bind(DIR, B); cleanup over the rooms mentioned in the instance.
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
STMT = re.compile(r"The ([a-z]+) is (north|south|east|west) of the ([a-z]+)\.")
Q1 = re.compile(r"What is (north|south|east|west) of the ([a-z]+)\?")
Q2 = re.compile(r"What is the ([a-z]+) (north|south|east|west) of\?")
INV = {"north": "south", "south": "north", "east": "west", "west": "east"}
DIRS = ["north", "south", "east", "west"]
PROMPT_BANKED = 0.4411          # V-51, better prompting arm on task 4


def build(split, tok, rooms, max_len=192):
    rows = []
    for text, y in split:
        enc = tok(text, truncation=True, max_length=max_len, padding="max_length",
                  return_tensors="pt", return_offsets_mapping=True)
        offs = enc["offset_mapping"].squeeze(0).tolist()
        st, ok = [], True
        for m in STMT.finditer(text):
            a, dr, b = m.group(1), m.group(2), m.group(3)
            if a not in rooms or b not in rooms:
                ok = False; break
            sa, sb = tok_span(offs, *m.span(1)), tok_span(offs, *m.span(3))
            if sa is None or sb is None:
                ok = False; break
            st.append((rooms[a], dr, rooms[b], sa, sb))
        if not ok or not st:
            continue
        m1, m2 = Q1.search(text), Q2.search(text)
        if m1:
            dr, b = m1.group(1), m1.group(2)
            if b not in rooms:
                continue
            qsp = tok_span(offs, *m1.span(2))
            qdir, qroom = dr, rooms[b]
        elif m2:
            a, dr = m2.group(1), m2.group(2)
            if a not in rooms:
                continue
            qsp = tok_span(offs, *m2.span(1))
            qdir, qroom = INV[dr], rooms[a]        # "What is the A north of?" == "What is south of A?" inverted
        else:
            continue
        if qsp is None:
            continue
        rows.append({"ids": enc["input_ids"].squeeze(0), "am": enc["attention_mask"].squeeze(0),
                     "stmts": st, "qdir": qdir, "qroom": qroom, "q_span": qsp, "label": int(y)})
    return rows


def sf(H, sp):
    return H[sp[0]:sp[1]].float().mean(0) if sp[1] > sp[0] else H[sp[0]].float()


def last_tok(H, r):
    return H[max(int(r["am"].sum()) - 1, 0)].float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["floor", "parser", "probe_argmax", "lora_direct"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    ap.add_argument("--n_train", type=int, default=2000)
    ap.add_argument("--n_test", type=int, default=500)
    ap.add_argument("--head_epochs", type=int, default=40)
    ap.add_argument("--probe_epochs", type=int, default=40)
    ap.add_argument("--lora_epochs", type=int, default=2)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="v58_logs/v58.json")
    a = ap.parse_args()

    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    d = load_babi(4, a.n_train, 200, a.n_test)
    rooms = {w: i for i, w in enumerate(d["vocab"])}
    tr = build(d["train"], tok, rooms)
    te = build(d["test"], tok, rooms)
    n_r = len(rooms)
    TOT = n_r + len(DIRS)
    print("===== V-58  bAbI task 4: two-argument relation, same-type spurious item =====", flush=True)
    print("  %d rooms, 4 direction rows, codebook %d; %d train / %d eval; banked prompting %.4f"
          % (n_r, TOT, len(tr), len(te), PROMPT_BANKED), flush=True)
    out = {"task": 4, "n_train": len(tr), "n_eval": len(te), "prompting_banked": PROMPT_BANKED, "arms": {}}

    llm = AutoModel.from_pretrained("microsoft/phi-2", trust_remote_code=True,
                                    torch_dtype=torch.float16).to(a.device).eval()
    for p in llm.parameters():
        p.requires_grad_(False)
    Htr, Hte = v13.cache(tr, llm, a.device), v13.cache(te, llm, a.device)

    def frozen_arm(arm, seed):
        torch.manual_seed(seed)
        sym = F.normalize(torch.randn(TOT, VSA_DIM,
                                      generator=torch.Generator().manual_seed(seed + 11)), dim=1).to(a.device)
        dvec = {dr: sym[n_r + i] for i, dr in enumerate(DIRS)}
        if arm == "floor":
            pr = nn.Linear(v13.HID, n_r).to(a.device)
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

        idh = None
        if arm == "probe_argmax":
            idh = nn.Linear(v13.HID, n_r).to(a.device)
            opt = torch.optim.AdamW(idh.parameters(), lr=1e-3)
            X, Y = [], []
            for b, r in enumerate(tr):
                H = Htr[b].to(a.device)
                for ra, dr, rb, sa, sb in r["stmts"]:
                    X.append(sf(H, sa)); Y.append(ra)
                    X.append(sf(H, sb)); Y.append(rb)
                X.append(sf(H, r["q_span"])); Y.append(r["qroom"])
            X = torch.stack(X).to(a.device); Y = torch.tensor(Y).to(a.device)
            for _ in range(a.head_epochs):
                p = torch.randperm(X.size(0), device=a.device)
                for i in range(0, p.numel(), 512):
                    s = p[i:i + 512]
                    loss = F.cross_entropy(idh(X[s]), Y[s]); opt.zero_grad(); loss.backward(); opt.step()

        def room_code(H, sp, rid):
            if arm == "parser":
                return sym[rid]
            return sym[int(idh(sf(H, sp)).argmax().item())]

        with torch.no_grad():
            ok = []
            for b, r in enumerate(te):
                H = Hte[b].to(a.device)
                mem = torch.zeros(VSA_DIM, device=a.device)
                mentioned = []
                for ra, dr, rb, sa, sb in r["stmts"]:
                    va, vb = room_code(H, sa, ra), room_code(H, sb, rb)
                    mem = mem + v13.bind(va, v13.bind(dvec[dr], vb))
                    mem = mem + v13.bind(vb, v13.bind(dvec[INV[dr]], va))
                    mentioned += [(ra, va), (rb, vb)]
                qv = room_code(H, r["q_span"], r["qroom"])
                got = F.normalize(v13.unbind(mem, v13.bind(dvec[r["qdir"]], qv)), dim=-1)
                cand = torch.stack([v for _, v in mentioned], 0)
                ids_ = [i for i, _ in mentioned]
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
            del m
            torch.cuda.empty_cache()
        out["arms"]["lora_direct"] = accs
        print("  %-13s %s mean=%.4f" % ("lora_direct", [round(x, 4) for x in accs],
                                        float(np.mean(accs))), flush=True)
        json.dump(out, open(a.out, "w"), indent=1)

    print("", flush=True)
    print("===== V-58 SCORING (bars from 86d5fec Part A, re-derived at 0c9080f) =====", flush=True)
    m = {k: float(np.mean(v)) for k, v in out["arms"].items()}
    for k in ("floor", "parser", "probe_argmax", "lora_direct"):
        if k in m:
            print("  %-13s %.4f" % (k, m[k]), flush=True)
    print("  prompting (banked) %.4f" % PROMPT_BANKED, flush=True)
    pa = out["arms"].get("parser", [])
    pq = out["arms"].get("probe_argmax", [])
    ld = out["arms"].get("lora_direct", [])
    if pa and pq and ld:
        cap = all(x <= 0.90 for x in pa)
        ft_wins = all(l - h >= 0.05 for h, l in zip(pq, ld))
        struct_wins = all(h - l >= 0.10 for h, l in zip(pq, ld))
        print("  parser <= .90 in 3/3 (the algebra caps): %s %s" % (cap, [round(x, 4) for x in pa]), flush=True)
        print("  lora - handover >= .05 in 3/3: %s | handover - lora >= .10 in 3/3: %s"
              % (ft_wins, struct_wins), flush=True)
        if cap and ft_wins:
            v = ("CONDITION PREDICTS CORRECTLY (the algebra caps on a same-type-spurious task and the fine-tune "
                 "passes it, as registered BEFORE the run)")
        elif struct_wins:
            v = ("CONDITION FAILS (structure wins on a reused-role task; the condition is withdrawn rather than "
                 "given an exception)")
        else:
            v = "GRADED (report as measured)"
    else:
        v = "INCOMPLETE"
    out["verdict"] = v
    json.dump(out, open(a.out, "w"), indent=1)
    print("  >>> V-58 VERDICT: %s" % v, flush=True)
    print("V58-DONE", flush=True)


if __name__ == "__main__":
    main()
