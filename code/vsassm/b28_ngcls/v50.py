"""V-50: train the language model ITSELF through the VSA (prereg 136ebe9, bars registered before this file).

Every previous run in this paper froze the encoder. The binding objective has never been backpropagated into the
model, so "the model learns to place symbols where they bind" has never been given a chance. Two arms:

  lora_direct        LoRA + cross-entropy on the answer token. No VSA. The supervised opponent bAbI never had.
  lora_through_vsa   LoRA + the loss taken AFTER bind, bundle, unbind and cleanup, gradients flowing through the
                     algebra into the model. Codes come from a jointly learned projection of span states and NO
                     identity labels are supplied.

Both arms see the same instances, the same epochs and the same LoRA configuration. The ordering rule is supplied
from outside the algebra, as everywhere on these tasks.
"""
import argparse, json, os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
sys.path.insert(0, "/path/to/work/ngcls")
import v13                                                            # noqa: E402
from v5 import load_babi                                              # noqa: E402

VSA_DIM = 2048
BANKED = {1: {"handover": 1.0, "prompt": 0.7920, "probe": 0.7713},
          2: {"handover": 1.0, "prompt": 0.3860, "probe": 0.2452}}


def spans_task1(r, n_p):
    """(query span, [(person span, loc span, loc_id)]) under the recency rule, from outside the algebra.

    BUG FIX, disclosed: v30.build stores the location symbol as len(persons) + locs[word], while r["label"] is
    the UNOFFSET location index. v30's own code absorbs the offset by slicing sym[len(persons):]; the first
    version of this file compared the offset id directly against the label, so the candidate list almost never
    contained the answer, training silently skipped nearly every row, and the arm scored below chance
    identically across seeds. n_p is subtracted here so the two indexings agree."""
    last = {}
    for j, s in enumerate(r["stmts"]):
        last[s[0]] = j
    st = [r["stmts"][j] for j in sorted(last.values())]
    return r["q_span"], [(s[2], s[3], s[1] - n_p) for s in st]


def spans_task2(r):
    """mem1 over object events, mem2 over person movements; oracle selection as in v37b."""
    return (r["q_span"],
            [(spf, spr) for oid, hid, spr, spf in r["obj_ev"]],
            [(psp, lsp, lid) for pid, lid, psp, lsp in r["moves"]])


def sf(H, sp):
    return H[sp[0]:sp[1]].mean(0) if sp[1] > sp[0] else H[sp[0]]


def build_rows(task, tok, n_train, n_test):
    if task == 1:
        from v30 import STMT, build
        d = load_babi(1, n_train, 200, n_test)
        locs = {l: i for i, l in enumerate(d["vocab"])}
        persons = {}
        for text, _ in d["train"]:
            for m in STMT.finditer(text):
                persons.setdefault(m.group(1), len(persons))
        return (build(d["train"], tok, locs, persons), build(d["test"], tok, locs, persons),
                d["vocab"], len(persons))
    import v37b
    d = load_babi(2, n_train, 200, n_test)
    locs = {l: i for i, l in enumerate(d["vocab"])}
    persons, objs = {}, {}
    for text, _ in d["train"]:
        for _, k, s_, o, _, _ in v37b.events(text):
            persons.setdefault(s_, len(persons))
            if k != "move":
                objs.setdefault(o, len(objs))
    tr, _ = v37b.build(d["train"], tok, persons, locs, objs)
    te, _ = v37b.build(d["test"], tok, persons, locs, objs)
    return ([r for r in tr if r["expressible"]], [r for r in te if r["expressible"]],
            d["vocab"], len(persons))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int, default=1)
    ap.add_argument("--arms", nargs="+", default=["lora_direct", "lora_through_vsa"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    ap.add_argument("--n_train", type=int, default=2000)
    ap.add_argument("--n_test", type=int, default=500)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="v50_logs/v50.json")
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    tok = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    tr, te, vocab, n_p = build_rows(a.task, tok, a.n_train, a.n_test)
    ans_ids, kept = [], []
    for i, w in enumerate(vocab):
        t = tok(" " + w, add_special_tokens=False)["input_ids"]
        if len(t) == 1:
            ans_ids.append(t[0]); kept.append(i)
    remap = {g: j for j, g in enumerate(kept)}
    tr = [r for r in tr if r["label"] in remap]
    te = [r for r in te if r["label"] in remap]
    print("===== V-50  task %d: training the model itself through the VSA =====" % a.task, flush=True)
    print("  %d train / %d eval instances, %d locations all single-token=%s"
          % (len(tr), len(te), len(vocab), len(ans_ids) == len(vocab)), flush=True)
    print("  banked references: handover %.4f | prompting %.4f | last-token probe %.4f"
          % (BANKED[a.task]["handover"], BANKED[a.task]["prompt"], BANKED[a.task]["probe"]), flush=True)

    out = {"task": a.task, "n_train": len(tr), "n_eval": len(te), "banked": BANKED[a.task], "arms": {}}

    def fresh():
        m = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", trust_remote_code=True,
                                                 torch_dtype=torch.bfloat16).to(a.device)
        cfg = LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.0, bias="none",
                         target_modules=["q_proj", "k_proj", "v_proj", "dense"], task_type="CAUSAL_LM")
        m = get_peft_model(m, cfg)
        m.train()
        return m

    def batch(rows):
        ids = torch.stack([r["ids"] for r in rows]).to(a.device)
        am = torch.stack([r["am"] for r in rows]).to(a.device)
        return ids, am

    def run(arm, seed):
        torch.manual_seed(seed)
        model = fresh()
        extra = []
        proj = None
        sym = None
        if arm == "lora_through_vsa":
            proj = nn.Linear(v13.HID, VSA_DIM).to(a.device).to(torch.bfloat16)
            extra = list(proj.parameters())
        params = [p for p in model.parameters() if p.requires_grad] + extra
        print("      %s trainable params: %d" % (arm, sum(p.numel() for p in params)), flush=True)
        opt = torch.optim.AdamW(params, lr=a.lr)

        def code(H, sp):
            return F.normalize(proj(sf(H, sp)).float(), dim=-1)

        def vsa_logits(H, r):
            """Retrieve through the algebra; returns (logits over candidate locations, their loc ids)."""
            if a.task == 1:
                qsp, st = spans_task1(r, n_p)
                mem = torch.stack([v13.bind(code(H, l), code(H, p)) for p, l, _ in st], 0).sum(0)
                got = F.normalize(v13.unbind(mem, code(H, qsp)), dim=-1)
                cand = torch.stack([code(H, l) for _, l, _ in st], 0)
                ids_ = [lid for _, _, lid in st]
            else:
                qsp, oe, mv = spans_task2(r)
                m1 = torch.stack([v13.bind(code(H, h), code(H, o)) for o, h in oe], 0).sum(0)
                hh = F.normalize(v13.unbind(m1, code(H, qsp)), dim=-1)
                hc = torch.stack([code(H, h) for _, h in oe], 0)
                hh = hc[int(F.cosine_similarity(hc, hh.unsqueeze(0)).argmax().item())]
                m2 = torch.stack([v13.bind(code(H, l), code(H, p)) for p, l, _ in mv], 0).sum(0)
                got = F.normalize(v13.unbind(m2, hh), dim=-1)
                cand = torch.stack([code(H, l) for _, l, _ in mv], 0)
                ids_ = [lid for _, _, lid in mv]
            return F.cosine_similarity(cand, got.unsqueeze(0)) * 10.0, ids_

        skipped = [0]
        seen = [0]
        g = torch.Generator().manual_seed(seed)
        for ep in range(a.epochs):
            perm = torch.randperm(len(tr), generator=g).tolist()
            tot, nb = 0.0, 0
            for k in range(0, len(perm), a.bs):
                rows = [tr[i] for i in perm[k:k + a.bs]]
                ids, am = batch(rows)
                if arm == "lora_direct":
                    lg = model(input_ids=ids, attention_mask=am).logits
                    last = am.sum(-1) - 1
                    sel = lg[torch.arange(lg.size(0), device=a.device), last].float()
                    y = torch.tensor([remap[r["label"]] for r in rows], device=a.device)
                    loss = F.cross_entropy(sel[:, ans_ids], y)
                else:
                    hs = model(input_ids=ids, attention_mask=am, output_hidden_states=True).hidden_states[-1]
                    ls = []
                    for b_, r in enumerate(rows):
                        lgts, idl = vsa_logits(hs[b_], r)
                        if r["label"] not in idl:
                            skipped[0] += 1
                            continue
                        ls.append(F.cross_entropy(lgts.unsqueeze(0),
                                                  torch.tensor([idl.index(r["label"])], device=a.device)))
                    seen[0] += len(rows)
                    if not ls:
                        continue
                    loss = torch.stack(ls).mean()
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
                tot += float(loss.item()); nb += 1
            if arm == "lora_through_vsa":
                print("      %s seed %d ep%d loss=%.4f  (rows skipped for label-not-in-candidates: %d of %d)"
                      % (arm, seed, ep + 1, tot / max(1, nb), skipped[0], seen[0]), flush=True)
                if seen[0] and skipped[0] / seen[0] > 0.5:
                    raise SystemExit("ABORT: more than half of training rows had the label outside the "
                                     "candidate list -- indexing mismatch, not a learning result")
            else:
                print("      %s seed %d ep%d loss=%.4f" % (arm, seed, ep + 1, tot / max(1, nb)), flush=True)

        model.eval()
        ok = []
        with torch.no_grad():
            for k in range(0, len(te), a.bs):
                rows = te[k:k + a.bs]
                ids, am = batch(rows)
                if arm == "lora_direct":
                    lg = model(input_ids=ids, attention_mask=am).logits
                    last = am.sum(-1) - 1
                    sel = lg[torch.arange(lg.size(0), device=a.device), last].float()[:, ans_ids]
                    pred = sel.argmax(-1).tolist()
                    ok += [int(p == remap[r["label"]]) for p, r in zip(pred, rows)]
                else:
                    hs = model(input_ids=ids, attention_mask=am, output_hidden_states=True).hidden_states[-1]
                    for b_, r in enumerate(rows):
                        lgts, idl = vsa_logits(hs[b_], r)
                        ok.append(int(idl[int(lgts.argmax().item())] == r["label"]))
        del model
        if proj is not None:
            del proj
        torch.cuda.empty_cache()
        return float(np.mean(ok))

    for arm in a.arms:
        accs = [run(arm, s) for s in a.seeds]
        out["arms"][arm] = accs
        print("  %-18s %s mean=%.4f" % (arm, [round(x, 4) for x in accs], float(np.mean(accs))), flush=True)
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(out, open(a.out, "w"), indent=1)

    print("", flush=True)
    print("===== V-50 SCORING task %d (bars registered at 136ebe9) =====" % a.task, flush=True)
    ld = out["arms"].get("lora_direct")
    lv = out["arms"].get("lora_through_vsa")
    hv = BANKED[a.task]["handover"]
    if ld:
        print("  lora_direct %s mean=%.4f  (prompting %.4f, probe %.4f, handover %.4f)"
              % ([round(x, 4) for x in ld], float(np.mean(ld)), BANKED[a.task]["prompt"],
                 BANKED[a.task]["probe"], hv), flush=True)
    if lv:
        print("  lora_through_vsa %s mean=%.4f" % ([round(x, 4) for x in lv], float(np.mean(lv))), flush=True)
    v = []
    if lv and all(x >= 0.95 for x in lv):
        v.append("THROUGH-VSA REACHES .95 on task %d" % a.task)
    if lv and ld and all(x <= y - 0.05 for x, y in zip(lv, ld)):
        v.append("THROUGH-VSA STILL FAILS on task %d (below lora_direct by >= .05 in 3/3)" % a.task)
    if ld:
        surv = all(hv - x >= 0.10 for x in ld)
        v.append("HANDOVER %s A SUPERVISED OPPONENT on task %d"
                 % ("SURVIVES" if surv else "DOES NOT SURVIVE", a.task))
    out["verdict"] = " | ".join(v) if v else "GRADED (report as measured)"
    json.dump(out, open(a.out, "w"), indent=1)
    print("  >>> V-50 VERDICT: %s" % out["verdict"], flush=True)
    print("V50-DONE", flush=True)


if __name__ == "__main__":
    main()
