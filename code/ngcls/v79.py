"""V-79: MuSiQue, frozen deployment -- does the 2Wiki store result replicate on a SECOND mainstream
multi-hop benchmark, and does the pool separation survive REAL distractor triples? (prereg 1fd4489;
bars, triple construction, exclusion rule, P1-P4 and the negative readings all committed before this file.)

Arms, all with phi-2 frozen: `prompt` (the 20 paragraphs in context), `prompt_ev` (gold decomposition
verbalised as facts), `hrr_gold` (HRR store over the gold decomposition triples, chain followed in
decomposition order). Pool factor for the latter two: gold-only, +50, +150 real triples drawn from other
MuSiQue instances. Primary metric: exact match on the covered slice, declared in the prereg.
"""
import argparse, json, os, re, string, sys
import numpy as np
import torch

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
VSA_DIM = 4096
POOLS = [0, 50, 150]
# 1500 rather than V-65's 1200: MuSiQue ships full paragraphs where 2Wiki ships sentences, so the ported
# budget starved the paragraph arm. A larger window can only HELP the baseline, so this deviation cannot
# flatter our own claim. Registered as a deviation in the ledger before the scored run.
PARA_BUDGET = 1500


def norm(s):
    s = str(s).lower().strip()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in string.punctuation)
    return " ".join(s.split())


def em(pred, gold):
    return int(norm(pred) == norm(gold))


# ---------- triple construction, exactly as registered ----------
def triples_of(decomp):
    """subject = left of '>>', relation = right of '>>', object = the hop's answer; '#k' -> hop k's answer.
    Returns None if ANY hop lacks '>>' (registered exclusion rule)."""
    out = []
    for hop in decomp:
        q = hop.get("question", "")
        if ">>" not in q:
            return None
        subj, rel = q.split(">>", 1)
        subj, rel = subj.strip(), rel.strip()
        for m in set(re.findall(r"#(\d+)", subj)):
            k = int(m) - 1
            if not (0 <= k < len(out)):
                return None
            subj = subj.replace("#" + m, str(out[k][2]))
        ans = str(hop.get("answer", "")).strip()
        if not subj or not rel or not ans:
            return None
        out.append((subj, rel, ans))
    return out or None


# ---------- HRR store over gold triples, with real distractor triples in the same bundle ----------
def hrr_answer(gold, distractors, rng):
    ev = list(gold) + list(distractors)
    strings = sorted({s for tr in ev for s in tr})
    idx = {s: i for i, s in enumerate(strings)}
    V = torch.tensor(rng.standard_normal((len(strings), VSA_DIM)) / np.sqrt(VSA_DIM), dtype=torch.float32)
    Vn = V / (V.norm(dim=1, keepdim=True) + 1e-8)

    def bind(a, b):
        return torch.fft.irfft(torch.fft.rfft(a) * torch.fft.rfft(b), n=VSA_DIM)

    def unbind(m, k):
        return torch.fft.irfft(torch.fft.rfft(m) * torch.conj(torch.fft.rfft(k)), n=VSA_DIM)

    mem = torch.zeros(VSA_DIM)
    for s, r, o in ev:
        mem = mem + bind(bind(V[idx[s]], V[idx[r]]), V[idx[o]])

    # entities that may be answers: every subject and object in the bundle, distractors included, so a
    # distractor can win the cleanup. Relations are excluded (they are never answers).
    ent = sorted({tr[0] for tr in ev} | {tr[2] for tr in ev})
    ent_ids = torch.tensor([idx[e] for e in ent])
    cur = gold[0][0]
    for s, r, o in gold:
        hat = unbind(mem, bind(V[idx[cur]], V[idx[r]]))
        hat = hat / (hat.norm() + 1e-8)
        sims = Vn[ent_ids] @ hat
        mask = torch.tensor([e != cur for e in ent])
        sims = sims.masked_fill(~mask, -1e9)
        cur = ent[int(sims.argmax())]
    return cur


# ---------- prompting ----------
FEWSHOT = [
    ("Facts:\nBlade Runner | director | Ridley Scott\nRidley Scott | place of birth | South Shields",
     "Where was the director of the film Blade Runner born?", "South Shields"),
    ("Facts:\nThe Matrix | director | Lana Wachowski\nLana Wachowski | country of citizenship | American",
     "What is the nationality of the director of the film The Matrix?", "American"),
]


def build_para_context(row, tok, budget, sup_only=False):
    """The 20 MuSiQue paragraphs, supporting ones first (favourable to the prompt arm), under a budget.
    sup_only keeps just the gold supporting paragraphs: the strongest natural-language baseline, and the
    one that answers "does the store only win because the paragraph arm was truncated?"."""
    paras = row["paragraphs"]
    sup = [p for p in paras if p.get("is_supporting")]
    rest = [] if sup_only else [p for p in paras if not p.get("is_supporting")]
    blocks = ["%s: %s" % (p.get("title", ""), p.get("paragraph_text", "")) for p in sup + rest]
    out, used, truncated = [], 0, False
    for b in blocks:
        n = len(tok(b)["input_ids"])
        if used + n > budget:
            truncated = True
            break
        out.append(b)
        used += n
    return "\n".join(out), truncated


def build_fact_context(gold, distractors, tok, budget, rng):
    """Gold + real distractor facts, SHUFFLED (you do not know which are gold when the pool is large).
    Reports whether every gold fact survived the budget."""
    facts = ["%s | %s | %s" % t for t in list(gold) + list(distractors)]
    order = rng.permutation(len(facts))
    gold_set = {"%s | %s | %s" % t for t in gold}
    out, used, truncated = [], 0, False
    for i in order:
        f = facts[i]
        n = len(tok(f)["input_ids"])
        if used + n > budget:
            truncated = True
            break
        out.append(f)
        used += n
    gold_kept = len(gold_set & set(out)) == len(gold_set)
    return "\n".join(out), truncated, gold_kept


def gen_answer(llm, tok, device, prompt_text):
    ids = tok(prompt_text, return_tensors="pt", truncation=True, max_length=1900).to(device)
    with torch.no_grad():
        out = llm.generate(**ids, max_new_tokens=16, do_sample=False, pad_token_id=tok.eos_token_id)
    txt = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
    return txt.split("\n")[0].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="v79_logs/v79.json")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("===== V-79 MuSiQue: second mainstream benchmark (prereg 1fd4489) =====", flush=True)
    ds = load_dataset("dgslibisey/MuSiQue", split="validation", streaming=True)

    rows, seen, excluded_no_arrow, skipped_unanswerable = [], 0, 0, 0
    for r in ds:
        seen += 1
        if not r.get("answerable", True):
            skipped_unanswerable += 1
            continue
        tr = triples_of(r.get("question_decomposition") or [])
        if tr is None:
            excluded_no_arrow += 1
            continue
        paras = r.get("paragraphs")
        if not paras or not isinstance(paras, list) or "paragraph_text" not in paras[0]:
            print("HARNESS FAULT: unexpected paragraphs structure: %s" % str(paras)[:200], flush=True)
            sys.exit(2)
        rows.append({"question": r["question"], "answer": r["answer"], "triples": tr, "paragraphs": paras})
        if len(rows) >= a.n:
            break
    if not rows:
        print("HARNESS FAULT: no usable instances", flush=True)
        sys.exit(2)

    excl_rate = excluded_no_arrow / max(1, seen - skipped_unanswerable)
    hops = [len(r["triples"]) for r in rows]
    print("instances: %d of %d scanned | excluded (no '>>'): %d (%.3f) | unanswerable skipped: %d"
          % (len(rows), seen, excluded_no_arrow, excl_rate, skipped_unanswerable), flush=True)
    print("hop distribution: %s" % {h: hops.count(h) for h in sorted(set(hops))}, flush=True)

    covered = [i for i, r in enumerate(rows) if norm(r["triples"][-1][2]) == norm(r["answer"])]
    coverage = len(covered) / len(rows)
    print("coverage (final answer == last hop object): %.4f (%d/%d)" % (coverage, len(covered), len(rows)),
          flush=True)

    out = {"prereg": "1fd4489", "n": len(rows), "coverage": coverage, "excl_rate": excl_rate,
           "hop_dist": {str(h): hops.count(h) for h in sorted(set(hops))}, "arms": {}, "diag": {}}
    json.dump(out, open(a.out, "w"), indent=1)

    if coverage < 0.60:
        out["verdict"] = ("P4 USABILITY GATE FAILS: coverage %.4f < .60, so per the prereg the covered "
                          "slice is too thin and the port is declared UNUSABLE; no EM is read." % coverage)
        json.dump(out, open(a.out, "w"), indent=1)
        print("  >>> V-79 VERDICT: %s" % out["verdict"], flush=True)
        print("V79-DONE", flush=True)
        return

    # global pool of REAL distractor triples drawn from other instances
    all_tr = [(i, t) for i, r in enumerate(rows) for t in r["triples"]]

    tok = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    llm = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", trust_remote_code=True,
                                               torch_dtype=torch.float16).to(a.device).eval()
    shots = "\n\n".join("%s\nQuestion: %s\nAnswer: %s" % fs for fs in FEWSHOT)

    def distractors_for(i, k, rng):
        if k == 0:
            return []
        cand = [t for (j, t) in all_tr if j != i]
        pick = rng.choice(len(cand), size=min(k, len(cand)), replace=False)
        return [cand[p] for p in pick]

    # ---- paragraph arms (no pool factor; the paragraphs are the corpus's own evidence) ----
    # `prompt`     = all 20 paragraphs, supporting first, under the window -> truncation is a real property
    #                of the corpus (its evidence exceeds phi-2's window), reported not hidden.
    # `prompt_sup` = gold supporting paragraphs only, untruncated: the strongest natural-language baseline.
    for arm, sup_only in (("prompt", False), ("prompt_sup", True)):
        per_seed, trunc = [], []
        for seed in a.seeds:
            hits, tn = [], 0
            for r in rows:
                ctx, tr_ = build_para_context(r, tok, PARA_BUDGET, sup_only=sup_only)
                tn += int(tr_)
                hits.append(em(gen_answer(llm, tok, a.device, "%s\n\nContext:\n%s\nQuestion: %s\nAnswer:"
                                          % (shots, ctx, r["question"])), r["answer"]))
            hits = np.array(hits)
            per_seed.append(float(hits[covered].mean()))
            trunc.append(tn / len(rows))
            print("  %-10s seed %d  EM covered=%.4f (trunc %.3f)" % (arm, seed, per_seed[-1], trunc[-1]),
                  flush=True)
        out["arms"][arm] = {"covered": per_seed}
        out["diag"]["%s_trunc" % arm] = trunc
        json.dump(out, open(a.out, "w"), indent=1)

    # ---- arms with the pool factor ----
    for arm in ("prompt_ev", "hrr_gold"):
        for k in POOLS:
            per_seed, gk, tn_all = [], [], []
            for seed in a.seeds:
                rng = np.random.default_rng(seed)
                hits, gkept, tn = [], 0, 0
                for i, r in enumerate(rows):
                    dis = distractors_for(i, k, rng)
                    if arm == "hrr_gold":
                        pred = hrr_answer(r["triples"], dis, rng)
                    else:
                        body, tr_, gok = build_fact_context(r["triples"], dis, tok, 1200, rng)
                        tn += int(tr_)
                        gkept += int(gok)
                        pred = gen_answer(llm, tok, a.device, "%s\n\nFacts:\n%s\nQuestion: %s\nAnswer:"
                                          % (shots, body, r["question"]))
                    hits.append(em(pred, r["answer"]))
                hits = np.array(hits)
                per_seed.append(float(hits[covered].mean()))
                gk.append(gkept / len(rows))
                tn_all.append(tn / len(rows))
                # gold-kept/trunc are meaningless for the store arm (no context window is involved);
                # print n/a rather than a 0.000 that a later reader could mistake for a measurement.
                if arm == "hrr_gold":
                    print("  %-10s pool+%-3d seed %d  EM covered=%.4f (gold-kept n/a, trunc n/a)"
                          % (arm, k, seed, per_seed[-1]), flush=True)
                else:
                    print("  %-10s pool+%-3d seed %d  EM covered=%.4f (gold-kept %.3f, trunc %.3f)"
                          % (arm, k, seed, per_seed[-1], gk[-1], tn_all[-1]), flush=True)
            out["arms"]["%s_p%d" % (arm, k)] = {"covered": per_seed}
            if arm != "hrr_gold":
                out["diag"]["%s_p%d_goldkept" % (arm, k)] = gk
                out["diag"]["%s_p%d_trunc" % (arm, k)] = tn_all
            json.dump(out, open(a.out, "w"), indent=1)

    # ---- scoring, strictly as registered ----
    M = {k: float(np.mean(v["covered"])) for k, v in out["arms"].items()}
    print("", flush=True)
    print("===== V-79 SCORING (prereg 1fd4489) =====", flush=True)
    for k in sorted(M):
        print("  %-16s %.4f" % (k, M[k]), flush=True)

    notes = []
    if M["prompt"] >= 0.85:
        notes.append("CONDITION GUARD FIRES: single pass >= .85, no gain claimable")
    p1 = all(c > b > a_ for a_, b, c in zip(out["arms"]["prompt"]["covered"],
                                            out["arms"]["prompt_ev_p0"]["covered"],
                                            out["arms"]["hrr_gold_p0"]["covered"]))
    p2 = (M["hrr_gold_p0"] - M["prompt"]) >= 0.20
    p3 = (M["hrr_gold_p150"] >= 0.70) and ((M["prompt_ev_p0"] - M["prompt_ev_p150"]) >= 0.20)
    notes.append("P1 ordering 3/3: %s | P2 gain>=.20: %s (%.4f) | P3 pool separation: %s "
                 "(hrr@150=%.4f, prompt_ev drop=%.4f) | P4 coverage>=.60: True (%.4f)"
                 % (p1, p2, M["hrr_gold_p0"] - M["prompt"], p3, M["hrr_gold_p150"],
                    M["prompt_ev_p0"] - M["prompt_ev_p150"], coverage))
    if M["hrr_gold_p0"] <= M["prompt_ev_p0"]:
        notes.append("NEGATIVE READING APPLIES: the algebra adds nothing over in-context evidence on "
                     "MuSiQue, so the 2Wiki gain does NOT generalise and this scope limit goes in the paper")
    if not p3:
        notes.append("NEGATIVE READING APPLIES: the large-pool separation does not reproduce with real "
                     "distractor triples, so the manuscript's large-pool sentence must be narrowed")
    # ADDED AFTER THE PREREG, and labelled as such: prompt_sup is the strongest natural-language baseline
    # (gold paragraphs, untruncated). It is NOT one of P1-P4 and carries no registered threshold; it exists
    # so the store's margin cannot be attributed to having starved the paragraph arm.
    notes.append("ADDED (not registered): strongest natural-language baseline prompt_sup=%.4f vs "
                 "hrr_gold_p0=%.4f -> store %s it" % (M["prompt_sup"], M["hrr_gold_p0"],
                 "beats" if M["hrr_gold_p0"] > M["prompt_sup"] else "does NOT beat"))
    out["verdict"] = " ; ".join(notes)
    json.dump(out, open(a.out, "w"), indent=1)
    print("  >>> V-79 VERDICT: %s" % out["verdict"], flush=True)
    print("V79-DONE", flush=True)


if __name__ == "__main__":
    main()
