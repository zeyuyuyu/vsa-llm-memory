"""V-131 Part C: MuSiQue without truncation -- was the baseline beaten, or just cut off?

The manuscript's MuSiQue single pass exceeds phi-2's window on 0.89 of instances and every +150 pooled
prompt is cut with only 46-51% of gold facts surviving. The 2026-09-11 audit corrected the ATTRIBUTION of
those collapses; it could not repair the baseline. This runs the same two arms with Qwen3-8B at a 32k window,
where nothing is cut, and reports truncation and gold-kept so the claim is checkable rather than asserted.

Everything that defines the measurement is imported from v79.py -- the row filter, the triple parser, the
context builders, the few-shot block, the normalisation and the metric -- so only the reader and the budget
change. The row set is gated on reproducing v79's banked n, hop distribution and coverage before any
generation happens.
"""
import argparse, json, os, sys, time

sys.path.insert(0, "/path/to/work/ngcls")
import numpy as np
import torch
from v79 import FEWSHOT, build_fact_context, build_para_context, em, norm, triples_of   # noqa: E402

BANKED = {"n": 212, "coverage": None, "prompt": 0.1650943396226415,
          "prompt_sup": 0.4009433962264151, "prompt_ev_p150": 0.1873, "hrr_gold_p150": 0.6556,
          "hrr_gold_p0": 0.9481, "prompt_trunc": 0.8868, "p150_goldkept": (0.4575, 0.5094)}


def gen_answer(llm, tok, device, prompt_text, max_len):
    ids = tok(prompt_text, return_tensors="pt", truncation=True, max_length=max_len).to(device)
    n_in = int(ids["input_ids"].shape[1])
    with torch.no_grad():
        o = llm.generate(**ids, max_new_tokens=16, do_sample=False, pad_token_id=tok.pad_token_id)
    return tok.decode(o[0][n_in:], skip_special_tokens=True).split("\n")[0], n_in


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    ap.add_argument("--para_budget", type=int, default=28000)
    ap.add_argument("--fact_budget", type=int, default=28000)
    ap.add_argument("--max_len", type=int, default=31000)
    ap.add_argument("--pool", type=int, default=150)
    ap.add_argument("--device", default="cuda:2")
    ap.add_argument("--out", default="v131_logs/v131c.json")
    a = ap.parse_args()

    if os.environ.get("PYTHONHASHSEED") != "0":
        print("ABORT: PYTHONHASHSEED must be 0 (standing rule, defect 024d9e0). Got %r."
              % os.environ.get("PYTHONHASHSEED"), flush=True)
        sys.exit(2)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("===== V-131 Part C: MuSiQue with a 32k reader, nothing truncated =====", flush=True)
    ds = load_dataset("dgslibisey/MuSiQue", split="validation", streaming=True)
    rows, seen, excluded, skipped = [], 0, 0, 0
    for r in ds:                                        # v79's filter, verbatim in effect
        seen += 1
        if not r.get("answerable", True):
            skipped += 1
            continue
        tr = triples_of(r.get("question_decomposition") or [])
        if tr is None:
            excluded += 1
            continue
        paras = r.get("paragraphs")
        if not paras or not isinstance(paras, list) or "paragraph_text" not in paras[0]:
            print("HARNESS FAULT: unexpected paragraphs structure", flush=True); sys.exit(2)
        rows.append({"question": r["question"], "answer": r["answer"], "triples": tr, "paragraphs": paras})
        if len(rows) >= a.n:
            break
    hops = [len(r["triples"]) for r in rows]
    covered = [i for i, r in enumerate(rows) if norm(r["triples"][-1][2]) == norm(r["answer"])]
    coverage = len(covered) / len(rows)
    print("  rows %d (banked %d) | hop dist %s | coverage %.4f"
          % (len(rows), BANKED["n"], {h: hops.count(h) for h in sorted(set(hops))}, coverage), flush=True)

    gate = (len(rows) == BANKED["n"])
    out = {"prereg": "6a2a52d", "config": vars(a), "n": len(rows), "coverage": coverage,
           "hop_dist": {str(h): hops.count(h) for h in sorted(set(hops))},
           "row_set_gate": gate, "banked": BANKED, "arms": {}, "diag": {}, "row_dumps": {}}
    json.dump(out, open(a.out, "w"), indent=1)
    if not gate:
        print("  ROW-SET GATE FAILS: %d rows against v79's banked %d. Nothing is read."
              % (len(rows), BANKED["n"]), flush=True)
        print("V131C-DONE", flush=True)
        return

    all_tr = [(i, t) for i, r in enumerate(rows) for t in r["triples"]]

    def distractors_for(i, k, rng):                     # v79's sampler, verbatim
        if k == 0:
            return []
        cand = [t for (j, t) in all_tr if j != i]
        pick = rng.choice(len(cand), size=min(k, len(cand)), replace=False)
        return [cand[p] for p in pick]

    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    llm = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16).to(a.device).eval()
    shots = "\n\n".join("%s\nQuestion: %s\nAnswer: %s" % fs for fs in FEWSHOT)
    print("  reader %s on %s, window %d" % (a.model, a.device, a.max_len), flush=True)

    # ---- arm 1: the single pass, all twenty paragraphs, nothing cut -------------------------------
    for arm, sup_only in (("single_pass", False), ("prompt_sup", True)):
        per_seed, trunc, ntok = [], [], []
        for seed in a.seeds:
            hits, tn, toks = [], 0, []
            t0 = time.time()
            for r in rows:
                ctx, tr_ = build_para_context(r, tok, a.para_budget, sup_only=sup_only)
                tn += int(tr_)
                pred, n_in = gen_answer(llm, tok, a.device,
                                        "%s\n\nContext:\n%s\nQuestion: %s\nAnswer:" % (shots, ctx, r["question"]),
                                        a.max_len)
                toks.append(n_in)
                hits.append(em(pred, r["answer"]))
            hits = np.array(hits)
            per_seed.append(float(hits[covered].mean()))
            trunc.append(tn / len(rows))
            ntok.append(float(np.mean(toks)))
            out["row_dumps"].setdefault(arm, {})[str(seed)] = [int(h) for h in hits]
            print("  %-12s seed %d  EM covered=%.4f  trunc=%.3f  mean prompt tokens=%.0f  (%.1f min)"
                  % (arm, seed, per_seed[-1], trunc[-1], ntok[-1], (time.time() - t0) / 60), flush=True)
        out["arms"][arm] = {"covered": per_seed}
        out["diag"]["%s_trunc" % arm] = trunc
        out["diag"]["%s_tokens" % arm] = ntok
        json.dump(out, open(a.out, "w"), indent=1)

    # ---- arm 2: the +150 pooled fact prompt, nothing cut ------------------------------------------
    arm = "prompt_ev_p%d" % a.pool
    per_seed, gk, tn_all, ntok = [], [], [], []
    for seed in a.seeds:
        rng = np.random.default_rng(seed)
        hits, gkept, tn, toks = [], 0, 0, []
        t0 = time.time()
        for i, r in enumerate(rows):
            dis = distractors_for(i, a.pool, rng)
            body, tr_, gok = build_fact_context(r["triples"], dis, tok, a.fact_budget, rng)
            tn += int(tr_); gkept += int(gok)
            pred, n_in = gen_answer(llm, tok, a.device,
                                    "%s\n\nFacts:\n%s\nQuestion: %s\nAnswer:" % (shots, body, r["question"]),
                                    a.max_len)
            toks.append(n_in)
            hits.append(em(pred, r["answer"]))
        hits = np.array(hits)
        per_seed.append(float(hits[covered].mean()))
        gk.append(gkept / len(rows)); tn_all.append(tn / len(rows)); ntok.append(float(np.mean(toks)))
        out["row_dumps"].setdefault(arm, {})[str(seed)] = [int(h) for h in hits]
        print("  %-12s seed %d  EM covered=%.4f  trunc=%.3f  gold_kept=%.3f  tokens=%.0f  (%.1f min)"
              % (arm, seed, per_seed[-1], tn_all[-1], gk[-1], ntok[-1], (time.time() - t0) / 60), flush=True)
    out["arms"][arm] = {"covered": per_seed}
    out["diag"]["%s_trunc" % arm] = tn_all
    out["diag"]["%s_goldkept" % arm] = gk
    out["diag"]["%s_tokens" % arm] = ntok
    out["covered_idx"] = covered
    json.dump(out, open(a.out, "w"), indent=1)
    print("V131C-DONE", flush=True)


if __name__ == "__main__":
    main()
