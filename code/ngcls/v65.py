"""V-65: 2WikiMultihopQA, frozen deployment (prereg 679939f, bars committed before this file).

Three arms over the same compositional two-hop instances: the model's single pass (prompt), the same gold
evidence verbalised in-context (prompt_ev), and an HRR store over the gold triples with string-identity codes
(hrr_gold). phi-2 frozen throughout. Primary metric: exact match on the covered slice (declared in prereg).
"""
import argparse, ast, json, os, re, string, sys
import numpy as np
import torch

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
VSA_DIM = 4096


def norm(s):
    s = s.lower().strip()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in string.punctuation)
    return " ".join(s.split())


def em(pred, gold):
    return int(norm(pred) == norm(gold))


# ---------- HRR over gold triples with string-identity codes ----------
def hrr_answer(evidences, question, answer, rng):
    """Follows the relation chain in gold-evidence order: hop1 object feeds hop2 subject."""
    strings = sorted({s for tr in evidences for s in tr})
    vec = {s: torch.tensor(rng.standard_normal(VSA_DIM) / np.sqrt(VSA_DIM)) for s in strings}
    def bind(a, b):
        return torch.fft.irfft(torch.fft.rfft(a) * torch.fft.rfft(b), n=VSA_DIM)
    def unbind(m, k):
        return torch.fft.irfft(torch.fft.rfft(m) * torch.conj(torch.fft.rfft(k)), n=VSA_DIM)
    mem = torch.zeros(VSA_DIM)
    for s, r, o in evidences:
        mem = mem + bind(bind(vec[s], vec[r]), vec[o])
    # chain: start at the first triple's subject, follow relations in evidence order
    ents = [tr[0] for tr in evidences] + [tr[2] for tr in evidences]
    cur = evidences[0][0]
    for hop, (s, r, o) in enumerate(evidences):
        hat = unbind(mem, bind(vec[cur], vec[r]))
        cands = sorted({e for e in ents if e != cur})
        sims = torch.stack([torch.dot(hat, vec[c]) / (hat.norm() * vec[c].norm() + 1e-8) for c in cands])
        cur = cands[int(sims.argmax())]
    return cur


# ---------- prompting ----------
def build_context(row, tok, budget):
    sup = set(zip(row["supporting_facts"]["title"], row["supporting_facts"]["sent_id"]))
    titles, sents = row["context"]["title"], row["context"]["content"]
    first = [f"{t}: {ss[i]}" for (t, i) in sup for tt, ss in zip(titles, sents) if tt == t and i < len(ss)]
    rest = [f"{t}: {' '.join(ss[:2])}" for t, ss in zip(titles, sents)
            if t not in {x[0] for x in sup}]
    out, used, truncated = [], 0, False
    for block in first + rest:
        n = len(tok(block)["input_ids"])
        if used + n > budget:
            truncated = True
            if block in first:
                continue
            break
        out.append(block); used += n
    return "\n".join(out), truncated


FEWSHOT = [
    ("Context:\nBlade Runner: Blade Runner is a 1982 film directed by Ridley Scott.\nRidley Scott: Ridley Scott was born in South Shields, England.",
     "Where was the director of the film Blade Runner born?", "South Shields, England"),
    ("Context:\nThe Matrix: The Matrix is a 1999 film directed by Lana Wachowski.\nLana Wachowski: Lana Wachowski is an American director.",
     "What is the nationality of the director of the film The Matrix?", "American"),
]


def gen_answer(llm, tok, device, prompt_text):
    ids = tok(prompt_text, return_tensors="pt", truncation=True, max_length=1900).to(device)
    with torch.no_grad():
        out = llm.generate(**ids, max_new_tokens=16, do_sample=False,
                           pad_token_id=tok.eos_token_id)
    txt = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
    return txt.split("\n")[0].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="v65_logs/v65.json")
    ap.add_argument("--dump_rows", default="", help="V-131: also write per-row hits here")
    a = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    try:
        ds = load_dataset("framolfese/2WikiMultihopQA", split="validation", streaming=True)
        split_used = "validation"
    except Exception:
        ds = load_dataset("framolfese/2WikiMultihopQA", split="train", streaming=True)
        split_used = "train (validation unavailable — deviation to note in ledger)"
    print("split: %s" % split_used, flush=True)
    rows = []
    for r in ds:
        if r["type"] != "compositional":
            continue
        def parse(x):
            if isinstance(x, str):
                try:
                    return ast.literal_eval(x)
                except Exception:
                    return None
            return x
        ev = parse(r["evidences"])
        sf = parse(r["supporting_facts"])
        cx = parse(r["context"])
        if not ev or len(ev) < 2 or not sf or not cx:
            continue
        if "content" not in cx:
            for k in ("sentences", "sent", "text"):
                if k in cx:
                    cx["content"] = cx[k]
                    break
        if "content" not in cx:
            continue
        rows.append({"question": r["question"], "answer": r["answer"], "evidences": ev,
                     "supporting_facts": sf, "context": cx})
        if len(rows) >= a.n:
            break
    print("instances: %d (type=compositional, dev)" % len(rows), flush=True)

    covered = [i for i, r in enumerate(rows) if any(norm(tr[2]) == norm(r["answer"]) for tr in r["evidences"])]
    print("answer-coverage of gold triples: %.4f (%d/%d)" % (len(covered) / len(rows), len(covered), len(rows)),
          flush=True)

    tok = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    llm = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", trust_remote_code=True,
                                               torch_dtype=torch.float16).to(a.device).eval()

    out = {"n": len(rows), "coverage": len(covered) / len(rows), "arms": {}, "trunc_rate": None}
    row_dump = {"n": len(rows), "covered": covered, "seeds": list(a.seeds), "arms": {}}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

    for arm in ("prompt", "prompt_ev", "hrr_gold"):
        per_seed_full, per_seed_cov = [], []
        for seed in a.seeds:
            rng = np.random.default_rng(seed)
            hits, trunc_n = [], 0
            for r in rows:
                if arm == "hrr_gold":
                    pred = hrr_answer(r["evidences"], r["question"], r["answer"], rng)
                else:
                    shots = "\n\n".join("%s\nQuestion: %s\nAnswer: %s" % fs for fs in FEWSHOT)
                    if arm == "prompt":
                        ctx, tr_ = build_context(r, tok, 1200)
                        trunc_n += int(tr_)
                        body = "Context:\n" + ctx
                    else:
                        body = "Facts:\n" + "\n".join("%s | %s | %s" % tuple(t) for t in r["evidences"])
                    pred = gen_answer(llm, tok, a.device,
                                      "%s\n\n%s\nQuestion: %s\nAnswer:" % (shots, body, r["question"]))
                hits.append(em(pred, r["answer"]))
            hits = np.array(hits)
            if a.dump_rows:
                row_dump["arms"].setdefault(arm, {})[str(seed)] = [int(h) for h in hits]
            per_seed_full.append(float(hits.mean()))
            per_seed_cov.append(float(hits[covered].mean()) if covered else float("nan"))
            if arm == "prompt":
                out["trunc_rate"] = trunc_n / len(rows)
            print("  %-10s seed %d  EM full=%.4f covered=%.4f" % (arm, seed, per_seed_full[-1], per_seed_cov[-1]),
                  flush=True)
        out["arms"][arm] = {"full": per_seed_full, "covered": per_seed_cov}
        json.dump(out, open(a.out, "w"), indent=1)

    print("", flush=True)
    print("===== V-65 SCORING (prereg 679939f) =====", flush=True)
    m = {k: float(np.mean(v["covered"])) for k, v in out["arms"].items()}
    print("  covered-slice EM: prompt=%.4f prompt_ev=%.4f hrr_gold=%.4f (coverage %.3f, trunc %.3f)"
          % (m["prompt"], m["prompt_ev"], m["hrr_gold"], out["coverage"], out["trunc_rate"] or -1), flush=True)
    notes = []
    if m["prompt"] >= 0.85:
        notes.append("CONDITION FAILS: single pass >= .85, no gain claimable")
    p1 = all(c >= b >= a_ for a_, b, c in zip(out["arms"]["prompt"]["covered"],
                                              out["arms"]["prompt_ev"]["covered"],
                                              out["arms"]["hrr_gold"]["covered"]))
    p2 = (m["hrr_gold"] - m["prompt"]) >= 0.20
    p3 = m["prompt"] < m["prompt_ev"] < m["hrr_gold"]
    notes.append("P1 ordering in 3/3: %s | P2 gain>=.20: %s (%.3f) | P3 strict middle: %s"
                 % (p1, p2, m["hrr_gold"] - m["prompt"], p3))
    if m["hrr_gold"] <= m["prompt_ev"]:
        notes.append("NEGATIVE READING APPLIES: algebra adds nothing over in-context evidence")
    out["verdict"] = " ; ".join(notes)
    json.dump(out, open(a.out, "w"), indent=1)
    print("  >>> V-65 VERDICT: %s" % out["verdict"], flush=True)
    if a.dump_rows:
        json.dump(row_dump, open(a.dump_rows, "w"))
        print("  per-row hits -> %s" % a.dump_rows, flush=True)
    print("V65-DONE", flush=True)


if __name__ == "__main__":
    main()
