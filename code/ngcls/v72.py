"""V-72: the bulky-evidence case (prereg c70f0df + addendum 07afab0). Each instance gets a pool of its own
gold triples plus K other instances' triples. Arms: whole pool in the prompt; HRR store over the pool with
two retrieved facts prompted; the store's own readout. Metrics: error rate and hallucination rate.
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
    picked = []
    for hop, (s, r, o) in enumerate(evidences):
        hat = unbind(mem, bind(vec[cur], vec[r]))
        cands = sorted({e for e in ents if e != cur})
        sims = torch.stack([torch.dot(hat, vec[c]) / (hat.norm() * vec[c].norm() + 1e-8) for c in cands])
        nxt = cands[int(sims.argmax())]
        picked.append((cur, r, nxt))
        cur = nxt
    return cur, picked


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
    ap.add_argument("--K", type=int, default=50)
    ap.add_argument("--out", default="v72_logs/v72.json")
    a = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    try:
        ds = load_dataset("framolfese/2WikiMultihopQA", split="validation", streaming=True)
        split_used = "validation"
    except Exception:
        ds = load_dataset("framolfese/2WikiMultihopQA", split="train", streaming=True)
        split_used = "train (validation unavailable)"
    print("split: %s" % split_used, flush=True)
    rows = {"compositional": [], "inference": []}
    for r in ds:
        t = r["type"]
        if t not in rows or len(rows[t]) >= a.n:
            if all(len(v) >= a.n for v in rows.values()):
                break
            continue
        def parse(x):
            if isinstance(x, str):
                try:
                    return ast.literal_eval(x)
                except Exception:
                    return None
            return x
        ev = parse(r["evidences"])
        if not ev or len(ev) < 2:
            continue
        rows[t].append({"question": r["question"], "answer": r["answer"], "evidences": ev})
    print("instances: compositional=%d inference=%d" % (len(rows["compositional"]), len(rows["inference"])), flush=True)

    tok = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    llm = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", trust_remote_code=True,
                                               torch_dtype=torch.float16).to(a.device).eval()

    REFS = {"compositional": {"prompt": 0.2533, "prompt_ev": 0.9500, "hrr_gold": 0.9744},
            "inference": {"prompt": 0.0933, "prompt_ev": 0.6833, "hrr_gold": 0.7622}}
    out = {"refs": REFS, "arms": {}, "diag": {}}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    shots = "\n\n".join("%s\nQuestion: %s\nAnswer: %s" % fs for fs in FEWSHOT)

    print("K=%d distractor instances per pool" % a.K, flush=True)
    for typ in ("compositional", "inference"):
        R = rows[typ]
        rng = np.random.default_rng(42)
        pools = []
        for idx in range(len(R)):
            others = rng.choice([j for j in range(len(R)) if j != idx], size=min(a.K, len(R) - 1), replace=False)
            pool = [tuple(t) for t in R[idx]["evidences"]]
            for j in others:
                pool += [tuple(t) for t in R[j]["evidences"]]
            rng.shuffle(pool)
            pools.append(pool)

        res = {}
        hits3 = []
        for idx, r in enumerate(R):
            pred, _ = hrr_answer(list(pools[idx]), r["question"], r["answer"], rng)
            hits3.append(em(pred, r["answer"]))
        res["hrr_gold_big"] = {"em": float(np.mean(hits3)), "hallucination": 0.0}
        print("  %-13s hrr_gold_big        EM=%.4f  hallu=0.0000 (by construction)"
              % (typ, res["hrr_gold_big"]["em"]), flush=True)

        def grounded(pred, pool):
            ents = {norm(x) for t in pool for x in (t[0], t[2])}
            return norm(pred) in ents

        hits1, hal1, trunc_n, ans_out = [], 0, 0, 0
        for idx, r in enumerate(R):
            allfacts = "\n".join("%s | %s | %s" % t for t in pools[idx])
            body = "%s\n\nFacts:\n%s\nQuestion: %s\nAnswer:" % (shots, allfacts, r["question"])
            ids = tok(body)["input_ids"]
            if len(ids) > 1900:
                trunc_n += 1
                if norm(r["answer"]) not in norm(tok.decode(ids[:1900])):
                    ans_out += 1
            pred = gen_answer(llm, tok, a.device, body)
            hits1.append(em(pred, r["answer"]))
            hal1 += int(not grounded(pred, pools[idx]))
        res["prompt_ev_big"] = {"em": float(np.mean(hits1)), "hallucination": hal1 / len(R),
                                "trunc_rate": trunc_n / len(R), "answer_truncated_out": ans_out / len(R)}
        print("  %-13s prompt_ev_big       EM=%.4f  hallu=%.4f  trunc=%.3f ans-out=%.3f"
              % (typ, res["prompt_ev_big"]["em"], res["prompt_ev_big"]["hallucination"],
                 res["prompt_ev_big"]["trunc_rate"], res["prompt_ev_big"]["answer_truncated_out"]), flush=True)

        hits2, hal2, hit2 = [], 0, 0
        for idx, r in enumerate(R):
            _, picked = hrr_answer(list(pools[idx]), r["question"], r["answer"], rng)
            facts = "\n".join("%s | %s | %s" % tuple(t) for t in picked)
            hit2 += int(any(norm(o) == norm(r["answer"]) for _, _, o in picked))
            pred = gen_answer(llm, tok, a.device,
                              "%s\n\nFacts:\n%s\nQuestion: %s\nAnswer:" % (shots, facts, r["question"]))
            hits2.append(em(pred, r["answer"]))
            hal2 += int(not grounded(pred, pools[idx]))
        res["hrr_then_prompt_big"] = {"em": float(np.mean(hits2)), "hallucination": hal2 / len(R),
                                      "retrieval_hit_rate": hit2 / len(R)}
        print("  %-13s hrr_then_prompt_big EM=%.4f  hallu=%.4f  hit-rate=%.4f"
              % (typ, res["hrr_then_prompt_big"]["em"], res["hrr_then_prompt_big"]["hallucination"],
                 res["hrr_then_prompt_big"]["retrieval_hit_rate"]), flush=True)
        out["arms"][typ] = res
        json.dump(out, open(a.out, "w"), indent=1)

    print("", flush=True)
    print("===== V-72 SCORING (prereg c70f0df + 07afab0) =====", flush=True)
    MIN = {"compositional": {"prompt_ev": 0.9500, "hrr_gold": 0.9744},
           "inference": {"prompt_ev": 0.6833, "hrr_gold": 0.7622}}
    notes = []
    for typ in ("compositional", "inference"):
        r = out["arms"][typ]
        p1 = MIN[typ]["prompt_ev"] - r["prompt_ev_big"]["em"] >= 0.10
        p2 = abs(r["hrr_gold_big"]["em"] - MIN[typ]["hrr_gold"]) <= 0.05
        p3 = r["hrr_then_prompt_big"]["em"] >= r["prompt_ev_big"]["em"] + 0.10
        notes.append("%s: P1 dump degrades >=.10: %s (%.3f->%.3f) | P2 store within .05: %s (%.3f) | P3 hybrid wins >=.10: %s (%.3f)"
                     % (typ, p1, MIN[typ]["prompt_ev"], r["prompt_ev_big"]["em"], p2,
                        r["hrr_gold_big"]["em"], p3, r["hrr_then_prompt_big"]["em"]))
        notes.append("%s: P4 hallucination dump=%.4f hybrid=%.4f readout=0" % (typ,
                     r["prompt_ev_big"]["hallucination"], r["hrr_then_prompt_big"]["hallucination"]))
    out["verdict"] = " || ".join(notes)
    json.dump(out, open(a.out, "w"), indent=1)
    print("  >>> V-72 VERDICT: %s" % out["verdict"], flush=True)
    print("V72-DONE", flush=True)


if __name__ == "__main__":
    main()
