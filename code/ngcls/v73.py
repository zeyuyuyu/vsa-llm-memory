"""V-73: the capacity curve (prereg 1c8bd0d). Readout arm only, K in {10,50,100,200,400}.
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


def hrr_answer_pool(store, chain, rng):
    """Store = pooled triples (bound and superposed); chain spec = the instance's own gold evidence."""
    strings = sorted({x for tr in store for x in tr})
    vec = {t: torch.tensor(rng.standard_normal(VSA_DIM) / np.sqrt(VSA_DIM)) for t in strings}
    def bind(a, b):
        return torch.fft.irfft(torch.fft.rfft(a) * torch.fft.rfft(b), n=VSA_DIM)
    def unbind(m, k):
        return torch.fft.irfft(torch.fft.rfft(m) * torch.conj(torch.fft.rfft(k)), n=VSA_DIM)
    mem = torch.zeros(VSA_DIM)
    for s_, r_, o_ in store:
        mem = mem + bind(bind(vec[s_], vec[r_]), vec[o_])
    ents = sorted({x for tr in store for x in (tr[0], tr[2])})
    cur, picked = chain[0][0], []
    for s_, r_, o_ in chain:
        if cur not in vec or r_ not in vec:
            return None, picked
        hat = unbind(mem, bind(vec[cur], vec[r_]))
        cands = [e for e in ents if e != cur]
        sims = torch.stack([torch.dot(hat, vec[c]) / (hat.norm() * vec[c].norm() + 1e-8) for c in cands])
        nxt = cands[int(sims.argmax())]
        picked.append((cur, r_, nxt))
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
    ap.add_argument("--out", default="v73_logs/v73.json")
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

    out = {"Ks": {}, "d": VSA_DIM}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    for K in (10, 50, 100, 200, 400):
        out["Ks"][str(K)] = {}
        for typ in ("compositional", "inference"):
            R = rows[typ]
            rng = np.random.default_rng(42)
            ems, sizes = [], []
            for idx in range(len(R)):
                others = rng.choice([j for j in range(len(R)) if j != idx],
                                    size=min(K, len(R) - 1), replace=False)
                pool = [tuple(t) for t in R[idx]["evidences"]]
                for j in others:
                    pool += [tuple(t) for t in R[j]["evidences"]]
                rng.shuffle(pool)
                sizes.append(len(pool))
                pred, _ = hrr_answer_pool(pool, [tuple(t) for t in R[idx]["evidences"]], rng)
                ems.append(em(pred, R[idx]["answer"]) if pred is not None else 0)
            out["Ks"][str(K)][typ] = {"em": float(np.mean(ems)), "mean_bindings": float(np.mean(sizes))}
            print("  K=%-4d %-13s EM=%.4f  bindings=%.1f" % (K, typ, float(np.mean(ems)), float(np.mean(sizes))), flush=True)
            json.dump(out, open(a.out, "w"), indent=1)

    print("", flush=True)
    print("===== V-73 SCORING (prereg 1c8bd0d) =====", flush=True)
    notes = []
    for typ in ("compositional", "inference"):
        seq = [out["Ks"][str(K)][typ]["em"] for K in (10, 50, 100, 200, 400)]
        mono = all(seq[i] >= seq[i+1] - 0.02 for i in range(len(seq)-1))
        notes.append("%s: P1 monotone(+-.02): %s %s" % (typ, mono, [round(x,3) for x in seq]))
    notes.append("P2 compositional K=400 >= .80: %s (%.4f)" % (out["Ks"]["400"]["compositional"]["em"] >= 0.80, out["Ks"]["400"]["compositional"]["em"]))
    notes.append("P3 K=50 reproduces V-72b: %s (%.4f/%.4f vs .9533/.7233)" % (
        abs(out["Ks"]["50"]["compositional"]["em"] - 0.9533) < 5e-3 and abs(out["Ks"]["50"]["inference"]["em"] - 0.7233) < 5e-3,
        out["Ks"]["50"]["compositional"]["em"], out["Ks"]["50"]["inference"]["em"]))
    out["verdict"] = " | ".join(notes)
    json.dump(out, open(a.out, "w"), indent=1)
    print("  >>> V-73 VERDICT: %s" % out["verdict"], flush=True)
    print("V73-DONE", flush=True)


if __name__ == "__main__":
    main()
