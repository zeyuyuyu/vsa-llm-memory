"""V-100 (prereg 4bf0d4f): the cost-accuracy crossover.

No training. At each evidence scale, the SAME gold facts go to two routes:
  prompt : verbalised into the context, the frozen model answers
  store  : hashed into a fixed codebook, exact two-hop unbind-and-cleanup
Per query we record accuracy, prompt tokens, and wall-clock -- including the store's own build and query
time, so its cost is measured rather than assumed to be zero.

Both routes receive gold triples, as in the manuscript's bulky-evidence experiment. This measures how to USE
evidence, not how to obtain it.
"""
import argparse, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v85_data import load_rows, em, norm, FEWSHOT   # noqa: E402
import v97_store as ST                        # noqa: E402


def gold_triples(row):
    out = []
    for t in row.get("evidences", []):
        if isinstance(t, (list, tuple)) and len(t) >= 3 and all(str(x).strip() for x in t[:3]):
            out.append((str(t[0]).strip(), str(t[1]).strip(), str(t[2]).strip()))
    return out


def head_entity(row):
    q = norm(row["question"])
    cands = [t[0] for t in gold_triples(row)]
    for c in cands:
        if norm(c) and norm(c) in q:
            return c
    return cands[0] if cands else ""


def pooled(rows, i, n_facts, rng):
    """This instance's own gold triples, padded with other instances' to reach n_facts. The instance's own
    facts always survive, so accuracy loss is distraction rather than missing evidence."""
    own = gold_triples(rows[i])
    others = []
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    for j in idx:
        if j == i:
            continue
        others.extend(gold_triples(rows[j]))
        if len(own) + len(others) >= n_facts:
            break
    pool = own + others[:max(0, n_facts - len(own))]
    rng.shuffle(pool)
    return pool, own


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--scales", type=int, nargs="+", default=[8, 16, 32, 64, 128, 256, 512])
    ap.add_argument("--n_eval", type=int, default=120)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="v100.json")
    a = ap.parse_args()

    import random
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cache = os.environ.get("HF_HOME")
    print("===== V-100 cost-accuracy crossover (prereg 4bf0d4f) =====", flush=True)
    print("  reference: manuscript's evidence-in-prompt arm 0.950 | store w/ gold 0.974", flush=True)

    tok = AutoTokenizer.from_pretrained(a.model, cache_dir=cache, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    ev = load_rows(max(a.n_eval, 400), split="validation", cache_dir=cache)
    print("  pool drawn from %d instances; %d evaluated per scale" % (len(ev), a.n_eval), flush=True)

    llm = AutoModelForCausalLM.from_pretrained(a.model, cache_dir=cache, dtype=torch.bfloat16,
                                              device_map="cuda", trust_remote_code=True).eval()

    out = {"prereg": "4bf0d4f", "curve": []}
    for n_facts in a.scales:
        rng = random.Random(a.seed)
        p_ok, s_ok, p_tok, p_lat, s_lat, s_build = [], [], [], [], [], []
        for i in range(a.n_eval):
            r = ev[i]
            pool, own = pooled(ev, i, n_facts, rng)
            if not own:
                continue
            # The manuscript's evidence-in-prompt format, few-shot examples included. My first attempt
            # used a bare Facts/Question/Answer prompt and its gate failed at .7833 against the banked
            # .950 -- a protocol difference of my own making, not a property of the route.
            facts = "\n".join("%s | %s | %s" % t for t in pool)
            shots = "\n\n".join("%s\nQuestion: %s\nAnswer: %s" % f for f in FEWSHOT)
            prompt = ("%s\n\nContext:\n%s\nQuestion: %s\nAnswer:" % (shots, facts, r["question"]))
            ids = tok(prompt, return_tensors="pt", truncation=True, max_length=30000).to("cuda")
            p_tok.append(int(ids["input_ids"].shape[1]))
            t0 = time.time()
            with torch.no_grad():
                o = llm.generate(**ids, max_new_tokens=16, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
            p_lat.append(time.time() - t0)
            pred = tok.decode(o[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).split("\n")[0]
            p_ok.append(em(pred, r["answer"]))

            t0 = time.time()
            ans, _, _ = ST.answer_two_hop(pool, head_entity(r), "cuda")
            s_lat.append(time.time() - t0)
            s_ok.append(em(ans, r["answer"]))

        rec = {"n_facts": n_facts, "n": len(p_ok),
               "prompt_acc": float(np.mean(p_ok)), "store_acc": float(np.mean(s_ok)),
               "prompt_tokens": float(np.mean(p_tok)), "prompt_latency_s": float(np.mean(p_lat)),
               "store_latency_s": float(np.mean(s_lat))}
        out["curve"].append(rec)
        print("  facts=%-4d prompt acc %.4f (%.0f tok, %.3fs) | store acc %.4f (%.4fs)"
              % (n_facts, rec["prompt_acc"], rec["prompt_tokens"], rec["prompt_latency_s"],
                 rec["store_acc"], rec["store_latency_s"]), flush=True)
        json.dump(out, open(a.out, "w"), indent=1)

    # ---- scoring ----
    cur = out["curve"]
    print("", flush=True)
    print("===== V-100 SCORING =====", flush=True)
    p1 = abs(cur[0]["prompt_acc"] - 0.950) <= 0.05
    notes = ["P1 gate: smallest scale prompt acc %.4f vs 0.950: %s" % (cur[0]["prompt_acc"], p1)]
    if not p1:
        notes.append("GATE FAILS: this is not the manuscript's evidence-in-prompt protocol, so no reading "
                     "is taken from the curve.")
        out["verdict"] = " ; ".join(notes)
        json.dump(out, open(a.out, "w"), indent=1)
        print("  >>> %s" % out["verdict"], flush=True)
        print("V100-DONE", flush=True)
        return

    tok_growth = cur[-1]["prompt_tokens"] / max(1e-9, cur[0]["prompt_tokens"])
    scale_growth = cur[-1]["n_facts"] / cur[0]["n_facts"]
    store_growth = cur[-1]["store_latency_s"] / max(1e-9, cur[0]["store_latency_s"])
    p2 = tok_growth >= 0.9 * scale_growth and store_growth <= 2.0
    notes.append("P2 prompt tokens grow %.1fx over a %.0fx scale increase while store query cost grows "
                 "%.2fx: %s" % (tok_growth, scale_growth, store_growth, p2))

    cross = [c for c in cur if c["store_acc"] > c["prompt_acc"]]
    if cross:
        c = cross[0]
        p3 = c["n_facts"] <= 256
        notes.append("P3 crossover at %d facts (store %.4f > prompt %.4f), <= 256: %s"
                     % (c["n_facts"], c["store_acc"], c["prompt_acc"], p3))
        notes.append("at the crossover: prompt %.0f tokens and %.3fs against the store's %.4fs -- "
                     "token ratio %.0fx, latency ratio %.0fx"
                     % (c["prompt_tokens"], c["prompt_latency_s"], c["store_latency_s"],
                        c["prompt_tokens"] / 1.0, c["prompt_latency_s"] / max(1e-9, c["store_latency_s"])))
    else:
        p3 = False
        notes.append("P3 NO CROSSOVER up to %d facts: the store never overtakes the prompt on accuracy, so "
                     "the regime where it pays is further out than any realistic prompt and the efficiency "
                     "argument is academic." % cur[-1]["n_facts"])
    degr = cur[-1]["store_acc"] < cur[0]["store_acc"] - 0.10
    if degr:
        notes.append("STORE ALSO DEGRADES with scale (%.4f to %.4f), so it has a capacity limit in the same "
                     "regime and neither route survives -- which the manuscript's capacity sketch predicts."
                     % (cur[0]["store_acc"], cur[-1]["store_acc"]))
    out["verdict"] = " ; ".join(notes)
    json.dump(out, open(a.out, "w"), indent=1)
    print("  >>> V-100 VERDICT: %s" % out["verdict"], flush=True)
    print("V100-DONE", flush=True)


if __name__ == "__main__":
    main()
