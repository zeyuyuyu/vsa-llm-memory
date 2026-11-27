"""B30b dev engine check (before registration; BABILong *training* split only, never the test release): does the B18
reader give the same outputs under vLLM (merged checkpoint) as under HF generate (b14.Reader, B18's settings)?

python3 b30b_devcheck.py prompts LENGTH K OUT.jsonl          cue-sentence prompts of training samples 4950..4950+K-1
python3 b30b_devcheck.py hf BASE ADAPTER PROMPTS OUT.jsonl   HF b14.Reader (base + LoRA merged on the GPU), bs 256,
                                                             budget 600,000 characters, greedy, 200 tokens (B18 test)
python3 b30b_devcheck.py compare PROMPTS HF.jsonl VLLM.jsonl identical text / identical parsed events
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "b30b_frozen"))
sys.path.insert(0, HERE)  # python -I leaves the script's directory off sys.path
import b18  # noqa: E402
import b30b_bab as BB  # noqa: E402

a = sys.argv
if a[1] == "prompts":
    L, K, out = a[2], int(a[3]), a[4]
    seen = set()
    with open(out, "w") as f:
        for t in BB.TASKS:
            for s in b18.load("train", L, t, K, offset=4950):
                for i, p in BB.sample_prompts(s["sents"]):
                    k = BB.key(p)
                    if k not in seen:
                        seen.add(k)
                        f.write(json.dumps({"id": k, "prompt": p}) + "\n")
    print(out, len(seen))
elif a[1] == "hf":
    import time
    import b14
    rows = [json.loads(l) for l in open(a[4])]
    rd = b14.Reader(a[2], a[3])
    t0 = time.time()
    outs = rd([r["prompt"] for r in rows], bs=256, budget=600000)
    with open(a[5], "w") as f:
        for r, (txt, n) in zip(rows, outs):
            f.write(json.dumps({"id": r["id"], "text": txt, "gen_tokens": n}) + "\n")
    print(f"hf {len(rows)} prompts in {time.time() - t0:.0f}s")
elif a[1] == "compare":
    pr = {json.loads(l)["id"]: json.loads(l)["prompt"] for l in open(a[2])}
    h = {json.loads(l)["id"]: json.loads(l)["text"] for l in open(a[3])}
    v = {json.loads(l)["id"]: json.loads(l)["text"] for l in open(a[4])}
    ks = sorted(set(h) & set(v))
    same_txt = sum(h[k].strip() == v[k].strip() for k in ks)
    sent = lambda k: pr[k].split("\nSentence: ", 1)[1].rsplit("\nEvents:", 1)[0]
    ev = lambda t, k: [e for e in b18.parse(t) if b18.grounded(e, sent(k))]
    same_ev = sum(ev(h[k], k) == ev(v[k], k) for k in ks)
    nonempty = sum(bool(ev(h[k], k)) for k in ks)
    print(f"{len(ks)} prompts: identical text {same_txt}, identical grounded events {same_ev} "
          f"({len(ks) - same_ev} differ; {nonempty} prompts with events under HF)")
    for k in [k for k in ks if ev(h[k], k) != ev(v[k], k)][:10]:
        print("  ", repr(sent(k))[:120], "| HF", ev(h[k], k), "| vLLM", ev(v[k], k))
