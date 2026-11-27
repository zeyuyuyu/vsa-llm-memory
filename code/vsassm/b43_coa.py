"""B43 part B (prereg/B43_natural_text_and_chunked_baseline_prereg.md): a Chain-of-Agents baseline (Zhang et al.,
NeurIPS 2024) with the paper's reader model, reading the same chunks as the paper's reader, with no symbolic store.

python3 b43_coa.py ruler OUT.jsonl --model DIR --root RULER_DIR --length L --tasks t1,t2 [--n 100] [--bs 16]
python3 b43_coa.py babilong OUT.jsonl --model DIR --length 128k --tasks qa1,...  [--n 100] [--bs 16]
python3 b43_coa.py windows --root RULER_DIR --length L --tasks ...   (CPU: window statistics, no model)

Protocol (fixed in the prereg):
- The text is cut exactly as the paper's reader cuts it (RULER: ruler_rm.chunks, 250-character sentence chunks;
  BABILong: b18's sentence split). Consecutive chunks are packed, in order, into windows of at most W = 8,192 Qwen3
  tokens (CoA's default window of 8k), joined by newlines. Every chunk the reader reads is in exactly one window.
- Worker i reads window i, the previous communication unit CU_{i-1} (empty for i = 1) and the question, with CoA's
  verbatim query-based worker prompt (Table 9 of arXiv 2406.02818), and writes CU_i.
- The manager reads CU_l and the question with CoA's verbatim manager prompt; {Task specific requirement} = the
  benchmark's own instruction (RULER: the first line of the input; BABILong: the official prompt's instruction).
  RULER's answer prefix starts the manager's turn (the RULER protocol, as for B19's direct arm).
- Qwen3 chat template, thinking off, greedy (temperature 0 in CoA), at most 1,024 new tokens per worker (CoA's
  setting); manager: 2 x RULER's tokens_to_generate (B19 direct), BABILong 64.
- Batches of samples move through their windows in lockstep; each call's prompt and generated tokens are counted.
"""
import argparse
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

W = 8192
WORKER_MAX = 1024

WORKER = ("{chunk}\n"
          "Here is the summary of the previous source text: {cu}\n"
          "Question: {q}\n"
          "You need to read current source text and summary of previous source text (if any) and generate a summary "
          "to include them both. Later, this summary will be used for other agents to answer the Query, if any. So "
          "please write the summary that can include the evidence for answering the Query:")
MANAGER = ("{req}\n"
           "The following are given passages. However, the source text is too long and has been summarized. You need "
           "to answer based on the summary:\n"
           "{cu}\n"
           "Question: {q}\n"
           "Answer:")


def pack(chunks, tok, w=W):
    """Consecutive chunks -> windows of at most w tokens (a single longer chunk would be its own window; none is)."""
    lens = [len(tok(c, add_special_tokens=False).input_ids) + 1 for c in chunks]
    out, cur, n = [], [], 0
    for c, l in zip(chunks, lens):
        if cur and n + l > w:
            out.append("\n".join(cur))
            cur, n = [], 0
        cur.append(c)
        n += l
    if cur:
        out.append("\n".join(cur))
    return out


# ---------------------------------------------------------------- RULER
GEN = {"niah": 128, "vt": 30}


def ruler_items(root, length, tasks, n):
    import ruler_rm as R
    for task in tasks:
        for pos, s in enumerate(R.load(root, length, task, n)):
            ctx, q = R.split(s)
            q = re.sub(r"^Question:\s*", "", q)  # VT's question line starts with "Question:"; CoA's template adds it
            req = s["input"].split("\n", 1)[0].strip()
            yield {"bench": "ruler", "task": task, "length": int(length), "pos": pos, "index": s["index"],
                   "outputs": s["outputs"],
                   "chunks": R.chunks(ctx), "q": q, "req": req, "prefix": s.get("answer_prefix", ""),
                   "gen": 2 * GEN["niah" if task.startswith("niah") else "vt"]}


# ---------------------------------------------------------------- BABILong
BAB_REQ = ("Answer the question about the people and objects in the text. Give only the answer: a single word (a "
           "place, a name, an object, a number word, yes / no / maybe), or a comma-separated list for 'what is ... "
           "carrying' (or 'nothing').")  # the answer instruction of the paper's whole-text arm (b18_llm.prep)


def babilong_items(length, tasks, n):
    sys.path.insert(0, os.path.join(HERE, "b30b_frozen"))
    import b18  # verbatim copy used by B30b; load() splits the text into the sentences the B18 reader reads
    for task in tasks:
        for i, s in enumerate(b18.load("test100", length, task, n)):
            yield {"bench": "babilong", "task": task, "length": length, "index": i, "target": s["target"],
                   "chunks": s["sents"], "q": s["question"], "req": BAB_REQ, "prefix": "", "gen": 64}


# ---------------------------------------------------------------- model
class LM:
    def __init__(self, path):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(path)
        self.tok.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda",
                                                          attn_implementation="sdpa").eval()
        eos = self.model.generation_config.eos_token_id
        self.eos = set(eos if isinstance(eos, list) else [eos])
        self.model.generation_config = GenerationConfig(do_sample=False, eos_token_id=sorted(self.eos),
                                                        pad_token_id=self.tok.pad_token_id)
        self.gpu = torch.cuda.get_device_name(0)
        self.splits = 0

    def text(self, prompt, prefix=""):
        return self.tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                            add_generation_prompt=True, enable_thinking=False) + prefix

    def gen(self, texts, max_new):
        """Greedy generation for a batch of chat texts; returns [(text, prompt_tokens, gen_tokens)]."""
        torch = self.torch
        try:
            enc = self.tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            with torch.no_grad():
                out = self.model.generate(**enc, max_new_tokens=max_new)[:, enc.input_ids.shape[1]:].tolist()
        except torch.cuda.OutOfMemoryError:
            if len(texts) == 1:
                raise
            self.splits += 1
            torch.cuda.empty_cache()
            h = len(texts) // 2
            return self.gen(texts[:h], max_new) + self.gen(texts[h:], max_new)
        res = []
        for t, ids in zip(texts, out):
            m = next((i + 1 for i, x in enumerate(ids) if x in self.eos), len(ids))
            res.append((self.tok.decode(ids[:m], skip_special_tokens=True).strip(),
                        len(self.tok(t, add_special_tokens=False).input_ids), m))
        return res


def run_batch(lm, items):
    """CoA over a batch of items in lockstep. Returns one record per item."""
    wins = [pack(it["chunks"], lm.tok) for it in items]
    cus = [""] * len(items)
    log = [[] for _ in items]
    ptok = [0] * len(items)
    gtok = [0] * len(items)
    for i in range(max(len(w) for w in wins)):
        act = [j for j in range(len(items)) if i < len(wins[j])]
        texts = [lm.text(WORKER.format(chunk=wins[j][i], cu=cus[j], q=items[j]["q"])) for j in act]
        for j, (txt, p, g) in zip(act, lm.gen(texts, WORKER_MAX)):
            cus[j] = txt
            log[j].append({"cu": txt, "p": p, "g": g, "capped": g >= WORKER_MAX})
            ptok[j] += p
            gtok[j] += g
    texts = [lm.text(MANAGER.format(req=it["req"], cu=cu, q=it["q"]), it["prefix"]) for it, cu in zip(items, cus)]
    outs, by_gen = [], {}  # the manager's budget differs by task: one call per distinct budget
    for k, it in enumerate(items):
        by_gen.setdefault(it["gen"], []).append(k)
    man = [None] * len(items)
    for g, ks in by_gen.items():
        for k, r in zip(ks, lm.gen([texts[k] for k in ks], g)):
            man[k] = r
    for k, it in enumerate(items):
        txt, p, g = man[k]
        rec = {kk: v for kk, v in it.items() if kk != "chunks"}
        rec.update(pred=txt, n_chunks=len(it["chunks"]), n_windows=len(wins[k]), workers=log[k],
                   manager={"p": p, "g": g}, prompt_tokens=ptok[k] + p, gen_tokens=gtok[k] + g, gpu=lm.gpu)
        outs.append(rec)
    return outs


def run(args):
    tasks = args.tasks.split(",")
    items = list(ruler_items(args.root, args.length, tasks, args.n) if args.mode == "ruler"
                 else babilong_items(args.length, tasks, args.n))
    done = set()
    if os.path.exists(args.out):
        done = {(r["task"], r.get("pos", r["index"])) for r in map(json.loads, open(args.out))}
    items = [it for it in items if (it["task"], it.get("pos", it["index"])) not in done]
    # execution only (B43 compute note): a job may take a slice of the samples (by file position) so that the rest of
    # a task whose job ran out of time can be split over several GPUs; prompts and decoding are unchanged
    items = [it for it in items if args.start <= it.get("pos", it["index"]) < args.stop]
    print(f"items to run: {len(items)} (done {len(done)})", flush=True)
    if not items:
        return
    lm = LM(args.model)
    print("gpu:", lm.gpu, flush=True)
    t0 = time.time()
    with open(args.out, "a") as f:
        for k in range(0, len(items), args.bs):
            t1 = time.time()
            for rec in run_batch(lm, items[k:k + args.bs]):
                f.write(json.dumps(rec) + "\n")
            f.flush()
            print(f"batch {k // args.bs}: {min(k + args.bs, len(items))}/{len(items)} in {time.time() - t1:.0f}s "
                  f"(total {time.time() - t0:.0f}s, OOM splits {lm.splits})", flush=True)


def windows(args):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    for it in ruler_items(args.root, args.length, args.tasks.split(","), args.n):
        w = pack(it["chunks"], tok)
        print(it["task"], it["index"], "chunks", len(it["chunks"]), "windows", len(w),
              "tokens", [len(tok(x, add_special_tokens=False).input_ids) for x in w][:3], flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["ruler", "babilong", "windows"])
    ap.add_argument("out", nargs="?")
    ap.add_argument("--model")
    ap.add_argument("--root")
    ap.add_argument("--length")
    ap.add_argument("--tasks")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--stop", type=int, default=10 ** 9)
    a = ap.parse_args()
    if a.mode == "windows":
        windows(a)
    else:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        run(a)
