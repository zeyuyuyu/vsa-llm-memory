"""B43: tokens per sample of the paper's RULER reader (B21), recounted from the banked raw outputs and the RULER inputs,
for comparison with the Chain-of-Agents baseline's logged tokens (prereg/B43_natural_text_and_chunked_baseline_prereg.md).

python3 b43_cost.py RULER_DIR TOKENIZER_DIR OUT.json

Per sample: chunks = ruler_rm.chunks(context); prompt tokens of a chunk = the chat-templated FACT_PROMPT (thinking off)
around it (template overhead measured exactly on 2,000 chunks; the per-chunk count is overhead + the chunk's own
tokens, checked against full tokenisation); generated tokens = tokens of the banked raw output + 1 (end of turn, as
b14.Reader counts). Three accountings:
  nocache      every chunk of the sample read once (one question per sample),
  dedup_sample identical chunk prompts within the sample read once,
  dedup_task   identical chunk prompts within the (task, length) run read once (B21's cache, as run), divided by 100.
"""
import collections
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ruler_rm as R  # noqa: E402

NV = ["niah_single_1", "niah_single_2", "niah_single_3", "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
      "niah_multivalue", "niah_multiquery", "vt"]


def main(root, tokdir, out):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokdir)

    def full(c):
        t = tok.apply_chat_template([{"role": "user", "content": R.FACT_PROMPT.format(chunk=c)}], tokenize=False,
                                    add_generation_prompt=True, enable_thinking=False)
        return len(tok(t, add_special_tokens=False).input_ids)

    res = {}
    for L in (65536, 131072):
        b21 = collections.defaultdict(list)
        for f in glob.glob(f"{os.path.dirname(os.path.abspath(__file__))}/b21_dl/b21_test/{L}_*.jsonl"):
            for l in open(f):
                r = json.loads(l)
                b21[r["task"]].append(r)
        overhead, checked, maxdev = None, 0, 0
        acc = collections.defaultdict(list)
        for t in NV:
            rows = R.load(root, L, t, 100)
            seen_task = {}
            task_p = task_g = 0
            for pos, (s, b) in enumerate(zip(rows, b21[t])):
                assert s["index"] == b["index"] and s["outputs"] == b["outputs"]
                ctx, _ = R.split(s)
                cs = R.chunks(ctx)
                assert len(cs) == len(b["raw"]), (t, pos, len(cs), len(b["raw"]))
                ct = [len(x) for x in tok(cs, add_special_tokens=False).input_ids]
                gt = [len(x) + 1 for x in tok(b["raw"], add_special_tokens=False).input_ids]
                if overhead is None:
                    overhead = full(cs[0]) - ct[0]
                if checked < 2000:
                    for c, n in list(zip(cs, ct))[:200]:
                        maxdev = max(maxdev, abs(full(c) - (overhead + n)))
                        checked += 1
                pt = [overhead + n for n in ct]
                acc["nocache_prompt"].append(sum(pt))
                acc["nocache_gen"].append(sum(gt))
                uniq = {}
                for c, p, g in zip(cs, pt, gt):
                    uniq.setdefault(c, (p, g))
                acc["dedup_sample_prompt"].append(sum(p for p, _ in uniq.values()))
                acc["dedup_sample_gen"].append(sum(g for _, g in uniq.values()))
                for c, (p, g) in uniq.items():
                    if c not in seen_task:
                        seen_task[c] = 1
                        task_p += p
                        task_g += g
                acc["chunks"].append(len(cs))
            acc["dedup_task_prompt"] += [task_p / 100] * 100
            acc["dedup_task_gen"] += [task_g / 100] * 100
            print(L, t, "done", flush=True)
        res[str(L)] = {k: float(np.mean(v)) for k, v in acc.items()}
        res[str(L)]["template_overhead"] = overhead
        res[str(L)]["overhead_check_max_abs_dev"] = maxdev
        res[str(L)]["overhead_checked_chunks"] = checked
        print(L, json.dumps(res[str(L)]), flush=True)
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:4])
