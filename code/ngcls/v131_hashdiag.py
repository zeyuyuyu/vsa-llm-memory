"""V-131 DIAGNOSTIC (not a scored arm): is the P0 miss on the `prompt` arm this project's own, already
documented context-assembly-order defect?

`v65.build_context` iterates `sup = set(zip(titles, sent_ids))`, so the order of the supporting blocks in the
context is the iteration order of a set of (str, int) tuples. Under randomized string hashing that order is a
fresh permutation per process. The banked V-65 run predates the standing PYTHONHASHSEED=0 rule (adopted at
V-123), so its 0.2533 is ONE DRAW from a permutation distribution -- the same effect the companion measured
as "context assembly order is worth about +-0.03".

This runs the prompt arm alone, once per PYTHONHASHSEED. If 0.2533 (banked) and 0.2400 (the PYTHONHASHSEED=0
re-run) both sit inside the observed spread, the P0 miss is that defect and not an environment change.

Run as:  PYTHONHASHSEED=<n> python v131_hashdiag.py    -> prints one RESULT line.
"""
import ast, os, sys

sys.path.insert(0, "/path/to/work/ngcls")
import torch
from v65 import FEWSHOT, build_context, em, gen_answer   # noqa: E402


def main():
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ds = load_dataset("framolfese/2WikiMultihopQA", split="validation", streaming=True)
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

        ev, sf, cx = parse(r["evidences"]), parse(r["supporting_facts"]), parse(r["context"])
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
        if len(rows) >= 300:
            break

    tok = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    llm = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", trust_remote_code=True,
                                               torch_dtype=torch.float16).to("cuda:0").eval()
    shots = "\n\n".join("%s\nQuestion: %s\nAnswer: %s" % fs for fs in FEWSHOT)

    hits, first_ctx = [], None
    for r in rows:
        ctx, _ = build_context(r, tok, 1200)
        if first_ctx is None:
            first_ctx = ctx[:120].replace("\n", " | ")
        hits.append(em(gen_answer(llm, tok, "cuda:0",
                                  "%s\n\nContext:\n%s\nQuestion: %s\nAnswer:" % (shots, ctx, r["question"])),
                       r["answer"]))
    print("RESULT hashseed=%s em=%.4f first_context_head=%s"
          % (os.environ.get("PYTHONHASHSEED"), sum(hits) / len(hits), first_ctx), flush=True)


if __name__ == "__main__":
    main()
