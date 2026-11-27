# B70 compute note 3 (dev choices, before any test run)

Date: 2026-11-23, about 17:15 (machine clock).

- **Local SDMt dev re-run (note 2's remedy):** 72 / 72 jobs, 17:05-17:10, local, 6 workers, one thread each, nice 5, no
  failure. The reproduction check again passes on the 4 language models (0 mismatches; their box rows and local rows are
  also identical, 0 of 39,312) and again **fails on the three sentence encoders**, now in fewer rows (e5 95 of 4,536, bge
  161 of 7,560, MiniLM 153 of 7,560; largest accuracy difference 2.7 points on one store and configuration).
- **Diagnosis, completed (two dev stores, dev data only):** the same code run with B64's own budget set (4, 8, 16)
  instead of B70's (16, 32 or 64) reproduces B64's 16-bit rows on MiniLM-L6 (M = 256, V = 16, store 0: 0 of 504 differ;
  with B70's set 6 of 504) and nearly on e5-base-v2 (M = 1,024, V = 2, store 0: 1 of 504; with B70's set 43 of 504). So
  b64_run.run_sdmt's query-side nearest-location search depends on the set of budgets in one call (its blocks advance to
  the sizes N that the requested budgets need, so the float32 similarity blocks and the tie order of near-equal cosines
  differ), on top of the machine dependence of note 2. On float16 language-model codes the products are exact and
  nothing depends on the path. This is a property of B64's implementation (its own R1 numbers have the same sensitivity
  in near-tie cases), not a change of the memory.
- **Registered rule applied:** "a mismatch stops the run before any choice is made", and the registered cut order allows
  SDMt to be cut. **SDMt and BEST7 are cut on the three sentence encoders** (b70_run.CUT_SDMT; plan and choice code skip
  them; no SDMt choice made for them). They stay on the 4 language models, where the check passes exactly on both
  machines. Everything else is unchanged; P5 (STORE - BEST7), VSAR - SDMt, STORE - SDMt and VSAR - BEST7 are reported on
  the 4 language models only; P4 (STORE - BEST36) covers all 7.
- **Dev choices** (results/b70_choice_{enc}.json, md5 qwen8 73452e91, b46 ab7f13d6, llama 19044739, mistral b8faebb7,
  e5 7836ddb5, bge 4c5a233a, minilm 167274b1; b70_run.py md5 037c173d after the cut lines). NLMSc at 32 / 64 bits per
  fact (20 cells): whiten in 18, principal features in 12 (at most n numbers per column, so 4 cells use under half of
  the budget, e.g. e5 M = 256 at 64 bits per fact uses 800 of 16,384 bits), last iterate in 18. SDMt (language models,
  9 cells): every chosen configuration uses all 5,621 training-code locations (6-26 kbit used of 8-66 kbit); BEST7 = SDMt
  in the 6 M = 256 cells, IDX in the 3 M = 1,024 cells.
- Sensitivity line (Q9) from these dev choices is computed by the verdict.
- Next: test once. NLMSc test jobs on the box (as its dev); SDMt test jobs (language models) locally (as its dev).
