# B70 compute note 2 (after dev, before any dev choice is committed and before any test run)

Date: 2026-11-23, about 17:05 (machine clock).

- Dev on the box finished 17:03: 216 / 216 jobs, 0 failures; job files pulled back (tar md5 1b9b7daa..., identical on
  both sides).
- **Dev-phase reproduction check (registered): failed for SDMt on the three sentence encoders.** The 16-bit dev rows of
  NLMSc equal B62's on every (configuration, store) on all 7 encoders (0 mismatches). SDMt's equal B64's on the 4 language
  models (0 mismatches) but not on e5-base-v2, bge-small and MiniLM-L6 (225 of 4,536, 292 of 7,560 and 320 of 7,560 rows
  differ; largest accuracy difference 1.06 points). What I read: only these counts and the largest difference (from
  b70_run.py choose's reproduction lines and a diagnosis script over the 16-bit dev rows); the first choose call also wrote
  choice files from the box rows, which I did not open and which are set aside unused (b70_out/jobs_box_sdmt/, not
  committed).
- **Diagnosis:** B64 ran its SDMt jobs on the local machine (B64 compute note 1). A local re-run of one B70 SDMt dev job
  (e5, M = 256, V = 2, store 0, 16 and 64 bits per fact) reproduces B64's 16-bit rows exactly (0 of 504 differ) and
  differs from the box's run of the same job in 12 of 504 rows at 16 and 12 at 64. So the box and the local machine
  rank near-tied nearest locations differently: float32 similarity products of the float32 sentence-encoder codes are
  rounded differently by the two machines' BLAS builds (the language-model codes are float16, whose products are exact in
  float32, and reproduce on both). The code is identical (md5 note 1).
- **Remedy (procedure only; no rule, grid or arm changed):** the registered rule stops the run before any choice; the
  check's purpose is that B70's SDMt is B64's SDMt. So every SDMt job (dev now, test later; all 7 encoders, for one
  machine throughout) is re-run on the local machine, where B64's rows reproduce, with the same code
  (b70_run.py runplan on the SDMt lines of plan_dev.jsonl, 6 workers, nice 5, one thread each). The box's SDMt dev rows
  are discarded. NLMSc keeps the box (it reproduces there exactly). Choices are made only after the local SDMt dev rows
  pass the reproduction check with 0 mismatches; if they do not, SDMt is cut (registered cut order) and this is noted.
- Disclosed consequence for the paper: SDMt's chosen configuration and accuracy can depend on the machine in near-tie
  cases on float32 codes; the size is reported from this diagnosis (up to about 1 point per dev row).
