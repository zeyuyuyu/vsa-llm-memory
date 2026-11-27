# V5 compute and cut note (2026-11-23, 17:33 machine clock; before any V5 output is compared with the pipeline)

- The A800 box could not be used: the ssh link moved about 1.3 MB in five minutes and the upload of the RULER stores
  timed out, so no V5 job ever started there (only GPT's two scripts and a partial copy of one data file reached
  ~/v5_run; nothing ran). Every V5 run is local (16 cores): Part A Qwen3-8B (8 workers, started 17:16), Part B all four
  memories at k = 1, 8, 16 (7 workers, started 17:32), Part A Llama-3.1-8B after Qwen3-8B finishes.
- Cut rule, fixed now: Part A Llama-3.1-8B enters the comparison only if its run is final (rc 0, result file written)
  by 18:55; otherwise (a) is reported on Qwen3-8B only, as the pre-registration's short-time scope. Part B k = 16
  enters only if the run's k = 16 requirements are final by 18:55; Part B's script runs k = 1, 8, 16 in one process, so
  if it is not finished by then, it is stopped by PID, and (b) is reported at whatever k levels the result file holds
  (at least k = 1 and 8 if it has them), stated in the verdict. This note is committed before any comparison.

## At the cut (18:43, before any part B comparison)

- Final: dense (k = 1, 8, 16) and the BCH static table (k = 1, 8, 16), each run to completion (b_result.json);
  the cuckoo table's O-BCH version at k = 1, 8, 16 (its test windows finished; the script's own partial file).
- Not final, stopped by PID at 18:43 because neither could finish by 18:55: the sparse-block store (its development
  sweep was at budget 1,448 of 4,096 with 18 (budget, k) evaluations left, no test run) and the cuckoo table's O-LDPC
  version (development sweep at budget 64, k = 1). So "the cuckoo insertable table" (the better of O-BCH and O-LDPC)
  and H-SBC have no V5 requirement: by the pre-registration they count as "not met" for (b). O-BCH alone is compared
  as a secondary row, not as the criterion's memory.
- Part A is complete on both encoders (no cut).
