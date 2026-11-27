# V-157 PREREG — B3: composition-hard, extraction-easy, with every review defect fixed

Registered 2026-10-15, before any V-157 generation. Runner `b3.py`, frozen scorer `v157_score.py`, independent second
scorer `v157_indep.py`, launcher `run_v157.sh`, all in this commit.

## Why V-157 exists: V-156 is VOID and was stopped

An independent code review (`review_B.md`, finding B2-1) showed that the V-156 generator leaks the answer. 30% of
distractor values re-used chain symbols, so the answer was the only frequent symbol that is never a key. I reproduced
this with my own script before acting, importing the box copy of b2.py (md5 3668c897 = HEAD 54178e3) and using the
runner's own RNG streams. The rule "most frequent never-key value" answers 100% of instances at N >= 64 and 95-97% at
N = 16, with no composition. V-156 was stopped (launcher and runner killed) at seed 0, cell N128_H2, **before its
bar cell N128_H4 had been produced for any seed**, so the stop selects nothing on outcome. Its partial artifact is
kept as `b_logs/v156_s0_STOPPED_partial.json`, with 7 cells: N4/16/64/128 at H1 and N4/16/64 at H2. V-156 is
recorded VOID and is not scored.

## What changed (review finding -> fix)

| Review | Defect | Fix in b3.py |
|---|---|---|
| B2-1 | frequency shortcut | every table is a random bijection level L -> level L+1. Each symbol is a key at most once and a value at most once. CPU check on all 1,800 registered instances: freq heuristic = chance (N16 .05-.09, N64 .01-.03, N128 .00-.01) |
| B2-2 | CoT truncation grows with N | CoT budget 256+64H (512 at H4). The new-token count and the EOS flag are saved, and `cot_truncated` is reported per cell |
| B2-3 | composers got the generator's start and H | the LM parses the question into start + table order (`p_query`). The composers use only the LM's parse. `query_ok` is reported |
| B2-4 | VSA vs table differ beyond superposition | one shared abstention rule (key absent from that extracted table -> no answer) and one shared duplicate rule (first line wins). The gold-binding walkers are reported beside the extracted ones |
| B2-5/6 | scorer never run on a real artifact; no completeness check | the scorer asserts 12 cells x 13 records x 150. It is run on a smoke artifact before launch (below) |
| B2-8 | float thresholds | every bar compares integer counts out of 150 |
| B2-9 | FS symbols inside instances | the demo symbols are excluded from the instance pool |
| F0 | no raw outputs | every generation, parse and prediction is saved per instance |

Not fixed, and disclosed:
- B2-7. Role-marking does no work in a bijection task, because keys and values of one table are disjoint sets.
- The single-vector arm (vsa1) uses a non-unitary composite key (table (x) symbol), so its loss mixes superposition
  capacity with composite-key unbinding noise. Gold vsa1 is already .88 at N128 H1, where vsa_shard is 1.00. **No
  capacity claim is to be read from vsa1.**

## Design

Qwen2.5-7B-Instruct, bf16, greedy. Grid: N in {4, 16, 64, 128} x H in {1, 2, 4}; 50 instances per cell per seed;
seeds 0, 1, 2 (data seeds: fresh RNG `7000003*seed + 1009*N + H + 17`, disjoint from V-156). The three seeds run in
sequence on GPU 6, which is shared with another 35 GB job, so only one 7B process fits. Pooled n = 150 per cell. Nothing is filtered.

## Bars (frozen scorer; counts out of 150)

- **G0 validity gate**: regex_ceiling = 150 in every cell, AND freq_heuristic <= 15 in every cell with N >= 16. If G0
  fails, the run is VOID.
- **P1** (the title claim, in its weak form: an exact external composer fed by the LM's own parse beats the LM's own
  reasoning under load). At N128_H4, count(extract_vsa_shard) − count(cot) >= +30, and the paired bootstrap 95% CI
  is > 0.
- **P2** (the load axis is real for the LM alone): count(cot @ N4_H1) − count(cot @ N128_H4) >= 45.
- **P3** (the sharded VSA is as good as a table on the same extracted bindings): |count(shard) − count(dict)| <= 4 in
  every cell. Per-instance disagreements are printed.
- **P4** (algebra reference): gold_vsa_shard >= 147 in every cell, and gold_vsa1 at N128_H4 <= 75.

## Predictions, written before the run

- G0: PASS (checked on CPU over all 1,800 instances).
- P4: HOLDS (checked on CPU: gold shard 1.00 everywhere).
- P3: HOLDS. dict and VSA now share every rule except retrieval, and gold shard retrieval is exact at these loads.
- P2: I expect it to HOLD, but with low confidence. The V-156 CoT drop was partly truncation on leaked distractor
  lines, which this design removes.
- P1: **no confident prediction.** It turns on the LM's copy accuracy over 512 bindings (needed for all 4 hops)
  against its CoT lookup accuracy at 4 hops over 128 per table.

## How a verdict will be read

- P1 HOLDS and P3 HOLDS: "an exact external composer, table or sharded HRR (indistinguishable), fed only by the
  frozen LM's own parse, beats the LM's own CoT at high binding load". This is evidence for the title in its weak
  form. It is **not** evidence that VSA beats a table.
- P1 FAILS: the title claim is not supported on this task either.

Either way the verdict is only a conclusion after the multi-angle checks:
- `v157_indep.py`, an independent re-parse from raw outputs with a CoT faithfulness check;
- manual reading of 20 sampled traces per bar cell;
- the same grid on a second model (Qwen2.5-14B-Instruct or Qwen3-8B) and with a second prompt wording. These are
  registered separately as V-158+ before they run.

## Smoke

Run before this commit: `--Ns 4 16 --Hs 1 2 --per_cell 4`, with both scorers run on the artifact.
- The frozen scorer stopped at its completeness assert ("cell N4_H1 arm direct has 4"), as intended.
- The independent scorer re-derived every prediction from the raw text with 0 disagreements.
- I read the raw outputs. One N4_H2 extraction mis-copied table 2, giving it table 1's keys: a genuine LM error,
  correctly scored.
- One N16_H2 CoT trace rambled ("H97 is not in Table 1", which is false) and hit the 160-token budget. **After the
  smoke and before this commit, the CoT budget was raised from 64+48H to 256+64H** so that truncation cannot hide a
  late correct answer. `cot_truncated` is still reported per cell.
