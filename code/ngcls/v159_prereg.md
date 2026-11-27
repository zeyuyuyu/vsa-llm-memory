# V-159 PREREG — B3 replication on two more models (another scale, another family)

Registered 2026-10-15, before any V-159 generation, while V-157 (Qwen2.5-7B-Instruct) is still running and before any
of its bar cells exist. Runner `b3.py` **unchanged** (md5 9d32c789, the V-157 commit 182687a). Frozen scorer
`v159_score.py`: it is v157_score.py restricted to four cells, with a model tag and its own bootstrap seed; the bars are
identical. Launcher `run_v159_spark.sh`.

## Why

The multi-angle rule: no B result is a conclusion until it holds on another model. V-159 asks whether V-157's verdict,
whatever it is, is a property of the task or of Qwen2.5-7B.

## Design

- **Models** (bf16, greedy with each model's own generation_config):
  - (a) `Qwen/Qwen2.5-14B-Instruct`: same family, 2x scale, 32k context;
  - (b) `ibm-granite/granite-3.3-8b-instruct`: different family, 128k context.
  - OLMo-2-7B was considered and dropped before any run: its 4,096-token context is shorter than the largest prompt
    plus output, about 10.4k tokens at N128_H4.
- **Grid:** N ∈ {4, 128} × H ∈ {1, 4}. These four cells contain both bar cells, N4_H1 (P2) and N128_H4 (P1/P2).
- **Instances:** seeds 0, 1, 2 with b3.py's own RNG, so they are **identical to V-157's instances in these cells**
  and all three models pair instance by instance. 50 per cell per seed; n = 150 per cell.
- **Hardware:** DGX Spark (GB10, transformers 5.19), batch size 16. The same smoke on Spark and A800 (Qwen2.5-7B) gave
  byte-identical direct, extraction and query outputs on 16/16 instances and identical CoT on 13/16; all three
  divergent CoTs reached the correct answer. If the Spark is too slow, a model may run on an A800 instead. The
  hardware does not change any bar.
- **Smoke before each model's run:** `--seed 99` (a seed that is not registered), cells N4 × H1/H4, 4 per cell. No
  registered instance is previewed (review C3-2).

## Bars (identical to V-157, per model, counts out of 150; frozen scorer)

- **G0:** regex_ceiling = 150 in every cell, and freq_heuristic ≤ 15 at N128. Fails → that model's run is VOID.
- **P1:** count(extract_vsa_shard) − count(cot) at N128_H4 ≥ +30, and the paired bootstrap CI > 0.
- **P2:** count(cot @ N4_H1) − count(cot @ N128_H4) ≥ 45.
- **P3:** |count(shard) − count(dict)| ≤ 4 in each of the four cells.
- **P4:** gold_vsa_shard ≥ 147 in every cell, and gold_vsa1 at N128_H4 ≤ 75.

## How the three models' verdicts will be read

- V-157's P1/P2 verdict **replicates** if both V-159 models give the same verdict.
- If they disagree, the conclusion is model-dependent and will be reported that way, with each model's numbers.
- Before reading P1 for any model, read `query_ok` and `extract_truncated` at N128_H4 (review C3-3).
- Every P1 statement carries the compute asymmetry: at N128_H4 the extract route generates about 5,100 tokens
  against CoT's 512 (review C3-5).

## Predictions

No confident prediction for P1. G0 and P4 are fixed by code and HOLD. For P2 I expect the larger model to drop less
than the 7B.
