# B21 pre-registration: RULER with a vector symbolic store in place of B19's fact table

Date: 2026-11-12. Written before any code for this experiment. Requested by the user ("需要跑实验的话，你就在pace上跑，
可以用付费账户"), so that the RULER result can be reported as a VSA-LLM result rather than a table result.

## Why
B19 (results/b19_verdict.txt) reached a 13-task average of 95.0 at 128k tokens (Qwen3-8B direct 77.3; leaderboard top
Jamba-1.5-large 95.1), but its memory was a plain fact table (prereg/B19_ruler_prereg.md: "more than the
256-dimensional vector store holds exactly, so this experiment uses a table, and says so"). B21 asks whether a
vector symbolic store, sharded so that each vector stays within its capacity, gives the same answers.

## Design (fixed here)
- Reader, prompts, chunking, grounding check, CWE/FWE counter and QA path: unchanged from B19 (ruler_rm.py).
- Data: the B19 test sets on PACE (official RULER scripts, Qwen3 tokenizer, seed 42, 100 samples per task and length),
  lengths 4k, 8k, 16k, 32k, 64k, 128k. Only the 9 tasks that use the memory are re-run: niah_single_1-3,
  niah_multikey_1-3, niah_multivalue, niah_multiquery, vt. CWE, FWE, qa_1, qa_2 do not use the memory; their B19
  predictions are reused for the 13-task average.
- Store (ruler_vsa.py): Fourier HRR, d = 4096, unit-modulus complex codes drawn from a seeded hash of each string.
  Every grounded pair (key, value) that B19 would put in its table, with value index j (the j-th distinct value read
  for that key), is written as bind(key, pos_j, value) into shard hash(key) mod S, S = ceil(pairs / 32).
  Read of key k: for j = 0, 1, ... unbind(key, pos_j) from its shard, clean up against the codes of all strings the
  reader wrote for this sample, accept the best row if its similarity is at least 0.5, stop at the first j below.
  NIAH answers list the accepted values of each queried key in order. VT resolves a variable by repeated reads
  (last accepted value; follow it while it is itself a variable), over the variables the reader named.
- Both stores answer from the same reader output in the same run; every prediction of both is saved, and the
  reader's raw per-chunk output is saved so that any store can be replayed on CPU.
- Scoring: official string_match_all (NIAH, VT), as B19.

## Bars
- P1: NIAH + VT mean accuracy of the VSA store >= 99.0 at every length (table, B19: 99.6-99.9).
- P2: the VSA and table predictions are identical on >= 99% of NIAH + VT samples at every length.
- P3: 13-task average at 128k with the VSA store (CWE/FWE/QA from B19) >= 94.0.
- Report the table arm of this run against B19 as a reproducibility check (same code path; greedy decoding).

## Predictions
P1-P3 hold. Disagreements, if any, fall on niah_multikey_3 at 64k and 128k, which has the most keys per sample.

## Compute
PACE gpu-h200, paid account (user approval 2026-11-12), one H200 per array task; budget estimate 10-15 GPU-hours.
Spend is reported in the verdict.

## Amendment 1 (2026-11-12, before any RULER sample was read by this code)
Two design changes found by the CPU self-test (`python3 ruler_vsa.py selftest`) before any run:
1. d = 8192 and at most 16 pairs per shard on average (was d = 4096, 32): with a few thousand candidate values in the
   clean-up, the expected maximum noise at 32 pairs per shard and d = 4096 (about 0.58) exceeds the stop threshold 0.5.
   At d = 8192 and 16 pairs per shard the noise s.d. is about 0.044.
2. Keys and values get separate codebooks. With one shared codebook a VT chain (b -> c, c -> 12345) read c as b,
   because binding commutes (the same alias as arena C; CLUTRR uses separate codebooks for the same reason).
Self-test after both changes: 0 of 50 / 154 / 150 probed keys differ from the table on synthetic stores of 50,
2,000 and 3,000 keys (7,444 pairs, up to 4 values per key; max shard load 37); the VT chain resolves correctly.
Bars and predictions unchanged.

## Amendment 2 (2026-11-12, compute only, before any B21 job started)
The first submission (job 15484950, gpu-h200 only) waited more than eight hours on priority and never started. It was
cancelled unrun and resubmitted to gpu-h200, gpu-h100, gpu-a100 and gpu-l40s (constraint H200|H100|A100|L40S, so no
V100, which lacks bf16), one GPU per array task, same script and parameters. B19 itself ran on two kinds of GPU (A800
for 4k-32k, H200 for 64k/128k). Greedy decoding on different hardware can change a few reader outputs, so the table-arm
reproducibility check against B19 may move slightly; P2 compares the two stores on the same reader output in the same
run and is unaffected. Each output records the GPU it ran on (from the job log). Design, bars and predictions unchanged.
