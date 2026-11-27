# B40 dev record, part 1 (dev split only; committed before any test run)

Written 2026-11-19, about 04:45 box time. Every number below is on the dev split (task N 800 rows, 100 per trained
type; task R 600 rows, depth 1-3 small; label-free depth-2 dev = 200 rows), seed 0, selected by the rules registered in
prereg/B40_inforward_vsa_prereg.md. Raw files: b40_out/out/ (scan, rd, dev, devpred, dev_choices.json,
noisecal_dev.json) and b40_out/devsum/devsum_*.txt. Part 2 (the depth-3 label-free grid of amendment 2, still running)
is appended before ITS test runs; its selection rule is mechanical (best dev step) and cannot see any test number.

## Layer (VSA encoder, 100 epochs; dev read exact)

| layer | 8 | 12 | 14 | 16 | 17 | 18 | 20 | 24 |
|---|---|---|---|---|---|---|---|---|
| N (type and both operands exact) | 0.005 | 0.000 | 0.028 | 0.711 | 0.836 | 0.819 | **0.865** | 0.666 |
| R (entity per mention) | 1.000 | 1.000 | 1.000 | 1.000 | **1.000** | 1.000 | 1.000 | 1.000 |

Choice: N layer 20 (DE25 used 17: 0.836 here, 0.03 below 20); R layer 17 (tie everywhere -> nearest 17).

## Reader length (dev read exact)

- N VSA epochs 100 / 300 / 1000: 0.865 / 0.894 / **0.896** -> 1000 (DE25's). Digit error 0.021; type 1.000.
- N DISC epochs 30 / 100 / 300: 0.890 / 0.929 / **0.946** -> 300. Digit error 0.009; type 1.000.
- R VSA 100 / 300 / 1000: 1.000 all -> 100. R DISC 30 / 100 / 300: 1.000 all -> 30.
- Seeds 1 and 2 (chosen lengths): N VSA 0.895 / 0.896, N DISC 0.946 / 0.946, R VSA 1.000 / 1.000, R DISC 1.000 / 1.000.

## Write-back (1,000 steps; dev first-token argmax accuracy at steps 0, 250, 500, 750, 1000)

| run | curve | best (step) |
|---|---|---|
| N VSA recon init, lr 1e-3 | .172 .911 .906 .912 .907 | **0.9125 (750)** |
| N VSA zero init, lr 1e-3 | .174 .911 .910 .912 .904 | 0.9125 (750) |
| N VSA recon, lr 3e-4 | .172 .896 .907 .907 .907 | 0.9075 |
| N VSA zero, lr 3e-4 | .174 .901 .907 .910 .909 | 0.910 |
| N DISC lr 1e-3 | .182 .552 .901 .922 .941 | **0.941 (1000)** |
| N DISC lr 3e-4 | .182 .256 .312 .349 .411 | 0.411 |
| N LORA-1L lr 1e-3 | .199 .236 .237 .250 .245 | 0.250 |
| N LORA-1L lr 3e-4 | .199 .255 .239 .261 .278 | **0.2775 (1000)** |
| R VSA lr 1e-3 | .710 1.0 1.0 1.0 1.0 | **1.000 (250)** |
| R VSA lr 3e-4 | .710 .998 1.0 1.0 1.0 | 1.000 (500) |
| R DISC lr 1e-3 | .712 .995 .998 1.0 1.0 | **1.000 (750)** |
| R DISC lr 3e-4 | .712 .928 .973 .985 .990 | 0.990 |

Choices: N VSA DE25 reconstruction init, lr 1e-3 (ties zero init; first listed); N DISC lr 1e-3; N LORA-1L lr 3e-4;
R VSA lr 1e-3; R DISC lr 1e-3. The N VSA arm sits at its read ceiling (0.896 read exact -> 0.9125 answer; a few
misread operands still give the right answer, e.g. a misread digit that does not change x AND y). DISC's write-back
needs about 750-1,000 steps at lr 1e-3 to reach its read ceiling (0.946); at lr 3e-4 it does not get there in 1,000
steps.

## LoRA (rank 6, all layers, 3,000 steps; dev every 500 steps)

| run | curve (0, 500, ..., 3000) | best (step) |
|---|---|---|
| N lr 1e-4 | .191 .291 .318 .335 .338 .333 .331 | **0.3375 (2000)** |
| N lr 3e-4 | .191 .276 .301 .316 .318 .323 .318 | 0.3225 |
| N lr 1e-3 | .191 .095 .092 .095 .102 .098 .104 | 0.104 (diverged) |
| R lr 1e-4 | .687 .998 1.0 1.0 1.0 1.0 1.0 | **1.000 (1000)** |
| R lr 3e-4 | .687 .998 .998 1.0 1.0 1.0 1.0 | 1.000 (1500) |
| R lr 1e-3 | .687 .407 .098 .097 .170 .185 .178 | 0.407 (diverged) |

Choices: N lr 1e-4, R lr 1e-4. The step-0 value is the frozen LLM (no injection): N 0.191, R 0.687 (depth 1-3 pooled).

## Label-free sub-experiment, depth 2 (two fixed hops; 3,000 steps)

| run | curve | best |
|---|---|---|
| E2E-VSA lr 1e-3 | .900 .965 .975 .985 .980 .980 .980 | **0.985 (1500)** |
| E2E-VSA lr 3e-3 | .900 .965 .955 .965 .970 .960 .980 | 0.980 |
| E2E-TAB lr 1e-3 | .895 .960 .990 .990 .995 .980 .995 | **0.995 (2000)** |
| E2E-TAB lr 3e-3 | .895 .965 .970 .980 .970 .975 .970 | 0.980 |
| E2E-CONST lr 1e-3 | .900 .975 .995 .995 .995 .995 .995 | **0.995 (1000)** |
| E2E-CONST lr 3e-3 | .900 .975 .995 .995 .990 .995 .990 | 0.995 |

All lr 1e-3. The floor (a learned constant vector, no module information) reaches 0.995: no headroom at depth 2
(amendment 2 adds a depth-3 version).

## VSA-soft clean-up (task R, the trained seed-0 VSA arm; dev accuracy with generation)

beta 10: 0.688; beta 30: 1.000; beta 100: 1.000 -> beta 30.

## Read-noise calibration

See prereg/B40_amendment1.md (dev read accuracy of the seed-0 readers at rho 0-8; grid extended by rho 4 and 8).

## Predictions that dev already speaks to (not scored here; the test decides)

- P6 (DISC reads >= VSA reads on N): dev 0.946 vs 0.896.
- P3 (LORA in [0.20, 0.55] on N): dev 0.3375. P11 (LORA >= 0.90 on depth 1-3 small): dev 1.000.
- P14 depth-2 part: no headroom (floor 0.995), so "E2E-TAB learns" cannot hold at depth 2 by construction.

# B40 dev record, part 2: depth-3 label-free grid (amendment 2; dev split only; committed before its test runs)

Written 2026-11-19, 08:35 box time, after the depth-3 dev grid finished. The selection rule (best dev step; ties ->
the first listed lr) is mechanical; no depth-3 E2E test run has happened. (Part 1's arms were tested before this
part existed; the rule here could not use any test number.)

Three fixed hops, 3,000 steps; dev = the 200 depth-3 dev rows; first-token argmax at steps 0, 500, ..., 3000:

| run | curve | best (step) |
|---|---|---|
| E2E-VSA lr 1e-3 | .225 .900 .925 .950 .935 .945 .960 | **0.960 (3000)** |
| E2E-VSA lr 3e-3 | .225 .865 .925 .905 .900 .920 .920 | 0.925 |
| E2E-TAB lr 1e-3 | .230 .970 .980 .965 .985 .990 .995 | **0.995 (3000)** |
| E2E-TAB lr 3e-3 | .230 .925 .955 .970 .970 .965 .990 | 0.990 |
| E2E-CONST lr 1e-3 | .235 .915 .945 .965 .965 .970 .970 | **0.970 (2500)** |
| E2E-CONST lr 3e-3 | .235 .960 .965 .960 .970 .970 .970 | 0.970 |

Choices: lr 1e-3 for all three. The floor again has almost no headroom: one learned vector mixed in at layer 17 lifts
the frozen LLM from 0.235 to 0.970 on depth-3 dev (S6's "learns" bar, floor + 0.10, cannot be met).
