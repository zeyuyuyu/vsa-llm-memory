# B62 compute note 2 (dev choices; written and committed before any test run)

Written 2026-11-23 about 02:48 (machine clock).

1. **Dev run.** plan_dev (252 jobs) ran on the box 02:31:29-02:43:57 with 48 workers; 252 / 252 jobs finished, no failure
   or traceback in the log; job times up to 312 s (language models, M = 1,024). Pulled back to b62_out/jobs by rsync.
2. **Dev choices** (b62_run.py choose, B60's rule; results/b62_choice_{enc}.json, 36 per encoder = 12 R1 cells x 3
   arms). The training-split feature norms used by RLSc / RLScS are committed as results/b62_trainnorm_{enc}.npz.
   Composition over the 84 R1 cells (42 per V):
   - NLMSc, V = 2: col 42; V-1 39; whiten:pca 36, whiten:rp 3, center:rp 3; mu 1: 6, 0.5: 15, 0.25: 14, 0.1: 2,
     0.03: 5; last 39, avg 3; q 1: 17, 2: 19, 4: 6.
   - NLMSc, V = 16: flat 39; V 42; whiten:pca 34, center:pca 5, center:rp 3; mu 0.5: 3, 0.25: 12, 0.1: 15, 0.03: 12;
     last 36, avg 6; q 1: 36, 2: 6.
   - RLSc, V = 2: beta 1: 24, 0.999: 15, 0.99: 3; alpha 1: 22, 0.1: 10, 10: 6, 100: 3, 1e-4: 1; whiten:pca 33.
   - RLSc, V = 16: beta 1: 29, 0.999: 13; alpha 1: 24, 10: 12; flat 37, V 39, whiten:pca 39.
   - RLScS (p <= 31): mixed (beta 1 in 46 of 84).
3. **Dev-only observation** (dev rows; no test output): mean over the R1 cells of the chosen configuration's dev
   accuracy minus LAMc's (B60) dev accuracy, V = 2 / V = 16, points: NLMSc Qwen3-8B -0.16 / -1.84, Granite -0.94 /
   -1.13, Llama -1.33 / -3.79, Mistral -1.96 / -0.99, e5 +0.36 / -4.70, bge -0.87 / -2.11, MiniLM -1.79 / -3.14; RLSc
   +0.04 to +0.64 / -0.26 to +0.44 (its grid is three times LAMc's, with the same algebra at beta = 1, so a small dev
   gain from selection is expected); RLScS -7.7 to -22.6 / -17.3 to -23.2.
   Already visible on dev, and therefore not a test prediction: Q6 (beta = 1 in >= 75 of 84) cannot hold (53 of 84)
   and Q7's "mu = 1 in at least half of the V = 2 cells" cannot hold (6 of 42). Both are scored on these committed
   choices as registered; they are dev-choice predictions, and they fail.
4. **Test plan.** plan_test.jsonl: 640 jobs (7 encoders x V 2 / 16 x (20 + 5) test stores x the preps that the chosen
   configurations use), 3,150 evaluations (12 cells x 3 arms per store set); md5 5f9e2de3bc79ea3740c41bc9be957945. It runs
   once on the box (48 workers) after this note and the choices are committed. No test output exists yet.
