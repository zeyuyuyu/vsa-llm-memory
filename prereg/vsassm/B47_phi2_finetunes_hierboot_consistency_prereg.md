# B47 pre-registration: Phi-2 generative and full fine-tunes in arena C, hierarchical seed-by-question intervals for every arena margin, and a within-symbol consistency sweep at zero coherence

Date: 2026-11-19, about 05:55 UTC (13:55 on the box clock, Asia/Shanghai; US Eastern 2026-11-19 00:55). Written, with
the B47 run code (b47_ft.py, b47_tok.py, b47_sweep.py, b47_select.py, PACE scripts), before any B47 run reads a test
label or scores a test instance. What was done before this file is listed under "Done before registration".
b47_boot.py, b47_verdict.py, b47_indep.py and b47_handread.py are written after this file is committed and before any
B47 test output is read; they implement what is fixed here and change nothing in it.

## Why (referee, seventh reading, M3)

1. "Arena C, Phi-2: the LoRA classifier read-out never fit its training split (train <= 0.64) and Phi-2 last-token
   states have mean pairwise cosine 0.95-0.98, so the read-out design may be the bottleneck. Add a standard generative
   (answer-token) fine-tune and a full fine-tune."
2. "Seed variance is large (Qwen3-8B fine-tune seeds 0.538-0.698); the arena margins need hierarchical seed-by-question
   bootstrap intervals."
3. "The code-coherence sweep manipulates a shared direction, which centring removes; the failure actually observed in
   the label-free codes is within-symbol inconsistency across mentions at near-zero coherence. Manipulate that, in the
   registered B31 pipeline with development-selected epochs, so that Fig. 2b,c can leave exploratory status."

The result goes into the paper whatever it shows. The paper is not edited by B47.

## Division of labour with B44 (running) and B42 (finished)

- B44 (prereg/B44_..., running on PACE) has the Qwen3-8B learning curves (800 / 8,000 / 80,000), the held-out variant,
  the scratchpad LoRA, CoT, the centred / whitened imposed sweep, the binding-loss map at initialisation and the Fig.
  2b,c data under the B31 protocol (results/b44_fig2bc.json: the imposed coherence sweep with perm on B31's arena C,
  nothing trained, threshold on the training split; the arm points from B31's finals at dev-selected epochs; the Fig.
  2c k-sweep with the naming head at dev-selected epochs). **B47 does not rerun any of it and uses no B44 GPU.** B47
  reuses B44's data generator (b44_data), tokeniser format (b44_tok), LoRA code (b44_lora, unchanged for the
  scratchpad arm) and its seeds.
- The published coherence sweep "in the registered pipeline": B44 already ran it (raw/perm, B31 chain and threshold).
  B47 recomputes it in its own sweep code path as gate G1b (it must equal B44's per-seed accuracies), so that the
  coherence curve and the new consistency curve come from one code path, and assembles the replacement data for Fig.
  2b,c (b47_out/b47_fig2bc.json) from B44's registered outputs plus B47's, every number with its source file and
  verbatim source line. Nothing in the imposed sweeps is trained, so they have no epoch; the only fitted quantity is
  B31's closed-form threshold on the training split. "Development-selected" applies to every trained arm point placed
  on the panels (B31 / B42 / B44 finals at each seed's dev-selected epoch) and to the Fig. 2c naming heads (B44).
- B42 (finished): the label-free arms' arena-C accuracies, recoveries and geometry are inputs to (2) and (3).

## (1) Phi-2 arena-C fine-tunes (b47_ft.py; PACE, >= 80 GB GPUs)

Data (b47_tok.py = b44_tok's code; self-test out/selftest_tok.json, done): **k32** = B31's arena C (training split seed
42, 800 instances: the same 800 inputs as B31's naming head and equalised LoRA; dev seed 777, 500; test seed 4242,
500); **std** = B44's std stream, first 8,000 instances (seed 44042; B44's 8,000 set); dev and test = B31's (token ids
identical to k32's, checked). Phi-2 = microsoft/phi-2 snapshot 810d3678 (the box's, on PACE since B44).

| key | arm | trained | read-out |
|---|---|---|---|
| gen | generative answer-token LoRA (**primary**) | LoRA on every linear layer (q, k, v, dense, fc1, fc2), alpha 2r, dropout 0.05, bf16 base, fp32 adapters (B31) | input = arena text + "\nAnswer:" (B44's scratchpad input); target = the one token " yes" / " no"; loss = cross-entropy over the full vocabulary at the answer position; score = logit(" yes") - logit(" no"); prediction score > 0 |
| fullgen | full fine-tune, answer token (**primary**) | all 2.78B weights; fp32 master weights, bf16 autocast, gradient checkpointing | as gen |
| fullcls | full fine-tune, B31 classifier read-out (secondary: separates read-out from capacity) | all weights + head | B31's equalised head: one linear layer on [query-span mean, candidate-span mean, last-token state], standardised with the initial model's statistics; prediction = arg-max |
| scratch | B44's scratchpad LoRA on Phi-2 (secondary; B44's strongest baseline on Qwen3-8B) | b44_lora.Scratch / b44_lora.run unchanged | target " <bridge> F<bridge hex> <answer> <yes/no>" + EOS; greedy generation, last yes/no |

Recipe (B31 / B44): AdamW, weight decay 0, warm-up 6% then cosine to 0 over the planned steps, batch 8, gradient clip
1.0; budget 800 -> 8 epochs, 8,000 -> 4 epochs (B44); 8 evaluation points evenly spaced, each scoring the fixed training
subset (first 500 training instances) and dev; the dev-best state is kept; early stop at dev >= 0.995 on two
consecutive evaluation points (B44). The free arg-max over the vocabulary (share of " yes"/" no" and its accuracy) is
logged for gen / fullgen.

Grids (seed 42, train + dev only):
- 800: gen rank {8, 64} x lr {3e-5, 1e-4, 3e-4} (B31's Phi-2 grid); fullgen and fullcls lr {3e-6, 1e-5, 3e-5}; scratch
  rank 8 x lr {1e-4, 3e-4} (B44's grid).
- 8,000 (gen and fullgen only): the same rate grid; gen at the rank selected at 800.
- **Selection (B31's rule):** "fits" = training-subset accuracy >= 0.95 at some evaluation point; among configurations
  that fit, the highest dev accuracy at the best evaluation point; if none fits, the highest dev overall; ties within
  0.004 to the smaller rank, then the smaller rate.
- **Finals:** the selected configuration, seeds 42 / 123 / 456. The seed-42 final is the seed-42 tuning run of the
  selected configuration (its dev-best state is saved; the test split is scored once from it after the selection is
  written); seeds 123 and 456 train afresh and score the test split once from their dev-best state (--final).
- **Naming head at 8,000 (secondary):** the Phi-2 naming head (B31 arm, b44_feats + b44_name unchanged, fp16 Phi-2
  features as B31 / B44) trained on the same 8,000; if it does not run, the 8,000 margin uses the 800 naming head and
  says so.

### Bars and readings (1)

N = B31's Phi-2 naming head + VSA (perm, named), arena C, test per seed (1.000 / 1.000 / 1.000 at 800). B+(n) = the
higher test mean of gen and fullgen at n (favours the baseline). M = mean(N) - mean(B+), m_s per seed.

- **F1 fit:** for gen and fullgen at 800: FITS if the selected configuration's seed-42 run reaches training-subset
  >= 0.95; otherwise "did not fit after k configurations".
- **F2 margin (primary, 800):** STANDS: M >= 0.10 and m_s >= 0.10 in 3/3; CLOSED: M <= 0.03; NARROWED otherwise. The
  same rule at 8,000. Secondary: the same rule against the strongest of all four arms at 800.
- Readings:
  - STANDS at 800 -> the paper keeps the Phi-2 arena-C margin, now stated against the generative and full fine-tunes,
    and replaces "a fine-tune that did not fit its own training split" with the new arms' fit status (if they fit but
    stay near chance: "fitted its training split and did not generalise").
  - NARROWED / CLOSED -> the paper must report the smaller margin and say that the classifier read-out (fullcls vs
    fullgen tells whether read-out or capacity) was the bottleneck on Phi-2.
  - Secondary CLOSED (e.g. by the scratchpad) -> the paper must say, for Phi-2 as B44 says for Qwen3-8B, that a
    fine-tune supervised with the bridge symbol reaches the naming head in arena C.
  - 8,000 -> reported per size as in B44's learning curve; CLOSED at 8,000 -> "with 10x the data a Phi-2 fine-tune
    reaches the naming head".
- Diagnostic (B44's): test accuracy by label and by whether the candidate is a table-F filler (the shortcut "yes iff
  the candidate is an F value" scores 0.798 on this test split).

## (2) Hierarchical seed-by-question intervals (b47_boot.py; local CPU)

Unit: per-instance correctness on the test split, recomputed from the saved per-instance outputs exactly as each
verdict does (B31 / B42 memory arms: saved final cosine against the saved training threshold at each seed's
dev-selected epoch; probe and classifier fine-tunes: sign of the saved margin; scratchpad: the saved generation's last
yes/no; B47: as above). Labels regenerated by b31_struct / b44_data. Every recomputed per-seed accuracy must equal the
logged one (reported).

Margins (every arena margin the paper cites, plus B47's):
- B31 (b) naming head (perm, named) - the strongest equalised baseline, and (c) - LoRA + identity loss, and LoRA + id -
  LoRA: Phi-2 and Qwen3-8B, arenas A, B, C (18).
- B31 exploratory: Qwen3-8B middle-layer naming head - equalised LoRA, arenas A, B, C (if its outputs are on disk).
- Recoveries R = (acc - F) / (N - F) (F = B31 probe, N = B31 naming head) of every label-free arm, arenas C and A, both
  encoders: B42's vq, dkvb, gumbel, st, cw, bind+pc, bind+pz and B31's bind, bindpub, nce, soft.
- B44 leads and differences whose cells have finished when b47_boot runs (naming head - strongest baseline per variant
  and n; scratch - lora; CoT - naming head); rerun at the end; unfinished cells listed as such.
- B47: N - each new fine-tune at 800 and 8,000.

Intervals, all with 10,000 replicates, 95% percentile:
- **H-nest (primary, as asked):** draw 3 seed slots with replacement for each arm independently (seeds of different arms
  are not paired); for each slot draw the n test questions with replacement (one question draw per slot, shared by the
  two arms of a margin, which are scored on the same questions); statistic = mean over slots of the slot's difference
  (recovery: the ratio of the slot-averaged differences).
- **H-cross (check):** as H-nest but ONE question draw per replicate shared by all slots (questions are crossed with
  seeds in these designs: every seed answers the same questions, so the nested draw understates the question
  component).
- **t-seed (check):** Student-t over the three seed-level differences of seed-sorted pairs (2 df).
- B31's instance-only bootstrap is quoted beside them.
- **Reading:** an interval "excludes 0" (or a bar's threshold) only if both H-nest and H-cross do; disagreements are
  flagged. With three seeds the seed draw has ten distinct multisets, so H-* understate the between-seed variance by
  about a third; the t-seed interval is the conservative bound and is reported for every margin.
- Output: b47_out/b47_hierboot.json (per margin: point estimate, per-seed values, all intervals, the files read and their
  md5).

## (3) Consistency sweep (b47_sweep.py; PACE CPU)

B31's arena C (train 800 for the threshold, dev 500, test 500), codebook seeds 42 / 123 / 456 (B31's rows), B31's chain
(two hops, clean-up to the instance's own value mentions after each hop, the winner's code as the next key, final
cosine with the candidate's code), perm (primary) and comm, B31's closed-form threshold fitted on the training split.

- **Manipulation (primary, "mention"):** coherence c = 0 (different symbols' canonical codes are B31's random rows,
  measured cosine ~0.000); every mention m of symbol s in every instance gets k_m = rho t_s + sqrt(1 - rho^2) n_m, n_m a
  unit vector orthogonal to t_s, independent per mention (the query and the candidate included), so cos(k_m, t_s) =
  rho exactly and two mentions of one symbol have cosine ~ rho^2; rho in {1.0, 0.95, ..., 0.25, 0.2} (17 levels).
  Noise drawn once per (codebook seed, split) and shared by all levels.
- **Secondary "form":** consistent within a written form, inconsistent across forms (hexadecimal keys / query vs decimal
  values / candidate; cross-form cosine rho): the arena-C alias in isolation.
- **Secondary grid:** c in {0.1, 0.2, 0.3} x the 17 rho (perm, mention), to place each learned arm at its measured
  (different-symbol, same-symbol) geometry.
- **G1b (coherence sweep in the same code path):** rho = 1 at B44's 14 published c levels, perm and comm.
- Measured per cell and seed: test and dev accuracy, AUROC, clean-up correctness at hop 1 / hop 2 / both (winner's
  symbol = the gold one), accuracy given the chain right / wrong, and the geometry of the codes as bound (B31
  protocol: mean cosine between different symbols and between mentions of one symbol).

### Gates, bars and readings (3)

- **G1:** rho = 1, c = 0 reproduces B31's exact + VSA test accuracy per seed (perm 1.000 / 1.000 / 1.000; comm 0.932 /
  0.932 / 0.938). **G1b:** the rho = 1 coherence curve equals B44's results/b44_xtalk_sweep.json raw/perm and raw/comm
  per-seed test accuracies at all 14 levels (within 0.002 per seed, i.e. one instance). If a gate fails, no (3) number
  is read until the cause is found and written in a dated note.
- **Observed inconsistency:** the arena-C label-free arms that kept codes apart (different-symbol cosine <= 0.25) have
  mean same-symbol cosine at most 0.30 (B31 InfoNCE Phi-2 0.297, Qwen3-8B 0.298; B42 dkvb Phi-2 0.258; every other
  arm lower), so s_obs = 0.30.
- A(s) = mean test accuracy (perm, mention, c = 0) interpolated linearly in the measured same-symbol cosine s of the
  sweep's codes. Floor = Phi-2 probe 0.472 (the published criterion's floor); floor + 0.10 = 0.572.
- **SUFFICIENT** if A(0.30) <= 0.572: inconsistency of the measured size, with zero coherence and exactly orthogonal
  symbols, is by itself enough to put exact codes at the floor. **NOT SUFFICIENT** if A(0.30) >= 0.90. **PARTIAL**
  otherwise.
- Tolerances (the published criterion applied to consistency): s_tol = the smallest measured same-symbol cosine with
  mean accuracy >= 0.572; s_95 = the smallest with >= 0.95 (and the corresponding rho).
- Readings:
  - SUFFICIENT -> the paper may say that within-symbol inconsistency of the size measured in the label-free codes is
    enough, at zero coherence, to put the memory at the floor (causal, under isotropic per-mention noise), and Fig.
    2b,c can show the consistency axis as a registered result.
  - PARTIAL -> the paper must say that inconsistency of that size costs part of the accuracy, not all, and that the
    label-free arms fall below what isotropic inconsistency of their measured size predicts (their errors are
    structured); hop-wise clean-up rates are compared with B42's.
  - NOT SUFFICIENT -> the paper must drop any causal reading of the inconsistency and say that the label-free arms fail
    for a reason the measured geometry does not capture.
  - Grid: for every arena-C label-free arm, the sweep accuracy at the grid cell nearest its measured (different, same)
    geometry against its observed accuracy (report only).

## Predictions (stated so they can be wrong)

- P1 the tokenisation self-test and gates G1 / G1b pass.
- P2 gen at 800 does not fit its training split (training-subset < 0.95) in any of its six configurations.
- P3 fullgen at 800 fits (>= 0.95) in at least one configuration; fullcls too.
- P4 test means at 800: gen <= 0.65 and fullgen <= 0.65; F2 STANDS at 800.
- P5 at 8,000: gen and fullgen test means between 0.70 and 0.85 (near the 0.798 shortcut); STANDS at 8,000.
- P6 scratch at 800 >= 0.90 (secondary margin CLOSED).
- P7 every B31 margin with |M| >= 0.10 keeps an H-nest and H-cross interval that excludes 0; for the Qwen3-8B arena-C
  margin over the equalised LoRA (+0.352) the H-nest interval is at least twice as wide as B31's instance-only
  interval and its lower end stays above +0.20.
- P8 every arena-C label-free recovery (B42's seven arms, B31's bind / bindpub / nce) has an H-nest upper end below
  0.25 on both encoders, except at most one cell.
- P9 consistency (perm, mention, c = 0): mean accuracy >= 0.99 for rho >= 0.85; <= 0.572 for rho <= 0.45; A(0.30)
  between 0.60 and 0.80 -> PARTIAL.
- P10 the comm curve lies below the perm curve at every rho >= 0.6.
- P11 at the level nearest s = 0.30, hop-1 clean-up correctness in the sweep >= 0.20, above every label-free arm's
  measured hop-1 rate (B42: 0.03-0.18).
- P12 form model: accuracy >= the mention model's at every rho (perm).
- P13 the second scorer agrees with the main runs on >= 0.99 of instances in every cell.
- P14 total GPU-hours <= 40.

## Checks from other angles (in the verdict)

- **Second scorer (b47_indep.py):** labels re-derived by a regex walk of the raw text; every B47 fine-tune accuracy
  recomputed from the saved margins / generations (a second yes/no parser written separately); a numpy float64
  re-implementation of the arena-C chain run on the sweep's codes for seed 42 at five rho levels (per-instance
  agreement); a plain-loop re-implementation of H-nest for three margins with another random stream (intervals within
  0.01).
- **Hand-reading:** at least 10 test failures of the best new fine-tune at 800 (and at 8,000 if run), read in the raw
  text (shortcut or wrong hop), and 10 sweep failures at the level nearest s = 0.30 (which hop failed).
- **Literature:** after the verdict (fine-tuning read-outs: classification head vs answer-token; full vs parameter-
  efficient fine-tuning on small data; representation consistency / identity in neural-symbolic binding; hierarchical
  bootstrap for seeds x items), recorded as supports / conflicts / no precedent.

## Compute and budget

- **GPU:** PACE Phoenix only (every A800 GPU was occupied at 05:10 UTC; B47 takes no GPU that B44 or anyone else holds),
  free tier gts-nimam6 first, paid gts-nimam6-paid only if free-tier jobs do not start within about an hour, spend
  reported; <= 1.5 h segments (backfill), resumable.
- **Projection:** 800: gen 8 runs x 0.25 h, fullgen 5 x 0.4 h, fullcls 5 x 0.4 h, scratch 4 x 0.2 h; 8,000: gen 5 x 1 h,
  fullgen 5 x 1.3 h; naming head at 8,000 0.5 h; smokes 0.5 h. About 19 GPU-hours. **Cap 40 GPU-hours.**
- Cuts if the cap or the deadline is at risk, in this order: (1) fullcls; (2) scratch; (3) the naming head at 8,000;
  (4) the 8,000 grid narrowed to the rate selected at 800; (5) 8,000 entirely. Every cut is reported as "not run".
- **CPU:** the sweeps on PACE cpu-small (sbatch), the bootstrap on the local machine; core-hours reported.
- **Deadline:** delivery by about 2026-11-19 23:00 UTC; runs not finished by 21:30 UTC are reported as not run and no
  test output of an unfinished cell is read.

## Done before registration (no test split read, no test instance scored)

- Code: b47_ft.py, b47_tok.py, b47_sweep.py, b47_select.py, b47_pace.sbatch, b47_pace_cpu.sbatch,
  b47_sweep_cpu.sbatch, b47_pace_submit.sh, b47_smoke_pace.sh.
- **Tokenisation self-test** (PACE job 15849140, data/selftest_tok.json): k32 train / dev / test texts equal B31's
  arena C; std dev / test token ids equal k32's; the dictionary walk equals the label on every instance of every split
  (800 / 500 / 500 / 8,000 / 500 / 500); max lengths <= 492 tokens; content md5 (b44_tokmd5) k32 train f9e46b4d...,
  dev 2036b1f9..., test 9f0c5775..., std train 08fee1e5...; the k32 checksums equal those of B44's independently
  built Phi-2 k32 cache on PACE.
- **Sweep smoke on the training split only** (local CPU, seed 42, no dev or test instance): training-split accuracy with
  the threshold fitted on that split 1.000 (rho 1), 0.945 (rho 0.7), 0.618 (rho 0.5), and 0.978 for the form model at
  rho 0.5; it showed that hop correctness must be scored by the winner's symbol (a filler can occur twice in a table;
  fixed). Predictions P9 and P12 were written after this smoke and take it into account.
- **GPU smoke** (PACE job 15849141, --smoke 64: the first 64 training instances, one epoch, "dev" = the same 64; no dev
  or test split read) queued, not yet run at registration; its timings and memory go into a dated compute note.

## Files

- Code: b47_ft.py, b47_tok.py, b47_sweep.py, b47_select.py, b47_boot.py, b47_verdict.py, b47_indep.py, b47_handread.py,
  PACE scripts; reused unchanged: b44_lora.py, b44_tok.py, b44_data.py, b44_tokmd5.py, b44_xtalk.py (geometry, auroc),
  b44_feats.py, b44_name.py, b31_arms.py, b31_prep.py, b31_struct.py, b27_common.py, b27_arena/.
- PACE: /path/to/pace-scratch/b47 (data/, out/, logs/).
- Results: results/b47_verdict.txt, b47_out/b47_hierboot.json, b47_out/b47_fig2bc.json, b47_out/sweep_*.json, per-run
  outputs in b47_out/ (gzipped).
