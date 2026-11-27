# B50 pre-registration: a standard out-of-distribution gate for B40's discrete pipeline at a matched open rate, and chain-of-thought with a 2,048-token budget

Date: 2026-11-19 (box clock Asia/Shanghai; US Eastern -13 h). Extension of B40 (prereg/B40_inforward_vsa_prereg.md and
its amendments, dev record and verdict). Written, with the code (b50_run.py), the calibration rule (b50_calib.py), the
verdict generator (b50_verdict.py), the independent score check (b50_check_scores.py), the hand-read sampler (b50_handread.py), the queue generator
(b50_queues.py) and the worker (b50_worker.sh),
before any B50 run of any kind. Done before this file: reading B40's committed record. A smoke run of the B50 code
paths on TRAINING rows only follows this commit and precedes every dev and test run; it and any code change it forces
are recorded in prereg/B50_smoke_record.md (B40's practice).

What I have seen of B40's test split (all of it already in B40's committed verdict and hand-read): accuracies and
gate-open rates at the 0.8 threshold (VSA gate open on 2.9% of addition and 0% of division items, 99.87% of trained-type
items; DISC's softmax gate open on 100% of addition, 34.4% of division, 100% of trained-type items), and that 517 of the
2,500 task-N CoT outputs and 85 of the 2,000 task-R CoT outputs end without "Final Answer" at 512 tokens. I have NOT
looked at any per-item score of DISC (softmax probabilities, logits, states) on held-out test items, and no
held-out-type item is in any dev or training split.

## Why

B40's secondary result S2: on the two operation types never trained (addition, integer division), the VSA module's
similarity gate (largest dot product of the decoded HRR vector with the ten type atoms > 0.8, DE25's rule) stayed shut
and the arm stayed near BASE (addition 0.971 vs BASE 1.000; division 0.960 vs 0.960), while DISC's gate (softmax
probability of the type head > 0.8, the same number) opened and cut addition to 0.065 and division to 0.655. A softmax
probability above a fixed number is not an out-of-distribution (OOD) detector. The referee (7th read): the held-out
result cannot be listed as something the vector form adds until DISC gets a standard OOD gate (max-softmax, energy, or
Mahalanobis distance on the classifier features) at a matched rate. Separately, 517 of 2,500 task-N CoT outputs hit
B40's 512-token limit, so B40's COT number (0.275 on the trained types) is understated.

## Part 1: OOD gates for DISC (task N)

**Fixed (B40, unchanged, read-only on the box under /path/to/work/b40):** the frozen Llama-3.1-8B-Instruct file set,
layer 20, the three trained DISC seeds (readers out/rd/N_disc_e300_s{0,1,2}; write-backs out/dev/N_disc_lr1e-3_s0,
out/fin/N_disc_s{1,2}), the three trained VSA seeds and their B40 test outputs (VSA is not re-run on test), BASE's B40
test output, the test split (2,000 trained-type items, 250 per type; 500 held-out items, 250 addition and 250 integer
division), greedy generation with 6 new tokens, batch 32, the parser. b40_core.py, b40_data.py and b40_run.py are
imported unchanged (md5 835e9f4f..., 67b29c80..., fef8081d..., identical on the box and in git).

**Only change: DISC's gate.** Every score is computed in the same forward pass from the same layer-20 last-token state
that DISC's heads read; higher = more in-distribution:
- **MSP**: the max softmax probability over the 8 type logits (Hendrycks & Gimpel 2017), computed as its log-odds
  log p_max - log(1 - p_max) (monotone in p_max, so the gate is the MSP gate; the log form prevents ties at p = 1.0
  from floating-point saturation).
- **ENERGY**: logsumexp of the 8 type logits (negative free energy, T = 1; Liu et al. 2020).
- **MAHA** (Lee et al. 2018): -min_c (h - mu_c)^T S^+ (h - mu_c), h = the 4,096-d state (the input of DISC's linear
  heads, i.e. its classifier features; DISC has no hidden layer), mu_c = the mean of the 16,000 B40 training states of
  type c, S = the tied empirical covariance of the class-centred training states, S^+ its pseudo-inverse (as in their
  code: sklearn EmpiricalCovariance, pinvh).
- **MAHA-LW**: the same with the Ledoit-Wolf shrinkage estimate of S (registered now because S is estimated from 16,000
  states in 4,096 dimensions).
- Secondary only (not in the bar): **MAXLOGIT** (largest type logit; Hendrycks et al. 2022).
MAHA and MAHA-LW depend only on the frozen LLM's states and the training labels, not on the DISC seed; their thresholds
are per seed.

**Matching rule (dev only).** The dev split has 800 items, 100 per trained type; it contains no held-out type, so no
gate or threshold sees a held-out item before the test run.
1. k_s = the number of the 800 dev items on which VSA seed s's gate opens (B40's VSA module, unchanged, evaluated live
   on dev by b50_run.py).
2. DISC seed s is evaluated live on dev with its gate forced open, logging every score. For each gate g,
   tau_{g,s} = the k_s-th largest of the 800 dev scores, and the gate opens iff score >= tau_{g,s}. So on dev, DISC-g
   seed s opens on exactly k_s items (ties reported): the open rate on trained types matches the VSA seed's.
3. The thresholds (out/thresholds.json) and the dev numbers are committed in prereg/B50_dev_record.md before any gated
   DISC test run.
This is the standard OOD protocol (threshold set on in-distribution data at a fixed in-distribution acceptance rate;
the false-open rate on held-out data is then the measurement). Setting a threshold so that the held-out false-open rate
matches would need held-out labels; it is reported only as a description of the ROC (S2b), never as a gate.

**Test runs (each once, the full 2,500-item test split), per DISC seed s:** gate forced open (every score logged);
gate p08 (B40's rule; must reproduce B40's DISC files: implementation check); one run per gate g in {MSP, ENERGY,
MAHA, MAHA-LW, MAXLOGIT} at tau_{g,s}. Also the layer-20 test states are cached (for the independent score check).

**Definitions.** TR(X) = accuracy over the 2,000 trained-type test items; HO(X) = over the 500 held-out items; per
item the seed-mean correctness, then the mean (B40's convention). Paired 95% CI: bootstrap over items, 10,000 resamples.
TR(DISC-0.8) = 0.9298 and HO(VSA) = 0.9653 are B40's registered test values.

## Bar (primary, B50-1; b50_verdict.py)

- **Eligible DISC gates**: g in {MSP, ENERGY, MAHA, MAHA-LW} with TR(DISC-g) >= TR(DISC-0.8) - 0.02 (a gate that buys
  held-out accuracy by closing on trained types does not count).
- **DISC-best** = the eligible gate with the highest HO. It is picked on test, which favours DISC, so the test is
  conservative for the claim under test (a VSA advantage).
- Delta = HO(VSA) - HO(DISC-best), paired CI.
- **SURVIVES** iff Delta >= +0.10 and CI lower > 0. Otherwise **DOES NOT SURVIVE**, sub-read as: DISC-best ahead (CI
  upper < 0); VSA ahead by less than 0.10 (CI lower > 0); no VSA advantage (CI contains 0). Per-seed differences are
  reported beside it.
- If no gate is eligible: **NO ELIGIBLE GATE** (reported as such, with every gate's TR and HO).
- How "trained-type accuracy within 0.02" is read: as a condition on the OOD-gated DISC (its gate must not cost more
  than 0.02 of DISC's trained-type accuracy). The other reading (VSA's own trained-type accuracy within 0.02 of
  DISC-best's) is reported as a number; it is already known from B40 that TR(VSA) = 0.8975 sits 0.032 below
  TR(DISC-0.8), so under that reading no held-out result could make the advantage survive. The eligibility under
  "TR(DISC-g) >= TR(VSA) - 0.02" is printed too.

## Secondary (reported, no headline)

- **S1** per held-out type: accuracy and gate-open rate of every gate, and "hurts" (arm - BASE <= -0.05 with CI upper
  < 0, B40's S2 rule); trained-type test open rates against the dev target; per-seed numbers.
- **S2** threshold-free separability: AUROC (trained-type vs held-out test items) of each DISC score (from the
  forced-open runs) and of the VSA gate's similarity (op_s, logged in B40's VSA files), per seed.
- **S2b** "matched false-open" operating point: for each DISC score, the threshold at which its held-out open count on
  TEST equals the VSA seed's held-out open count on test; the trained-type open rate and the accuracies there
  (composed, below). It uses test labels to set a threshold, so it describes the ROC; it is not a gate.
- **S3** sensitivity: thresholds at dev open rates 0.99 and 0.95 (composed).
- **S4** exploratory: B40's VSA-argmax runs (exploratory 1; held-out 0.9707 / 0.9600) against each DISC gate.
- "Composed" accuracy: per item, the forced-open run's correctness if the gate opens, BASE's (B40) if it closes. This
  equals a real gated run when the gate is a per-item function and the closed branch leaves the stream untouched (the
  injector's torch.where); the composition check below measures how exactly.

## Part 2: CoT with a 2,048-token budget

**COT2048** = B40's COT arm unchanged (task N: DE25's CoT system prompt and "Solve the following problem step by step:
..." user turn, no exemplars; task R: B40's analogue; greedy; B40's parser, b40_core.parse; the same batch composition:
items sorted by prompt length, batch 48 on N and 32 on R as in B40) except max_new_tokens = 2,048 instead of 512. Full
test splits: task N 2,500 items (2,000 trained-type + 500 held-out), task R 2,000. Logged per item: the number of
generated tokens up to and including the first end token (ids 128001, 128008, 128009) and whether the budget was hit;
per batch: generation wall time.

Readings (no bar on CoT itself; it corrects a baseline):
- COT2048 accuracy on the primary sets (N: 2,000 trained-type items; R: all 2,000) with Wilson intervals; per type /
  cell; N's held-out types.
- R1 re-read with COT2048 in place of COT (B40's bar: VSA - COT2048 >= +0.10, CI lower > 0, all seeds); the R1 label
  (REPLICATED / PARTLY / NOT) re-derived with BASE, COT2048 and LORA.
- Truncation left: items still at 2,048 tokens without an end token; upper bound UB = accuracy if every such item were
  counted correct.
- Cost: generated tokens (mean, median, 95th percentile, max, total); A800 generation seconds per item and GPU-hours;
  against the module arms' 6 new tokens and one forward pass.

## Predictions (stated so they can fail)

- P1 Every VSA seed's dev open count k_s >= 796 of 800 (>= 0.995; B40 test 0.9987), so the matched thresholds sit in
  the extreme lower tail of DISC's dev scores.
- P2 MAHA on test: open on <= 5% of the items of each held-out type in every seed, and on >= 99% of trained-type items.
- P3 MSP on test: open on >= 30% of addition items in at least 2 of 3 seeds (the type head's overconfidence on unseen
  operations persists even in the extreme tail of its dev confidence).
- P4 ENERGY opens on fewer held-out items than MSP (pooled over seeds).
- P5 B50-1 = DOES NOT SURVIVE; DISC-best is MAHA or MAHA-LW; Delta in [-0.03, +0.02].
- P6 All four primary gates are eligible.
- P7 AUROC: MAHA >= 0.99 and VSA op_s >= 0.99 in every seed; MSP <= 0.95 in every seed.
- P8 gate p08 reproduces B40's DISC files: pred and gate identical on >= 2,498 of 2,500 items per seed.
- P9 Composition mismatches <= 0.5% of items for every gate.
- P10 COT2048 task N: trained-type accuracy in [0.28, 0.36]; >= 40% of B40's 517 truncated outputs still hit 2,048
  tokens; R1 vs COT2048 on N passes.
- P11 COT2048 task R: accuracy in [0.80, 0.85]; R1 vs COT2048 on R passes.
- P12 >= 99% of the items that end before 512 tokens have text identical to B40's COT output (same GPU type, same
  batches, greedy).
- P13 Second scorer: 0 gold mismatches; 0 correctness mismatches on non-CoT files; <= 1% on CoT files.
- P14 Independent score check: Spearman >= 0.999 between offline and live scores for every gate and seed.
- P15 Total B50 GPU-hours <= 6.

## Checks from other angles

- **Second scorer**: b40_check.py (B40's registered independent scorer: gold rebuilt from the prompt text with its own
  arithmetic, outputs re-normalised by its own rules) on every B50 test file; CoT files are named *_cot.jsonl so that
  it applies its CoT rules.
- **Gate re-derivation**: each gated file's decisions recomputed from its logged score and tau.
- **B40 reproduction**: gate-p08 files against B40's disc_s{s}.jsonl, item by item (pred and gate).
- **Composition**: each gated run's per-item correctness against (gate open ? forced-open correctness : BASE's).
- **Independent score implementation** (b50_check_scores.py, numpy / sklearn on cached states): MAHA and MAHA-LW refit
  with sklearn's EmpiricalCovariance and LedoitWolf classes (not the torch code), MSP / energy / max-logit from DISC's
  saved head weights in numpy, on the dev cache (B40) and a test cache (B50); compared with the live-logged scores
  (Spearman, median relative difference, gate agreement at tau). Cached states come from a separate batched forward
  pass, so small numeric differences are expected; agreement is a check, not an identity.
- **CoT reproducibility**: items ending before 512 tokens vs B40's text; longer items: B40's 512-token text is a prefix.
- **Hand-read**: (a) every held-out item on which DISC-best opens (up to 30) and every trained-type item on which it
  closes (up to 30): what DISC wrote and whether the decision was right; the same for MSP if it is not DISC-best;
  (b) COT2048: 20 items B40 truncated that now end (scored right?), 10 still at 2,048 tokens (what loops?), 10 random.
- Full test sets only; per-type numbers beside pooled ones.

## What the paper may say (decided now)

- **DOES NOT SURVIVE**: B40's held-out-type result is not something the vector form adds. With a standard OOD score on
  the discrete pipeline's own features (name it), at the VSA gate's trained-type open rate, DISC is as safe on the
  held-out operations (Delta [CI]). The paper must not list held-out gating as a VSA property. It may say that DE25's
  similarity gate rejects these held-out operations without a separately fitted detector, as a convenience, and only
  if the numbers show it.
- **SURVIVES**: on two held-out operation types, the VSA gate keeps the model's own answer more reliably than the best
  standard OOD gate on DISC at a matched trained-type open rate (Delta [CI]); restricted to Llama-3.1-8B-Instruct, task
  N and these two operation types; with the trained-type accuracy gap (VSA 0.032 below DISC) stated beside it.
- **NO ELIGIBLE GATE**: reported with every gate's numbers; no VSA claim is made from it without a further registered
  test.
- **CoT**: B40's COT number is replaced by COT2048 in every table, with the remaining truncation and the upper bound;
  R1 vs COT is re-read from COT2048. If more than 5% of items still hit 2,048 tokens, the text says the CoT baseline
  remains a lower bound and gives UB.
- In every case the paper must not generalise beyond Llama-3.1-8B-Instruct, these two task families, and (Part 1)
  these two held-out operation types.

## Compute

A800 box, one idle GPU claimed by the shared lock rule (never GPU 3; nobody else's process touched); PACE (sbatch,
free tier) only if the box has no idle GPU. Cap 20 GPU-hours (expected about 3-4). Per-run wall time to
/path/to/work/b50/logs/gpu_hours.txt. No paid API.
