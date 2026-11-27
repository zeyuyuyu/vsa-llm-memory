# B44 pre-registration: arena C with 10x and 100x the training data, a held-out-pair variant, a scratchpad fine-tune, chain-of-thought prompting, and the cross-talk sweep with centred / whitened codes

Date: 2026-11-18, about 22:20 UTC (06:20 on 11-19 on the box clock, Asia/Shanghai; US Eastern 2026-11-18 17:20).
Written, with all B44 code, gates and readings, before any B44 run reads a test label or scores a test instance. What
was done before this file (no test split read) is listed under "Done before registration".

## Why (referee M2 and M4, "critical for minor revision")

- **M2.** "The arena comparison is structurally unequal and the fine-tune baseline is data-starved: the arena-C
  training split is 800 instances; the Qwen3-8B fine-tune fits its training split exactly yet reaches 0.631 -- the
  signature of too little data. On Phi-2 the fine-tune never fits. Missing: learning curves for the fine-tune and the
  naming head at 800 / 8k / 80k training instances; a CoT-prompted Qwen3-8B baseline; a scratchpad-supervised fine-tune
  (emitting the bridge symbol); the dictionary row; arena C is not held out (every test pair occurs in some training
  instance)."
- **M4.** "The cross-talk causal test uses v_s = sqrt(1-c) s + sqrt(c) u with a single shared direction u -- rank-one
  common mode that centring removes. Repeat the sweep with centred or whitened codes; report whether the binding-loss
  map's codes, once centred, separate symbols. Fig. 2b,c come from the earlier Phi-2 pipeline (commutative binding,
  best epoch selected on the test set); regenerate them with the B31 protocol (permuted binding, development-selected
  epochs)."

B44 answers both on Qwen3-8B (Phi-2 only where the paper's panels are Phi-2: Fig. 2b,c). Its result goes into the paper
whatever it shows.

## Division of labour with B42 (registered 2026-11-18 22:30 UTC, running)

B42 (prereg/B42_label_free_discrete_prereg.md) already registers, on B31's arenas and features, **the B31 binding-loss
map read with post-hoc centring (bind+pc) and post-hoc ZCA whitening (bind+pz)** and **binding-loss maps trained with
centred / whitened codes (cw)**, with B31's geometry protocol. That is the referee's "report whether the binding-loss
map's codes, once centred, separate symbols". **B44 does not rerun these arms.** B44's verdict quotes B42's bind+pc /
bind+pz / cw cells (accuracy and different- / same-symbol cosine) when B42's verdict exists, and adds what B42 does not
measure: the imposed-cross-talk sweep with centred and whitened codes (item 3) and the binding-loss map at
initialisation, raw and centred (the open diamond of Fig. 2b). B44 uses no GPU that B40, B42, B43 or B46 holds.

## Arena C data (b44_data.py; self-test below)

- **Generator:** v13.Arena / b31_struct.build_C, re-implemented line by line with Python's `random.Random(seed)` and no
  file read (no tokenizer, no word list). Self-test: seeds 42 / 777 / 4242 reproduce B31's arena-C train / dev / test
  texts exactly (800 / 500 / 500 identical).
- **std** (the published arena C): training stream seed 44042, 80,000 instances; **the 800- and 8,000-instance sets are
  its first 800 and 8,000 instances (nested)**. Dev = B31's dev (seed 777, 500), test = **B31's test (seed 4242, 500)**,
  so every B44 number is on the same 500 test instances as the paper's arena C.
- **ho (held-out pairs):** a fixed partition of the 64 x 64 (role, filler) symbol pairs, 32 fillers per role on each
  side (seed 4401). Training instances (stream seed 44142, 80,000, nested 800 / 8,000) use only training-half pairs,
  at both levels (the chain and every distractor); dev (seed 44777, 500) and test (seed 44242, 500) use only test-half
  pairs. A pair is the written (role symbol, filler symbol) pair regardless of level, so **no pair that occurs anywhere
  in a test instance occurs anywhere in a training instance** (checked: 0 shared pairs). Every symbol still occurs as a
  role and as a filler in training (only pairs are held out). Negative candidates are drawn from the same half of the
  bridge symbol as the true answer, in training and in test, so the half membership of the candidate carries no label
  information.
- **Fig. 2c sweep (k pairs per level):** the std generator with k in {2, 4, 8, 16, 24, 32} pairs per level, train seed
  42 (800), dev 777 (500), test 4242 (500), as V-14 (which had no dev split); k = 32 is B31's arena C.
- **Dictionary row:** the dictionary walk on the raw text (regex parse, b44_data.dict_answer) equals the label on every
  instance of every split (self-test), so the dictionary arm is 1.000 by construction on std and ho; it is reported
  beside every table, with exact symbols + VSA (perm and comm).

## Arms (Qwen3-8B, the B31 snapshot b968826d; frozen states = B31's final-layer recipe)

| key | arm | trained on | sizes |
|---|---|---|---|
| dict | exact symbols + dictionary | nothing | - |
| exact | exact symbols + VSA (perm; comm read too), B31 threshold fitted on the training split | nothing | each n (threshold) |
| name | naming head + VSA (perm, named), **B31's registered arm unchanged**: identity loss on every mention + the answer loss on scale / bias, AdamW 3e-4, batch 4, clip 1.0, 4 epochs, test at each seed's dev-best epoch | identity labels + answers | 800 / 8,000 / 80,000, std and ho |
| probe | B31 equalised probe (grid selected on dev, 3 seeds) | answers | 800 / 8,000 (80,000 if the budget allows, cut 2) |
| lora | **B31 equalised LoRA unchanged** (b31_lora.Clf, arena C): LoRA on every linear layer, alpha 2r, dropout 0.05, head on [query span, candidate span, last token], standardised with initial-model statistics of the first min(n, 2,000) training instances (= B31 at 800) | answers | 800 / 8,000 / 80,000, std and ho |
| scratch | scratchpad-supervised LoRA: the same LoRA on the causal LM; input = the arena text + "\nAnswer:"; target " <bridge, decimal> F<bridge, hex> <answer symbol, decimal> <yes or no>" + EOS (e.g. " 37 F25 12 yes"), loss on the target tokens only; **scored by greedy generation** (max 16 new tokens), label = the last yes / no of the generated text, none = wrong | answers + the two bridge forms + the answer symbol | 800 / 8,000 / 80,000, std; ho (cut 1 / 3) |
| cot | Qwen3-8B prompted, no training (b44_cot.py): zs-nothink, zs-think, fs-nothink, fs-think | nothing | - |

### Training budget, evaluation and selection (LoRA arms; b44_lora.py)

- Recipe (B31): AdamW, weight decay 0, warm-up 6% then cosine to 0 over the planned steps, batch 8, clip 1.0, bf16
  base, fp32 adapters and heads.
- **Planned budget:** 800 -> 8 epochs (B31), 8,000 -> 4 epochs, 80,000 -> 1 epoch (6,400 / 32,000 / 80,000
  presentations); **8 evaluation points** evenly spaced (every epoch / half epoch / 10,000 instances), each scoring the
  fixed training subset (the first 500 training instances) and the dev split; the dev-best state is kept.
- **Early stop:** dev >= 0.995 at two consecutive evaluation points (saturated). Divergence (non-finite loss) stops.
- **Grid (seed 42):** rank 8 x lr in {1e-4, 3e-4} (B31's arena-C selection was rank 8, 3e-4; rank 64 was worse at
  both rates in B31). lora std: the grid at every size. lora ho and scratch std: the grid at 800 and 8,000; at 80,000
  the rate selected at 8,000 for the same arm and variant. scratch ho: the rate selected for scratch std at each size.
- **Selection:** highest dev accuracy at the best evaluation point; ties within 0.004 to the smaller rate.
- **Finals:** the selected configuration with seeds 42 / 123 / 456. **The seed-42 final is the seed-42 tuning run of the
  selected configuration** (its dev-best state is saved; the test split is scored once from that state after the
  selection is written). Seeds 123 and 456 train afresh; each scores the test split once from its own dev-best state.
- Runs are resumable (checkpoint at every evaluation point); PACE jobs run in <= 3 h segments.

### Naming head and probe (b44_name.py: b31_arms.run and b31_probe, unchanged)

- Features: B31's recipe (b31_prep.feats) on the B44 splits: span mean of every mention's final-layer state, the last
  token's state and the mean over the input; inputs untruncated and unpadded (length-bucketed batches with an attention
  mask). Span means are stored in fp16 (the 80,000-instance sets would need 170 GB in fp32) and read back as fp32.
- The naming head and the exact arm are B31's code path (perm, named; threshold on the training split), on these
  features. The probe is B31's grid and selection rule.

### CoT prompting (b44_cot.py)

- The prompt lists every pair, one per line, grouped by table in the text's order (level 0 = table R, level 1 = table
  F), explains the notation (keys in hexadecimal, values in decimal, one shared set of 64 symbols, with a worked
  conversion), states the two-step question in words with the query key and the candidate, and asks for a final line
  "Answer: yes" or "Answer: no". Few-shot: three training instances of the std stream (the first positive, the first
  negative, the second positive), each followed by a worked solution, as earlier chat turns.
- Decoding: thinking off: greedy, max 2,048 new tokens; thinking on (enable_thinking=True): the model card's sampling
  (temperature 0.6, top-p 0.95, top-k 20), torch seed = batch offset, max 8,192 new tokens. A reply without a parsable
  final answer (including thinking cut at the limit) is wrong; the rate is reported.
- **Selection:** the four arms on the first 100 std dev instances; the CoT baseline = the arm with the highest dev
  accuracy (ties: thinking off before on, then zero-shot before few-shot). **All four arms are scored on the full std
  test (500)** and reported; the selected arm also on the ho test (500).

## Cross-talk (M4; b44_xtalk.py)

- **Sweep:** exactly the published imposed sweep (V-95 ConeArm / V-163: row s -> l2n(sqrt(1-c) s + sqrt(c) u), u one
  fixed random direction, generator seed + 4242; the 14 published levels 0-0.9), now on B31's arena-C splits and
  codebooks (train 800 for the threshold, test 500; codebook seeds 42 / 123 / 456), B31's chain (two hops, clean-up
  after each) under **perm** (B31 primary) and comm (the published sweep), with the codes read in four forms before
  binding: **raw** (published), **centred** (minus the mean code of all training-split mentions, re-normalised),
  **whitened** (ZCA with shrinkage: W = S_a^(-1/2), S_a = (1-a) S + a (tr S / d) I, S the covariance of the
  training-split mention codes; a = 0.1 primary, 0.5 sensitivity). Accuracy with B31's closed-form threshold refitted at
  each level, AUROC, and the measured cross-talk of the form that is bound.
- **Gate G1:** at c = 0, raw, the sweep code reproduces B31's exact + VSA test accuracy per seed (perm 1.000 / 1.000 /
  1.000; comm 0.932 / 0.932 / 0.938).
- **Binding-loss map at initialisation** (the open diamond of Fig. 2b): b31_arms.Arm("C", "bind") with B31's seeds
  (same initial parameters as B31's runs), on B31's features, both encoders: geometry (B31 protocol) and test accuracy
  (threshold on the training split), raw and centred.
- **Trained binding-loss map, centred / whitened:** B42 (bind+pc, bind+pz, cw); quoted, not rerun.

## Fig. 2b,c under the B31 protocol (data only; the paper is not edited)

- **2b:** the sweep curve (perm, raw; comm beside it), the floor = B31's equalised probe on Phi-2 arena C (0.472), the
  tolerance = the largest measured cross-talk with mean accuracy >= floor + 0.10 (the published criterion), the arm
  points from B31's Phi-2 arena-C finals at each seed's dev-selected epoch (naming head, naming head read without
  snapping, binding-loss map perm/named and published form, InfoNCE map: x = different-symbol cosine, y = test
  accuracy), and the binding-loss map at initialisation (B44). Qwen3-8B beside it.
- **2c:** accuracy against pairs per level k in {2, 4, 8, 16, 24, 32}: exact + VSA (perm; comm beside it), the naming
  head (perm, named, dev-selected epoch, 3 seeds) and the equalised probe (floor; grid selected on dev, 3 seeds), on
  Phi-2 (the published panel) and Qwen3-8B.
- Written to results/b44_fig2bc.json with, for every number, the source file and the verbatim source line or JSON path.

## Bars and readings (fixed now)

Notation per variant v and size n: test accuracies per seed; N(n) = naming head mean; B*(n) = the strongest
equalised baseline = the highest test mean among lora, scratch and probe at n (the choice favours the baseline); lead
M(n) = N(n) - B*(n), m_s per seed. Intervals: Wilson 95% per seed; means and differences by a bootstrap over test
instances, paired across seeds and arms, 10,000 resamples.

1. **Learning curve (std, M2 "data-starved").** At each n: **STANDS** M >= 0.10 and m_s >= 0.10 in 3/3; **CLOSED**
   M <= 0.03; **NARROWED** otherwise. Also n* = the smallest n at which lora or scratch reaches N(800) - 0.03.
   - CLOSED at 8,000 or 80,000 -> the paper must say that in arena C a fine-tune of the same encoder reaches the naming
     head with 10x (or 100x) the training data: the memory's lead at 800 instances is a data-efficiency result, and
     the paper may not say the fine-tune "cannot" compose the chain.
   - STANDS at 80,000 -> the paper may say the naming head's lead survives 100x the training data.
   - NARROWED -> the paper reports the margins per size.
2. **Held-out pairs (ho).** The same rule on ho. Also the held-out drop per arm and n: arm_std(n) - arm_ho(n).
   - If lora_ho(n) <= lora_std(n) - 0.10 at the largest n -> the fine-tune's arena-C accuracy relies on pairs seen in
     training, and the paper must give the held-out figure as its generalisation number.
   - If the lead STANDS on ho where it is CLOSED on std -> the paper must say that the fine-tune closes the gap only on
     pairs it has seen.
3. **Scratchpad.** scratch - lora per n (paired by seed): **HELPS** if >= +0.05 with the interval above 0.
4. **CoT.** The dev-selected arm C* against N(800) and against lora(800). **MATCHES** if C* >= N(800) - 0.03 on std
   test -> the paper must say that prompting the same model with reasoning (no training) reaches the naming-head
   pipeline in arena C. **FALLS SHORT** if C* <= N(800) - 0.10.
5. **Sweep (M4).** **REMOVABLE** if centred (and whitened, a = 0.1) perm accuracy is within 0.01 of the c = 0 raw
   accuracy at every c in 3/3 seeds. REMOVABLE -> the paper must say that the imposed cross-talk is a common mode that
   centring removes, so the sweep measures the memory's tolerance to a shared offset, and that whether a learned map's
   failure is cross-talk depends on its centred codes (B42). Otherwise the sweep's conclusion stands for centred codes.

## Predictions (stated so they can be wrong)

- P1 the data self-test and gate G1 pass.
- P2 dictionary = 1.000 on std and ho test.
- P3 exact + VSA (perm) = 1.000 on std test and >= 0.99 on ho test in every seed.
- P4 lora std mean: 800 <= 0.75; 8,000 >= 0.85; 80,000 >= 0.95.
- P5 (bar 1) the lead is CLOSED at 80,000 on std.
- P6 lora ho at 80,000 >= 0.90 and within 0.05 of lora std at 80,000.
- P7 scratch >= lora + 0.05 at 800 and 8,000 on std (HELPS), and scratch std at 800 >= 0.80.
- P8 naming head >= 0.97 at every n on std and ho, and varies by <= 0.02 across n.
- P9 CoT: zs-nothink <= 0.65; the best thinking arm >= 0.80; the dev-selected arm is a thinking arm.
- P10 sweep: REMOVABLE; raw perm tolerance (Phi-2 floor) between 0.15 and 0.30.
- P11 binding-loss map at initialisation: raw different-symbol cosine >= 0.60 on both encoders; centred <= 0.10; centred
  accuracy <= 0.70.
- P12 Fig. 2c (perm): exact = 1.000 at every k; Phi-2 naming head >= 0.98 at every k; Qwen3-8B naming head >= 0.95 at
  every k.
- P13 the second scorer agrees with the main runs on >= 0.99 of instances in every cell.
- P14 total GPU-hours <= 80.

## Checks from other angles (in the verdict)

- **Recomputation and second scorer (b44_indep.py):** labels re-derived by an independent regex walk of the raw text;
  every test accuracy recomputed from saved per-instance outputs (lora: sign of the saved margin; name / exact / probe:
  saved scores and thresholds; scratch and CoT: the saved generations re-parsed by a second parser written separately
  from the first); per-instance agreement; the held-out property re-checked from the raw texts (pairs parsed by regex).
- **Hand-reading:** at least 10 test failures each of lora (largest n, std and ho), scratch (largest n), the selected
  CoT arm and the naming head, read in the raw text (which hop failed; for CoT, where the reasoning goes wrong).
- **Diagnostics:** training-subset vs dev curves (fit); scratch's bridge, bridge-hex and answer-symbol accuracies; CoT
  tokens used, limit hits, unparsed rate; the share of test pairs seen in training per n (std: expected ~1.0 from 800
  on; ho: 0).
- **Literature:** after the verdict (sample efficiency of fine-tuning on pointer chasing / multi-hop lookup; scratchpad
  and chain-of-thought supervision; held-out compositional splits; anisotropy and centring of LM representations),
  recorded as supports / conflicts / no precedent.

## Compute and budget

- **Where:** PACE Phoenix (sbatch only; free tier gts-nimam6 first, paid gts-nimam6-paid only if free-tier jobs do not
  start, spend reported) for the LoRA runs, in <= 3 h segments, and for every other GPU job while the A800 box has no
  free GPU (at 22:15 UTC all seven usable GPUs were claimed by B40 / B42 workers); the A800 box (tokenisation on CPU;
  GPU jobs only on idle GPUs (b44_worker.sh = b31_worker.sh: < 500 MiB used and no compute process, re-checked after 20 s,
  shared claim directory, never GPU 3, never another process).
- **Projection** (A800 / A100 rates from B31 and the smokes below: Qwen3-8B LoRA ~0.2 s per training presentation at
  batch 8; features ~0.04 s per instance; H100 / H200 about half): lora std 23.6 h; lora ho 19.7 h; scratch std 21 h;
  scratch ho 19 h; features 2 h; naming head 3 h; probe 2.5 h; CoT 5 h; Fig. 2c 1 h. Total about 96 GPU-hours before
  early stops; early stops at saturation are expected to remove a large part of the 80,000-instance runs.
- **Cap: 80 GPU-hours.** If the projection (updated as runs finish) passes the cap, cut in this order: (1) scratch ho at
  80,000; (2) probe at 80,000; (3) scratch ho at 8,000; (4) scratch ho entirely; (5) the 80,000 grid of lora std (use
  the 8,000 selection). Every cut is reported as "not run".
- **Deadline:** the parent asked for delivery within 24-30 h (by about 2026-11-20 03:00 UTC). Runs not finished by then
  are reported as not run; no test output of an unfinished cell is read.

## Done before registration (no test split read)

- Code: b44_data.py, b44_tok.py, b44_feats.py, b44_lora.py, b44_name.py, b44_cot.py, b44_xtalk.py, b44_worker.sh,
  b44_pace.sbatch, b44_pace_submit.sh, b44_ssh.sh, b44_select.py, b44_indep.py (b44_verdict.py and b44_handread.py are
  written after this file, before any registered test output is read).
- **Data self-test** (box, out/selftest_data.json): seeds 42 / 777 / 4242 reproduce B31's arena-C train / dev / test
  texts (800 / 500 / 500 identical); the dictionary walk on the raw text equals the label on every instance of every
  split (std and ho: 80,000 / 500 / 500; k-sweep: every k and split); positive rates std 0.501 / 0.506 / 0.474, ho
  0.503 / 0.478 / 0.468 (train / dev / test); no training text equals a dev or test text; **std: 100% of the test
  pairs (and both chain pairs of 100% of test instances) occur in the training set already at 800 instances; ho: 0% at
  every size** (0 shared pairs, dev included); ho negatives: 1,019 / 1,019 training and 266 / 266 test negative
  candidates lie in the same half of the bridge symbol as the answer.
- **Disclosure: one code smoke read the test split.** The sweep code (b44_xtalk.py sweep) was smoke-run at two levels
  (c = 0 and c = 0.5), all four forms, both encodings, three codebook seeds, on B31's arena-C test split (no encoder,
  nothing trained). Its output was read after prediction P10 and the forms (raw, centred, whitened with a = 0.1 / 0.5)
  were written and before this file was committed. Nothing was changed because of it. It showed: c = 0 raw
  reproduces B31's exact + VSA test accuracies per seed (perm 1.000 / 1.000 / 1.000; comm 0.932 / 0.932 / 0.938: gate
  G1 passes); at c = 0.5 raw perm 0.524 / 0.512 / 0.554, centred and whitened perm 1.000 in every seed. The registered
  sweep is rerun in full after the commit.
- Smokes of the GPU code (train and dev only) run after the commit; their timings go into a dated compute note.

## Files

- Code: as above; reused unchanged: b31_arms.py, b31_lora.py (Clf logic copied into b44_lora.Cls for arena C),
  b31_probe.py, b31_struct.py, b31_prep.py, b27_common.py, b27_arena/.
- Box: /path/to/work/b44 (data/, feats/, out/, logs/, q/); PACE: /path/to/pace-scratch/b44.
- Results: results/b44_verdict.txt, results/b44_scores.json, results/b44_fig2bc.json, results/b44_xtalk_sweep.json.
