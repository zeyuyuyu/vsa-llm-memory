# B68 pre-registration: label-free readers of the input embeddings and early layers (referee M4)

Date: 2026-11-23, about 07:10 UTC (15:10 on the box clock, Asia/Shanghai; US Eastern 2026-11-23 02:10). Written, with
all B68 code, bars, readings and the verdict generator, before any B68 run reads a dev or test label. What was run
before this file (feature extraction of every split, without scoring; checks and smokes on training rows only) is listed
under "Done before registration". The file is committed and time-stamped with OpenTimestamps before any dev run.

## Why (referee, M4)

The abstract says "Maps from frozen language models' states to VSA vectors failed without identity labels". The tests
behind it (B31, B42, B48, B52) read the FINAL layer of Phi-2 and Qwen3-8B, and Qwen3-8B's layer 27 in arena C (B52).
B52 pre-registered the input-embedding cells (Qwen3-8B, arenas A and C) and the arena-A middle layer and cut them for a
deadline (B52 compute note 2), and the paper now says "readers of input embeddings or early layers were not tested
(their pre-specified cells were cut)". The referee: "Run the cut input-embedding and early-layer readers, which are
cheap." Input embeddings and early layers are where a label-free reader has the best chance: a symbol's span state
there is close to a function of its own tokens, so the mentions of one written form get (nearly) one vector, which is
the consistency the final-layer readers lacked (B42, B48: the codes followed a mention's position). B68's result goes
into the record whatever it shows; the paper is not edited by B68.

## Sources (feature sets)

- **Phi-2** (32 blocks) and **Qwen3-8B** (36 blocks; base, snapshot b968826d), frozen.
- **L0 = the input embeddings** (`hidden_states[0]`, the token-embedding lookup: no context, no position term in either
  model), **L2** and **L4** (`hidden_states[2]`, `hidden_states[4]`: the outputs of blocks 2 and 4; 6% and 12-13% of
  the depth). Chosen now, not on data.
- Span means exactly as B31 (`b68_feats.py`): the same instances, token ids, spans and pooling
  (`b31_prep.pool_instance`); Phi-2 through the B27 rows and batch sizes in fp16 as the B27 caches were made, Qwen3-8B
  through `b31_prep.feats_layers`' batching in bf16. Gates (training split, checked before registration, below): the same
  forward pass's `last_hidden_state` reproduces B31's final-layer feature files, and Qwen3-8B's L0 reproduces B52's
  embedding-lookup files.

## Arenas

A, B, C with B31's splits (A 2,000 / 500 / 1,000; B 1,000 / 500 / 500; C 800 / 500 / 500), chains (perm encoding,
named convention, d = 2,048) and scoring.
- **Arena A is the informative arena.** Every symbol has one written form per kind (`[R3A]`, `[X..]`, `[Y..]`, decimal
  fillers), three hops, and the final-layer naming head is far above the probe on Phi-2 (gap 0.49).
- **Arena B has no headroom by B52's rule** (the final-layer probe reaches 0.956 on Phi-2 and 0.994 on Qwen3-8B, so
  N - F < 0.10 at every layer): B is informative only as accuracies against F and the same-layer naming head
  (a label-free reader at or below the floor shows no usable code; one at the naming head shows codes as good as the
  labelled ones), never as a recovery.
- **Arena C is near-impossible for a label-free reader by construction**: each symbol is written in two unrelated forms
  (hexadecimal as a role, `[R3F]`, decimal as a filler, `{63}`), and the second hop must link them. Nothing in a
  context-free input embedding links "3F" to "63", and a label-free reader gets only the memory's binary answer
  (800 inputs) to learn it. A C failure therefore says little about label-free coupling; C is run for completeness and is
  the first arena cut.

## Readers (all label-free: no identity label anywhere; trained only through the memory's answer)

The seven B42 / B52 label-free interfaces, unchanged code (`b42_arms.py` through `b68_run.py`, which only widens the
`--enc_model` choices): **vq** (VQ, commitment, EMA codebook, k-means++ initialisation, dead-code restart), **dkvb**
(discrete key-value bottleneck, frozen k-means keys, learned values), **gumbel** (Gumbel-softmax over the naming head's
fixed codebook), **st** (straight-through arg-max), **cw** (binding-loss map with input whitening / code centring or
ZCA), **bind+pc**, **bind+pz** (B31's binding-loss map with post-hoc centring / ZCA). Plus **bind**, B31's
gradient-trained label-free map itself (the binding-loss map; plain reading of the same final). All seven are run (no
family is dropped for time; the time cut is by cell, below). B31's contrastive map (InfoNCE between two renderings of
an instance) is not run: it needs second-view features at every layer, and the brief names the binding-loss map.

**Grids and selection.** B42's registered grids: vq 16, dkvb 8, gumbel 12, st 12 configurations (b42_queues.grid), cw 12
at B42's rate for the model and arena (as B48: Phi-2 3e-4; Qwen3-8B A and C 1e-3, B 3e-4). The binding-loss map's
lr in {3e-4, 1e-3} is chosen on the cell's dev split (B52's rule: highest best-epoch dev, ties within 0.002 to 3e-4);
bind, bind+pc and bind+pz are read from that one final. Per family: the configuration with the highest best-epoch dev
accuracy (perm/named), ties within 0.002 to the first listed (b42_select rule). 10 epochs, seeds 42 / 123 / 456 for the
finals, test read at each seed's dev-best epoch. `b68_queues.py` (tune, select, final, merge).

**Gate G1** (labelled positive control, dev, B42's): gumbel and st with the identity loss added (lr 3e-4, hard
clean-up), reported per cell; a failed G1 means that arm's failure in that cell cannot be attributed to the missing
labels (B42: G1 failed in arena A for an optimiser reason).

## References

- **F** = B31's equalised probe (final layer; the floor of B42 and B52): Phi-2 A 0.485, B 0.956, C 0.472; Qwen3-8B A
  0.492, B 0.994, C 0.512.
- **N** = the naming head (perm, named, identity labels on every mention) read from the SAME source, run here with B31
  explore1's exact arguments (4 epochs, AdamW 3e-4, seeds 42 / 123 / 456, `b31_arms.py`). Reported for every cell as the
  identity-labelled reference. Also listed: the naming head at the final layer (B31) and, for Qwen3-8B, layers 9 / 18 /
  27 (B31 explore1), and the best of all.
- **Feature hash** (reference without training, computed in the verdict): one fixed random code per distinct
  span-feature vector, the published chain, the closed-form threshold from the training split. At L0 this is exactly
  "hashing the written tokens"; it is not one of the readers under the bar.

## Bar (B52's, per cell) and readings (fixed now)

Recovery R = (mean - F) / (N - F) with a paired bootstrap interval (10,000 resamples over test instances, paired across
seeds and with the references). A reader **MATCHES** if R >= 0.90 (per-seed R_s >= 0.80 in 3/3 flagged); **PARTIAL**
if R >= 0.25; **FAIL** below. If N - F < 0.10 the cell has no headroom and is reported as accuracies only. Sensitivity
(reported, not a bar): recovery with F at chance (0.5), and each reader against the best naming head at any layer.

Readings:
- **No reader MATCHES or is PARTIAL in any A cell** (both models, L0, L2, L4): the paper may say that the label-free
  interfaces failed from the input embeddings, early layers (2, 4), middle and final layers of two models in arena A
  (and in C), and may drop "readers of input embeddings or early layers were not tested".
- **A reader MATCHES in an A cell at L0 only:** the abstract's "Maps from frozen language models' states to VSA vectors
  failed without identity labels" must be restricted to contextual states (layers 2 and above, as tested); the paper
  must say that from the input embeddings, where every mention of a written form has one vector, the named reader
  matched the naming head in arena A, i.e. the failure lies in reading contextual states, not in coupling as such. If
  the L0 naming head is itself below 0.90, the reading is "matches a naming head that fails on the embeddings" (B52's
  rule) and the comparison against the best naming head is quoted with it.
- **A reader MATCHES at L2 or L4:** the abstract sentence must be dropped or restricted to the layers where every reader
  failed (named), and the paper must name the reader, layer and model that sufficed.
- **PARTIAL anywhere in A:** the paper reports that recovery with its interval and does not say "failed" for that
  source and model.
- **Arena C:** any result is reported; a C failure is attributed to the two written forms only if the same source
  succeeds in A (the C-by-construction reading), otherwise it adds nothing. A C success would be reported as the
  strongest evidence against the sentence.
- **Arena B:** accuracies only (no headroom).
- The note after the test states what the abstract's coupling sentence may and must not say.

## Predictions (stated so they can be wrong)

Informed only by the training-row facts under "Done before registration" (no dev or test row read):

- **P1** G1 (labelled control) passes in every arena-B and arena-C cell run, and fails for gumbel in at least one arena-A
  cell (B42's optimiser conflict does not depend on the layer).
- **P2** The naming head (identity labels) reaches >= 0.95 in arena A at L0, L2 and L4 on both models (the feature hash
  reaches 0.99 on Qwen3-8B's L0 training rows despite 454 of 832 forms sharing a vector, so anagram collisions cost
  little; Qwen3-8B's final-layer digit-identity loss (0.751) has not happened yet at layer 4).
- **P3** Arena A, L0: at least one label-free reader MATCHES the same-layer naming head on each model (most likely dkvb,
  whose frozen k-means keys separate the distinct input vectors as the feature hash does).
- **P4** Arena A, L4: no label-free reader MATCHES on either model.
- **P5** Arena A: the best label-free reader's recovery does not rise with depth on either model (R(L0) >= R(L2) - 0.05
  and R(L2) >= R(L4) - 0.05).
- **P6** Arena C: every label-free reader FAILS (R < 0.25) at every source on both models, and the feature hash at L0 is
  <= 0.60 on test (training rows: 0.52 on both models).
- **P7** Arena B, L0: the best label-free reader reaches >= 0.97 on both models.
- **P8** B31's binding-loss map (bind, plain reading) FAILS in every arena-A cell.
- **P9** The second scorer agrees with the main run on >= 0.99 of test instances for every discrete final.
- **P10** At most 12 A800 GPU-hours; every arena-A and arena-B cell completes within the time box (a cut, if any, falls
  in arena C).

## Checks from other angles (in the verdict)

- **Recomputation:** every test accuracy from the saved scores and thresholds against labels regenerated by b31_struct
  (b42_verdict.memory_arm); "RECOMPUTED != LOGGED" flagged as in B42 (ties at 6-decimal scores).
- **Second scorer:** b42_indep.py (numpy float64, its own regex parse of the raw text, the gold answer by a dictionary
  walk) on every final: codes of every discrete final (vq, dkvb, gumbel, st) rebuilt from the saved assignments and
  codebook / values / table, chain recomputed, per-instance agreement with the main run; accuracy of every final against
  the dictionary-walk labels. P-check: agreement >= 0.99 per discrete final.
- **Sample reading by the AI-assisted process:** in the arena-A cells at L0 (both models), for the label-free reader with
  the highest selected dev accuracy, 6 correct and 6 wrong test instances per model (seed 42): raw text, gold chain,
  clean-up winners per hop, which mentions shared a code with another symbol (collisions), written down before the
  verdict text is drafted.
- **Literature:** after the verdict, a search for precedents (token identity in input embeddings and early layers;
  detokenisation stages; label-free discrete bottlenecks on embeddings), each marked supports / conflicts / no
  precedent.

## Compute and cut order

A800 box (`ssh cavin`), GPUs 1, 2, 4, 5, 6, 7 only, each only if idle at claim (memory.used < 1,000 MiB, no compute
process, re-checked after 20 s; claim lock shared with the other experiments; never GPU 0 or 3; never another user's
process; `b68_worker.sh`, 7 lanes x 2 threads per GPU). The finals queue (q/final) is served before the tuning queue
(q/tune). Run counts: per cell 64 dev runs (2 bind lr, 2 G1, 60 grid) and 19 test processes (6 families x 3 seeds, one
naming-head process); 18 cells. Projection from B42's GPU timings: about 41 lane-hours, about 8-10 A800 GPU-hours if 7
lanes share a GPU at B42's speed. **Time box: 3 hours from the start of the experiment (06:45 UTC), i.e. all runs ended
by about 09:15 UTC and the record by 09:45 UTC.**

**Priority order** (queue order of the dev runs): arena A (Qwen3-8B L0, L4, L2; Phi-2 L0, L4, L2), arena B (same), arena
C (same). **Cut order = the reverse** (Phi-2 C first, then Qwen3-8B C, then Phi-2 B, ...). A cut is written in a compute
note before any test output of the affected or any other cell is read; a cut cell's dev runs, if complete, are reported
as dev only, and its test is not run (or, if already run, not read).

**Dev choices before test, in batches.** The dev choices of a batch of cells (A, then B, then C) are committed (with the
selection file's md5) before that batch's finals are released into q/final; finals of other batches never run before
their own commit. No test output is read before the compute note that follows the last test run.

## Done before registration (no dev or test label read, nothing scored on dev or test)

- **Code** written: b68_feats.py, b68_run.py, b68_queues.py, b68_worker.sh, b68_selftest.py, b68_verdict.py,
  b68_sample.py. The reused files on the box (/data0/path/to/HOME/b68) were copied from this repository and are
  md5-identical to B42's box copies (b31_arms.py ab0abce2..., b42_arms.py 700bcb8b...).
- **Features of every split** (train, dev, test; A, B, C; L0, L2, L4; both models) were extracted 06:54-06:58 UTC on GPUs
  1, 2, 4, 5, 6, 7 (6 workers, one lane each). No model output was scored; the files carry the instance labels, as every
  B31 feature file does.
- **Gates (training split, rerun with --check into feats_check/, 06:58-07:01 UTC):** the same forward pass's
  last_hidden_state, pooled the same way, equals B31's final-layer feature files exactly (max |diff| 0.0 in all six
  model x arena cells), so the instances, token ids, spans, batching and pooling are B31's; Qwen3-8B's L0 equals B52's
  embedding-lookup files (A 3.0e-8, the CPU-vs-GPU fp32 mean; C 0.0; B has no B52 file); Phi-2's L0 equals
  embed_tokens(ids) exactly. Phi-2 returns 33 hidden states, Qwen3-8B 37, so L2 / L4 are the outputs of blocks 2 / 4.
  The rerun's L0 / L2 / L4 training-split files equal the first extraction exactly (18 of 18 files, max |diff| 0.0;
  b68_out/feature_determinism.txt); gate lines in b68_out/feature_gates.txt.
- **Self-test on training rows** (b68_selftest.py; out/selftest_*.json, copied to results/b68_selftest.json):

  | model | source | arena | written forms | forms sharing their vector with another form | centred cosine, same form / different forms | feature hash, training rows |
  |---|---|---|---|---|---|---|
  | Qwen3-8B | L0 | A / B / C | 832 / 320 / 128 | 454 / 262 / 89 | 1.000 / 0.40, 1.000 / 0.02, 1.000 / 0.03 | 0.990 / 0.994 / 0.520 |
  | Phi-2 | L0 | A / B / C | 832 / 320 / 128 | 160 / 60 / 60 | 1.000 / 0.33, 1.000 / 0.04, 1.000 / 0.01 | 0.992 / 0.998 / 0.520 |
  | Qwen3-8B | L2 | A / B / C | (every mention its own vector) | 0 | 0.86 / 0.47, 0.94 / 0.35, 0.79 / 0.30 | 0.51 / 0.51 / 0.53 |
  | Qwen3-8B | L4 | A / B / C | | 0 | 0.72 / 0.45, 0.81 / 0.33, 0.68 / 0.29 | 0.51 / 0.51 / 0.52 |
  | Phi-2 | L2 | A / B / C | | 0 | 0.84 / 0.41, 0.97 / 0.18, 0.89 / 0.17 | 0.52 / 0.53 / 0.53 |
  | Phi-2 | L4 | A / B / C | | 0 | 0.80 / 0.40, 0.95 / 0.19, 0.84 / 0.18 | 0.52 / 0.53 / 0.53 |

  At L0 every mention of a written form has exactly one vector (forms with several vectors: 0 in every cell), so the
  per-mention consistency that the final-layer readers lacked is given; collisions are anagrams of single-digit tokens
  (Qwen3-8B) and merged number tokens (Phi-2, e.g. X10-X15). In arena C the feature hash is at chance (0.52): the hex
  role and the decimal filler of one symbol get unrelated codes, as the arena's construction implies. At L2 / L4 the
  same form's centred cosine falls to 0.68-0.97, so no exact hash exists there.
- **Smokes** (one epoch on the first 64 training instances, the same rows standing in for dev; every family and the
  naming head ran through the unchanged code path on all 18 cells, 0.2-15 s each). Their accuracies are on rows the
  reader was trained on (dkvb memorises: 0.80-1.00 everywhere, including arena C), so nothing was read from them.
- A first self-test attempt crashed (a numpy object-array of tuples) before computing anything, and the --check flag was
  first ignored (an empty dict is falsy); both fixed before the runs above. Neither touched dev or test rows.

## Files

Code: b68_feats.py, b68_run.py, b68_queues.py, b68_worker.sh, b68_selftest.py, b68_verdict.py; reused unchanged
(md5 equal to B42's box copies): b31_arms.py, b42_arms.py, b31_prep.py, b31_struct.py, b27_common.py, b27_arena/,
b42_queues.py, b42_verdict.py, b42_indep.py. Box: /data0/path/to/HOME/b68 (feats/, out/, logs/, q/). Results:
results/b68_verdict.txt, results/b68_scores.json, results/b68_selftest.json, results/b68_sample_reading.txt,
results/b68_literature.txt; ledger b68_out/gpu_ledger.txt.
