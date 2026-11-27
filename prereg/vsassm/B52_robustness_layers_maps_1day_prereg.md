# B52 pre-registration: three robustness checks for the referee's 8th read (label-free codes at other layers; the compression result across random maps; the hardware ranking at one day)

Date: 2026-11-20, about 04:10 UTC (12:10 on the box clock, Asia/Shanghai; US Eastern 2026-11-19 23:10). Written, with all
B52 code, bars and verdict generators, before any B52 run reads a test label or writes a test prediction. What was done
before this file is listed under "Done before registration". Every result goes into the record whatever it shows; the
paper is not edited by B52.

## Why (referee, 8th read)

- **M3a.** "Codes read from frozen states needed identity labels" rests on label-free interfaces trained on FINAL-layer
  states (B42), while the supervised naming head on Qwen3-8B fails at the final layer in arenas A/B and succeeds at a
  middle layer (B31 exploratory 1, layers 9 / 18 / 27, dev-chosen). Required: the seven B42 label-free interfaces on
  the dev-chosen layer and on the input embeddings.
- **M1f.** The compression result's interval (B36, Qwen3-8B, region R1: +6.77 [+5.82, +7.74]) is a cluster bootstrap
  over entities; it omits the randomness of the store's random feature map and value codes and of configuration
  selection.
- **M5b.** The hardware comparison (B38) uses the device model at 30 days, extrapolating a model fitted to about one
  day.

## Part (a), primary: B42's label-free interfaces on Qwen3-8B's dev-chosen layer and input embeddings

**Cells** (feature set x arena; Qwen3-8B only, as the referee asks):
- **C x layer 27** (primary 1) and **C x input embeddings** (primary 2). B31 exploratory 1 chose layer 27 for arena C on
  dev (dev 1.000, tied with layer 9, ties to the later layer by its registered rule) and layer 9 for arena A (dev
  0.998). These choices are taken as made; nothing is re-chosen.
- **A x layer 9** and **A x input embeddings** (secondary).
- Input embeddings = `hidden_states[0]` = `embed_tokens(input_ids)` (bf16), span-mean pooled exactly as B31
  (`b52_embfeats.py`: a lookup into the checkpoint's embedding matrix on B31's token ids and spans, checked bit-for-bit
  against `AutoModel(...).hidden_states[0]` on two instances). Layers 9 / 27: B31's cached exploratory-1 features.

**Arms (unchanged B42 code; `b52_run.py` only widens the `--enc_model` choices and lets b31_arms record a CPU device):**
vq, dkvb, gumbel, st, cw (trained centred / whitened map), bind+pc, bind+pz (post-hoc centring / ZCA whitening of the
binding-loss map; read from the same `bind` final). Same arenas, splits, chains (perm, named, d = 2,048), training,
threshold, 10 epochs, seeds 42 / 123 / 456, test at each seed's dev-best epoch, geometry and discrete measurements.

**Grids and selection (B42's).** vq 16, dkvb 8, gumbel 12, st 12 configurations (b42_queues.grid), cw 12 at the
binding-loss lr of the cell. No B31 dev choice of that lr exists for these layers, so it is chosen here by the B31 / B42
rule on the cell's dev split: lr in {3e-4, 1e-3}, seed 42, highest best-epoch dev accuracy, ties within 0.002 to 3e-4;
the bind finals use it too. Per family: the configuration with the highest best-epoch dev accuracy (perm/named), ties
within 0.002 to the first listed (b42_select rule). Finals: the selected configuration, seeds 42 / 123 / 456.
(`b52_queues.py`; an orchestrator appends the cw grid after the lr choice and the finals after each cell's grid.)

**References.** F = B31's equalised probe (final layer; B42's floor; arena C 0.512, A 0.492). N = the naming head
(perm, named) read from the SAME layer: B31 explore1 for layers 9 / 27 (C layer 27 0.998, A layer 9 0.997); for the input
embeddings it is run here with B31 explore1's exact arguments (4 epochs, AdamW 3e-4, seeds 42 / 123 / 456).
**Gate G1** (labelled positive control, dev, B42's): gumbel and st with the identity loss added, within 0.02 of the
naming head's best dev on the same cell; reported, and a failed G1 means that arm's failure in that cell cannot be
attributed to the missing labels (B42 found G1 fails in arena A for an optimiser reason).

**Bar (B42's, per cell).** Recovery R = (mean - F) / (N - F) with a paired bootstrap interval. An arm **MATCHES** if
R >= 0.90 (the per-seed condition R_s >= 0.80 in 3/3 flagged); PARTIAL if R >= 0.25; FAIL below. If N - F < 0.10 the
cell has no headroom and is reported as accuracies only. Sensitivity (reported, not a bar): the MATCH threshold with F
at chance (0.5), and each arm against the best naming head at any layer.

**Readings (fixed now).**
- No arm MATCHES in C at layer 27 or on the input embeddings: the paper may say that the label-free interfaces fail at
  the final layer, at the dev-chosen intermediate layer where the naming head reaches 0.998, and on the input
  embeddings, with the best recovery and its interval per layer; the M3a claim then rests on three layers.
- An arm MATCHES in C at layer 27: the paper must drop "codes read from frozen states needed identity labels" for
  Qwen3-8B, say which layer and interface suffice, and recast the naming head as one way to supply codes.
- An arm MATCHES only on the input embeddings while the embedding naming head itself is below 0.90: reported as "matches
  a naming head that fails on the embeddings", not as label-free success; the comparison against the best naming head
  (any layer) is then the one quoted.
- PARTIAL anywhere: the paper reports the recovery and its interval and does not say "label-free codes fail" for that
  layer.
- Arena A: the same rules, secondary; if A finals are not complete by 2026-11-20 19:30 UTC, the incomplete A cells are
  reported as "not run" (dev only) and no test output of them is read.

**Checks.** Recomputation of every test accuracy from saved scores and thresholds against labels regenerated by
b31_struct (b42_verdict.memory_arm; "RECOMPUTED != LOGGED" flagged as in B42); the second scorer b42_indep.py (numpy
float64, its own regex parse of the raw text, codes rebuilt from the saved assignments / codebooks) on every discrete
final; hand-reading with b42_handread.py (12 arena-C failures per C cell of the label-free arm with the highest selected
dev); geometry (cross-talk), mention consistency, purity and the hex / decimal alias rate as in B42.

**Predictions (a).**
- PA1 G1 passes in both C cells; it fails for gumbel in at least one A cell.
- PA2 The naming head on the input embeddings is below the layer-27 one in C and between 0.55 and 0.92 there (span
  means of single-digit tokens cannot tell anagrams apart: 13 of the 32 two-digit decimal pairs and 6 hex pairs of
  arena C's 64 symbols collide; counted on the dev split, below).
- PA3 C x layer 27: every label-free arm FAILS (R < 0.25).
- PA4 C x input embeddings: no arm MATCHES; at most one arm is PARTIAL; every arm is at least 0.15 below the best naming
  head at any layer.
- PA5 A x layer 9: every label-free arm FAILS. PA6 A x input embeddings: no arm MATCHES.
- PA7 For every discrete final the second scorer agrees with the main run on >= 0.99 of instances (the CPU
  straight-through reruns of B42 were not bit-reproducible; such cases are reported).

## Part (b), primary: region R1 of B36 across three new random maps and value codes

**Design.** Qwen3-8B codes (mean layer-9 sentence codes, B32's), ideal accounting, B32's stores, region R1 = V in
{2, 16} x bpf in {4, 8, 16} x M in {256, 1024} (12 cells), test queries as B36 (about 4,500 per cell). Three new maps
m = 1, 2, 3: tag `qwen8#b52m<m>` replaces the encoder name in every seed of the memories' random components (map 0 =
tag `qwen8` = B32 / B36's seeds, used only for the gates).
- **VSA:** projections and value codes (`Feat` blocks) and the quantizer's tie signs drawn from the map's tag; B32's
  grid (prep x fmap x q x stored share = 72) on the 18 R1 dev stores; B32's choice rule per cell (best dev mean over the
  dev stores, ties within 0.1 points to the first in grid order); test once.
- **BEST:** B36's chosen competitor family and configuration per R1 cell (results/b36_choice_qwen8.json: FBF, CM, IDX,
  SF, SDM depending on the cell), NOT re-selected, with every random component drawn from the map's tag: the projection
  pool (SimHash, PQ sub-spaces, fly cells, SDM addresses), PQ codebooks refitted on the training split from the map's
  initialisation seeds, SDM value codes, tie jitter, and the index's LSH hyperplanes.
- Secondary comparisons: each new-map VSA against B36's original BEST predictions; B36 itself as a fourth map.
(`b52_maps.py`; local CPU.)

**Gates (before the new maps' test run).** G-b1: the map-0 dev path reproduces B32's committed VSA choices in the R1
cells. G-b2: the map-0 test path reproduces B36's per-query test predictions (VSA and BEST, every R1 bpf) on rep 0 of each
(M, V). A failed gate is investigated before the test run and reported.

**Statistics.** Per map: R1 difference = mean over the 12 cells of VSA - BEST (points), 95% cluster bootstrap (clusters
= an entity within a store, resampled within each cell; 10,000 draws; B36's method). Across maps: the mean of the three
per-map R1 differences; **primary interval = two-level bootstrap** (maps drawn with replacement from the three, entity
clusters resampled once per draw and shared across maps because every map answers the same queries; 10,000 draws);
also reported: the between-map SD, the entity-only interval of the map-averaged difference, a normal interval with
entity variance + between-map variance / 3, and the t interval over the three map means (df 2).

**Bar.** ROBUST if every new map has R1 VSA - BEST >= +2.0 with its 95% cluster interval above 0, AND the two-level
interval lies above 0. Otherwise NOT ROBUST.
**Readings.** ROBUST: the paper may quote the R1 result with the across-map interval (two-level) and per-map values and
say that the result does not hinge on one random map or on one configuration choice. NOT ROBUST: the paper must report
the across-map numbers and qualify or drop the R1 claim accordingly (a map below +2 or an interval reaching 0 is named).
Selection randomness of BEST is not re-sampled (its configuration stays B36's; stated as a limit).

**Checks.** Second angle: a numpy float64 re-implementation of the VSA read with the same generators (own arithmetic,
own quantizer) re-derives the predictions of 10% of the test stores per (map, M, V) at every R1 bpf (agreement >= 0.99
required; near-ties may flip between float32 and float64); an independent re-scoring of every cell from the saved
predictions (|difference| < 1e-9). Hand-reading: 10 queries of the R1 cell with the largest between-map range where
two maps' VSA answers differ (alias, label, values).

**Predictions (b).** PB1 G-b1 in >= 11 of 12 cells; G-b2 per-query agreement >= 0.995 (VSA) and 1.000 (BEST). PB2 each
new map's R1 VSA - BEST between +4.5 and +9.0. PB3 between-map SD <= 1.0 point. PB4 two-level interval lower bound
>= +3.0. PB5 ROBUST. PB6 re-derivation agreement >= 0.99 per map.

## Part (c): the hardware ranking at one day

**Design.** B38 unchanged (b38_eq / b38_design / b38_check) except the rate table: i.i.d. sign errors at the device
model's **one-day** bit-error rates from results/b34_ber.json (k1 0, k2 8.2e-6, k4 0.0097645, k8 0.1017177, k16
0.25640955, k32 0.36504285; the 5% control e05 kept), i.i.d. conditions only (H1 is the i.i.d. comparison), arms on H1's
two sides: the VSA reads with regenerated codebooks (V-pub, V-Snt, V-am, V-last) and the non-HD tables (T-rep, E-BCH,
E-LDPC, E-CAM). B38's full pipeline: dev grid on B34's 12 synthetic dev stores, designs and dev N* by b38_design.choose
at the one-day rates, test window dev N* +- 4 grid steps plus B = 512, B38's extension rule repeated until every N* is
bracketed (at most 6 passes), on the 50 RULER test stores (4,776 probed keys). (`b52_b38_1d.py`; box CPUs, the B38
environment.) V-cb and T-vhdd are not rerun (H1cb / H2 keep their 30-day numbers).
**Bar and readings (B38's H1 rule).** Per condition, Delta = grid steps of the best table's N* minus the best VSA read's
N*; "table fewer cells" if Delta <= -2, "VSA fewer cells" if >= +2, tie otherwise. The ranking is UNCHANGED if every
condition gives the same label as at 30 days (table fewer cells at every k). Readings: unchanged -> the paper may say
the 2-4x table advantage holds at the fitted horizon (one day) as well as at the extrapolated 30 days and give both;
changed at some k -> the paper must report the one-day labels and restrict the claim to the conditions where it holds.
**Checks.** B38's closed form (b38_check closed) at the one-day rates on the simulated cells (flags |sim - closed form|
> max(1 point, 4 SE)); N*(1 day) <= N*(30 days) for every arm and condition (rates are lower; violations listed); k1
(rate 0 at both horizons) compared with B38's 30-day k1 N* (differences can come only from tie-breaks at the next-harder
rate, which changed).
**Predictions (c).** PC1 TABLE WIN at every k at one day (ranking unchanged). PC2 at most 2 monotonicity violations. PC3
Delta between -6 and -3 at k8, k16 and k32. PC4 best table E-BCH at k1-k4, E-LDPC at k8-k32, as at 30 days. PC5 at most
10 closed-form flags.

## Done before registration (no test label read, no test prediction written)

- Code: b52_run.py, b52_embfeats.py, b52_queues.py, b52_worker.sh, b52_verdict_a.py, b52_maps.py, b52_b38_1d.py. The
  reused B31 / B42 files on the box were checked identical (md5) to the committed ones.
- The input-embedding features (`qwen8L0_{A,C}_{train,dev,test}.pt`) were extracted (no model output involved; the files
  carry the instance labels as every B31 feature file does; nothing was scored).
- Dev split only: the count of anagram collisions among arena C's surface forms (64 decimal forms 0-63, 13 colliding
  pairs; 64 hex forms 00-3F, 6 colliding pairs).

## Compute

(a) A800 box CPU (`b52_worker.sh cpu`, 8 lanes x 6 threads, `nice`), plus one worker that takes a GPU only if one is idle
(never GPU 3, never another process; GPU-hours from its ledger); (c) box CPU (B38 environment, `nice`, <= 24 workers);
(b) local CPU. No PACE planned (the layer features are 1-3 GB per file on the box). No paid API. Cap 30 GPU-hours.
Ledger: b52_out/compute_ledger.txt.

## Files

Code above; box /path/to/work/b52 (out/, logs/, q/, out_c/); local b52_out/. Results: results/b52a_verdict.txt,
results/b52a_scores.json, results/b52b_verdict.{txt,json}, results/b52b_check.json, results/b52c_verdict.{txt,json},
results/b52_literature.txt, hand-reading files under results/.
