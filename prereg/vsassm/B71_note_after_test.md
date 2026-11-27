# B71 note after the test (test run once; results/b71_verdict.txt, b71_audit.txt, b71_side.txt, b71_d2_verdict.txt, b71_check.txt, b71_predictions.txt, b71_sample_reading.txt)

Pre-registration 417dab0 (17:28:12, OpenTimestamps 00f4bd7); dev 17:35-17:53 (box; compute notes 1-4); dev choices
8e332d5 (17:59) and D2 choices e410f08; test once 18:00:34-18:11 on the box (1,767 jobs, 0 failures); D2 tests 18:06-18:20;
verdict, checks and sample reading after compute note 6 (18:12). No cut. Deviations: worker count above 48 for about six
minutes during dev (note 4); dev moved between machines (notes 1-4); a dev job-name collision fixed before any output was
read (note 3); check-code corrections after the first check run and D1's label recomputation not executed (note 7). R1 only
(V 2 / 16 x 4, 8, 16 bits per fact x M 256 / 1,024), ideal accounting. Labels: B60's rule on the one-level 95% interval
(entity within store); two-level = test stores, then entities (B66's scheme; 1.0-1.5 times as wide here).

## (A) The strongest Hebbian variants against least squares (R1, points; one-level interval; two-level label if different)

| encoder | VSAPcX - LAMc | VSAPcX - NLMSc | VSAS - LAMc | VSAS - NLMSc |
| --- | --- | --- | --- | --- |
| Qwen3-8B | -2.01 [-2.61, -1.43] LAMc | -0.06 [-0.67, +0.54] level | -3.73 [-4.42, -3.06] LAMc | -1.78 [-2.46, -1.11] NLMSc |
| Granite-3.3-8B | -3.48 [-4.10, -2.85] LAMc | -0.80 [-1.44, -0.16] NLMSc (two-level: level) | -5.01 LAMc | -2.33 NLMSc |
| Llama-3.1-8B | -4.02 [-4.70, -3.34] LAMc | +0.03 [-0.62, +0.66] level | -5.14 LAMc | -1.09 [-1.82, -0.38] NLMSc |
| Mistral-7B | -3.91 LAMc | -3.47 [-4.14, -2.79] NLMSc | -5.39 LAMc | -4.94 NLMSc |
| e5-base-v2 | -5.31 LAMc | -2.26 [-2.85, -1.65] NLMSc | -6.34 LAMc | -3.29 NLMSc |
| bge-small | -3.47 LAMc | -1.32 [-1.98, -0.65] NLMSc | -3.47 LAMc | -1.32 NLMSc |
| MiniLM-L6 | -3.66 LAMc | -1.52 [-2.21, -0.79] NLMSc | -3.98 LAMc | -1.84 NLMSc |

Least squares solved over all facts leads every store variant on every encoder, with test stores resampled too. Against
the one-pass online write the strongest variant is level on Qwen3-8B and Llama-3.1-8B (and on Granite under the two-level
interval) and behind on Mistral-7B and the three sentence encoders. The point estimates of these four contrasts were
derivable from B60 / B62 before B71 (pre-registration); the intervals and two-level labels are new. Two-level labels
differ from one-level ones in 4 of 84 (contrast x encoder x region) cases, all becoming "level".

**Write rule vs features (exact per-query decompositions, R1):**

| encoder | batch: write VSAPcX - LSFM | features LSFM - LAMc | online: write VSAPcX - NLMSFM | features NLMSFM - NLMSc | VSAS: batch write / features | VSAS: online write / features |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen3-8B | -1.82 LS | -0.19 level | +0.18 level | -0.24 level | -1.18 / -2.55 | +0.10 level / -1.88 |
| Granite-3.3-8B | -1.76 LS | -1.72 LAMc | +0.04 level | -0.84 NLMSc | -1.34 / -3.68 | -0.44 level / -1.89 |
| Llama-3.1-8B | -3.07 LS | -0.95 LAMc | +0.30 level | -0.27 level | -1.26 / -3.88 | +0.55 level / -1.64 |
| Mistral-7B | -2.82 LS | -1.10 LAMc | -2.19 online | -1.27 NLMSc | -1.64 / -3.75 | -0.75 online (two-level: level) / -4.19 |
| e5-base-v2 | -1.58 LS | -3.73 LAMc | -0.15 level | -2.10 NLMSc | -1.52 / -4.82 | -0.66 online (two-level: level) / -2.63 |
| bge-small | -1.56 LS | -1.92 LAMc | +0.49 level | -1.81 NLMSc | -1.56 / -1.92 | +0.49 level / -1.81 |
| MiniLM-L6 | -1.68 LS | -1.98 LAMc | +0.49 level | -2.00 NLMSc | -1.56 / -2.43 | +0.42 level / -2.26 |

(LSFM / NLMSFM = least squares / the one-pass delta rule with LAMc's options on exactly the features that VSAPcX chose in
each cell: the final design's random map where VSAPcX = VSAS, VSAPc's principal map where VSAPcX = VSAPc.) Every write term
in the batch columns has its interval below 0 (least squares on identical features ahead on 7 of 7 for both variants);
every feature term against NLMSc for VSAS has its interval below 0 (7 of 7). Registered readings for VSAPcX - LAMc: write
rule on Qwen3-8B, Llama, Mistral; features on e5; both on Granite, bge, MiniLM. For VSAS - LAMc: features on 5, both on bge
and MiniLM. Against NLMSc: features on Granite, e5, bge, MiniLM (VSAPcX) and on 7 of 7 (VSAS); "both" where the total is
near zero (Qwen3-8B, Llama) and on Mistral. At identical features the one-pass online write is behind its own batch solve
on 7 of 7 (NLMSFR - LSFR -0.86 to -2.05; NLMSFP - LSFP -0.45 to -3.98, Mistral level).

## (B) The Hebbian store on rotated principal features (VSAPR; the store's 72 configurations x p, d not capped)

| encoder | VSAPR - VSAPc at V = 16 | VSAPR - STORE | VSAPR - VSAPcX | VSAPR - LAMc | VSAPR - NLMSc |
| --- | --- | --- | --- | --- | --- |
| Qwen3-8B | +1.66 [+0.71, +2.65] | +0.49 level | -1.56 VSAPcX | -3.57 [-4.25, -2.92] LAMc | -1.62 [-2.27, -0.97] NLMSc |
| Granite-3.3-8B | +5.49 | +1.93 VSAPR | -0.14 level | -3.62 LAMc | -0.94 [-1.59, -0.30] NLMSc |
| Llama-3.1-8B | +3.90 | +1.23 VSAPR | -0.31 level | -4.33 LAMc | -0.28 [-0.93, +0.34] level |
| Mistral-7B | +3.49 | +1.34 VSAPR | +0.02 level | -3.90 LAMc | -3.45 NLMSc |
| e5-base-v2 | +10.38 | +0.69 VSAPR | -0.54 level | -5.85 LAMc | -2.79 NLMSc |
| bge-small | +13.41 | +0.61 level | -0.10 level | -3.57 LAMc | -1.42 NLMSc |
| MiniLM-L6 | +13.14 | +0.62 level | -0.02 level | -3.68 LAMc | -1.54 NLMSc |

The rotation removes the sixteen-value loss of the principal-feature store (+1.7 to +13.4 points at V = 16; with two values
VSAPc stays ahead by up to 1.3 on four encoders) and lifts the Hebbian store above its random-feature final design by 0.5 to
1.9 points (ahead on 4, level on 3). It does not change the comparison with least squares: LAMc leads by 3.6 to 5.9 on all
seven (11-35% of the store's gap closed), and the gap to NLMSc is closed only on Llama-3.1-8B (registered rule: 1 of 7).
VSAPR is level with VSAPcX on 6 of 7. Dev chose linear features in nearly every cell and p below n in 4 to 8 of the 12
language-model cells (p = 1,024 or 256; n = 4,096 in the rest).

## (C) The store written by a delta rule on its own bundle (same vector, value codes, read and bits)

| encoder | VSADR1 - STORE | VSADR1 - VSAS | VSADR1 - LAMc | VSADR1 - NLMSc | several passes VSADRE - VSADR1 | VSADRE - NLMSc |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen3-8B | +0.42 [+0.07, +0.77] VSADR1 | +0.09 level | -3.64 LAMc | -1.69 NLMSc | +0.73 | -0.96 NLMSc |
| Granite-3.3-8B | +0.08 level | -0.46 level | -5.47 LAMc | -2.79 NLMSc | +1.33 | -1.46 NLMSc |
| Llama-3.1-8B | +0.05 level | -0.37 level | -5.51 LAMc | -1.46 NLMSc | +1.27 | -0.19 level |
| Mistral-7B | -0.41 level | -0.26 level | -5.65 LAMc | -5.20 NLMSc | +1.61 | -3.59 NLMSc |
| e5-base-v2 | +0.20 level | +0.01 level | -6.33 LAMc | -3.28 NLMSc | +1.27 | -2.01 NLMSc |
| bge-small | +0.25 level | -0.45 level | -3.92 LAMc | -1.77 NLMSc | +1.79 | +0.02 level |
| MiniLM-L6 | +0.12 level | -0.20 level | -4.18 LAMc | -2.03 NLMSc | +0.99 | -1.04 NLMSc |

One pass of the normalised delta rule (or the misread-only variant, chosen in 1-3 cells per encoder) on the store's own
bundle closes the gap to neither least-squares memory on any encoder (0 of 7 each). It is level with B56's retrained /
decorrelating-read store on 7 of 7 and with the Hebbian store on 6 of 7 (+0.42 on Qwen3-8B). Up to ten passes with
full-precision accumulation and quantization at the end add +0.73 to +1.79 points on every encoder (VSADRE - VSADR1, all
intervals above 0; region means put VSADRE 0.8 to 1.4 points above B56's VSAS and 1.1 to 2.0 above the store, point
estimates only, not registered contrasts), level with NLMSc on Llama-3.1-8B and bge-small and still behind LAMc on all
seven (-2.1 to -5.1). The delta rule on VSAPc's principal bundle (VSADP1) does not beat VSAPc (-1.3 to +0.5).

**Why B56's retraining gained at most 0.8 while NLMSc led the store by 1.5 to 4.8 (registered decomposition, R1):**
NLMSc - VSAS (+1.09 to +4.94) = features NLMSc - NLMSFR (+1.64 to +4.19, NLMSc ahead on 7 of 7; the largest term on 7 of 7;
mean +2.33) + layout NLMSFR - VSADR1 (least squares' shared features and columns vs the store's residue classes and Hadamard
codes, -0.23 to +1.01, level on 4) + write rule VSADR1 - VSAS (-0.46 to +0.09, level on 7 of 7). The main cause is the
features: NLMSc's dev-chosen whitened principal coordinates. On the store's own random features one pass of any
error-driven write gains about what B56's retraining gained.
**Dev audit of B56's settings (B53's configuration, dev stores, means over R1 cells):** float64 accumulator, Hebbian start,
updates only on misread facts, no margin, no step normalisation, eta 1 / 0.1, 1-10 epochs, quantized after training. On dev
the retrained bundle read 0.3-1.5 points above the Hebbian one; the extended grid (eta down to 0.01, 30 epochs) added
+0.18 to +0.35 (not named); quantizing after training cost the retrained bundle 2.0-3.2 points but cost the Hebbian bundle
as much (1.5-3.4), and reduced the retraining gain itself by at most 0.7 points (sentence encoders; none on the language
models) — the registered rule names quantisation (loss >= 0.5 on 7 of 7), but the loss is common to every bundle at these
budgets, not specific to retraining; updating every fact beat updating only misread ones by 1.3-3.3 points in one pass
(named), yet the dev-chosen one-pass delta rule read only -0.4 to +0.9 above B56's multi-epoch retraining. The Hebbian
bundle reads 69-78% of its own stored keys correctly, so the first retraining epoch corrects 16-29% of the facts.

## (D) Side information and operations (chosen configurations, R1; full table results/b71_side.txt)

| arm | fixed components (uncharged) | LLMs: bits at 8 per entry, median [range] | x budget (median) | sentence encoders | x budget | write MAC per fact (LLM median) |
| --- | --- | --- | --- | --- | --- | --- |
| store, first design, delta store | seed; mean; whitening matrix where whitened | 6.7e7 [3.3e4, 6.7e7] | 16,000 | 5.9e5 [3.1e3, 2.4e6] | 150 | 1.7e7 (features) |
| VSAS | + LMMSE read: 512-1,024 training codes where chosen | 6.7e7 [3.3e4, 1.0e8] | 16,000 | 2.2e6 | 530 | 1.7e7 |
| VSAPcX / VSAPR | + principal basis (n x p; derivable from the whitening matrix when whitened) | 6.7e7 [1.7e7, 1.3e8] | 16,000 | 0.7-1.1e6 | 240-360 | 1.7e7 |
| LAMc / NLMSc | seed or principal basis; mean; whitening matrix | 6.7e7 [3.3e4, 1.3e8] | 16,000 | 5.9e5 | 150 | 4.5e6 / 4.1e6 |
| SDMt | 2,818-5,621 training codes as hard locations (N x n); mean; whitening | 1.6e8 [3.7e6, 2.5e8] | 38,000 | 9.0e6 | 2,300 | 1.2e7 |
| learned hashing | LDA map n x r (r = 256) + mean | 8.5e6 | 2,100 | 7.9e5 | 190 | 1.0e6 |

**D1 (full charge per store, 8 bits per entry):** on the 4 language models every arm's side information exceeds the whole
budget in 12 of 12 R1 cells ("no arm storable"); a stored whitening matrix would have to be shared by about 1e5 stores to
fall to 10% of a store's budget. On the sentence encoders a centring-only store or least-squares memory fits its side
information in 1-3 of 12 cells; labels were not recomputed (note 7). **D2 (charged beyond what the compared store design
itself uses, 8 bits per entry):** LAMc - STORE over R1 +3.47 to +5.73, LAMc leads on 7 of 7 (labels unchanged; LAMc re-chosen
in the 12 cells where the store only centres); SDMt - first design: the SDM's lead reverses on 6 of 7 (first design ahead by
2.5 to 8.2) and survives on Mistral-7B (+2.82), where the first design whitens in every cell and SDMt keeps 2,049 locations.
Where the first design only centres, the rule leaves SDMt one location, so D2's SDM numbers depend on this allowance rule.

## Predictions

Held: Q2, Q3, Q4, Q5, Q6, Q9, Q12 (D1 on the language models; D2 LAMc labels unchanged 7 / 7, SDMt lead lost on 6 / 7).
Failed: Q1 in part (Granite NLMSc leads, predicted level), Q7 in part (VSAPR closes the NLMSc gap on 1 of 7, predicted >= 3),
Q8 (the one-pass delta store leads the store on 1 of 7, not >= 5, and B56's VSAS on 0, not >= 4), Q10 in part (the write
term VSADR1 - VSAS positive on 2 of 7, not >= 5; the quantisation-loss part), Q11 (several passes add 0.7-1.8, not +-0.5),
Q13 in part (reproduction bar in 3 of 84 cells; re-derivation bar in 1 of 56 VSAPR stores after the check corrections).
I expected the delta rule on the bundle to recover most of least squares' write-rule advantage; it recovers almost none in
one pass and about a point in ten.

## Checks (results/b71_check.txt)

Plain-loop scorer: 1,512 cell accuracies and 756 region statistics, max |difference| 1.4e-17. Bits: 303,527 dev and test
rows, 0 over budget, 0 not maximal. Reproduction: B56's retraining choice re-made in 84 / 84 cells, dev accuracies identical
in 81 / 84 (3 sign-feature cells at q = 2 differ by 0.007-0.08 points; exact lattice ties; same result on both machines).
Float64 re-derivation by independent code: NLMSFR 56 / 56 >= 0.994 and VSADR1 56 / 56 >= 0.994; VSAPR 55 / 56 >= 0.99 (one
MiniLM sgn3 store 0.966) after the corrections of note 7 (first runs: NLMSFR 4 of 40 and VSAPR 3 of 56 below 0.99 from
the check's own quantizers). Sample reading by the AI-assisted process: no scoring artefact; the delta store's gain over
the Hebbian one is the difference of two large sets of near-copy aliases read differently (8-10% of a cell's queries
change each way); the rotated store's gain at sixteen values is a systematic recovery of clear aliases that the capped
principal store loses.

## Literature (supporting / conflicting / new; recalled, not re-fetched)

Supporting: that error-driven writes approach the least-squares (projection) memory only with enough passes, and that one
LMS / Kaczmarz pass falls short of the batch solve when facts outnumber features, is the classic picture (Widrow-Hoff;
Kohonen's optimal associative mapping; Diederich and Opper 1987 on iterative learning reaching the projection rule's
capacity). In hyperdimensional classifiers, single-pass updates weighted by the error on every sample (OnlineHD,
Hernández-Cano et al. 2021) beat misread-only perceptron retraining, and multi-epoch retraining adds further accuracy, as in
our dev audit and the several-pass arm. Random rotations that spread variance evenly over coordinates before quantization
or splitting are standard (iterative quantization, Gong and Lazebnik 2011; optimized product quantization, Ge et al. 2013),
consistent with (B). Conflicting: none found. What is ours: the decomposition at equal stored bits on frozen language-model
codes showing that the one-pass online advantage is the features, and that a VSA-native delta write of the same bundle
recovers about a point only with several passes.

## The paper's claims (current text; the paper is not edited by B71)

- Results 2.1, "by 2.0 to 5.3 points against the strongest store variant (random or principal-component features chosen
  per condition)": SURVIVES (also with test stores resampled); B71 adds that the rotated principal store (VSAPR) is not
  stronger (level with VSAPcX on 6 of 7) and also trails least squares, by 3.6 to 5.9.
- "Written one fact at a time by a normalised delta rule ... least squares was still ahead, by 1.5 to 4.8 points": TRUE
  ONLY AGAINST THE FINAL DESIGN. Against the strongest variant the one-pass online write is level on Qwen3-8B and
  Llama-3.1-8B (and on Granite with test stores resampled) and ahead by 0.8 to 3.5 on the other five; the sentence must say so.
- "Retraining the bundle or a decorrelating read added at most 0.8 points": SURVIVES; B71 adds the reason (above) and that
  a delta-rule write of the same bundle adds no more in one pass and 0.7-1.8 points in ten.
- "fitted to the store's own random features it led by 1.5 to 2.3": SURVIVES (B66); B71 adds the online counterpart
  (level) and the per-encoder split for the strongest variant.
- "so that the write rule alone pays is likely but not settled": REFINE. At identical features the batch least-squares
  write leads every store variant on every encoder (1.2 to 3.1 points, all intervals below 0); one online pass does not.
- SN 7, "With sixteen values the store lost ... Hadamard value codes split the principal coordinates into sixteen fixed
  classes": ADD that a random rotation of the principal coordinates (and letting the stored vector use the budget instead of
  stopping at the code's dimension) removes the loss (+1.7 to +13.4 points at sixteen values).
- No current sentence states side information; the Methods' "fixed or fitted on the training split and free to all
  memories" must be accompanied by (D).

PLAIN SENTENCES FOR THE PAPER (proposals; the authors' wording decides):
1. **Write rule vs features.** Fitted to exactly the features of the strongest store variant in each condition, a
   least-squares memory solved over all facts still answered 1.6 to 3.1 points more queries over the region on every
   encoder, so the batch least-squares write beats the Hebbian write at identical features; the rest of its lead (0.2 to
   3.7 points, level on Qwen3-8B) comes from its own principal-component features.
2. **The online claim against the strongest variant.** Written one fact at a time by a single pass of the normalised delta
   rule, least squares was level with the strongest store variant on Qwen3-8B and Llama-3.1-8B and ahead by 0.8 to 3.5
   points on the other five encoders; at identical features that one-pass write was level with the store variant on six of
   seven encoders, so its lead is the features, not the online write rule.
3. **(B).** A Hebbian store on randomly rotated principal-component coordinates, free to use the whole budget, removed the
   principal-feature store's loss with sixteen values (+1.7 to +13.4 points) and led the store's final design by 0.5 to 1.9
   points (interval above zero on four encoders), but least squares still led it by 3.6 to 5.9 points on every encoder and the one-pass delta rule on six of
   seven.
4. **(C).** Writing the store's own bundle with a normalised delta rule (the same vector, value codes, read and bits)
   changed it by -0.4 to +0.4 points in one pass and gained 0.7 to 1.8 more with up to ten passes, closing the gap to the
   batch least-squares memory on no encoder and, with several passes, to the one-pass delta rule on two. B56's retraining
   gained little because an error-driven write gains little on the store's own random features: the one-pass delta rule's
   1.1 to 4.9-point lead over the retrained store came from least squares' principal-component features (1.6 to 4.2 points
   on every encoder), not from its update rule;
   the retraining's step and epoch grid were not limiting (+0.2 to +0.35 points from a wider grid on development data).
5. **Side information.** Every memory compared uses fixed components fitted on the training split that the budget does
   not charge: the whitening matrix (8.4 million numbers for a language-model code, thousands of times a store's budget) for
   the store and least squares alike, 2,818 to 5,621 stored training codes for the SDM, and a learned 4,096 x 256 map for
   learned hashing. Charged at 8 bits per number beyond what the store itself uses, least squares still led the store on all
   seven encoders (by 3.5 to 5.7 points), whereas the SDM's lead over the first design reversed on six of seven; charged in
   full per store, no memory, the store included, fits the language-model budgets.

PAPER MUST NOT SAY: that the one-pass online least-squares write beats every store variant on every encoder; that the
write rule alone explains the online lead; that a delta-written VSA store closes the gap (it closes none); that the
rotated principal store matches least squares; that the comparisons are free of side information or that least squares
carries more of it than the store (they use the same whitening matrix); the D2 SDM numbers without the allowance rule.

## Disclosed analysis additions after the test (no arm, choice or bar changed)

b71_verdict.audit_panel (the registered audit's summary); the region-mean point differences VSADRE - VSAS and VSADRE -
STORE quoted above (no interval; not registered); the cell-level explanation of D2's SDM reversal (cells where the first
design centres); the basis comparison for the MiniLM re-derivation store (note 7).
