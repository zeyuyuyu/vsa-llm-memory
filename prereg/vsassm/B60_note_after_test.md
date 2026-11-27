# B60 note after the test (test run once; results/b60_verdict.txt, b60_predictions.txt, b60_check.txt, b60_sample_reading.txt)

Qwen3-8B's outcome was known in advance from the exploratory diagnosis (store - corrected least squares at V = 16
-3.61); the other six encoders were blind. Compute notes 1 and 2 record the interruption (03:40-08:06), the reboot that
erased the diagnosis files (its step5 code was reconstructed verbatim and reproduces all 24 accuracies it printed), and
the one cut: map statistics on the R1 budgets only.

## Primary (registered bar): R1 store (B53) - LAMc, ideal accounting, 95% cluster bootstrap

| encoder | V = 2 | V = 16 | R1 |
| --- | --- | --- | --- |
| Qwen3-8B | -3.94 [-4.81, -3.09] LAMc leads | -4.17 [-5.25, -3.14] LAMc leads | -4.06 [-4.76, -3.40] LAMc leads |
| Granite-3.3-8B | -4.09 [-5.04, -3.17] LAMc leads | -7.01 [-8.03, -5.97] LAMc leads | -5.55 [-6.24, -4.87] LAMc leads |
| Llama-3.1-8B | -4.41 [-5.57, -3.25] LAMc leads | -6.72 [-7.94, -5.52] LAMc leads | -5.56 [-6.37, -4.72] LAMc leads |
| Mistral-7B | -5.47 [-6.47, -4.50] LAMc leads | -5.00 [-6.21, -3.81] LAMc leads | -5.23 [-6.03, -4.46] LAMc leads |
| e5-base-v2 | -2.97 [-3.73, -2.21] LAMc leads | -10.10 [-11.21, -9.02] LAMc leads | -6.53 [-7.19, -5.87] LAMc leads |
| bge-small | -1.05 [-1.96, -0.16] LAMc leads | -7.30 [-8.27, -6.34] LAMc leads | -4.18 [-4.85, -3.51] LAMc leads |
| MiniLM-L6 | -0.19 [-1.27, +0.85] level | -8.41 [-9.32, -7.49] LAMc leads | -4.30 [-4.99, -3.60] LAMc leads |

Counts: V = 16 LAMc leads on 4 of 4 language models and 3 of 3 sentence encoders, the store on none; V = 2 LAMc leads
on 4 of 4 and 2 of 3 (MiniLM level); R1 LAMc leads on 7 of 7. No encoder is within the +-2 margin over R1.
Chance-rescaled, R1 LAMc leads on all 7 (-4.7 to -8.4); real accounting gives the same labels (MiniLM V = 2 level).

## How much the fix matters: LAMc - LAM (B49)
R1-V16 +4.5 to +9.7 points (interval above 0 on all 7; language models +8.5 to +9.7); R1-V2 -0.4 to +2.0 (above 0
only on Mistral, +1.97); R1 +2.1 to +5.9; R0 +1.4 to +3.7; RH -0.5 (bge) to +4.5; R2 (V = 1,024) +1.8 to +4.3. Dev
chose the corrected whiten principal map in 45 of 48 language-model R1 cells; at V = 16 flat scales in 36 of 42 R1
cells and all V columns in 40 of 42; at V = 2 the per-column scale and V - 1 columns (equivalent there: one column).

## Secondary
- Store - BESTc over R1: -3.13 (Qwen3-8B) to -6.87 (e5), interval below -2 on all 7 (B49's H1 passes on 0). At
  V = 16 -2.3 to -10.1. BESTc picks least squares in 68 of the 84 R1 conditions (B49's BEST: 51), including all 24
  language-model conditions with two values.
- First design (B49's store, random value codes) - LAMc: R1-V2 -7.2 to -9.1 on the language models, R1-V16 -5.8 to
  -8.0 (LAMc leads, all 7 encoders); first design - BESTc over R1 -6.1 to -8.2 (real -5.3 to -7.8).
- B56's VSAS - LAMc over R1: LAMc leads on 7 of 7 (-3.5 to -6.3); at V = 2 level on bge and MiniLM.
- B57's VSAP - LAMc: LAMc leads on 7 of 7 (-4.9 to -12.7); VSAPX - LAMc: LAMc leads on 7 of 7 (-3.3 to -5.2).
- VSAPc (the store on the CORRECTED principal basis; B60 re-run of B57's procedure): VSAPc - VSAP over R1 -0.6 to
  +2.3 (above 0 on Qwen3-8B +0.84, e5 +1.13, MiniLM +2.28; level elsewhere). VSAPc - LAMc: LAMc leads on 7 of 7
  (-4.0 to -10.9); VSAPcX (dev-best of VSAS and VSAPc) - LAMc: LAMc leads on 7 of 7 (-2.0 to -5.3). B57's VSAP and
  VSAPX numbers are superseded by VSAPc / VSAPcX; the direction (behind least squares) is the same and stronger.
- R0 (1-2 bits per fact): LAMc leads on 7 of 7 (-2.1 to -4.7). RH (32-64 bits): LAMc leads on 5, level on bge and
  MiniLM; at RH-V2 the store leads on e5, bge and MiniLM (+1.7 to +4.7). R2 (V = 1,024, B49's store): level on
  Qwen3-8B, LAMc leads on 5, the store leads on bge (+1.57 [+1.08, +2.05]).
- Useful accuracy (90% of the exact index within 64 bits per fact, 28 settings): store 0, first design 0, LAM 4,
  LAMc 2, BEST 19, BESTc 19 (at 12.8 to 63.2 bits per fact).

## Map variance (three new random maps per encoder for the store and LAMc's rp blocks; R1 budgets, compute note 2)
Between-map SD of R1 store - LAMc 0.10-0.59 points (all 7 encoders); map means -4.11 (Qwen3-8B) to -6.32 (e5); every
map-mean R1 and R1-V16 label is "LAMc leads" (two-level 95% intervals); R1-V2 map means: LAMc leads on 5, level on bge
(-0.47 [-1.36, +0.46]; registered map -1.05, LAMc leads) and MiniLM. The map-mean label differs from the registered
map's in 1 of 21 cases (bge R1-V2). VSAS - LAMc: the same picture (SD 0.07-0.76 over R1). The registered map was
not systematically the most favourable (rank 1-4 across encoders). This replaces B59 part (1), which used B49's LAM.

## Predictions
Held: P1 (V = 16 LAMc leads on 4 of 4 LLMs, store leads on 0), P2, P3 (7 of 7), P4 (LAMc - LAM at V = 16 +4.5 to +9.7,
all intervals above 0; |V = 2| <= 1 on 6 of 7), P5, P6, P7, P8 (|VSAPc - VSAP| <= 1.5 on 6 of 7), P9 (SD <= 0.8 on all
7; labels 20 of 21). Failed: P10 (|R2 LAMc - LAM| <= 2 on 1 of 7, not >= 5: the fix also matters with 1,024 values,
+1.8 to +4.3). P11 (checks) failed in one part: the step5 agreement bar (>= 0.99 on every identical-map configuration)
missed in 1 of 1,728 cells (0.9878; flat scales, q = 4: B60's quantizer uses the group's std, step5's the root mean
square, a difference the prereg named); mean agreement 0.9998, largest accuracy difference 0.29 points.

## Checks
Plain-loop scorer: 1,967 values, max |difference| 4e-17. Bits: 1,794,374 dev and test rows, 0 over budget, 0 not
maximal. Overlap selftest: LAMc 1.000 everywhere (B49's whiten basis 0.07-0.98). Step5 agreement: above. Float64
re-derivation (independent code, 84 test cells): agreement min 0.998. Sample reading by the AI-assisted process
(results/b60_sample_reading.txt): the queries only LAMc reads are mostly near-copies of the label that the store loses
to cross-talk at V = 16; no scoring artefact.

## The paper's equal-bit claims (current text, which already carries "[B60 pending]" placeholders)
- MUST GO / FLIPS: "with sixteen values the store led least squares" (any version: B53's +2.3 to +4.7, the first
  design's +1.3 to +2.9, Introduction / Discussion "whereas with sixteen values the store led"). With the corrected
  memory least squares leads with sixteen values on every encoder: final design by 4.2 to 7.0 points on the language
  models (rescaled 4.5 to 7.5), 7.3 to 10.1 on the sentence encoders; first design by 5.8 to 8.0 on the language models.
- MUST GO: "averaged over the region the store was within the +-2-point margin of least squares on the four language
  models"; LAMc leads over R1 on all seven encoders by 4.1 to 6.5 points.
- MUST GO: "a store choosing per condition between random and principal-component features led least squares by 0.7
  to 1.6 points on the four language models" (VSAPX); with the corrected memory it trails by 3.4 to 4.3 (VSAPcX 2.0 to
  4.0).
- SURVIVES, numbers updated: least squares leads with two values on the language models (3.9 to 5.5 points; rescaled
  7.9 to 10.9); the first design trails it with two values by 7.2 to 9.1.
- SURVIVES, stronger: below the best of ten on all seven encoders, now by 3.1 to 6.9 points and beyond the margin on
  all seven (was 0.9 to 4.2, beyond on six); first design below by 6.1 to 8.2 (was 3.0 to 6.2); least squares the
  development-chosen best of ten in 68 of 84 conditions (was 51); rescaled for chance the best competitor reached 24
  to 55 (was 21 to 51; per encoder and number of values over R1, the definition that reproduces 20 to 38 / 21 to 51).
- SURVIVES unchanged: the best competitor reached 90% of the exact index within 64 bits in 19 of 28 settings (now at
  13 to 63 bits per fact); neither store did; decorrelating read / retraining added at most 0.8 points (store-internal).
- STANDS AS A V = 2 STATEMENT ONLY: least squares on the store's own random features (B56's LAMRF) and the one-pass delta
  rule (B57's NLMS) led with two values; LAMRF has no basis defect and at V = 2 one column, so its column and scale
  choices do not bind; NLMS used B49's features and quantizer and was not re-run (disclose).
- REPLACE: the map-variance sentences against least squares (B60: SD 0.10-0.59, map means -4.1 to -6.3, all LAMc leads).

PAPER CAN SAY: at equal bits a least-squares associative memory with principal-component features, a column per value
and shared block scales answered more alias queries than the VSA store over the pre-specified region on every encoder,
by 4.1 to 6.5 points (on the language models 3.9 to 5.5 with two values and 4.2 to 7.0 with sixteen), and at 1-2 bits
per fact on every encoder, with the same R1 verdicts on the means over four random maps (map-to-map SD 0.1 to 0.6
points); exceptions to name: at 32-64 bits per fact with two values the store led on the three sentence encoders, and
with 1,024 values it led on bge-small and was level on Qwen3-8B; the earlier lead of the store with sixteen values was an artefact of a
defective least-squares baseline (wrong principal basis, a missing column, per-column scales), which the corrected run
and a clean-room implementation agree on; the Hebbian (matched-filter) write is not the best superposed memory at
these budgets, and neither the store's VSA-internal read rules nor principal-component features close the gap.

PAPER MUST NOT SAY: that the store leads least squares with sixteen values; that the store is within the margin of
least squares over the region on any encoder; that the per-condition random / principal-feature store leads least
squares; that superposition per se loses (the comparison is between write rules of superposed memories); any number
from B49 / B53 / B56 / B57 / B59 for least squares, the best of ten, VSAP / VSAPX, NLMS or RLS at V = 16 or over R1
without marking it as using the defective baseline.

## Disclosed analysis additions after the test (no arm, choice or bar changed)
First-design minus BESTc over R1 (ideal and real); real-accounting store - BESTc; chance-rescaled accuracy ranges per
(encoder, V) over R1 (to match the paper's definition); LAMc's Qwen3-8B R1 cell accuracies beside step5's selection;
the step5 agreement note naming the one cell under 0.99.
