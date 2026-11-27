# B65 note after test: verdict, predictions, checks, sample reading, and what the paper may say about the forward pass

Written 2026-11-23, about 14:40 (machine clock), after the 18 registered test evaluations (LAST 06:06-06:10Z, OPS
06:25-06:28Z; compute note 4, committed before any test output was read) and after results/b65_verdict.txt,
b65_scores.json, b65_check.txt, b65_handread.txt and b65_literature.txt.

## Registered result (task N, the 2,000 trained-type test items, 3 seeds; Delta in points)

| read position | VSA-PUB | VSA-BM | DISC |
|---|---|---|---|
| LAST (last prompt token, layer 20; B40's design) | 0.8975 (seeds .9005 / .8930 / .8990) | 0.9607 (.9595 / .9615 / .9610) | 0.9477 (.9455 / .9450 / .9525) |
| OPS (operand tokens, layer 17; the clean room's arm A) | 0.9773 (.9865 / .9675 / .9780) | 0.9797 (.9875 / .9760 / .9755) | 0.9852 (.9950 / .9825 / .9780) |

| cell | Delta, two-level 95% interval (seeds, then items) | label (four-way rule) | one-level items (B40's convention) |
|---|---|---|---|
| LAST, published read - DISC | -5.02 [-6.45, -3.58] | **BEHIND** | [-6.30, -3.72], all seeds |
| LAST, best match - DISC | +1.30 [+0.35, +2.25] | **AHEAD** | [+0.42, +2.15], all seeds |
| OPS, published read - DISC | -0.78 [-2.10, +0.47] | **INCONCLUSIVE** | [-1.25, -0.37], mixed seeds |
| OPS, best match - DISC | -0.55 [-1.68, +0.50] | **EQUIVALENT** | [-0.87, -0.27], all seeds behind |

Answer to the question. Under DE25's published read the vector module never leads: the discrete module wins at the
last-token read (-5.0) and at operand reads the comparison is inconclusive (point -0.8; the interval misses the
+/-2 equivalence band by 0.1 point at its lower end). Under best match the vector module leads at the last-token read
(+1.3) and is equivalent at operand reads (-0.55). By the registered answer rule neither module "wins" under either
read (no read gives the same label at both positions); the answer must name the position.

Secondary (no headline): best match - published +6.32 [+5.17, +7.52] at LAST, +0.23 [-0.90, +1.43] (equivalent) at
OPS; OPS - LAST +7.98 (VSA-PUB), +1.90 (VSA-BM), +3.75 (DISC), all AHEAD; each arm at its dev-best position (OPS for
all three) gives the OPS cells. Reads on test (trained types): read exact LAST 0.888 / 0.970 / 0.952, OPS 0.992 /
0.998 / 1.000 (PUB / BM / DISC); "followed" (correct when the module's answer is right and the gate open) 0.983-0.987
everywhere. Held-out types (addition, division): the VSA gate stays closed (open 0.7-2.1%) and the vector arms keep
the frozen LLM's answers (0.959-0.973); DISC opens on 57-67% and falls to 0.340 (LAST) / 0.441 (OPS), as in B40 / B50.
LAST VSA-PUB is B40's VSA arm reproduced bit for bit (same chosen settings; identical seed accuracies), so B40's
-3.2 moves to -5.0 only because DISC is now tuned (0.930 -> 0.948).

Reading of the tuned discrete module: its chosen write-back lr is 1e-2 at both positions (B40 used 1e-3); on dev the
gain is +1.0 point at LAST and +1.0 at OPS, on test DISC at LAST rises from B40's 0.930 to 0.948 at the same reader.

## Predictions

| | prediction | outcome |
|---|---|---|
| Q1 | DISC's chosen lr >= 3e-3 at both positions (dev) | held (1e-2 at both; 3e-2 tied, extended once each) |
| Q2 | VSA dev best spans <= 0.010 over the first grid; no VSA extension | held (0.005 / 0.0013 at LAST, 0 / 0 at OPS) |
| Q3 | OPS layer 17; OPS dev reads >= 0.98; LAST readers reproduce B40 within 0.005; lr 1e-4 not better | held (exact reproduction, weights identical) |
| Q4 | LAST PUB - DISC BEHIND, point in [-7, -3] | held (-5.02) |
| Q5 | LAST BM - DISC point in [-0.5, +2.5], not BEHIND | held (+1.30, AHEAD) |
| Q6 | OPS PUB - DISC and BM - DISC both EQUIVALENT, |point| <= 1 | partly failed: BM EQUIVALENT (-0.55) held; PUB INCONCLUSIVE (-0.78, lower bound -2.10) failed |
| Q7 | every arm at OPS >= at LAST; VSA-PUB gains >= +5 | held (+7.98 / +1.90 / +3.75) |
| Q8 | DISC at LAST >= 0.94 | held (0.948) |
| Q9 | second scorer 0 / 0; plain loop exact; determinism 2,500 / 2,500; dev optima interior | held |
| Q10 | compute <= 12 GPU-hours | held on claimed GPU time (4.82 GPU-hours); the per-process sum (13.51 lane-hours, up to four lanes per GPU) exceeds 12 |

One prediction failed outright (the published-read half of Q6); Q10 depends on the accounting and both numbers are
given.

## Checks (results/b65_check.txt)

- Second scorer (b40_check.py, gold rebuilt from the prompt text) on all 18 test files: 0 gold and 0 correctness
  mismatches. Plain-loop scorer (pred == gold) vs the verdict's numpy: max |difference| 0.
- Determinism: the three seed-0 LAST evaluations re-run give identical predictions and identical module records on
  2,500 / 2,500 items each.
- Reproduction of B40: the LAST lr-1e-3 reader snapshots equal B40's separately trained readers weight for weight
  (max |W - W_B40| = 0 at every epoch count, both encoders), and the LAST write-back dev curves at lr 1e-3 and 3e-4
  (VSA-PUB, DISC) equal B40's dev record to the printed digit.
- Dev-optimum-interior: all six (arm, position) choices are interior of their final grids. For DISC at both positions
  the optimum is a plateau (1e-2 = 3e-2, at its read ceiling) resolved by the registered tie rule after one extension.
- Procedure (compute notes 2-4): six OPS VSA dev runs died of out-of-memory at their first step (another user's job
  joined two of our GPUs) and were re-run once unchanged before any of them had produced a dev number; dependent dev
  phases and the test were chained onto the workers that already held GPUs; nothing was re-run on the test split.

## Sample reading by the AI-assisted process (results/b65_handread.txt; seed 0; all disagreements tagged, 12 per cell read)

- LAST, published vs DISC (206 disagreements): of the 148 items DISC gets and VSA-PUB misses, 142 are VSA misreads,
  and five of the six read drop or zero a digit under the 0.5 threshold (300 -> 0, 605 -> 5, 593 -> 93, 888 -> 88)
  or shift one (890 -> 880); the sixth is a closed type gate (similarity 0.785). Over all 148: 142 misreads, 2 closed
  gates, 4 write-backs not followed. The 58 the other
  way are mostly DISC misreads by one digit (660 -> 650, 139 -> 149). The -5.0 is DE25's thresholded digit read at
  the last-token state.
- LAST, best match vs DISC (92 disagreements): of the 60 items VSA-BM gets and DISC misses, 49 are DISC misreads (one
  digit off by one: 460 -> 450, 748 -> 738, 500 -> 400) and 11 DISC write-backs not followed (two of the six read are
  four-digit OR answers, 1023 written as 1027); the 32 the other way are 27 VSA-BM misreads of the same digit-space
  kind (109 -> 119, 493 -> 593), 3 closed gates, 2 not followed. The +1.3 is a read-accuracy difference at the last
  token (read exact 0.970 vs 0.952), not a write-back difference.
- OPS (31 and 23 disagreements): reads are almost perfect, and what is left is (i) four-digit answers (bitwise OR,
  answers 1000-1023, 52 of the 2,000 items): the write-back sets only the first token ("102"), the LLM chooses the
  second, and every module is near 0.4-0.5 there (analysis only: OR answers >= 1000, seed-mean, LAST 0.47 / 0.50 /
  0.39, OPS 0.49 / 0.39 / 0.48 for PUB / BM / DISC); all 11 VSA wins at OPS (7 + 4) are DISC write-backs not
  followed on bitwise OR (9 four-digit answers written 1024 / 1000 for 1023 / 1003 / 1006, and 876 written 878 twice);
  (ii) the VSA type gate: 3 + 5 items with type similarity 0.73-0.80, just under DE25's 0.8, where DISC's softmax
  gate is open; (iii) a few VSA operand reads that lose a two-digit operand's tens digit (21 -> 1,
  79 -> 9) or, with best match, copy x into y's role (y = 1 read as 21) - a cross-talk of the summed encoder
  enc(h_x) + enc(h_y). On the 1,942 trained-type items with answers below 1,000 (analysis only, no bar): OPS 0.9907 /
  0.9957 / 0.9993, LAST 0.9094 / 0.9731 / 0.9633.

## Literature (results/b65_literature.txt)

Read position dominating the comparison: SUPPORTED by the DE25 authors' follow-up ([D26T-ch8]: "encoding accuracy
plays a pivotal role") and by digit probes at the number's own token (Levy & Geva, NAACL 2025: per-digit base-10
probes on Llama 3 8B, errors close in digit space; snippet only). No precedent for a VSA-vs-discrete comparison across
read positions.

## What the paper may say about the forward pass

- Inside the frozen Llama-3.1-8B-Instruct forward pass on task N, with every module's write-back learning rate tuned on
  dev and three seeds per arm, the comparison depends on where the module reads:
  - reading the last prompt token (DE25's and B40's design), DE25's published vector read trails the same-shape
    discrete module by 5.0 points (0.898 vs 0.948, [-6.5, -3.6]); the best-match vector read leads it by 1.3 points
    (0.961 vs 0.948, [+0.4, +2.3]);
  - reading the operand tokens, all three modules read the operands almost exactly (read exact 0.992-1.000) and score
    0.977-0.985; best match is equivalent to the discrete module (-0.55 [-1.7, +0.5]) and the published read is
    inconclusive (-0.8 [-2.1, +0.5]); neither vector read leads.
- The read position matters more than the format: every module gains from operand reads (+8.0, +1.9, +3.8 points), and
  the residual errors at operand reads are shared by all modules (first-token-only write-back on four-digit answers).
- B40's -3.2 and B55's +3.0 are superseded and must be reported as such: the discrete arm there was under-tuned (lr
  1e-3; tuned 1e-2: 0.930 -> 0.948), and the best-match cell is now blind (write-back trained on argmax reads).
- The best-match read is our deviation from DE25's published module (B40 exploratory 1 / B55); the +1.3 must carry
  that qualifier and the read position.

## What the paper must not say

- That the vector format wins (or loses) on task N without naming the read position and the read rule: no read gives
  the same label at both positions.
- That the vector module leads at operand-position reads: the clean room's +0.95 (one seed, discrete lr 1e-3, its own
  items) is not reproduced with a tuned discrete module (-0.78, inconclusive; best match -0.55, equivalent).
- That B40's DISC WIN (-3.2) or B55's +3.0 is the tuned comparison, or that the published read's deficit is a property
  of the vector format in general: it is DE25's thresholded digit read at the last-token state (142 of the 148 seed-0
  items it loses to DISC are misreads), and it shrinks to -0.8 at operand reads.
- That the vector format is needed for the in-forward-pass gain: a discrete module of the same shape reaches 0.948
  (LAST) and 0.985 (OPS) with 0.65-1.06 M parameters against 16.8 M.
- Anything beyond Llama-3.1-8B-Instruct, task N's eight trained types, these two read positions, these two vector read
  rules and this write-back (last prompt token, first pass, 50/50 mix). The held-out-type gating result is unchanged
  from B40 / B50 (the vector gate stays closed; B50's caveat that a least-squares readout reproduces it still applies).

## Compute

13.51 lane-hours summed over processes (write-back training 12.45, readers 0.47, evaluations 0.55, caches 0.05) on
A800 GPUs 1, 2, 4, 5, 7 (6 claimed once, briefly, with no job), 4.82 GPU-hours of claimed GPU wall time; GPUs 0 and 3
never touched; another user's processes never touched. No paid API. Wall: registration 13:09, test finished 14:28.
