# B56 note after the test (test run once; results/b56_verdict.txt, b56_predictions.txt, b56_check.txt)

Primary (registered bar): R1 VSAS - LAM, not behind (95% lower bound >= -2 points) on 5 of 7 encoders (all four LLM
families and bge; TOST-equivalent at +-2 on the same 5), behind by more than the margin on e5-base-v2 (-2.70 [-3.42,
-1.99]) and MiniLM-L6 (-1.89 [-2.64, -1.15]); ahead on none. Per the fixed readings, "a least-squares associative
memory beats the store at equal bits over R1" stands for e5 and MiniLM.

What the VSA-internal rules add over B53's matched-filter store (R1, points): LMMSE read (training-split statistics)
-0.03 to +0.55; retrained bundle -0.33 to +0.82; widened grid with sparse ternary maps -0.48 to +0.48; the
dev-selected VSAS -0.16 to +0.70. None closes the V = 2 gap to LAM on the LLMs (R1-V2 VSAS - LAM -2.5 to -3.9).

The R1 level result on the LLMs is an average of a V = 2 loss and a V = 16 win. Chance-normalised (per-cell
difference / (1 - 1/V)), R1 VSAS - LAM is below 0 with the interval below 0 on all 7 encoders (-1.1 to -4.4).

Feature confound: LAM restricted to the store's own features (LAMRF) loses 0.8-3.3 points over R1 against B49's LAM on
every encoder (interval below 0 on 7 of 7). Like-for-like (same features), R1 VSAS - LAMRF is +0.5 to +1.6 (interval
above 0 on 5 of 7, level on e5 and bge); chance-normalised it is level on 6 and ahead on Mistral. At V = 2 alone least
squares on the same features is still ahead by 1.3-1.7 points on 6 of 7 (MiniLM level): that is the remaining
write-rule gap; at V = 16 the store is ahead of LAMRF by 1.7-4.7 (LAM's c = V - 1 columns).

Failed predictions: P2 (retraining helped at V = 2 with the interval above 0 on 2 of 7, not >= 5), P3 (not behind on
5, not >= 6), P5 (dev chose a sparse map in 5 of 12 R1 cells on e5; its second half held). Held: P1, P4, P6, P7, P8.
Checks: plain-loop scorer 0 difference; 19,600 logged sizes, 0 over budget, 0 not maximal; VSAL at alpha = inf
reproduces B53's VSA (agreement 0.995-1.000; every one of the 568 differing queries is an exact tie on the quantizer
lattice, sgn features with q = 2); float64 re-derivation 0.992-1.000 (worst cells 0.97, e5 / bge, sign features with
integer scores, where ties decide reads and retraining steps).

Disclosed analysis additions after the test (no arm, choice or bar changed): b56_check.py 'ties' (to test the
reproduction differences) and two more chance-normalised pairs in the verdict (VSAS - LAMRF, VSA - LAM).
