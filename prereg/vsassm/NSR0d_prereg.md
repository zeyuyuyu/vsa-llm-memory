# NSR0d pre-registration: does V-delta beat every symbolic table when one story per chain is noisy?

Date: 2026-11-06. Written after NSR0b's exploratory upper bound (parameters chosen on test; nsr/out/nsr0b_upper_score.log)
showed the best V-delta above the best T-prune in the m = 1 noisy cells (L_p0.1_m1 0.700 vs 0.549; L_p0.2_m1 0.446 vs
0.301; L_p0.3_m1 0.302 vs 0.201; F_p0.1/0.2_m1 +0.02/+0.04). Committed before any NSR0d code runs. This is the one
regime left where the NSR0 conclusion ("VSA adds nothing over a table") could be wrong.

## Why (diagnosis from the upper-bound predictions, L_p0.2_m1)

Plain T learns 176 rules (117 when clean), and 77% of its answers are AMBIG: a wrong k = 2 rule spawns wrong derived
rules in closure. V-delta keeps 121 rules and 41% AMBIG: a derived key is known only after several closure writes,
while single-vote spurious keys sit near tau. T-prune cannot express this, because it applies one threshold inside and
after closure: s >= 2 also deletes every single-vote k = 2 rule, so closure never starts.

## Arms

- **T-stage:** count table with separate thresholds for rules first written from k = 2 stories (s2, q) and for rules
  first created in closure (sc, q); `known` applies them inside closure and at inference.
  Grid: s2 in {1, 2}, sc in {1, 2, 3}, q in {0, 0.5, 0.7}.
- **T-stage+vote:** T-stage, and when CYK derives several values, answer the value with the most derivations
  (bracketings) instead of AMBIG; ties stay AMBIG. Same grid.
- **V-delta:** tau in {0.3, 0.4, 0.5, 0.6, 0.7}, alpha in {0.3, 0.5, 1.0}.
- Tuning, every arm, validation only: mean validation accuracy over five cells, m = 1 clean, m = 1 label noise 0.2,
  m = 3 clean, m = 3 label noise 0.2, m = 3 fact noise 0.2; 3 seeds; ties: first in grid order.
- **Best symbolic arm** = whichever of T-stage and T-stage+vote has the higher validation criterion (decided before
  test).
- Cells: every NSR0 S, C, L and F cell; **fresh seeds 20-29**.

## Bars

- **D1 (VSA beats every symbolic table):** V-delta beats the best symbolic arm by >= 0.02 overall in some cell, paired
  difference > 0 in >= 9 of 10 seeds, V-delta >= 0.50 there.
- **D2:** in the m = 1 label-noise cells, the best symbolic arm is within 0.02 of V-delta or above it (reported per
  cell; usefulness floor not applied, since these cells are the point).
- Independent scorer `nsr/nsr0d_score.py`, written before the run.

## Predictions

D1 fails and D2 holds in all three m = 1 label-noise cells: staged support pruning plus derivation voting matches the
graded accumulation. Confidence is moderate (about 60%). If D1 passes, the verdict changes to: VSA's graded
accumulation helps when every rule is seen once under label noise, which a count table with staged thresholds and
derivation voting does not reproduce.
