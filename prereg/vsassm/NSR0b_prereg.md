# NSR0b pre-registration: is V-delta's advantage under noise a VSA property, or rule pruning?

Date: 2026-11-06. Written while NSR0 (prereg/NSR0_prereg.md, amendment 1) was still running, after seeing its
per-cell log lines and before scoring it; committed before any NSR0b code runs.

## Why

NSR0's log shows V-delta far above the table T in some noisy cells, e.g. label noise 0.3 at m = 3: V-delta 0.780,
T 0.331; label noise 0.2 at m = 3: 0.803 vs 0.590. A likely mechanism has nothing to do with vectors: V-delta's
`known` test (top cleanup similarity >= tau = 0.4 with alpha = 0.3) passes only after two or more consistent writes,
so it drops rules with weak or split support, and CYK then routes around them. T keeps every rule with one vote, and
under noise those rules make CYK return several distinct results (AMBIG, counted wrong). A table can prune the same way.
If NSR0's A2 bar passes, it would be a true but misleading result unless this control is run.

## Design

- **T-prune:** NSR0's count table, except `known(a, b)` requires the winning value to have >= s votes and a vote
  share >= q. The same rule applies inside closure (as V-delta's `known` does).
- **Tuning (validation only, NSR0's criterion and draws):** s in {1, 2, 3}, q in {0, 0.5, 0.6, 0.7}; mean
  validation accuracy over (m = 3, clean) and (m = 3, label noise 0.2), 3 seeds; ties: first in grid order. V-delta keeps
  NSR0's frozen tau = 0.4, alpha = 0.3.
- **Cells:** every NSR0 S, C, L and F cell, 10 seeds, the same training sets and codebooks as NSR0 (same seeding code,
  imported unchanged).

## Bars

- **B1 (VSA beyond pruning):** V-delta beats T-prune by >= 0.02 overall in some cell, paired difference > 0 in >= 9 of
  10 seeds, V-delta accuracy >= 0.50 in that cell. If B1 fails everywhere, V-delta's NSR0 wins are attributed to
  pruning, not to the vector store.
- **B2 (pruning explains the gap):** in every cell where NSR0's A2 bar passes for V-delta, T-prune is within 0.02 of
  V-delta or above it.
- **Exploratory upper bound (no bar, reported as such):** for each cell, the best (s, q) for T-prune and the best
  (tau, alpha) for V-delta chosen on the test cell itself; this checks that the comparison is not an artifact of the
  validation-chosen parameters. It favours both arms equally and is never quoted as a result.

## Verification

Independent scorer `nsr/nsr0b_score.py` (written before the run, recomputes from predictions and test.csv); manual
inspection of 20 test stories where V-delta is right and T is wrong in NSR0 (is T's error AMBIG from a weakly supported
rule?).

## Predictions

B1 fails in every cell; B2 holds: T-prune matches or beats V-delta wherever V-delta beat T. In the upper bound,
best-in-cell T-prune >= best-in-cell V-delta in every cell.
