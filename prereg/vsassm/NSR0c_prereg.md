# NSR0c pre-registration: confirmatory re-test of V-delta vs a pruned table, both tuned with fact noise in view

Date: 2026-11-06. Written after NSR0b was scored (B1 passed in F_p0.1/0.2/0.3_m3; B2 failed there; my prediction was
wrong for fact noise) and BEFORE running NSR0b's exploratory upper bound, so nothing tuned on test informs this design.

## Why

NSR0b's T-prune was tuned on validation cells that contain label noise but no fact noise. There, s = 1 and s = 2 tied
(0.9761) and the tie rule picked s = 1. Fact noise creates spurious keys with a single vote (share 1.0), which a share
threshold cannot remove but a minimum support of 2 can; V-delta (alpha = 0.3, tau = 0.4) needs about two consistent
writes before a key is known, which is a minimum support in disguise. The hypothesis: V-delta's fact-noise advantage
is support pruning, which a table does natively. Alternative: the vector store's graded accumulation does something a
support threshold cannot.

## Design

- **Arms:** T-prune (s in {1, 2, 3}, q in {0, 0.5, 0.6, 0.7}) and V-delta (tau in {0.3, 0.4, 0.5, 0.6, 0.7}, alpha in
  {0.3, 0.5, 1.0}); plain T for reference.
- **Tuning, both arms, validation only:** mean validation accuracy over three m = 3 cells (clean, label noise 0.2,
  fact noise 0.2), 3 seeds (NSR0's validation draws, fact-noise cell added with the same draw procedure); ties: first in
  grid order. Both arms get exactly the same tuning budget and criterion.
- **Cells:** every NSR0 S, C, L and F cell, **fresh seeds 10-19** (new training draws and new noise; the test split is
  unchanged). Same code paths as NSR0/NSR0b.

## Bars

- **C1 (VSA beyond pruning):** V-delta beats T-prune by >= 0.02 overall in some cell, paired difference > 0 in >= 9 of
  10 seeds, V-delta >= 0.50 there.
- **C2:** in every cell where V-delta >= 0.50, T-prune >= V-delta - 0.02.

## Predictions

C1 fails in every cell and C2 holds: with fact noise in the tuning criterion, T-prune picks s >= 2 and matches or beats
V-delta. If C1 passes, the vector store's graded accumulation does something a support threshold does not, and I will
say so.
