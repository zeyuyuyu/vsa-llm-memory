# B58 compute note 2 (during dev pass 1; written before dev pass 2 and before any test run; no test output exists)

Measured speed: the four 6,000-pair dev stores take 3-6 minutes per job of 6 P-LDPC (budget, design) entries; the
cost is belief propagation on words that do not converge (60 iterations each) while the store still reads, so no
censoring stops them (e.g. b = 76, R = 9/10, B = 256, k8: all 446 words fail, 149 of 150 keys exact, 15 s). Dev pass 1
will end about 21:25 machine clock; the registered plan (full pass 2, then the test with 23 conditions and the E-LDPC
re-score) would not finish in the time budget. Cuts, fixed now:

1. **Dev pass 2, P-LDPC:** only the 3 quarter steps below each (condition)'s pass-1 N* (where the dev N* can move),
   not the 3 above (pass 1 already reaches the bar there, and the octave designs above remain). P-BCH (4 designs)
   keeps the registered +-3 steps.
2. **Design at a test budget that dev did not evaluate:** the nearest dev-evaluated budget's design among those that
   fit at that budget (b / R <= B), ties to the larger budget; if none fits, the nearest one (as registered). Without
   this, budgets just below a dev budget would inherit a design that does not fit and could not be evaluated,
   which would bias the protected store's N* upwards.
3. **Test order:** the 7 i.i.d. conditions (primary) run first, with their extensions; the 16 damage conditions
   (secondary) run after them, beside the E-LDPC re-score. If the damage conditions have not finished their
   registered window and extensions by 23:20 machine clock, the unfinished ones are reported as not run (no partial
   N* is reported for them).

Nothing else changes (designs, grids, rules, bars, predictions).
