# B59 compute note 5 (03:11 machine clock; after the i.i.d. part 2 verdict was first computed and read)

**A plan defect, found while reading the first i.i.d. part 2 verdict (its coverage table).** The full-grid plan
(b59_store.plan, and plan_dmg) skipped a budget when b / R > B, with b taken from the choice file, where it is stored
rounded to 4 decimals (b58_store.choose's json; 76.1093 for 76.10925...). Where a dev-chosen design fills the budget
exactly at one copy (b / R = B: 76.1 / 0.5 = 152.2, 76.1 / 0.25 = 304.4, 90.5 / 0.25 = 362.0) the rounded value made
it look infeasible, so the plan dropped budgets that dev had evaluated and that the registered plan requires ("every
grid budget from 90.5 to 4,096 with that family's dev design"): P-LDPC at 152 (k1, k2, k4, e05, k8), at 304 and 362
(k8); in the damage plan P-LDPC at 152 (all four conditions) and 304 (k8 conditions). (B58's extension rule has the
same check; B58 is not re-run.) The fix: a relative tolerance of 1e-5 in the fit test (b59_store.fits); nothing else
changes. The corrected plans are results/b59_plan_full2.json and results/b59_plan_dmg2.json; the dropped entries form
tier T6 (results/b59_plan_fix.json: 3 (family, design, budget) entries, 650 store evaluations), run now on the box.
The verdict is recomputed on the corrected plan; the first verdict output (results/b59_store_verdict.*, written 03:10)
is overwritten and is not a result. Budgets where the dev design genuinely does not fit (all dev designs censored
there, so B58's rule copied the nearest budget's design; e.g. P-LDPC k32 at 91-128 with b = 128, R = 9/10) stay
not evaluable, below each family's N*, and are listed in the verdict's coverage table.

This fix is mechanical (the registered plan already required these budgets); it was not chosen by its effect. Its
possible effect is stated here before the T6 output is read: the added budgets lie at or above some families' test
N* (P-LDPC k8: 152, 304, 362), so a failure there can only raise that family's N*; P-best's N* is set by U at every
k in the first read and U is complete on the full grid, so H1-P can change only if a coded design passes at a budget
below U's N* (152 at k1-k4 and e05 is above U's 76 / 91, so it cannot).

T5 (damage, registered): 400 jobs finished at about 03:11; its extension (B38's rule, b59_store.extend_dmg, with
fits()) and T6 run next on the box.
