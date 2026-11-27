# B71 compute note 7 (after the test; corrections of check code and one unexecuted sub-rule, stated before the note)

Date: 2026-11-23, about 18:52 (machine clock). The verdict (b71_verdict.py part / panel), D2's verdict and the dev audit
summary were computed once from the committed choices and the test jobs; no arm, choice, bar or rule was changed after
the test. The following changes are to CHECK code and were made after its first output had been read:

1. **NLMSFR re-derivation quantizer.** The first float64 re-derivation (b71_check.py rederive) used an unscaled sign for
   q = 1. Least squares' registered 1-bit quantizer (b60_run.quantW) is the sign times the group's mean |w|, which matters
   with per-column groups. The check now uses B66's independent quantizer (b66_lsfc.own_quant). First run: 4 of 40 stores
   below 0.99 (min 0.903); corrected: 56 of 56 >= 0.994.
2. **Polyak average in the check's misread-only loop.** The check's own 'err' loop did not implement the averaged output,
   which the dev rule chose in a few cells; b46 and bge stopped with an assertion. Implemented (equal to b71_run.delta_write
   to 3e-17 on toy data) and re-run for b46 and bge.
3. **The store's tie rule in the VSAPR re-derivation.** The check sent exact zeros of the bundle to +1; the store's
   quantizer (B32) sets them to +-1 from a generator seeded ('Q', store seed, d, q, stored count, block). With sign
   features the bundle holds integer sums, so exact zeros are common. Re-implemented from that specification and re-run
   for VSAPR on all 7 encoders: 55 of 56 >= 0.99 (first run 53 of 56); the remaining store (MiniLM, centre, sgn3, p = 256,
   V = 2) agrees 0.966 although its own basis equals the pipeline's to 5e-7. Both runs are in results/b71_rederive_*.json
   (suffix _NLMSFR / _VSAPR = the corrected runs) and results/b71_check.txt.
4. **D1 labels not recomputed (registered sub-rule not executed).** The pre-registration said that D1's labels are
   recomputed if at least one arm of a pair is storable in at least one cell. On the 4 language models no arm is storable
   (every arm's side information exceeds the whole budget in 12 of 12 cells), so D1 is "no arm storable" there, as
   registered. On the sentence encoders the store (centring only: a mean of 384 or 768 numbers) is storable in 1 to 3 of 12
   cells, LAMc in 0 to 2, SDMt in none; but storing the side information leaves less than the stored vector's bits, so a
   recomputed label would need re-runs at budgets reduced by the side information, which the pre-registration did not
   specify. Not done; reported as a deviation.
5. results/b71_check_bits.json now holds the bits audit of all 303,527 dev and test rows (the dev-only file of note 5 had
   296,862 rows; both 0 over budget, 0 not maximal).
