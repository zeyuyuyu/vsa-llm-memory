# B58 compute note 5 (22:38 machine clock; hard deadline from the user: results by about 23:05; no test output read)

The registered test (700 i.i.d. jobs, then extensions) and the E-LDPC re-score (1,108 jobs) would take hours more. Both
runs are stopped now; every job already finished and saved in their checkpoints (results/b58_test_iid1.json.partial,
results/b58_ldpc_rescore1.json.partial; saved every 5 minutes, later jobs lost) is kept. The remaining work is cut by
these rules, fixed before any test output is read:

**Part (a), i.i.d. (primary bar).**
- Variants: both P-LDPC and P-BCH at k1, k2, k4 (cheap budgets); at e05, k8, k16, k32 only the variant with the
  smaller dev N* (e05 P-BCH 108 vs 128; k8 P-BCH 181 vs 256; k16 P-LDPC 1024 vs 1722; k32 P-LDPC 4096 vs not
  reached). P-best at those four levels is then that variant's test N*; if the other variant would have been better
  on test, P-best is overstated (against the store).
- Budgets: dev N* +- 2 grid steps (5 budgets) instead of +- 4, with the dev designs (compute note 2's nearest-fitting
  rule between dev budgets). No extension passes. Where the test N* is not bracketed in this window, it is reported as
  a bound (">= top of window" not reached, or "<= bottom of window").
- Every store (all 50) at every kept budget, so N* keeps its pooled definition. Evaluations already in the checkpoint
  count; outside the kept budgets they are reported but not used for N* unless all 50 stores have them.
- Part (a) damage conditions: not run (compute note 4).

**Part (c) re-score of E-LDPC with OWN.**
- i.i.d. conditions: not re-scored. The primary bar uses B38's E-LDPC i.i.d. N* (PKG); evidence that OWN = PKG on
  i.i.d. inputs: c2 channel A (same words in every setting), B38's exploratory own-decoder run (identical shares at
  k8 / k16), c1's i.i.d. cells (all agree). Stated as such.
- The 16 Fig. 5b conditions, budgets fixed from results that existed before B58: natural layout, the 5 grid budgets
  from 4 steps below to the octave N* of B38's exploratory own-decoder run (lines k4 128, e05 512, k8 4,096; clusters
  k4 128, e05 512, k8 1,024, k16 2,048; lines k16 never reached -> 4,096 only); interleaved, B38's E-LDPC N* +- 1 step
  (lines 27 / 45 / 76 / 215, clusters 27 / 38 / 64 / 304 at k4 / e05 / k8 / k16), designs as B38 (dev rate for the
  matched i.i.d. condition and budget). No extension; unbracketed N* reported as bounds.

**Checks** run on reduced samples (second scorer on 6 cells, decoder agreement on 2 P-LDPC and 1 P-BCH cells), all
else as registered.
