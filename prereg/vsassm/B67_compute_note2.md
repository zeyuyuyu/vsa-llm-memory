# B67 compute note 2 (during (a); before any language-model theory number)

Machine clock 2026-11-23 15:11 (Asia/Shanghai). Correction: compute note 1 was written at about 15:03, not 15:12.

1. **What was seen.** (b) ran to completion at 15:09 (results/b67_b.json); its numbers were printed in the log and
   read. In (a) the three sentence encoders' registered theory (36 cells) finished first, and b67_verdict.py was run on
   them (to test the verdict code) and its output read: theory H (store) about 2 points under the measured accuracy,
   theory L about 5 to 6.5 points under the measured least squares, so the predicted gaps had the wrong sign in most
   cells. No language-model theory number had been computed when this note was written.
2. **Diagnosis on synthetic data only** (scratchpad diagL.py; isotropic toy keys of the C3 selftest): the Gaussian race
   on the exact per-query moments of a direct ridge simulation reproduces the simulation (0.437 vs 0.425), the
   deterministic-equivalent signal matches (0.1485 vs 0.1488), but the registered cross-talk r + g over-states the
   simulated cross-talk (0.189 vs 0.149). Cause, a derivation slip: r + g drops the cross term between the stored key's
   residual and the alias perturbation. Through the leave-one-out inverse (Sherman-Morrison) the cross-talk of query q
   with the other stored keys is (kappa' / M) v^T Sigma (Sigma + kappa)^-2 v with v = f(q) - mu f(k); r + g keeps only
   its two square terms (with 1 - h for 1 - mu).
3. **Exploratory variant L', post hoc and labelled so everywhere.** b67_theory.theory_L(corrected=True) uses that
   cross-talk; nothing else changes. Synthetic selftest: L' 0.621 / 0.716 / 0.425 / 0.650 against simulation 0.623 /
   0.718 / 0.425 / 0.629 (registered L 0.608 / 0.695 / 0.372 / 0.646). L' was written after the sentence encoders'
   registered (a) verdict had been read, so on those encoders it is not a prediction; on the four language models it is
   computed in the same pass as the registered L, before either is read. The registered L stays the primary theory;
   every (a) statistic is reported for L and, separately and labelled exploratory, for L'.
4. **Incident.** A `pkill -f` whose pattern also matched its own shell line killed that shell and the four
   language-model (a) processes (no cell had finished); they and the sentence encoders were restarted at 15:10 with
   the code above (the registered L / H code paths are unchanged; the sentence encoders' registered numbers are
   recomputed by the same deterministic code).
