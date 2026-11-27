# B35 note after the test run (2026-11-18, written by the parent session after the B35 agent handed back)

**What happened.** b35_data._make_names drops candidate names that appear in /usr/share/dict/words. That file is absent on the A800 box (no filtering) and is linux.words (479,826 lines) on PACE. The name pool therefore differs between machines, and so do the test instances. Box and PACE copies of the same run share 43 of 3,600 question texts. Every run within one machine shares its questions exactly (22/22 PACE runs, 3,600/3,600).

**Consequences.**
- The registered primary comparison (VSA vs learned tokens at the same B and seed) is paired within one machine for every (B, seed). B4/B8 seed 0 were scored from the box for both arms; every other pair was scored from PACE. Its paired CIs are valid.
- The reference columns (T-B, Vq, ext) were generated on one machine's questions. Where they are set beside box-scored VSA cells (B4/B8 seed 0) they are unpaired and descriptive. No bar uses them.
- In b35_collect.py, "same correctness" between box and PACE copies compares different questions, so it is not meaningful. The per-cell accuracy differences (max 0.023) are an unpaired reproducibility check.

**Second scorer.** b35_check.py regenerates the questions locally, where the dictionary differs again, so at first it flagged every run. Re-run under each machine's dictionary (b35_out/test runs from the box: no dictionary; runs from PACE: PACE's linux.words, copied), every run passes with 0 mismatches except test_tfull_zeroshot. There the run script and the checker each apply a strict rule (whole string vs first line). Neither is the registered lenient rule (gold as a whole word in the first line), which b35_verdict.py applies. The zero-shot reference carries no bar.

**Hand-read.** The soft-memory arms answer the prior: 11–18 distinct answers over 3,600 questions, dominated by one value per attribute (e.g. 'alpaca' 630, 'cello' 537). This holds for VSA and learned tokens alike.

**Fix for any follow-up.** Remove the dependence on /usr/share/dict/words (ship a fixed name list).
