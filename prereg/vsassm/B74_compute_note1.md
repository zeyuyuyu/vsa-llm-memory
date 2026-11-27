# B74 compute note 1 (2026-11-23, about 22:50 machine clock; before any development or test run)

1. **Pre-registration** committed 71499dd (22:40:07) with b74_run.py, b74_verdict.py and the selftest log; OpenTimestamps
   stamp committed c47ee65 (22:40). The first background poll loop called `ots upgrade` on the .md file instead of the
   .ots file (error "not a timestamp file", once, 22:40:39); stopped and restarted on the .ots file at 22:44 (first
   answer: pending at all four calendars). Polling every 10 minutes, `ots upgrade` only.
2. **Wording of the pre-registration, clarified (no analysis changed).** "Path 1 ... path 2 ... they differ by exactly
   I_X" means: both paths sum to the same gap G_X; the write terms of the two paths differ by I_X (W_X - W_R = I_X) and
   the feature terms by -I_X (F_X|H - F_X|L = -I_X). The checks' "path 1 - path 2 = I_X" is accordingly checked as
   W_X - W_R = I_X, F_X|L - F_X|H = I_X and path 1 = path 2 = G_X (to 1e-12). The attribution rule is unchanged.
3. **Scheduling change in b74_run.py (before any run):** FX jobs at M = 1,024 are one store per job (job name carries the
   store indices) so that the box's workers are used; M = 256 jobs keep all stores of a group. Results depend only on the
   registered seeds (a store's FX computation does not depend on the other stores in its job).
4. **b74_check.py** written (scorer, bits, re-derivation, reproduction, determinism, report); not run.
5. **Box:** ~/b74_run made from ~/b72_run's code (md5s of the 12 shared files equal this repository's; B49's 7 panel
   code files md5-equal to the local ones), data via symlinks (dat/ for b53_run's per-encoder directories, B53_DATA);
   b57_run.py, b60_vsapc.py, b74_run.py uploaded (md5 check pending). Plans generated (dev FX: 84 language-model jobs on
   the box, 68 sentence-encoder jobs locally; test label plans 932 / 649 jobs). No dev or test store answered.
