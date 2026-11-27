# B72 compute note 4 (2026-11-23, about 20:15 machine clock; no test output read)

- Language models, (A) / (D): development on the box finished at 20:03 (453 jobs, 0 failures); dev choices step 1
  committed (6b4589d, 20:12). LAMc+ kept LAMc's choice in all 48 language-model R1 cells (no sign-feature configuration
  beat LAMc's dev accuracy by more than 0.1 points), so there is no language-model LAMP test job.
- The language models' (A) test runs LOCALLY (465 OPT jobs, 6 workers, started 20:13) instead of on the box: the box's
  48 workers are busy with (B)'s development, and running the test here also removes the packed transport of compute
  note 2 for (A). Same code, same committed choices, same seeds.
- (B) development on the box is slower than planned (about 24 jobs per minute; each worker re-derives the panel's
  whitening and principal bases once per encoder in each engine). At 20:13: 216 of the 2,376 remaining (B) jobs done,
  0 failures. Decision point fixed now: if (B)'s development has not finished by 20:50, (B) is cut in the registered
  order (NLMSC, then VSAR - SDMT, then K = 1e3) so that what remains can be tested once and analysed inside the time box.
- Local checks started on the sentence encoders: float64 re-derivation (b72_check.py rederive; output not read).
