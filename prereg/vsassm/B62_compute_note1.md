# B62 compute note 1 (during dev; no test store built, no dev or test output read)

Written 2026-11-23 about 02:36 (machine clock), before any dev output is read.

1. **Pre-registration time.** The header says "about 02:30"; the pre-registration and code were committed at 02:28:19
   (ae3c72f) and the OpenTimestamps proof at 02:28:21 (1b73db8). The commit times are the record.
2. **Selftest change before the commit (toy data only).** The first selftest run failed one assertion of the toy
   store: one pass of NLMS (mu = 1) read 0.75 of a 200-fact toy store at 64 bits per fact, where p is about M (one pass
   of the delta rule cannot fit M facts exactly with p ~ M features; the dual-form and recursion checks had passed). The
   toy budget was moved to 256 bits per fact (p >> M), where every toy configuration reads >= 0.9; the code comment
   records this. The committed code is the code after this change; no dev, test or training store was involved.
3. **Box.** ~/b62_run (fresh), code md5 b32_common d6e1231b, b32_run bf5510f5, b49_common f2228be9, b60_run 520af3db,
   b62_run ca210944 (= the committed files); panel codes copied from ~/b60_run's B49 copies, md5 identical to B60's
   table. Training-split feature norms (b62_run.py trainnorm, 2,000 training codes per encoder) ran 02:29-02:31; sanity:
   rp s_p at p = 16,384 is p / n (4.0 for n = 4,096; 21.3 for 768; 42.7 for 384), pca s_n = 1.000 (unit rows).
4. **Dev.** plan_dev.jsonl (252 jobs: 7 encoders x V 2 / 16 x 9 dev stores x 2 preps, every NLMSc / RLSc / RLScS
   configuration at 4 / 8 / 16 bits per fact, ideal; md5 235c3409...) started at 02:31:29 with 48 workers. The timing run
   on a training-split store (b62_run.py timing) was not run; the dev jobs' own times are logged.
5. **Verdict and check code** (b62_verdict.py, b62_check.py) committed at 02:33:22 (c042f9d), during dev and before any
   dev output was read; the verdict was exercised on fabricated B62 rows only (predictions = gold with random noise);
   its STORE - LAMc for MiniLM reproduced B60's (-0.19 / -8.41 / -4.30).
