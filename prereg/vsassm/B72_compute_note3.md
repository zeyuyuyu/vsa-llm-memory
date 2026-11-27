# B72 compute note 3 (2026-11-23, about 20:00 machine clock; no test output read)

- Sentence encoders: development (348 jobs, local, 0 failures), dev choices committed (38ad411, 19:51), test run once
  locally (405 jobs, 19:51-19:55, 0 failures). (C) on the four headline contrasts (saved outputs of B60 / B62 / B71
  only) computed by b72_verdict.py cpart for all seven encoders; part files of the sentence encoders computed; none of
  their numbers read (only exit status, run time and the "missing" count).
- Box development re-ordered (procedure only): the runner (PID 4084043) ran the jobs in its built-in "long jobs first"
  order, which put (A)'s OPT jobs last although (A) has the highest registered priority. At 19:58 the runner and its 48
  workers were stopped by exact PIDs (432 of 3,261 jobs finished, 0 failed; no partial file left) and restarted as a
  chain on the same 48 workers: first the (A) / (D) jobs (OPT, LAMP: 453), then the (B) jobs (2,376 left). Jobs already
  finished are skipped; results depend on registered seeds only.
- To start (A) / (D)'s language-model test while (B)'s development still runs, the dev choice is made in two steps
  with the same registered rules: first (A) / (D) only (b72_run.py choose ENC --ks none; committed), then the full
  choice including (B) after (B)'s development (the (A) / (D) entries recomputed identically; committed again before
  (B)'s test). b72_run.py's command line and b72_box.sh's choose / plantest steps changed accordingly (procedure only;
  job code unchanged); the box copy replaced atomically (temporary file + mv) while the chain ran.
