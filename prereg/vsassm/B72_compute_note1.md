# B72 compute note 1 (2026-11-23, about 19:50 machine clock; no development or test output read)

- Pre-registration ec233f4 (19:40:33), OpenTimestamps proof d7d6186 (19:40:44); both before any development run.
- Development split by machine (procedure only; every job's result depends on registered seeds only):
  box (~/b72_run, 48 workers, nice 5): the four language models' OPT, LAMP and every (B) job (3,261 jobs);
  local (this machine): the three sentence encoders' OPT and LAMP jobs (348 jobs).
- Local workers: started with 10 at 19:42; the machine's free memory fell to about 4 GB (other users' jobs hold
  about 14 GB), so the run was stopped by exact PIDs at 19:43 (10 finished job files kept; partial .tmp files
  deleted) and restarted with 6 workers. No output was read.
- The link to the box is slow (about 3 KB/s for an uncompressed copy); plans and choice inputs go compressed, the dev
  choices are computed on the box by the registered rule (b72_run.py choose, the same code) and only the choice files
  come back; test predictions will be packed and copied back after the test.
- Box code md5s equal the repository's (b32_common, b32_run, b36_common, b49_common, b53_run, b56_run, b60_run,
  b62_run, b64_run, b66_lsfc, b71_run, b72_run); panel codes checked against b71_box.sh's md5 table; the B53 / B56 /
  B60 choice files used by the box are copies of results/ (md5 listed in ~/b72_run/choice_md5_box.txt).
