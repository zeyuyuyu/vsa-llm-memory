# B75 compute note 1 (written about 23:05 machine clock, while waiting for the Bitcoin attestation; no development or test run has started and no development or test store has been read by B75 code)

- **Registration.** Pre-registration and code committed eaa649d (22:48:43 machine clock; the header's "about 22:50"
  was written ahead); `ots stamp` and the pending proof committed 504b449 (22:48:51). Verdict and check code (exercised
  on fabricated records and synthetic stores only) committed 0a95bca (22:54:37) and 43f2c4b (about 23:00; the source
  audit no longer flags the local name `vals`, a false positive). A background loop runs only `ots upgrade` on
  prereg/B75_strongest_options_prereg.md.ots every 10 minutes (at most 14 tries) and stops at the first Bitcoin block
  attestation. Development waits for it (or for 2 hours, i.e. until 00:49, said in a later note).
- **Shared scratch space.** The session scratch directory is shared with other agents (B74 is running at the same
  time); an unrelated agent writes to a log file of the same name as my OpenTimestamps loop's, so the loop's log
  interleaves with theirs. The loop's decision reads `ots info` of the B75 proof only; my later files are in a b75/
  subdirectory.
- **Box.** ~/b75_run on the A800 box: the unchanged code and stores copied on the box from ~/b63_run and ~/b61_run
  (b38_eq, b34_inmem, b25_pcm, b22_robust, b26b_pcm_table, ruler_vsa, ruler_rm, b63_online, b61_hd, the dev and test
  stores: MD5-identical to the local files), the new files uploaded as one tarball (b75_strong 9372036..., b75_box,
  b75_check, b75_verdict, b69_vocab, b73_cells, results/b75_plan_dev.json 1bf27c4...: MD5-identical). The link stalled
  twice (one scp of the files timed out at 300 s, one ssh at 60 s; both retried once by hand after a read-only
  connectivity probe; nothing on the box was changed by the failed attempts except partial copies that were
  overwritten). The box's quick self-test (synthetic stores) is run before development to check its environment.
- **Development plan** (b75_strong.py devplan, the registered windows and designs): 1,194 (memory, parameter, rate,
  budget) entries, 1,932 evaluations per store, 3,000 jobs on the 12 dev stores.
- **Timing on synthetic stores** (seed tag b75-selftest; this machine, one thread): H-SBC-J lam 0.5 at 304 cells per
  pair and x16 took 159 s for one evaluation (free accounting, all T) on 5,993 pairs, so H-SBC-J at x16 dominates the
  cost (about 12 CPU-hours of development); development in all is about 25-30 CPU-hours, about 35 minutes on 48 box
  workers.
