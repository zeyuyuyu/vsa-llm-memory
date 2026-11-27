# B75 compute note 2 (23:41 machine clock; development running; no development output has been read except job-completion lines)

- **Bitcoin attestation.** The OpenTimestamps proof of prereg/B75_strongest_options_prereg.md was upgraded at 23:37:58
  machine clock: **Bitcoin block 977225** (attestations from the alice and bob OpenTimestamps calendars), 49 minutes after
  the stamp (22:48:51). The upgraded proof is committed as a005f02 (23:38:09). (A copy of the proof was upgraded first in
  the scratch directory to see whether the calendars had the attestation; the background loop, which polled every 10
  minutes, was then stopped and `ots upgrade` run once by hand on the registered file.)
- **Development started after the attestation**: 23:38:28 on the A800 box's CPUs (~/b75_run, `b75_box.sh dev
  results/b75_plan_dev.json results/b75_dev.json`, 48 workers, nice 10; b75_strong.py, the plan and the dev stores
  MD5-identical to the local commit: 9372036..., 1bf27c4..., c3ee82d...). 3,000 jobs (the registered windows and designs,
  every memory at once, largest jobs first).
- **Fresh stores on the box** for the secondary part (D): b73_out/test_stores.json.gz delivered in MD5-verified pieces
  (the whole file 31ba48c5..., identical to B73's). Not read by any B75 code yet.
- **Box environment check** (synthetic pairs from b63_online.synth only, sent on standard input, no file left on the
  box): every memory's exact counts identical to the same check on this machine (U-LS, U-J both accountings and every T,
  H-SBC-J, O2-BCH, O2-LDPC). The box's own quick self-test could not run there (the system word list that the synthetic
  B73-style stores use is not installed on the box); nothing else depends on it.
