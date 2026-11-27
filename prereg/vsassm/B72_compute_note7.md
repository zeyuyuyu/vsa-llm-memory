# B72 compute note 7 (2026-11-23, 20:35 machine clock; written BEFORE any B72 test output is read)

- Tests run once: sentence encoders (A) / (D) 405 jobs (local, 19:51-19:55); language models (A) 465 jobs (local,
  20:13-20:30; no language-model LAMP job: LAMc+ = LAMc in all 48 cells); 0 failures. (B) test launched once on the
  box at 20:33 from the committed choices (4e96c0f; md5s printed by the box script equal the repository's): 3,970 jobs
  in four per-encoder runners of 12 workers, followed on the box by the (B) verdict and bits audit.
- (C) was computed on saved outputs (cpart, all seven encoders) and the sentence encoders' part files and float64
  re-derivations were computed; none of their numbers has been read.
- **Cut rule fixed now, mechanically, before reading anything:** (B) is reported in full if the box's (B) test, verdict
  and bits audit are back here by 21:30; otherwise (B) is cut in the registered order (NLMSC first, then VSAR - SDMT, then
  K = 1e3): whatever is not complete at 21:30 is reported as not run, and nothing about it is claimed. No other cut:
  (A), (C), (D) are complete for all seven encoders. VSAPcX - LAMc under (B) and (B) on the sentence encoders were cut
  in the pre-registration (before any run).
- From here on the (A), (C), (D) outputs are read while the (B) test runs; nothing read can change (B), whose choices are
  committed and whose jobs run unattended.
