# B66 compute note 4 (test runs of (A) and (B) finished; before their verdicts are computed or any of their test output is read, with one disclosed exception)

Date: 2026-11-23, about 14:06 (machine clock).

- **Test runs, each once, no failure:** SDMTCR 350 / 350 jobs (box, 13:46-13:53); SDMTm 3 x 350 jobs (box, 13:55-13:58);
  VSARm 21 (encoder, map) runs, 84 files (box, 13:57-14:01); LSfc 835 / 835 jobs (box, 13:57-14:02; 0 FAILED in the
  log); clean-room code test phase (b66_cr.py, local, started 13:58, both encoders, 12 conditions each; still running at this note: the A1 verdict is computed only after it ends, nothing else depends on it). (C)'s SDMT / LAMC test
  jobs are still running on the box (545 jobs); a separate note precedes reading them.
- **Disclosed exception:** b66_cr.py's test phase prints one line per condition with that condition's test accuracies
  (cr.py's own logging, kept). While checking progress at 14:01 I saw three such lines (Qwen3-8B, M = 1,024: V = 2 at 4
  bits per fact, V = 16 at 4 and at 8 bits per fact). All B66 choices had been committed before (9991d44 for the clean
  room); nothing was chosen or changed after it. Progress is now checked by counting lines only.
- **No cut** in (A), (B) or (D). (C): KF6 cut as in compute note 2.
- Order now: b66_verdict.py a1 / a2 / a2panel; b66_lsfc.py verdict; checks (b66_check.py all; b66_lsfc.py check); sample
  reading; then (C) after its own note.
