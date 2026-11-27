# B74 compute note 2 (2026-11-23, 23:35 machine clock; written before any development or test run)

1. **Bitcoin attestation.** `ots upgrade` (polled every 10 minutes from 22:44) reported "Success! Timestamp complete" at
   23:34:27: BitcoinBlockHeaderAttestation(977225) (merkle root cbbeb278...). The upgraded proof is committed 91144cb
   (23:34). The stamp was made at 22:40, so the block arrived about 54 minutes later, before any B74 development or
   test run; the development runs start after this note.
2. **Box environment** (TRAINING-split stores only): b74_boxtest.py ran the FX arm (kinds P and T, V = 16, M = 256,
   4 bits per fact) for the four language models on ~/b74_run with the box's Python environment (the default python3 of
   a non-interactive ssh has no numpy; first attempt failed at import, recorded). All ran (setup about 70 s per encoder).
   Uploads verified by md5 (code tarball, b74_box.sh, plans). Two uploads were cut by the link ("lost connection") and
   repeated once; one short read-only status command (uptime / CPU by user) ran as a foreground ssh with a timeout.
3. **Placement.** Development: FX on the box for the four language models (84 jobs, 48 workers), FX locally for the three
   sentence encoders (68 jobs, 6 workers); NL72's dev choices (20 draws) computed locally from B62's existing dev rows.
   Tests after the choices are committed.
