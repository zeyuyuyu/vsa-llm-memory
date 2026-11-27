# B74 compute note 6 (2026-11-24, 00:39 machine clock; written before the language-model label and NL72 outputs are read)

1. The box test run (launched once at 23:42:06) ended at about 00:37: 1,393 of 1,393 jobs written, 0 failures (the
   training-norm files of compute note 4 were in place before any box LABN job started). Outputs (label engines'
   files, LABS / LABL / LABN / LABV / LABP / NL72 job files) downloaded as one tarball, md5 2b692ecb... on both ends.
2. Determinism sample for the language models (b74_check.py detplan: the first test job of every (encoder, job type),
   32 jobs) launched on the box at 00:38 in a fresh directory ~/b74_run/b74_det with 16 workers instead of 48 (PID
   4167102); compared after it ends.
3. No cut. Time used so far: 2 h 20 min of the 4-hour box (from 22:19). The verdict, checks, sample reading and note
   follow.
