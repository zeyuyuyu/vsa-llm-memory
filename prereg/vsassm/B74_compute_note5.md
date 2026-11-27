# B74 compute note 5 (2026-11-24, 00:23 machine clock; written before any language-model test output is read)

1. **Language-model test, state at 00:21:** the box run (PID 4151736, launched once at 23:42:06) has finished every
   FX test job (126 of 126, 0 failures); its label (LAB) and NL72 jobs are still running (about 970 of 1,393 jobs done,
   0 failures so far; the M = 256 label jobs of LAMc / NLMSc take 2-3.5 minutes each because B60's / B62's original test
   specifications include every budget from 1 to 64 bits per fact, kept so that the stored memories are bit-identical).
2. The 126 FX test job files were downloaded as one tarball (md5 7751c505... on both ends) and are read from here on
   for (A). The label and NL72 outputs are read only after the run ends (compute note 6). No choice, plan or bar is
   changed by reading (A): all tests were launched once, before any output existed.
3. Sentence encoders (read since compute note 4): determinism sample 24 / 24 jobs identical (fresh directory, 3 workers
   instead of 6); bits audit of their rows 0 / 0 / 0.
