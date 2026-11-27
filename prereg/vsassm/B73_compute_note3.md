# B73 (equal bits) compute note 3 (about 20:55 machine clock; written before any test output is read)

- **Box link.** Between 20:36 and 20:55 the box answered small commands but stalled or refused larger transfers and
  some logins (connect time-outs). The code and choice files the box needs are there and MD5-identical to this
  repository: most copied from earlier run directories (~/b60_run, ~/b62_run, ~/b66_run, ~/b71_run, /path/to/work/b46),
  the rest from the part of an earlier upload that arrived (its complete members) and a 60 kB upload. The fresh entity
  file (xz, 220 kB) is being uploaded; the string list is rebuilt on the box by b32_embed.strings() and checked against
  this repository's MD5 (774a3a13...).
- **Cut rule, fixed now (mechanical, by the clock), before any test output is read.** The language-model part runs on
  the box in b73_run.py plan's order restricted to the four language models: core arms (LAMC, NLMSC, STORE, VSAS, VSAPC)
  for Qwen3-8B, Llama-3.1-8B, Mistral-7B, then Granite-3.3-8B, then VSAR, then SDMT. (1) If no language-model code has
  been encoded by 21:40, the language-model part is cut whole (C1-C4 then "not decided" on those encoders). (2) At 22:25
  any unit not finished is stopped by PID and counted as cut; an encoder or arm with a missing unit is reported "not
  run" for that arm (no partial cells). Because the cut depends only on the clock and the registered order, the
  sentence encoders' results (their test finished at 20:35, compute note 2) are read from now on; nothing they show can
  change what runs on the box.
- Determinism re-run (sentence encoders, done after compute note 2, before any output was read): 9 units (LAMC, NLMSC,
  STORE, VSAS, VSAPC, VSAR, SDMT) re-run into b73_out/run_det, 29 output files compared, predictions identical in 29
  (results/b73_check_same_sent.json). Bits and frozen-configuration audit of the sentence encoders' 3,150 test rows: 0
  over budget, 0 not maximal, 0 configuration mismatches (results/b73_check_bits_sent.json).
